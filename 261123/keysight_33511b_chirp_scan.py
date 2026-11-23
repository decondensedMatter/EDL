"""Shared acquisition for the two continuous-chirp entry scripts.

The original regular-waveform and PUND scripts are not modified. Run an entry
script with --preview to inspect its complete plan without importing PyVISA.
"""

import argparse
import csv
from dataclasses import dataclass
import datetime
import math
import os
import time

from keysight_33511b_chirp_plan import CHIRP_VPP_LEVELS, WINDOW_WIDTHS_V, WINDOW_MARGINS_V, WINDOW_NOTES, make_plan, voltage_levels
from keysight_33511b_chirp_generator import (
    arb_point_count, configure_arb, configure_native, release_arb, start_output,
)


@dataclass
class ScanSettings:
    sample_name: str = "0.1M NaClO4"
    device_name: str = "Xin_0p1M_NaClO4_electrolyte"
    sweep_mode: str = "native"
    waveform_types: tuple = ("sine", "square", "triangle")
    native_leg_duration_s: float = 1.0
    arb_leg_duration_s: float = 1.0
    arb_sample_rate_hz: float = 160e6
    arb_max_points: int = 1_000_000
    min_frequency_hz: float = 10.0
    frequency_limit_fraction: float = 0.1
    sweep_cycles: int = 2
    voltage_margin_v: float | None = None
    generator_load_ohms: str = "INF"
    use_current_amp: bool = True
    current_preamplifier_gain_a_per_v: float = 1e-3
    electrode_area_m2: float | None = (35.2e-6)**2
    voltage_scale_factor: float = 1.0
    max_allowed_amplitude_v: float = 4.0
    target_time_step_s: float = 1e-8
    function_generator_address: str = "USB0::0x0957::0x2707::MY57301577::INSTR"
    oscope_address: str = "USB0::0x0957::0x9005::MY50140103::INSTR"
    base_save_dir: str = r"D:\Ferroelectric_BTO_optimization"
    capture_margin_s: float = 0.01
    transfer_chunk_points: int = 100_000


def sample_voltage_levels(settings):
    margin = settings.voltage_margin_v
    if margin is None:
        margin = WINDOW_MARGINS_V[settings.sample_name]
    cap = voltage_levels(WINDOW_WIDTHS_V[settings.sample_name], margin)[-1]
    levels = CHIRP_VPP_LEVELS[settings.sample_name]
    # A manually tightened margin can restrict the prescribed schedule.
    cap = min(cap, levels[-1])
    return [level for level in levels if level < cap] + [cap]


def scan_plans(settings):
    if settings.sample_name not in WINDOW_WIDTHS_V:
        raise ValueError("Select one sample_name from " + ", ".join(WINDOW_WIDTHS_V))
    if settings.sweep_mode not in {"native", "arb"}:
        raise ValueError("sweep_mode must be native or arb")
    duration = (settings.native_leg_duration_s if settings.sweep_mode == "native"
                else settings.arb_leg_duration_s)
    # Finish every shape at this voltage before increasing the cell excursion.
    return [make_plan(shape, vpp, duration, f_min=settings.min_frequency_hz,
                      max_fraction=settings.frequency_limit_fraction,
                      cycles=settings.sweep_cycles, mode=settings.sweep_mode)
            for vpp in sample_voltage_levels(settings)
            for shape in settings.waveform_types]


def print_plan(settings, plans):
    print(f"Sample: {settings.sample_name}; mode: {settings.sweep_mode}")
    print(WINDOW_NOTES[settings.sample_name])
    print("Two sample electrodes; CH1=current, CH2=cell voltage, CH3=generator SYNC.")
    print(f"Current amplifier flag: {settings.use_current_amp}; conversion retained: "
          f"{settings.current_preamplifier_gain_a_per_v:g} A/V "
          f"x {2 if settings.use_current_amp else 1}")
    print("Voltage levels (Vpp):", sample_voltage_levels(settings))
    for shape in settings.waveform_types:
        plan = next(p for p in plans if p.waveform == shape)
        print(f"{shape}: {plan.f_max:g} -> {plan.f_min:g} -> {plan.f_max:g} Hz, "
              f"{plan.cycles} round trips; {plan.leg_duration_s:g} s/leg; "
              f"{plan.total_duration_s:g} s total")
        capture_s = plan.total_duration_s + settings.capture_margin_s
        print(f"  Scope request: {math.ceil(capture_s / settings.target_time_step_s) + 1:,} "
              f"points/channel at {settings.target_time_step_s:g} s/point; averaging OFF")
        if settings.sweep_mode == "arb":
            try:
                count = arb_point_count(plan, settings.arb_sample_rate_hz, settings.arb_max_points)
                print(f"  ARB: {count:,} stored samples")
            except ValueError as error:
                print(f"  ARB DOES NOT FIT: {error}")
    print(f"{len(plans)} measurements for this sample. Change sample_name/device_name for the next sample.")


def validate_settings(settings, plans):
    for name in ("target_time_step_s", "electrode_area_m2", "max_allowed_amplitude_v"):
        value = getattr(settings, name)
        if name == "electrode_area_m2" and value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be positive and finite")
    if not math.isfinite(settings.capture_margin_s) or settings.capture_margin_s < 0:
        raise ValueError("capture_margin_s must be non-negative and finite")
    if type(settings.transfer_chunk_points) is not int or settings.transfer_chunk_points < 1:
        raise ValueError("transfer_chunk_points must be a positive integer")
    for plan in plans:
        if plan.voltage_vpp / 2 > settings.max_allowed_amplitude_v:
            raise ValueError("Requested peak voltage exceeds max_allowed_amplitude_v")
        if settings.sweep_mode == "arb":
            arb_point_count(plan, settings.arb_sample_rate_hz, settings.arb_max_points)


def check_scope_errors(scope, stage):
    errors = []
    for _ in range(32):
        reply = scope.query(":SYSTem:ERRor?").strip()
        if int(reply.split(",", 1)[0]) == 0:
            break
        errors.append(reply)
    if errors:
        raise RuntimeError(stage + ": " + " | ".join(errors))


def configure_scope(scope, plan, settings):
    """Retain the original channels/scales/HRES; capture the entire finite chirp."""
    duration = plan.total_duration_s + settings.capture_margin_s
    requested_points = math.ceil(duration / settings.target_time_step_s) + 1
    scope.write("*RST")
    scope.write("*CLS")
    scope.write(":SYSTEM:HEADER OFF")
    scope.query(":STOP;*OPC?")
    for channel in (1, 2, 3):
        scope.write(f":CHANNEL{channel}:DISPLAY ON")
    scope.write(f":TIMEBASE:SCALE {duration / 10.0}")
    scope.write(":TIMEBASE:REFERENCE LEFT")
    scope.write(":TIMEBASE:POSITION 0")
    scope.write(":CHANNEL1:SCALE 0.1")
    scope.write(":CHANNEL2:SCALE 1")
    scope.write(":CHANNEL3:SCALE 1")
    for channel in (1, 2, 3):
        scope.write(f":CHANNEL{channel}:OFFSET 0")
    scope.write(":WAVEFORM:FORMAT ASCII")
    scope.write(":ACQUIRE:SRATE:AUTO OFF")
    scope.write(f":ACQUIRE:SRATE {1.0 / settings.target_time_step_s}")
    scope.write(f":ACQUIRE:POINTS {requested_points}")
    scope.write(":ACQUIRE:MODE HRESOLUTION")
    # 256 averages would need 256 repeated acquisitions, not two sweep cycles.
    scope.write(":ACQUIRE:AVERAGE OFF")
    scope.write(":TRIGGER:MODE EDGE")
    scope.write(":TRIGGER:EDGE:SOURCE CHANNEL3")
    scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
    scope.write(":TRIGGER:HYSTERESIS NREJECT")
    scope.write(":TRIGGER:SWEEP TRIGGERED")
    scope.write(":TRIGGER:LEVEL CHANNEL3,0.2")
    check_scope_errors(scope, "Scope configuration")
    actual_rate = float(scope.query(":ACQUIRE:SRATE?"))
    actual_points = int(float(scope.query(":ACQUIRE:POINTS?")))
    if not math.isclose(actual_rate, 1 / settings.target_time_step_s, rel_tol=0.02):
        raise RuntimeError(f"Scope accepted {actual_rate:g} Sa/s, not requested "
                           f"{1 / settings.target_time_step_s:g} Sa/s")
    if actual_points < math.ceil(duration * actual_rate):
        raise RuntimeError(f"Scope accepted {actual_points:,} acquisition points; "
                           f"this chirp needs {math.ceil(duration * actual_rate):,}. "
                           "Shorten the sweep or explicitly change target_time_step_s; "
                           "the script will not silently reduce sampling.")
    # Post-acquisition metadata is also checked; these are configuration readbacks.
    return duration


def wait_for_scope(scope, query, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if int(float(scope.query(query))) & 1:
            return
        time.sleep(0.01)
    raise TimeoutError(f"Scope did not signal {query} within {timeout_s:g} s")


def acquire(scope, fg, plan, settings):
    try:
        configure_scope(scope, plan, settings)
        scope.query(":AER?")
        scope.query(":ADER?")
        scope.write(":SINGLE")
        wait_for_scope(scope, ":AER?", 10.0)
        start_output(fg, plan.mode)
        wait_for_scope(scope, ":ADER?",
                       plan.total_duration_s + settings.capture_margin_s + 30.0)
    finally:
        fg.write("OUTP1 OFF")


def waveform_metadata(scope, channel):
    scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
    scope.write(":WAVEFORM:FORMAT ASCII")
    dt = float(scope.query(":WAVEFORM:XINCREMENT?"))
    origin = float(scope.query(":WAVEFORM:XORIGIN?"))
    points = int(float(scope.query(":WAVEFORM:POINTS?")))
    return dt, origin, points


def read_chunk(scope, channel, start, count):
    scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
    raw = scope.query(f":WAVEFORM:DATA? {start},{count}").strip()
    values = [float(item) for item in raw.split(",") if item.strip()]
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"CH{channel}: invalid/incomplete waveform chunk at point {start}")
    return values


def save_capture(scope, plan, settings, run_date):
    meta1 = waveform_metadata(scope, 1)
    meta2 = waveform_metadata(scope, 2)
    dt, origin, points = meta1
    if points < 2 or dt <= 0 or not all(math.isfinite(x) for x in (dt, origin)):
        raise RuntimeError("Scope returned invalid waveform metadata")
    if (meta2[2] != points or not math.isclose(meta2[0], dt, rel_tol=1e-9)
            or not math.isclose(meta2[1], origin, abs_tol=dt * 0.01)):
        raise RuntimeError("CH1/CH2 waveform time axes do not match")
    if not math.isclose(dt, settings.target_time_step_s, rel_tol=0.02, abs_tol=1e-12):
        raise RuntimeError(f"Scope spacing {dt:g} differs from requested {settings.target_time_step_s:g} s")
    if origin > 2 * dt or origin + (points - 1) * dt < plan.total_duration_s - 2 * dt:
        raise RuntimeError("Scope record does not cover both complete chirp round trips")
    mode_tag = "amp" if settings.use_current_amp else "noamp"
    folder = os.path.join(settings.base_save_dir, run_date,
                          f"{settings.device_name}_{plan.mode}_{plan.waveform}_{mode_tag}")
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    area_tag = "area_unmeasured" if settings.electrode_area_m2 is None else f"area{settings.electrode_area_m2:.6g}m2"
    filename = (f"{settings.device_name}_{plan.mode}_{plan.waveform}_"
                f"f{plan.f_min:g}-{plan.f_max:g}Hz_vpp{plan.voltage_vpp:g}V_"
                f"leg{plan.leg_duration_s:g}s_cycles{plan.cycles}_"
                f"{area_tag}_"
                f"gain{settings.current_preamplifier_gain_a_per_v:g}AperV_{stamp}.csv")
    path = os.path.abspath(os.path.join(folder, filename))
    disk_path = ("\\\\?\\" + path) if os.name == "nt" else path
    partial_path = disk_path + ".partial"
    with open(partial_path, "x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (s)", "Current (A)", "Voltage (V)"])
        gain = settings.current_preamplifier_gain_a_per_v * (2 if settings.use_current_amp else 1)
        for start in range(0, points, settings.transfer_chunk_points):
            count = min(settings.transfer_chunk_points, points - start)
            raw_current = read_chunk(scope, 1, start, count)
            raw_voltage = read_chunk(scope, 2, start, count)
            writer.writerows((origin + (start + i) * dt, current * gain,
                              voltage * settings.voltage_scale_factor)
                             for i, (current, voltage) in enumerate(zip(raw_current, raw_voltage)))
    os.replace(partial_path, disk_path)
    print("Data saved to", path)
    return path


def run_scan(settings):
    plans = scan_plans(settings)
    print_plan(settings, plans)
    validate_settings(settings, plans)
    import pyvisa
    run_date = datetime.date.today().strftime("%Y%m%d")
    rm = pyvisa.ResourceManager()
    fg = scope = None
    try:
        fg = rm.open_resource(settings.function_generator_address)
        fg.write_termination = fg.read_termination = "\n"
        fg.timeout = 100000
        scope = rm.open_resource(settings.oscope_address)
        scope.write_termination = scope.read_termination = "\n"
        scope.timeout = 100000
        print("Oscilloscope ID:", scope.query("*IDN?"))
        for index, plan in enumerate(plans, 1):
            print(f"\nMeasurement {index}/{len(plans)}: {plan.waveform}, {plan.voltage_vpp:g} Vpp")
            owned_arb = None
            try:
                if plan.mode == "native":
                    configure_native(fg, plan, settings.generator_load_ohms)
                else:
                    owned_arb = configure_arb(fg, plan, settings.arb_sample_rate_hz,
                                              settings.arb_max_points, settings.generator_load_ohms)
                acquire(scope, fg, plan, settings)
                save_capture(scope, plan, settings, run_date)
                scope.write(":RUN")
            finally:
                fg.write("OUTP1 OFF")
                if owned_arb is not None:
                    fg.write("ABOR")
                    release_arb(fg, owned_arb)
    finally:
        try:
            if fg is not None:
                fg.write("OUTP1 OFF")
        finally:
            for instrument in (scope, fg):
                if instrument is not None:
                    try:
                        instrument.close()
                    except Exception:
                        pass
            rm.close()


def main(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Print plan without connecting instruments")
    parser.add_argument("--mode", choices=("native", "arb"), help="Override sweep_mode for this run")
    args = parser.parse_args()
    if args.mode:
        settings.sweep_mode = args.mode
    if args.preview:
        print_plan(settings, scan_plans(settings))
    else:
        run_scan(settings)
