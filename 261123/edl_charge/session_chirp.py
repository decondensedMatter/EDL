"""Calibrated, trigger-relative views of finite native Keysight sweeps."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from uuid import uuid4

from keysight_33511b_chirp_generator import configure_native, start_output
from keysight_33511b_chirp_plan import make_plan
from .measure import (AcquisitionDeadline, FG_ADDRESS, SCOPE_ADDRESS, _save,
                      _scope_setup, _shutdown, _validate, _wait)


def plan_windows(shape, *, leg_duration_s=1.0):
    plan = make_plan(shape, 1., leg_duration_s)
    windows = []
    for name, center in (("fast", 0.), ("mid", .5), ("slow", 1.)):
        t = center * leg_duration_s
        duration = min(.15 * leg_duration_s, 8 / plan.frequency_at(t))
        start = max(0., t - duration / 2)
        end = start + duration
        peak_hz = max(plan.frequency_at(start), plan.frequency_at(end))
        windows.append(dict(name=name, start_s=start, duration_s=duration,
                            sample_rate_hz=math.ceil(80 * peak_hz),
                            leg_duration_s=leg_duration_s))
    return windows


def _reset_zero(fg, seconds, deadline, sleep):
    fg.write("OUTP1 OFF")
    fg.write("ABOR")
    fg.write("SOUR1:APPL:DC DEF,DEF,0")
    if float(fg.query("SOUR1:VOLT:OFFS?")) != 0 or fg.query("SOUR1:FUNC?").strip().upper() != "DC":
        raise RuntimeError("Generator rejected zero DC reset")
    deadline.check(seconds)
    fg.write("OUTP1 ON")
    fg.query("*OPC?")
    deadline.sleep(seconds, sleep)
    _shutdown(fg, "OUTP1 OFF")


def acquire_chirp_window(shape, amplitude_v, window, hardware_settings, outputdir,
                         metadata, *, deadline_remaining_s=None, resource_manager_factory=None,
                         clock=time.monotonic, sleep=time.sleep):
    """amplitude_v is symmetric cell peak voltage. Each view repeats both round trips."""
    settings = dict(current_offset_v=0., voltage_offset_v=0., voltage_gain=1.,
                    voltage_scale_factor=1., voltage_tolerance_v=0., transfer_chunk_points=8192,
                    max_capture_points=100000, chirp_reset_s=1.)
    settings.update(hardware_settings)
    deadline = AcquisitionDeadline(deadline_remaining_s)
    deadline.check()
    start, duration, rate = _validate(None, settings, metadata, window)
    bound = float(metadata["voltage_limit_v"])
    if not math.isfinite(bound) or not 0 < amplitude_v <= bound:
        raise ValueError("Chirp requires an in-bound symmetric cell peak")
    plan = make_plan(shape, 2 * amplitude_v / settings["voltage_gain"], window["leg_duration_s"])
    if not .002 <= plan.voltage_vpp <= 20:
        raise ValueError("Calibrated generator Vpp exceeds the 33511B range")
    end = start + duration
    extrema = [plan.frequency_at(start), plan.frequency_at(end)]
    extrema.extend(plan.frequency_at(i * plan.leg_duration_s) for i in range(1, 4)
                   if start <= i * plan.leg_duration_s <= end)
    if rate < 80 * max(extrema) or math.ceil(duration * rate)+1 > settings["max_capture_points"]:
        raise ValueError("Chirp view exceeds point budget or undersamples its fastest cycle")
    measured_bw = float(settings["current_bandwidth_hz"])
    bandwidth = min(400e3,measured_bw) if settings["use_current_amp"] else measured_bw
    reset = float(settings["chirp_reset_s"])
    if not math.isfinite(reset) or reset <= 0:
        raise ValueError("Specify a positive chirp_reset_s")
    sample = re.sub(r"[^\w.-]+", "_", str(metadata["sample_name"])).strip(" ._") or "sample"
    now = datetime.now().astimezone()
    folder = Path(outputdir)/now.strftime("%Y%m%d")/sample/(
        f"chirp_{shape}_{uuid4().hex[:12]}")
    folder.mkdir(parents=True)
    stem = f"{now:%Y%m%d}_{sample}_chirp_{shape}_{window['name']}_{2*amplitude_v:g}Vpp_{now:%H%M%S_%f}"
    record = dict(protocol=dict(metadata), chirp=asdict(plan), capture_view=window,
                  commanded_cell_peak_v=amplitude_v, actual_reset_s=reset,
                  selected_frequency_min_hz=min(extrema), selected_frequency_max_hz=max(extrema),
                  protocol_duration_s=plan.total_duration_s, hardware_settings=settings,
                  capture_csv=stem+".csv", raw_channels_csv=stem+"_raw_channels.csv",
                  status="incomplete", started_utc=datetime.now(timezone.utc).isoformat(),
                  use_current_amp=settings["use_current_amp"],
                  effective_current_bandwidth_hz=bandwidth,
                  window_all_fundamentals_within_current_bandwidth=max(extrema)<=bandwidth,
                  complete_sweep_fundamentals_within_current_bandwidth=plan.f_max<=bandwidth,
                  bandwidth_flags_apply_to="fundamental frequency only; harmonics and path response require qualification",
                  partial_record=True, whole_protocol_energy_available=False)
    info = folder/"metadata.json"
    info.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if resource_manager_factory is None:
        import pyvisa
        resource_manager_factory = pyvisa.ResourceManager
    with ExitStack() as stack:
        manager = resource_manager_factory()
        stack.callback(manager.close)
        scope = deadline.open(manager, settings.get("oscope_address", SCOPE_ADDRESS), stack)
        fg = deadline.open(manager, settings.get("function_generator_address", FG_ADDRESS), stack)
        record.update(scope_id=scope.query("*IDN?").strip(), generator_id=fg.query("*IDN?").strip())
        for key, model, serial in (("scope_id", "DSA90604A", "MY50140103"),
                                   ("generator_id", "33511B", "MY57301577")):
            if record[key].split(",")[1:3] != [model, serial]:
                raise ValueError("Unexpected instrument identity")
        try:
            deadline.check(2 * reset + plan.total_duration_s)
            _reset_zero(fg, reset, deadline, sleep)
            configure_native(fg, plan, load_ohms="INF")
            accepted = _scope_setup(scope, settings, start, duration, rate)
            record["scope_sample_rate_hz"] = accepted
            scope.query(":AER?")
            scope.query(":ADER?")
            scope.write(":SINGLE")
            bounded_sleep = lambda seconds: deadline.sleep(seconds, sleep)
            _wait(scope, ":AER?", 15, clock, bounded_sleep)
            deadline.check(plan.total_duration_s + reset)
            start_output(fg, "native")
            # Wait from INIT acknowledgement before removing drive. The scope
            # retains its single acquisition independently of host transfer.
            deadline.sleep(plan.total_duration_s, sleep)
            _shutdown(fg, "OUTP1 OFF")
            _reset_zero(fg, reset, deadline, sleep)
            _wait(scope, ":ADER?", 10, clock, bounded_sleep)
            obs = record["observed"] = _save(scope, folder, settings, start, duration, accepted, record, deadline)
            ok = max(abs(obs["observed_min_v"]), abs(obs["observed_max_v"])) <= bound + settings["voltage_tolerance_v"]
            record.update(voltage_bound_ok=ok, voltage_bound_applies_to="observed view only",
                          status="captured" if ok else "bound_exceeded")
        finally:
            try:
                _shutdown(fg, "OUTP1 OFF")
            finally:
                info.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return folder
