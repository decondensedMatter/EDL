"""One triggered 33511B protocol and one calibrated Infiniium acquisition view."""

import csv
import json
import math
import re
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from keysight_33511b_chirp_scan import check_scope_errors, read_chunk, waveform_metadata
from keysight_33511b_chirp_generator import _arb_catalog, _contains_only_default
from .awg import configure_sequence, start_output


FG_ADDRESS = "USB0::0x0957::0x2707::MY57301577::INSTR"
SCOPE_ADDRESS = "USB0::0x0957::0x9005::MY50140103::INSTR"


class AcquisitionDeadline:
    """Bound ordinary VISA calls and leave time to turn the output off."""
    def __init__(self, remaining=None, reserve_s=3.0):
        self.remaining = remaining
        self.reserve_s = reserve_s

    def check(self, required=0.0):
        available = math.inf if self.remaining is None else self.remaining() - self.reserve_s
        if available <= required or math.isnan(available):
            raise TimeoutError("Acquisition deadline reached")
        return available

    def open(self, manager, address, stack):
        if self.remaining is None:
            device = manager.open_resource(address)
        else:
            device = manager.open_resource(address, open_timeout=max(1, int(1000 * min(10, self.check()))))
        stack.callback(device.close)
        device.write_termination = device.read_termination = "\n"
        device.timeout = 100000
        return _TimedDevice(device, self) if self.remaining else device

    def sleep(self, duration, sleep):
        self.check(duration)
        sleep(duration)


class _TimedDevice:
    def __init__(self, device, deadline):
        self.device = device
        self.deadline = deadline

    def __getattr__(self, name):
        method = getattr(self.device, name)
        if not callable(method):
            return method
        def call(*args, **kwargs):
            self.device.timeout = max(1, int(1000 * min(100, self.deadline.check())))
            return method(*args, **kwargs)
        return call


def _shutdown(device, command):
    if isinstance(device, _TimedDevice):
        device = device.device
        device.timeout = 2000
    device.write(command)


def _number(obj, name, *, positive=False):
    value = float(obj[name])
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid {name}")
    return value


def _validate(p, settings, metadata, view):
    if p is not None and metadata.get("status") != "validated":
        raise ValueError("Hardware output requires an independently replayed, validated protocol")
    if p is not None:
        bound = _number(metadata, "voltage_limit_v", positive=True)
        if max(abs(p.voltage_min), abs(p.voltage_max), p.voltage_peak) > bound or not p.duration_s > 0:
            raise ValueError("Compiled sequence exceeds the protocol window")
    if settings.get("calibration_mode") == "sample_only_nominal" and "voltage_limit_v" in metadata:
        if _number(metadata,"voltage_limit_v",positive=True) > _number(settings,"specimen_peak_cap_v",positive=True):
            raise ValueError("Protocol voltage limit exceeds the specimen peak cap")
    tol = _number(settings, "voltage_tolerance_v")
    if tol < 0:
        raise ValueError("voltage_tolerance_v must be nonnegative")
    for key in ("current_gain_a_per_v", "current_bandwidth_hz", "ch1_scale_v_per_div",
                "ch2_scale_v_per_div", "voltage_gain", "voltage_scale_factor"):
        _number(settings, key, positive=True)
    for key in ("current_offset_v", "voltage_offset_v"):
        _number(settings, key)
    if settings["current_sign"] not in (-1, 1) or type(settings["use_current_amp"]) is not bool:
        raise ValueError("Specify current_sign ±1 and boolean use_current_amp")
    if settings["ch1_input"] != "DC50":
        raise ValueError("DSA90604A inputs are DC-coupled 50 ohms; calibrate the complete front end under that load")
    if not str(settings.get("voltage_measurement_location") or "").strip():
        raise ValueError("Specify the CH2 voltage measurement location")
    start = _number(view, "start_s")
    duration = _number(view, "duration_s", positive=True)
    rate = _number(view, "sample_rate_hz", positive=True)
    if rate > 20e9:
        raise ValueError("Requested rate exceeds the 20 GSa/s DSA90604A limit")
    if not view.get("name") or start + duration <= 0 or (p is not None and start >= p.duration_s):
        raise ValueError("Capture view must intersect the finite protocol")
    if int(settings["transfer_chunk_points"]) != settings["transfer_chunk_points"] or settings["transfer_chunk_points"] < 1:
        raise ValueError("Invalid transfer_chunk_points")
    return start, duration, rate


def observed_voltage_limits(settings, low, high):
    tolerance = settings["voltage_tolerance_v"]
    limits = [low-tolerance, high+tolerance]
    if settings.get("calibration_mode") == "sample_only_nominal":
        cap = _number(settings,"specimen_peak_cap_v",positive=True)
        limits = [max(-cap,limits[0]),min(cap,limits[1])]
    return limits


def _scope_setup(scope, settings, start, duration, rate, passive=False):
    scope.write(":SYSTEM:HEADER OFF")
    scope.query(":STOP;*OPC?")
    # Reset inherited probe, averaging and trigger settings before reapplying calibration.
    scope.write("*RST")
    scope.write(":SYSTEM:HEADER OFF")
    scope.query(":STOP;*OPC?")
    for channel, scale, coupling in ((1, settings["ch1_scale_v_per_div"], settings["ch1_input"]),
                                     (2, settings["ch2_scale_v_per_div"], "DC50"), (3, 1.0, "DC50")):
        scope.write(f":CHANNEL{channel}:DISPLAY ON")
        scope.write(f":CHANNEL{channel}:INPUT {coupling}")
        scope.write(f":CHANNEL{channel}:SCALE {scale}")
        scope.write(f":CHANNEL{channel}:OFFSET 0")
    for cmd in (":ACQUIRE:MODE RTIME", ":ACQUIRE:AVERAGE OFF", ":TIMEBASE:REFERENCE LEFT",
                f":TIMEBASE:RANGE {duration:.15g}", f":TIMEBASE:POSITION {start:.15g}",
                ":ACQUIRE:SRATE:AUTO OFF",
                ":WAVEFORM:FORMAT ASCII", ":TRIGGER:MODE EDGE", ":TRIGGER:EDGE:SOURCE CHANNEL3",
                ":TRIGGER:EDGE:SLOPE POSITIVE", ":TRIGGER:SWEEP AUTO" if passive else ":TRIGGER:SWEEP TRIGGERED",
                ":TRIGGER:LEVEL CHANNEL3,0.2"):
        scope.write(cmd)
    check_scope_errors(scope, "Scope configuration")
    requested = rate
    while True:
        scope.write(f":ACQUIRE:SRATE {requested:.15g}")
        accepted = float(scope.query(":ACQUIRE:SRATE?"))
        if math.isfinite(accepted) and accepted >= rate*(1-1e-10):
            break
        if requested >= 20e9 or not math.isfinite(accepted) or accepted <= 0:
            raise RuntimeError("Scope cannot meet minimum requested sample rate")
        requested = min(20e9, requested*2)
    points = math.ceil(duration * accepted) + 1
    if "max_capture_points" in settings and points > settings["max_capture_points"]:
        raise ValueError("Scope depth exceeds session point limit")
    scope.write(f":ACQUIRE:POINTS {points}")
    check_scope_errors(scope, "Scope acquisition rate/depth")
    actual_points = int(float(scope.query(":ACQUIRE:POINTS?")))
    if "max_capture_points" in settings and actual_points > settings["max_capture_points"]:
        raise ValueError("Scope accepted record length exceeds session point limit")
    if (not math.isclose(float(scope.query(":ACQUIRE:SRATE?")), accepted, rel_tol=1e-10)
            or actual_points < points):
        raise RuntimeError("Scope cannot capture the requested view at the requested rate and depth")
    if abs(float(scope.query(":TIMEBASE:POSITION?")) - start) > 1/rate:
        raise RuntimeError("Scope rejected the specified delayed capture position")
    return accepted


def _wait(scope, event, timeout, clock, sleep):
    deadline = clock() + timeout
    while not int(float(scope.query(event))) & 1:
        if clock() >= deadline:
            raise TimeoutError(f"Scope timed out waiting for {event}")
        sleep(.01)


def _save(scope, folder, settings, start, duration, rate, record, deadline=None):
    axes = [waveform_metadata(scope, ch) for ch in (1, 2, 3)]
    dt, t0, count = axes[0]
    if not math.isfinite(t0) or not math.isclose(dt, 1/rate, rel_tol=1e-6) or count < 2:
        raise RuntimeError("Invalid post-acquisition time axis")
    for dx, origin, n in axes[1:]:
        if n != count or not math.isclose(dx, dt, rel_tol=1e-10) or abs(origin-t0) > .01*dt:
            raise RuntimeError("CH1/CH2/CH3 time axes disagree")
    if "max_capture_points" in settings and count > settings["max_capture_points"]:
        raise RuntimeError("Returned record exceeds the session point limit")
    last = t0 + (count-1)*dt
    if t0 > start + dt or last < start + duration - dt:
        raise RuntimeError("Returned waveform does not span the requested trigger-relative view")
    high, low = -math.inf, math.inf
    current_gain = settings["current_gain_a_per_v"] * settings["current_sign"]
    partial = folder / (record["capture_csv"]+".partial")
    raw_partial = folder / (record["raw_channels_csv"]+".partial")
    with partial.open("x", newline="") as handle, raw_partial.open("x", newline="") as raw_handle:
        writer = csv.writer(handle)
        raw_writer = csv.writer(raw_handle)
        writer.writerow(["Time (s)", "Current (A)", "Voltage (V)", "Sync (V)"])
        raw_writer.writerow(["Time (s)", "raw_CH1_V", "raw_CH2_V", "Sync (V)"])
        for index in range(0, count, settings["transfer_chunk_points"]):
            n = min(count-index, settings["transfer_chunk_points"])
            current, voltage, sync = (read_chunk(scope, ch, index, n) for ch in (1, 2, 3))
            for j, (i, v, marker) in enumerate(zip(current, voltage, sync)):
                if deadline is not None and j % 1024 == 0:
                    deadline.check()
                timestamp = t0+(index+j)*dt
                if not all(map(math.isfinite, (i, v, marker))):
                    raise RuntimeError("Nonfinite value in acquired scope data")
                raw_writer.writerow((timestamp, i, v, marker))
                v = (v-settings["voltage_offset_v"])*settings["voltage_scale_factor"]
                high, low = max(high, v), min(low, v)
                writer.writerow((timestamp, (i-settings["current_offset_v"])*current_gain, v, marker))
    raw_partial.rename(folder/record["raw_channels_csv"])
    partial.rename(folder/record["capture_csv"])
    return dict(start_s=t0, end_s=last, sample_interval_s=dt, points=count,
                observed_max_v=high, observed_min_v=low)


def _release(fg, name, compiled):
    if not name or float(fg.query("OUTP?")) != 0:
        return False
    prefix = name[:-1].upper()
    assets = {prefix+str(i) for i in range(len({s.name for s in compiled.segments}))}
    owned = assets | {x+".ARB" for x in assets} | {name.upper(), name.upper()+".SEQ"}
    entries = _arb_catalog(fg)
    if not entries.intersection(owned) or not _contains_only_default(entries-owned):
        return False
    fg.write("FUNC SIN")
    fg.write("DATA:VOL:CLE")
    return True


def acquire_protocol(compiled, hardware_settings, outputdir, metadata, capture_view, *,
                     resource_manager_factory=None, clock=time.monotonic, sleep=time.sleep,
                     deadline_remaining_s=None, _zero_duration_s=None):
    """Return a fresh run directory. Each call performs the *whole* protocol once."""
    deadline = AcquisitionDeadline(deadline_remaining_s)
    deadline.check()
    original_sleep = sleep
    def sleep(duration):
        deadline.sleep(duration, original_sleep)
    settings = dict(voltage_gain=1., voltage_scale_factor=1., current_offset_v=0.,
                    voltage_offset_v=0., voltage_tolerance_v=0., transfer_chunk_points=100000)
    settings.update(hardware_settings)
    start, duration, rate = _validate(compiled, settings, metadata, capture_view)
    local = datetime.now().astimezone()
    day, hour = local.strftime("%Y%m%d"), local.strftime("%H%M%S_%f")
    def tag(value):
        value = re.sub(r'[^\w.-]+', '_', str(value)).strip(" ._")[:60]
        return value or "unspecified"
    sample = tag(metadata.get("sample_name", "sample"))
    test = tag(metadata.get("test_name", capture_view["name"]))
    folder = Path(outputdir) / day / sample / (test+"_"+hour+"_"+uuid4().hex[:8])
    folder.mkdir(parents=True)
    stem = "_".join((day, sample, test, hour))
    record = dict(protocol=metadata, capture_view=capture_view, hardware_settings=settings,
                  started_utc=datetime.now(timezone.utc).isoformat(), status="incomplete",
                  capture_csv=stem+".csv", raw_channels_csv=stem+"_raw_channels.csv")
    if settings.get("calibration_mode") == "sample_only_nominal":
        record.update(calibration_mode="sample_only_nominal", data_hardware_calibrated=False,
                      conversion_basis="Nominal channel gains, offsets and response; connected specimen")
    if compiled is not None:
        record.update(actual_charge_s=compiled.actual_charge_s, actual_hold_s=compiled.actual_hold_s,
                      actual_discharge_s=compiled.actual_discharge_s, actual_reset_s=compiled.actual_reset_s,
                      protocol_duration_s=compiled.duration_s, generator_sample_rate_hz=compiled.sample_rate_hz)
    info = folder/"metadata.json"
    info.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if resource_manager_factory is None:
        import pyvisa
        resource_manager_factory = pyvisa.ResourceManager
    with ExitStack() as stack:
        manager = resource_manager_factory()
        stack.callback(manager.close)
        def connect(address):
            return deadline.open(manager, address, stack)
        scope = connect(settings.get("oscope_address", SCOPE_ADDRESS))
        scope_id = scope.query("*IDN?").strip()
        if "DSA90604A" not in scope_id.split(",") or "MY50140103" not in scope_id.split(","):
            raise ValueError("Instrument identity differs from the specified experimental setup")
        record.update(scope_id=scope_id, scope_nominal_analog_bandwidth_hz=6e9)
        if compiled is None:
            accepted = _scope_setup(scope, settings, start, duration, rate, passive=True)
            record.update(scope_sample_rate_hz=accepted, generator_connected=False)
            fg = None
            if _zero_duration_s is not None:
                fg = connect(settings.get("function_generator_address", FG_ADDRESS))
                fg_id = fg.query("*IDN?").strip()
                if fg_id.split(",")[1:3] != ["33511B", "MY57301577"]:
                    raise ValueError("Unexpected generator identification")
                record.update(generator_connected=True, generator_id=fg_id,
                              zero_reset_duration_s=_zero_duration_s)
            try:
                if fg is not None:
                    fg.write("OUTP1 OFF")
                    fg.write("ABOR")
                    fg.write("OUTP1:LOAD INF")
                    fg.write("SOUR1:APPL:DC DEF,DEF,0")
                    if fg.query("SOUR1:FUNC?").strip().upper() != "DC" or float(fg.query("SOUR1:VOLT:OFFS?")) != 0:
                        raise RuntimeError("Generator rejected zero DC reset")
                    deadline.check(_zero_duration_s + duration)
                    fg.write("OUTP1 ON")
                    fg.query("*OPC?")
                    sleep(_zero_duration_s)
                scope.query(":ADER?")
                scope.write(":RUN")
                try:
                    _wait(scope, ":ADER?", max(0,start+duration)+30, clock, sleep)
                finally:
                    if deadline_remaining_s is None:
                        scope.query(":STOP;*OPC?")
                    else:
                        _shutdown(scope, ":STOP")
            finally:
                if fg is not None:
                    _shutdown(fg, "OUTP1 OFF")
            record["observed"] = _save(scope, folder, settings, start, duration, accepted, record, deadline)
            record["status"] = "captured"
            info.write_text(json.dumps(record, indent=2), encoding="utf-8")
            return folder
        fg = connect(settings.get("function_generator_address", FG_ADDRESS))
        fg_id = fg.query("*IDN?").strip()
        if "33511B" not in fg_id.split(",") or "MY57301577" not in fg_id.split(","):
            raise ValueError("Unexpected generator identification")
        record["generator_id"] = fg_id
        name = None
        try:
            name = configure_sequence(fg, compiled, metadata["voltage_limit_v"], settings["voltage_gain"])
            accepted = _scope_setup(scope, settings, start, duration, rate)
            record["scope_sample_rate_hz"] = accepted
            # Play the idle zero before the trigger, also on the very first run.
            deadline.check(compiled.actual_reset_s + compiled.duration_s + max(0, start+duration))
            fg.write("OUTP ON")
            fg.query("*OPC?")
            sleep(compiled.actual_reset_s)
            scope.query(":AER?")
            scope.query(":ADER?")
            scope.write(":SINGLE")
            _wait(scope, ":AER?", 15, clock, sleep)
            deadline.check(compiled.duration_s + max(0, start+duration))
            start_output(fg)
            _wait(scope, ":ADER?", max(0, start+duration)+max(30, 2048/compiled.sample_rate_hz), clock, sleep)
            # ADER proves the physical marker occurred, even with an unknown
            # trigger-to-first-sample delay in a repeatTillTrig idle segment.
            # Conservatively keep output active for a full protocol after ADER.
            finish = clock() + compiled.duration_s
            try:
                obs = record["observed"] = _save(scope, folder, settings, start, duration, accepted, record, deadline)
            finally:
                while (remaining := finish-clock()) > 0:
                    sleep(min(remaining, 1))
            bound = metadata["voltage_limit_v"]
            low = -bound if compiled.voltage_min < 0 else 0.
            high = bound if compiled.voltage_max > 0 else 0.
            limits = observed_voltage_limits(settings,low,high)
            ok = limits[0] <= obs["observed_min_v"] and obs["observed_max_v"] <= limits[1]
            record.update(voltage_bound_ok=ok, voltage_bound_applies_to="observed view only",
                          voltage_bound_v=[low, high], observed_acceptance_limits_v=limits,
                          status="captured" if ok else "bound_exceeded")
        finally:
            try:
                _shutdown(fg, "OUTP OFF")
                record["owned_memory_released"] = _release(fg, name, compiled)
            finally:
                info.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return folder


def acquire_baseline(hardware_settings, outputdir, metadata, capture_view, **options):
    """Capture passive OCV/zero by scope auto-trigger. The AWG is never opened."""
    test = str(metadata.get("test_name", "")).lower()
    if not test:
        raise ValueError("Passive acquisition requires a test_name")
    if ("ocv" in test or "open" in test) and hardware_settings.get("ocv_buffer_verified") is not True:
        raise ValueError("OCV on a 50-ohm scope requires a verified high-input-impedance voltage buffer/probe")
    return acquire_protocol(None, hardware_settings, outputdir, metadata, capture_view, **options)


def acquire_zero_reset(settings, outputdir, metadata, duration_s, *, capture_view=None, **options):
    """Hold zero DC for duration_s, then measure the baseline while still driving zero."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Specify a positive zero-reset duration")
    if capture_view is None:
        capture_view = dict(name="driven_zero", start_s=0., duration_s=.05, sample_rate_hz=20000.)
    return acquire_protocol(None, settings, outputdir, metadata, capture_view,
                            _zero_duration_s=duration_s, **options)
