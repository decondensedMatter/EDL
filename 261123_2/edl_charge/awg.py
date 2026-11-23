"""Compile positive, piecewise-linear device-voltage protocols for the 33511B."""

from dataclasses import dataclass, replace
from math import ceil, isfinite
from uuid import uuid4

import numpy as np


MAX_RATE = 160_000_000
MAX_REPEAT = 1_000_000


@dataclass(frozen=True)
class Segment:
    name: str
    samples_volts: np.ndarray
    repeat_count: int = 1
    control: str = "once"
    marker: str = "maintain"


@dataclass(frozen=True)
class CompiledSequence:
    segments: tuple
    sample_rate_hz: float
    voltage_peak: float
    actual_charge_s: float
    actual_hold_s: float
    actual_discharge_s: float
    actual_reset_s: float
    unique_sample_points: int
    allocated_points: int
    charge_marker_time_s: float = 0.0
    voltage_min: float = 0.0
    voltage_max: float = 0.0

    @property
    def duration_s(self):
        return self.actual_charge_s + self.actual_hold_s + self.actual_discharge_s + self.actual_reset_s

    def control_knots(self):
        return compiled_control(self)


def _curve(times, volts, bound, signed):
    t, v = np.asarray(times, dtype=float), np.asarray(volts, dtype=float)
    if t.ndim != 1 or v.shape != t.shape or t.size < 2:
        raise ValueError("Supply matching 1-D times and voltages, including both endpoints")
    if not np.all(np.isfinite(t)) or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Curve times must increase strictly from zero")
    if not np.all(np.isfinite(v)) or np.any(v < (-bound if signed else 0)) or np.any(v > bound):
        raise ValueError("Voltage outside the specified window")
    return t, v


def _ticks(seconds, rate):
    # Suppress roundoff in products which are exact integer ticks mathematically.
    return int(ceil(np.nextafter(seconds * rate, -np.inf)))


def compile_protocol(charge_times_s, charge_voltages_v, hold_s=0.0, discharge_times_s=None,
                     discharge_voltages_v=None, reset_s=0.0, *, voltage_limit_v,
                     sample_rate_hz=None, max_points=1_000_000, nominal_step_rise=16.8269e-9,
                     _signed_calibration=False):
    """Each linear interval rounds upward; actual durations include that quantization.

    Without an explicit discharge, ramp linearly to zero in the charge duration.
    A finite phase includes its left endpoint. Its right endpoint is the first
    sample of the following phase (the terminal idle supplies the final zero).
    """
    if not isfinite(voltage_limit_v) or voltage_limit_v <= 0:
        raise ValueError("Invalid voltage limit")
    if not all(isfinite(x) and x >= 0 for x in (hold_s, reset_s)):
        raise ValueError("Hold and reset durations must be finite and nonnegative")
    if not isfinite(nominal_step_rise) or nominal_step_rise <= 0:
        raise ValueError("Invalid nominal step rise")
    ct, cv = _curve(charge_times_s, charge_voltages_v, voltage_limit_v, _signed_calibration)
    if cv[0] != 0 or cv[-1] == 0:
        raise ValueError("Charge must start at zero and end at a nonzero Vfinal")
    if discharge_times_s is None and discharge_voltages_v is None:
        discharge_times_s, discharge_voltages_v = [0, ct[-1]], [cv[-1], 0]
    dt, dv = _curve(discharge_times_s, discharge_voltages_v, voltage_limit_v, _signed_calibration)
    if dv[0] != cv[-1] or dv[-1] != 0:
        raise ValueError("Discharge must connect Vfinal to zero")
    shortest = min(np.min(np.diff(t)[np.diff(v) != 0]) for t, v in ((ct, cv), (dt, dv)))
    # At least 20 ticks per shortest changing interval where available, and
    # 10 nominal STEP rise times per interval before reaching the rate ceiling.
    # Pick a whole number of ticks across the shortest interval first (e.g.
    # 27 ticks in 168.75 ns gives exactly 160 MSa/s).
    preferred_rate = ceil(27 / shortest * 1e6) / 1e6
    rate = min(MAX_RATE, max(1e-6, preferred_rate)) if sample_rate_hz is None else float(sample_rate_hz)
    if not isfinite(rate) or not 1e-6 <= rate <= MAX_RATE:
        raise ValueError("Sample rate outside 33511B range")
    if abs(rate - round(rate, 6)) > 1e-10:
        raise ValueError("Rate must be representable to 1 microSa/s")
    if shortest + 1e-18 < max(nominal_step_rise, 0.35 / (0.13 * rate)):
        raise ValueError("A varying control interval is shorter than the nominal STEP edge")
    segments = [Segment("idle", np.zeros(8), 0, "repeatTilTrig", "lowAtStart")]

    def curve_segment(name, t, v, marker):
        counts = [_ticks(float(interval), rate) for interval in np.diff(t)]
        n = sum(counts)
        if n < 8 or n > max_points:
            raise ValueError(f"{name} needs {n} samples; change the curve or sample rate")
        knots = np.concatenate(([0], np.cumsum(counts)))
        samples = np.interp(np.arange(n), knots, v)
        segments.append(Segment(name, samples, marker=marker))
        return n

    def plateau(name, seconds, value):
        n = _ticks(seconds, rate)
        if not n:
            return 0
        tile = min(n, 1024)
        chunks, tail = divmod(n, tile)
        if tile < 8:
            prev = segments[-1]
            segments[-1] = replace(prev, samples_volts=np.append(prev.samples_volts, np.full(n, value)))
            return n
        if ceil(chunks / MAX_REPEAT) + len(segments) > 510:
            raise ValueError("Constant dwell exceeds sequence step capacity")
        data = np.full(tile, value)
        while chunks:
            count = min(chunks, MAX_REPEAT)
            segments.append(Segment(name, data, count, "repeat" if count > 1 else "once"))
            chunks -= count
        if tail:
            # Borrow the last tile to make an 8+ point remainder segment.
            last = segments[-1]
            if last.repeat_count == 1:
                segments.pop()
            else:
                segments[-1] = replace(last, repeat_count=last.repeat_count - 1)
            segments.append(Segment(name + "end", np.full(tile + tail, value)))
        return n

    charge = curve_segment("charge", ct, cv, "highAtStart")
    hold = plateau("hold", hold_s, cv[-1])
    discharge = curve_segment("discharge", dt, dv, "lowAtStart")
    reset = plateau("reset", reset_s, 0)
    segments.append(Segment("idle", segments[0].samples_volts, 0, "repeatInf", "lowAtStart"))
    unique = {s.name: len(s.samples_volts) for s in segments}
    allocated = sum(ceil(n / 128) * 128 for n in unique.values())
    if allocated > max_points or len(segments) > 512:
        raise ValueError(f"Sequence needs {allocated} allocated points and {len(segments)} steps")
    low, high = float(min(min(cv), min(dv))), float(max(max(cv), max(dv)))
    zero_code = 32767 * (2 * (0 - low) / (high - low) - 1)
    if abs(zero_code - round(zero_code)) > 1e-8:
        raise ValueError("A zero reset needs an exact DAC code; use one polarity per run or symmetric bipolar extrema")
    return CompiledSequence(tuple(segments), rate, max(abs(low), abs(high)),
                            charge/rate, hold/rate, discharge/rate, reset/rate,
                            sum(unique.values()), allocated, voltage_min=low, voltage_max=high)


def compile_calibration_protocol(*args, **kwargs):
    """Calibrate either polarity within ±voltage_limit_v and finish at zero."""
    return compile_protocol(*args, **kwargs, _signed_calibration=True)


def _codes(volts, compiled):
    span = compiled.voltage_max - compiled.voltage_min
    normalized = 2 * (volts - compiled.voltage_min) / span - 1
    return np.rint(normalized * 32767).astype("<i2")


def compiled_control(compiled):
    """Return exact linear-interpolant knots of the uploaded DAC samples.

    The sampled waveform is before analog filtering. Infinite idle periods are
    omitted. Phase boundaries and the terminal zero are retained explicitly.
    """
    ticks, codes, offset = [], [], 0
    for segment in compiled.segments[1:-1]:
        data = _codes(segment.samples_volts, compiled).astype(np.int32)
        count = len(data) * segment.repeat_count
        if np.all(data == data[0]):
            pos = np.array([0, count-1], dtype=np.int64)
            values = np.repeat(data[0], 2)
        else:
            if segment.repeat_count != 1:
                raise ValueError("Unexpected repeated nonconstant protocol segment")
            pos = np.r_[0, np.flatnonzero(np.diff(data, n=2)) + 1, len(data)-1]
            values = data[pos]
        ticks.append(pos+offset)
        codes.append(values)
        offset += count
    x = np.concatenate([*ticks, [offset]])
    y = np.concatenate([*codes, _codes(np.zeros(1), compiled)]).astype(np.int64)
    boundaries = np.rint(np.cumsum([0, compiled.actual_charge_s, compiled.actual_hold_s,
                                    compiled.actual_discharge_s, compiled.actual_reset_s])
                         * compiled.sample_rate_hz).astype(np.int64)
    phases = np.setdiff1d(boundaries, x)
    if len(phases):
        yp = np.rint(np.interp(phases, x, y)).astype(np.int64)
        order = np.argsort(np.r_[x, phases])
        x, y = np.r_[x, phases][order], np.r_[y, yp][order]
    dx, dy = np.diff(x), np.diff(y)
    changes = dy[:-1]*dx[1:] != dy[1:]*dx[:-1]
    keep = np.r_[True, changes, True] | np.isin(x, boundaries)
    volts = compiled.voltage_min + ((y[keep]+32767)/65534)*(compiled.voltage_max-compiled.voltage_min)
    return x[keep]/compiled.sample_rate_hz, volts


def _check(fg):
    msg = fg.query("SYST:ERR?").strip()
    if int(msg.split(",", 1)[0]) != 0:
        raise RuntimeError(f"33511B reported {msg}")


def configure_sequence(fg, compiled, voltage_limit_v, voltage_gain=1.0):
    """Upload to fresh volatile names. Gain = measured device V / AWG high-Z V."""
    fg.write("OUTP OFF")
    if not isfinite(voltage_gain) or voltage_gain <= 0 or not isfinite(voltage_limit_v) or not 0 < compiled.voltage_peak <= voltage_limit_v:
        raise ValueError("Invalid measured gain or voltage window")
    peak = (compiled.voltage_max - compiled.voltage_min) / voltage_gain
    midpoint = (compiled.voltage_max + compiled.voltage_min) / (2 * voltage_gain)
    if not 0.002 <= peak <= 20 or compiled.voltage_peak / voltage_gain > 10:
        raise ValueError("Requested waveform exceeds high-Z generator amplitude/offset range")
    if int(float(fg.query("DATA:VOL:FREE?"))) < compiled.allocated_points:
        raise ValueError("Insufficient free waveform memory")
    prefix = "E" + uuid4().hex[:8]
    name = prefix + "S"
    names = {}
    for cmd in ("BURS:STAT OFF", "FREQ:MODE CW", "AM:STAT OFF", "FM:STAT OFF",
                "PM:STAT OFF", "FSK:STAT OFF", "SUM:STAT OFF", "PWM:STAT OFF",
                "OUTP:POL NORM", "OUTP:LOAD INF",
                "VOLT:UNIT VPP", "TRIG:SOUR BUS", "FORM:BORD SWAP"):
        fg.write(cmd)
    for s in compiled.segments:
        if s.name in names:
            continue
        key = prefix + str(len(names))
        names[s.name] = key
        fg.write_binary_values(f"DATA:ARB:DAC {key},", _codes(s.samples_volts, compiled),
                               datatype="h", is_big_endian=False)
    fields = [f'"{name}"']
    for s in compiled.segments:
        fields.extend((f'"{names[s.name]}"', str(s.repeat_count), s.control, s.marker, "4"))
    payload = ",".join(fields).encode("ascii")
    size = str(len(payload)).encode("ascii")
    fg.write_raw(b"DATA:SEQ #" + str(len(size)).encode("ascii") + size + payload + b"\n")
    for cmd in (f"FUNC:ARB {name}", "FUNC ARB", "FUNC:ARB:ADV SRAT", "FUNC:ARB:FILT STEP",
                f"FUNC:ARB:SRAT {compiled.sample_rate_hz:.6f}", "VOLT:OFFS 0", f"VOLT {peak:.15g}",
                f"VOLT:OFFS {midpoint:.15g}", "OUTP:SYNC:MODE MARK", "OUTP:SYNC:POL NORM", "OUTP:SYNC ON"):
        fg.write(cmd)
    if int(fg.query("*OPC?").strip()) != 1:
        raise RuntimeError("Waveform upload incomplete")
    _check(fg)
    for query, expected, tolerance in (
            ("FUNC:ARB:SRAT?", compiled.sample_rate_hz, 1e-10),
            ("VOLT?", peak, 5e-4), ("VOLT:OFFS?", midpoint, 5e-4)):
        actual = float(fg.query(query))
        if not np.isclose(actual, expected, rtol=tolerance, atol=1e-12):
            raise RuntimeError(f"33511B {query} returned {actual}, expected {expected}")
    if fg.query("FUNC:ARB:FILT?").strip().upper() != "STEP":
        raise RuntimeError("33511B STEP filter readback mismatch")
    _check(fg)
    return name


def start_output(fg):
    """Call after arming the scope for a rising CH3 marker."""
    try:
        fg.write("OUTP ON")
        if int(fg.query("*OPC?").strip()) != 1:
            raise RuntimeError("Generator did not enable output")
        _check(fg)
        fg.write("*TRG")
        _check(fg)
    except Exception:
        fg.write("OUTP OFF")
        raise
