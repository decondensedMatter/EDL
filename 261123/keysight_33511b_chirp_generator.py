"""SCPI control of a one-channel Keysight 33511B. Call start_output after scope arming."""

from array import array
import csv
from hashlib import sha256
import math
import re

from keysight_33511b_chirp_plan import WAVEFORM_LIMITS_HZ


def _check_errors(fg, stage):
    errors = []
    for _ in range(32):
        reply = fg.query("SYST:ERR?").strip()
        if reply.split(",", 1)[0].strip() in {"0", "+0"}:
            break
        errors.append(reply)
    if errors:
        raise RuntimeError(f"33511B {stage}: " + " | ".join(errors))


def _expect(fg, command, expected):
    reply = fg.query(command).strip()
    if isinstance(expected, str):
        matches = reply.upper() == expected
    else:
        matches = math.isclose(float(reply), expected, rel_tol=1e-7, abs_tol=1e-12)
    if not matches:
        raise RuntimeError(f"33511B {command} returned {reply!r}; expected {expected!r}")


def _prepare(fg, native):
    ident = fg.query("*IDN?").strip().split(",")
    if len(ident) != 4 or ident[1].strip().upper() != "33511B":
        raise RuntimeError(f"Expected Keysight 33511B; *IDN? returned {','.join(ident)!r}")
    fg.write("OUTP1 OFF")
    fg.write("INIT1:CONT OFF")
    fg.write("ABOR")
    if native:
        revision = re.match(r"(\d+)\.(\d+)", ident[3].strip())
        if not revision or int(revision.group(1)) < 4:
            raise RuntimeError(f"Logarithmic return requires firmware 4.00 or later; found {ident[3]}")
    # Configure sweep/burst explicitly without resetting user state or clearing ARB memory.
    fg.write("SOUR1:BURS:STAT OFF")
    fg.write("SOUR1:FREQ:MODE CW")
    fg.write("OUTP1:POL NORM")
    fg.write("TRIG1:DEL 0")
    fg.write("OUTP:SYNC:SOUR CH1")
    fg.write("OUTP1:SYNC:MODE NORM")
    fg.write("OUTP1:SYNC:POL NORM")
    fg.write("OUTP:SYNC ON")
    _check_errors(fg, "initialization")


def _set_levels(fg, plan, load_ohms):
    load = str(load_ohms).strip().upper()
    if load not in {"INF", "INFINITY"}:
        value = float(load)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("load_ohms must be positive or INF")
        load = format(value, ".15g")
    fg.write(f"OUTP1:LOAD {load}")
    fg.write("SOUR1:VOLT:UNIT VPP")
    fg.write("SOUR1:VOLT:OFFS 0")
    fg.write(f"SOUR1:VOLT {plan.voltage_vpp:.15g}")


def configure_native(fg, plan, load_ohms="INF"):
    """Configure counted hardware sweeps. While waiting the generator uses f_start."""
    if plan.mode != "native" or plan.f_max > 0.1 * WAVEFORM_LIMITS_HZ[plan.waveform]:
        raise ValueError("Native sweep exceeds the selected waveform's one-tenth limit")
    if plan.leg_duration_s < 0.001 or 2 * plan.leg_duration_s > 500:
        raise ValueError("Native logarithmic sweep needs >=1 ms per leg and <=500 s per round trip")
    _prepare(fg, native=True)
    shape = {"sine": "SIN", "square": "SQU", "triangle": "RAMP"}[plan.waveform]
    # Set a shared valid fixed frequency before switching to the lower-bandwidth ramp.
    fg.write("SOUR1:FUNC SIN")
    fg.write("SOUR1:FREQ 100")
    fg.write(f"SOUR1:FUNC {shape}")
    if shape == "SQU":
        fg.write("SOUR1:FUNC:SQU:DCYC 50")
    elif shape == "RAMP":
        fg.write("SOUR1:FUNC:RAMP:SYMM 50")
    _set_levels(fg, plan, load_ohms)
    fg.write(f"SOUR1:FREQ:STAR {plan.f_max:.15g}")
    fg.write(f"SOUR1:FREQ:STOP {plan.f_min:.15g}")
    fg.write("SOUR1:SWE:SPAC LOG")
    fg.write(f"SOUR1:SWE:TIME {plan.leg_duration_s:.15g}")
    fg.write("SOUR1:SWE:HTIM 0")
    fg.write(f"SOUR1:SWE:RTIM {plan.leg_duration_s:.15g}")
    fg.write("TRIG1:SOUR IMM")
    fg.write(f"TRIG1:COUN {plan.cycles}")
    fg.write("SOUR1:FREQ:MODE SWE")
    _check_errors(fg, "native sweep configuration")
    for command, expected in [
        ("INIT1:CONT?", 0), ("TRIG1:COUN?", plan.cycles),
        ("SOUR1:SWE:SPAC?", "LOG"), ("SOUR1:FREQ:STAR?", plan.f_max),
        ("SOUR1:FREQ:STOP?", plan.f_min), ("SOUR1:SWE:TIME?", plan.leg_duration_s),
        ("SOUR1:SWE:RTIM?", plan.leg_duration_s), ("SOUR1:SWE:HTIM?", 0),
        ("SOUR1:VOLT?", plan.voltage_vpp), ("OUTP1?", 0),
    ]:
        _expect(fg, command, expected)
    _check_errors(fg, "native sweep readback")


def arb_point_count(plan, sample_rate_hz, max_points):
    """Offline feasibility check; no instrument commands or waveform allocation."""
    if plan.mode != "arb" or plan.f_max > 2e6:
        raise ValueError("Stored ARB sweep requires ARB mode and f_max <= 2 MHz")
    if not math.isfinite(sample_rate_hz) or not 80 * plan.f_max <= sample_rate_hz <= 160e6:
        raise ValueError("ARB requires >=80 samples per fastest cycle and <=160 MSa/s")
    if type(max_points) is not int or not 8 <= max_points <= 16_000_000:
        raise ValueError("ARB max_points must be an integer from 8 to 16,000,000")
    count = math.ceil(plan.total_duration_s * sample_rate_hz)
    if count > max_points:
        raise ValueError(f"Stored ARB needs {count:,} points; capacity {max_points:,} permits "
                         f"{max_points / sample_rate_hz:.9g} s total or "
                         f"{max_points / (2 * plan.cycles * sample_rate_hz):.9g} s per leg. "
                         "Choose native mode or shorten the leg duration.")
    if count < 8:
        raise ValueError("Stored ARB requires at least eight samples")
    return count


def _dac_data(plan, sample_rate_hz, count):
    def code(index):
        phase = plan.phase_cycles_at(index / sample_rate_hz) % 1.0
        if plan.waveform == "square":
            return 32767 if phase < 0.5 else -32767
        sine = math.sin(math.tau * phase)
        level = 2 * math.asin(sine) / math.pi if plan.waveform == "triangle" else sine
        return round(32767 * level)
    return array("h", (code(i) for i in range(count)))


def _arb_catalog(fg):
    raw = fg.query("SOUR1:DATA:VOL:CAT?").strip()
    return {s.strip().upper() for s in next(csv.reader([raw], skipinitialspace=True)) if s.strip()}


def _contains_only_default(entries):
    # DATA:VOL:CLE automatically reloads this factory waveform.
    return entries.issubset({r"INT:\BUILTIN\EXP_RISE.ARB"})


def release_arb(fg, name: str | None) -> bool:
    """Release this acquisition's fresh upload after output-off if no other user ARBs exist."""
    if name is None:
        return False
    if not re.fullmatch(r"C[0-9A-F]{11}", name):
        raise ValueError("Expected an acquisition-owned ARB name")
    if float(fg.query("OUTP1?")) != 0:
        raise RuntimeError("Disable output before releasing the ARB waveform")
    entries = _arb_catalog(fg)
    owned = {name, name + ".ARB"}
    if not entries.intersection(owned) or not _contains_only_default(entries - owned):
        return False
    fg.write("SOUR1:BURS:STAT OFF")
    fg.write("SOUR1:FUNC SIN")
    fg.write("SOUR1:DATA:VOL:CLE")
    _check_errors(fg, "release owned ARB")
    return True


def configure_arb(fg, plan, sample_rate_hz=160e6, max_points=1_000_000, load_ohms="INF") -> str | None:
    """Return a releasable fresh upload's name, or None for shared/pre-existing memory."""
    count = arb_point_count(plan, sample_rate_hz, max_points)
    _prepare(fg, native=False)
    # Use a recipe-specific name so repeated amplitudes can reuse the same normalized samples.
    recipe = (1, plan.waveform, plan.f_min, plan.f_max, plan.leg_duration_s, plan.cycles,
              sample_rate_hz, count)
    name = "C" + sha256(repr(recipe).encode("ascii")).hexdigest()[:11].upper()
    catalog = _arb_catalog(fg)
    release_name = None
    if not catalog.intersection({name, name + ".ARB"}):
        free = int(float(fg.query("SOUR1:DATA:VOL:FREE?")))
        if free < count:
            raise ValueError(f"ARB needs {count:,} points; instrument reports {free:,} free. "
                             "Choose native mode or shorten the leg duration.")
        fg.write("FORM:BORD SWAP")
        fg.write_binary_values(f"SOUR1:DATA:ARB:DAC {name},", _dac_data(plan, sample_rate_hz, count),
                               datatype="h", is_big_endian=False)
        _check_errors(fg, "ARB upload")
        if _contains_only_default(catalog):
            release_name = name
    try:
        _configure_arb_burst(fg, plan, name, count, sample_rate_hz, load_ohms)
    except Exception:
        release_arb(fg, release_name)
        raise
    return release_name


def _configure_arb_burst(fg, plan, name, count, sample_rate_hz, load_ohms):
    fg.write(f"SOUR1:FUNC:ARB {name}")
    fg.write("SOUR1:FUNC ARB")
    fg.write("SOUR1:FUNC:ARB:ADV SRAT")
    filt = "STEP" if plan.waveform == "square" else "NORM"
    fg.write(f"SOUR1:FUNC:ARB:FILT {filt}")
    fg.write(f"SOUR1:FUNC:ARB:SRAT {sample_rate_hz:.15g}")
    _set_levels(fg, plan, load_ohms)
    fg.write("SOUR1:BURS:MODE TRIG")
    fg.write("SOUR1:BURS:NCYC 1")
    fg.write("SOUR1:BURS:PHAS 0")
    fg.write("TRIG1:SOUR BUS")
    fg.write("TRIG1:COUN 1")
    fg.write("SOUR1:BURS:STAT ON")
    _check_errors(fg, "ARB burst configuration")
    for command, expected in [
        ("SOUR1:FUNC:ARB:POIN?", count), ("SOUR1:FUNC:ARB:SRAT?", sample_rate_hz),
        ("SOUR1:BURS:NCYC?", 1), ("TRIG1:SOUR?", "BUS"),
        ("SOUR1:VOLT?", plan.voltage_vpp), ("OUTP1?", 0),
    ]:
        _expect(fg, command, expected)
    _check_errors(fg, "ARB burst readback")


def start_output(fg, mode):
    """Start after arming the oscilloscope. Caller must turn output off in finally."""
    if mode not in {"native", "arb"}:
        raise ValueError("mode must be native or arb")
    try:
        fg.write("OUTP1 ON")
        fg.write("INIT1")
        if mode == "arb":
            fg.write("*TRG")
        _check_errors(fg, "start")
    except Exception:
        fg.write("OUTP1 OFF")
        raise
