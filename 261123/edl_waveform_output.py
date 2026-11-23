"""33511B continuous waveform settings for the original CH3-triggered acquisition.

The caller retains ownership of VISA, instrument reset, output enable/disable,
scope setup, averaging, timing and saving. No software trigger is sent here.
"""

from dataclasses import replace
import math

from keysight_33511b_chirp_generator import (
    _arb_catalog, _check_errors, _contains_only_default, _expect, configure_arb,
    configure_native,
)


# The 33511B can clear all volatile waveforms only. Remember exact uploads from
# this process so an unattended scan can reuse memory without clearing user data.
_owned = {}


def _release_previous(fg, identity):
    names = _owned.get(identity, set())
    if not names:
        return
    entries = _arb_catalog(fg)
    if entries.intersection(names) and _contains_only_default(entries - names):
        _expect(fg, "OUTP1?", 0)
        fg.write("SOUR1:FUNC SIN")
        fg.write("SOUR1:DATA:VOL:CLE")
        _check_errors(fg, "release previous waveform")
        _owned.pop(identity, None)


def _periodic_sequence(compiled):
    """Remove software-trigger waits; preserve every finite dwell and charge marker."""
    segs = compiled.segments
    if len(segs) < 4 or segs[0].control != "repeatTilTrig" or segs[-1].control != "repeatInf":
        raise ValueError("Expected one initial trigger wait and one terminal idle")
    body = segs[1:-1]
    if (compiled.actual_reset_s <= 0 or body[-1].samples_volts[-1] != 0 or
            body[0].samples_volts[0] != 0 or body[0].marker != "highAtStart" or
            sum(s.marker == "highAtStart" for s in body) != 1 or
            not any(s.marker == "lowAtStart" for s in body) or
            any(s.control not in ("once", "repeat") or s.repeat_count < 1 for s in body)):
        raise ValueError("Continuous protocol requires a charge marker, discharge and finite zero reset")
    points = sum(len(s.samples_volts)*s.repeat_count for s in body)
    if not math.isclose(points/compiled.sample_rate_hz, compiled.duration_s, rel_tol=1e-12):
        raise ValueError("Compiled marker repetition period disagrees with protocol duration")
    sizes = {s.name: len(s.samples_volts) for s in body}
    return replace(compiled, segments=body, unique_sample_points=sum(sizes.values()),
                   allocated_points=sum(math.ceil(n/128)*128 for n in sizes.values()))


def configure_output(fg, case):
    """Configure one repeating waveform and return its nominal SYNC period in seconds.

    For native chirps this period is one down/up sweep plus device rearm overhead.
    For stored ARBs and compiled protocols it is one whole DAC sequence. All
    checks precede the caller's existing OUTP ON.
    """
    if not case.get("ready", False):
        raise ValueError(case.get("reason", "Waveform case is unavailable"))
    kind = case["waveform_kind"]
    if kind == "passive":
        raise ValueError("A passive input provides no generator CH3 edge for the original acquisition")
    if kind not in {"native", "stored_arb", "compiled"}:
        raise ValueError(f"Unknown waveform kind {kind!r}")
    if not case.get("repeat", True):
        raise ValueError("Original scope averaging requires a repeating waveform")
    reset = float(case.get("reset_s", 0))
    if kind != "compiled" and reset != 0:
        raise ValueError("A zero-voltage dwell requires a compiled sequence with an explicit reset segment")
    identity = fg.query("*IDN?").strip()
    if len(identity.split(",")) < 2 or identity.split(",")[1].strip().upper() != "33511B":
        raise RuntimeError(f"Expected 33511B; received {identity!r}")
    _expect(fg, "OUTP1?", 0)
    _release_previous(fg, identity)
    if kind == "native":
        plan = case["plan"]
        configure_native(fg, plan, case.get("generator_load_ohms", "INF"))
        fg.write("TRIG1:COUN 1")
        fg.write("INIT1:CONT ON")
        _expect(fg, "INIT1:CONT?", 1)
        period = 2 * plan.leg_duration_s
    elif kind == "stored_arb":
        plan, rate = case["plan"], case["sample_rate_hz"]
        name = configure_arb(fg, plan, rate, case["max_points"], case.get("generator_load_ohms", "INF"))
        if name:
            _owned.setdefault(identity, set()).update((name, name+".ARB"))
        # An unmodulated single arbitrary waveform runs continuously with burst
        # disabled. Normal Sync rises at sample zero of each complete record.
        fg.write("SOUR1:BURS:STAT OFF")
        fg.write("OUTP1:SYNC:MODE NORM")
        _expect(fg, "SOUR1:BURS:STAT?", 0)
        period = int(float(fg.query("SOUR1:FUNC:ARB:POIN?"))) / rate
    else:
        from edl_charge.awg import configure_sequence
        body = _periodic_sequence(case["compiled"])
        bound = case.get("voltage_limit_v", case["voltage_peak_v"])
        if "generator_load_ohms" in case and str(case["generator_load_ohms"]).upper() not in {"INF","INFINITY"}:
            raise ValueError("Compiled calibration specifies the high-impedance generator voltage convention")
        seq = configure_sequence(fg, body, bound, case.get("voltage_gain", 1.0))
        assets = {seq[:-1].upper()+str(i) for i in range(len({s.name for s in body.segments}))}
        _owned.setdefault(identity, set()).update(assets | {x+".ARB" for x in assets} | {seq.upper(), seq.upper()+".SEQ"})
        _expect(fg, "SOUR1:BURS:STAT?", 0)
        period = body.duration_s
    _check_errors(fg, "continuous waveform setup")
    _expect(fg, "OUTP1?", 0)
    return period
