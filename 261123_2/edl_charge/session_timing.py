"""Voltage-dependent dwell selection and offline acquisition cost estimates."""
import math
from copy import deepcopy

from .planning import duration_grid, final_voltage_grid, get_profile, hardware_profile


def default_timing_policy():
    """Seconds. Target dwell is an upper design target absent measured floor effects."""
    return dict(hold_min_s=.1, hold_target_s=1., reset_min_s=.2, reset_target_s=2.,
                minimum_scale=.10, maximum_scale=1., io_s_per_capture=2., cpu_s_per_pair=2.,
                shutdown_reserve_s=5.)


def _number(value, label, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid {label}")
    return value


def _voltage_magnitude(value):
    if isinstance(value,bool):
        raise ValueError("Invalid voltage")
    try:
        return _number(abs(float(value)),"voltage magnitude")
    except (ValueError,TypeError) as exc:
        raise ValueError("Invalid voltage") from exc


def _settings(policy):
    p = default_timing_policy()
    p.update(policy or {})
    for key in default_timing_policy():
        p[key] = _number(p[key], key)
    for phase in ("hold", "reset"):
        if not 0 < p[phase+"_min_s"] < p[phase+"_target_s"]:
            raise ValueError(f"{phase} target must exceed its positive minimum")
    if not 0 < p["minimum_scale"] <= p["maximum_scale"] <= 1:
        raise ValueError("Require 0 < minimum_scale <= maximum_scale <= 1")
    return p


def _phase(phase):
    if phase not in ("hold_s", "reset_s", "chirp_reset_s"):
        raise ValueError("Unknown dwell phase")
    return "hold" if phase == "hold_s" else "reset"


def proposed_dwell_seconds(voltage, cap, phase, policy=None, scale=None):
    """Initial unqualified quadratic proposal, including for calibration trials."""
    p = _settings(policy)
    v, cap = _voltage_magnitude(voltage), _number(cap,"positive cap",True)
    scale = _number(p["maximum_scale"] if scale is None else scale,"scale")
    if v > cap or not p["minimum_scale"] <= scale <= p["maximum_scale"]:
        raise ValueError("Voltage or dwell scale exceeds selection range")
    prefix = _phase(phase)
    low, target = p[prefix+"_min_s"], p[prefix+"_target_s"]
    return low + scale*(target-low)*(v/cap)**2


def _measured_floor(p, voltage, phase):
    field = _phase(phase)+"_s"
    fallback = p.get("observed_settling_s")
    levels = {}
    for signed_level, value in p.get("settling_by_voltage_s",{}).items():
        level = _voltage_magnitude(signed_level)
        measured = value.get(field) if isinstance(value,dict) else value
        levels[level] = max(levels.get(level,0.),_number(measured,field,True))
    floor = 0.
    for level, value in sorted(levels.items()):
        floor = max(floor,value)
        if level >= voltage-1e-12:
            break
    else:
        floor = max(floor,_number(fallback,"observed_settling_s",True))
    if phase == "chirp_reset_s":
        floor = max(floor,_number(fallback,"observed_settling_s",True))
    return floor


def _at(p, voltage, phase, scale):
    prefix = _phase(phase)
    low, target = p[prefix+"_min_s"], p[prefix+"_target_s"]
    floor = _measured_floor(p,voltage,phase)
    # Allow the measured floor to shift the curve upward. This preserves a
    # strict voltage ordering even when multiple amplitudes have the same floor.
    slope = scale*(target-low)
    q = (voltage/p["cap_v"])**2
    dwell = max(low,floor) + slope*q
    # A replay extension is already a full dwell, including the quadratic term
    # at its voltage. Propagate only the incremental term to larger voltages.
    field = prefix+"_s"
    for anchor, required in p.get("model_required_dwell_by_voltage_s",{}).items():
        anchor = float(anchor)
        if anchor <= voltage and field in required:
            dwell = max(dwell, required[field]+slope*(q-(anchor/p["cap_v"])**2))
    return dwell


def _snapshot(config, policy):
    p = _settings(policy)
    cap = get_profile(config["sample_name"])["peak_bound_v"]
    if "cap_v" in p and p["cap_v"] != cap:
        raise ValueError("Dwell policy belongs to a different voltage range")
    if p.get("sample_name",config["sample_name"]) != config["sample_name"]:
        raise ValueError("Dwell policy belongs to a different specimen")
    for key in ("settling_by_voltage_s","observed_settling_s"):
        if key in p and p[key] != config.get(key,{} if key=="settling_by_voltage_s" else None):
            raise ValueError("Recorded dwell policy differs from the current measured settling evidence")
    replay = {}
    for source in (config, p):
        for anchor, row in source.get("model_required_dwell_by_voltage_s",{}).items():
            anchor = _voltage_magnitude(anchor)
            if not 0 < anchor <= cap or not isinstance(row,dict):
                raise ValueError("Invalid replay dwell entry")
            dest = replay.setdefault(str(anchor),{})
            for phase in ("hold_s","reset_s"):
                if phase in row:
                    dest[phase] = max(dest.get(phase,0.),_number(row[phase],phase))
    p["model_required_dwell_by_voltage_s"] = replay
    p.update(sample_name=config["sample_name"],cap_v=cap,
             settling_by_voltage_s=deepcopy(p.get("settling_by_voltage_s",config.get("settling_by_voltage_s",{}))),
             observed_settling_s=p.get("observed_settling_s",config.get("observed_settling_s")))
    return p


def dwell_seconds(config, dwell_policy, voltage, phase):
    """Apply the recorded scale to the measured monotone floor envelope."""
    p = _snapshot(config,dwell_policy)
    v = _voltage_magnitude(voltage)
    scale = _number(p.get("scale",p["maximum_scale"]),"scale")
    if v > p["cap_v"] or not p["minimum_scale"] <= scale <= p["maximum_scale"]:
        raise ValueError("Voltage or dwell scale exceeds selection range")
    return _at(p,v,phase,scale)


def _capture_cost(config,p,selected,scale,levels,leg,hw_floor,*,measured):
    def dwell(v,phase):
        if measured:
            return _at(p,v,phase,scale)
        return proposed_dwell_seconds(v,p["cap_v"],phase,p,scale)
    chirp_views = 9*len(levels)
    excited = 0.
    max_dwell = 0.
    for v,t in selected:
        v = abs(v)
        h = dwell(v,"hold_s")
        z = dwell(v,"reset_s")
        obs = max(h,config.get("observe_hold_s",0.))
        max_dwell = max(max_dwell,obs,z)
        rate = min(160e6,1200/t,(config.get("max_points",1_000_000)-8192)/(1.2*t))
        d = max(hw_floor,27/rate)
        for hold,view in ((obs,"charge"),(obs,"hold"),(h,"reset"),(0.,"reset")):
            period = t+hold+d+z
            end = {"charge":1.05*t,"hold":t+1.05*hold,"reset":period+.05*z}[view]
            excited += z+end+period
    sweep = 0.
    for vpp in levels:
        z = dwell(vpp/2,"chirp_reset_s")
        max_dwell = max(max_dwell,z)
        sweep += 9*(4*leg+2*z)
    io = (4*len(selected)+chirp_views)*p["io_s_per_capture"]
    cpu = len(selected)*p["cpu_s_per_pair"]
    return dict(optimized_stimulus_and_scope_s=excited,chirp_and_reset_s=sweep,
                setup_transfer_write_s=io,optimize_cpu_s=cpu,total_s=excited+sweep+io+cpu,
                largest_dwell_s=max_dwell)


def schedule_sample(config, pairs, *, remaining_s, policy=None, reserved_s=0., chirp_leg_s=1., chirp_levels=None):
    """Fit the complete chirp grid and matched four-view charge groups to this share.

    remaining_s is active time available now. reserved_s is an outstanding
    allocation for future work; already elapsed calibration must not be subtracted twice.
    """
    from .workflow import timing
    p = _snapshot(config,policy)
    remaining = _number(remaining_s,"remaining_s")
    reserved = _number(reserved_s,"reserved_s")
    leg = _number(chirp_leg_s,"chirp_leg_s",True)
    if not .001 <= leg <= 250:
        raise ValueError("Invalid native sweep leg")
    cap = p["cap_v"]
    pairs = [(math.copysign(_voltage_magnitude(v),float(v)),_number(t,"charge duration",True))
             for v,t in pairs]
    if len(set(pairs)) != len(pairs) or any(not 0<abs(v)<=cap for v,t in pairs):
        raise ValueError("Invalid or repeated charge pair")
    times = sorted({t for v,t in pairs})
    levels = get_profile(config["sample_name"])["chirp_vpp_levels"] if chirp_levels is None else chirp_levels
    chirp_views = 9*len(levels)
    hw_floor = timing(config)["min_charge_time_s"] if pairs else 0.
    if any(t < hw_floor*(1-1e-12) for v,t in pairs):
        raise ValueError("Pair below qualified charge-time floor")
    available = remaining-reserved-p["shutdown_reserve_s"]
    maximum = config.get("max_hold_s") or math.inf
    def cost(selected, scale):
        return _capture_cost(config,p,selected,scale,levels,leg,hw_floor,measured=True)

    def fits(c):
        return c["total_s"] <= available and c["largest_dwell_s"] <= maximum

    def result(selected,scale,c,feasible,reason):
        keep = set(selected)
        return dict(feasible=feasible, full_grid=feasible and len(selected)==len(pairs),
                    selected_pairs=[list(pair) for pair in selected],
                    omitted_pairs=[list(pair) for pair in pairs if pair not in keep],
                    planned_chirp_captures=chirp_views, remaining_s=remaining,
                    reserved_s=reserved,available_scan_and_cpu_s=available,reason=reason,
                    cost_breakdown_s=c, dwell_policy=dict(p,scale=scale),
                    estimates_assume_optimized_peak_equals_target=True)

    # Drop interior durations across all amplitudes together; never drop an
    # individual voltage to make an incomplete matched group appear feasible.
    counts = range(len(times),min(2,len(times))-1,-1)
    last = None
    for n in counts:
        chosen = ({times[round(i*(len(times)-1)/(n-1))] for i in range(n)}
                  if n>1 else set(times))
        selected = [pair for pair in pairs if pair[1] in chosen]
        lo,hi = p["minimum_scale"],p["maximum_scale"]
        last = cost(selected,lo)
        if not fits(last):
            continue
        if fits(cost(selected,hi)):
            lo = hi
        else:
            for _ in range(32):
                mid = (lo+hi)/2
                if fits(cost(selected,mid)):
                    lo = mid
                else:
                    hi = mid
        return result(selected,lo,cost(selected,lo),True,"all feasible durations" if n==len(times)
                      else "interior durations omitted uniformly across voltages")
    return result([],p["minimum_scale"],last,False,
                  "Required chirps and endpoint charge groups exceed the available time or maximum dwell")

def allocate_sample_share(names, remaining_s, *, policy=None, chirp_leg_s=1.,
                          planning_min_charge_s=.02, calibration_s_per_trace=12.,
                          fit_s_per_level=90., small_signal_frequency_count=1, use_current_amp=False):
    """Allocate remaining active seconds before specimen calibration, after shared standards.

    The 20 ms loaded minimum, 12 s/trace and 90 s for two polarity fits are forecasts only.
    Repeat with current remaining time when moving to the next specimen.
    """
    remaining = _number(remaining_s,"remaining_s")
    leg = _number(chirp_leg_s,"chirp_leg_s",True)
    minimum = _number(planning_min_charge_s,"planning_min_charge_s",True)
    cal_s = _number(calibration_s_per_trace,"calibration_s_per_trace")
    fit = _number(fit_s_per_level,"fit_s_per_level")
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply distinct remaining specimen names in run order")
    if type(small_signal_frequency_count) is not int or small_signal_frequency_count<1:
        raise ValueError("At least one calibration frequency required")
    if not .001 <= leg <= 250:
        raise ValueError("Invalid native sweep leg")
    if type(use_current_amp) is not bool:
        raise ValueError("use_current_amp must be boolean")
    minimum = max(minimum,hardware_profile(use_current_amp)["min_charge_time_s"])
    grid = duration_grid(minimum) if minimum<=1. else []
    times = [grid[i] for i in sorted({round(j*(len(grid)-1)/5) for j in range(6)})] if grid else []
    rows = {}
    for name in names:
        profile = get_profile(name)
        cap = profile["peak_bound_v"]
        volts = final_voltage_grid(cap)
        pairs = [(sign*v,t) for v in volts for t in times for sign in (1,-1)]
        p = dict(_settings(policy),cap_v=cap)
        c = _capture_cost({},p,pairs,p["maximum_scale"],profile["chirp_vpp_levels"],leg,minimum,measured=False)
        levels = sorted(set(volts) | {vpp/2 for vpp in profile["chirp_vpp_levels"]})
        counts = [4 for _ in levels]  # Two signed steps and two independent validation captures.
        counts[0] += 5+3*small_signal_frequency_count  # OCV, zero, bipolar and two extra holdouts.
        counts[-1] += 2*small_signal_frequency_count  # Near-cap bias tones.
        traces = sum(counts)
        rows[name] = dict(calibration_traces=traces,calibration_s=cal_s*traces,
                          voltage_levels_v=levels,calibration_traces_by_level=counts,
                          fit_s=fit*len(volts),scan_and_optimize_s=c["total_s"],shutdown_s=p["shutdown_reserve_s"])
    overhead = sum(x["calibration_s"]+x["fit_s"]+x["shutdown_s"] for x in rows.values())
    scan = sum(x["scan_and_optimize_s"] for x in rows.values())
    for row in rows.values():
        reserve = row["calibration_s"]+row["fit_s"]+row["shutdown_s"]
        if overhead and remaining<=overhead:
            row["share_s"] = remaining*reserve/overhead
        else:
            row["share_s"] = reserve+(remaining-overhead)*row["scan_and_optimize_s"]/scan
    return dict(share_s=rows[names[0]]["share_s"],by_sample=rows,remaining_s=remaining,
                expected_non_scan_s=overhead,planning_min_charge_s=minimum,use_current_amp=use_current_amp,
                calibration_s_per_trace=cal_s,fit_s_per_level=fit)
