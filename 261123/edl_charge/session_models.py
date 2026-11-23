"""Budgeted session grid and measured-model qualification of one charge target."""
import hashlib
import json
import math

from . import workflow as w
from .actuator import ActuatedModel
from .awg import _codes
from .planning import VARIANTS, duration_grid, final_voltage_grid, get_profile


def balanced_pairs(config):
    """Cycle all voltages per round, staggering six logarithmic time indices."""
    hw = w.timing(config)
    if hw["min_charge_time_s"] > 1.:
        return []
    times = duration_grid(hw["min_charge_time_s"])
    times = [times[i] for i in sorted({round(j*(len(times)-1)/5) for j in range(6)})]
    volts = final_voltage_grid(get_profile(config["sample_name"])["peak_bound_v"])
    return [(sign*v, times[(r+i) % len(times)]) for r in range(len(times))
            for i, v in enumerate(volts) for sign in (1.,-1.)]


def compiled_digest(cp):
    fields = dict(rate=cp.sample_rate_hz, low=cp.voltage_min, high=cp.voltage_max,
                  segments=[(s.name, s.repeat_count, s.control, s.marker,
                             hashlib.sha256(_codes(s.samples_volts, cp).tobytes()).hexdigest())
                            for s in cp.segments])
    return w.digest(fields)


def _floor(config, policy, v, phase):
    table = policy.get("settling_by_voltage_s", config.get("settling_by_voltage_s", {}))
    rows = sorted((float(k), value) for k, value in table.items())
    floors = []
    def add(value):
        x = value[phase] if isinstance(value,dict) else value
        if x is None or not math.isfinite(x) or x <= 0:
            raise ValueError("Positive measured settling time required")
        floors.append(float(x))
    for level,value in rows:
        add(value)
        if level >= v-1e-12:
            break
    else:
        add(config.get("observed_settling_s"))
    return max(floors)

def signed_config(config,vf):
    """Select physical parameters independently qualified on the requested voltage sign."""
    if not math.isfinite(vf) or vf == 0:
        raise ValueError("Select a finite nonzero charge target")
    polarity = "positive" if vf>0 else "negative"
    if "polarity_fits" in config:
        config = config["polarity_fits"][polarity]
    if "fit_polarity" in config and (config.get("fit_report") or {}).get("fit_polarity") != config["fit_polarity"]:
        raise ValueError("Stored signed fit report and selected specimen polarity differ")
    if config.get("fit_polarity","positive") != polarity:
        raise ValueError("No matching independent "+polarity+" specimen fit")
    return config


def optimize_one(config, vf, T, holds_policy=None):
    from .session_timing import dwell_seconds
    config = signed_config(config,vf)
    policy = config.get("session_dwell_policy",{}) if holds_policy is None else holds_policy
    calibrated = w.model_for(config)
    hw = w.timing(config)
    cap = w.peak_limit(config)
    if T < hw["min_charge_time_s"]*(1-1e-12):
        raise ValueError("Duration below measured hardware minimum")
    rate = round(min(160e6, 1200/T, (config["max_points"]-8192)/(1.2*T)), 6)
    if rate <= 0:
        raise ValueError("Insufficient waveform memory")
    transition = max(config["hardware"]["measured_step_rise_s"], .35/(.13*rate))
    model = ActuatedModel(calibrated.cell_model, transition)
    n = config["optimization_knots"]
    res = w.optimize_charge(model, vf, T, cap, minimum_transition_s=transition, knots=n,
                            starts=config["starts"], maxiter=config["maxiter"],
                            charge_tolerance=config["charge_tolerance"],
                            terminal_voltage_tolerance=config["voltage_tolerance"])
    common = {k:v for k,v in res.items() if k != "prediction"}
    common.update(config_sha256=w.digest(config), sample_name=config["sample_name"],
                  effective_command_rise_s=transition, voltage_limit_v=cap, dwell_policy=policy,
                  fit_polarity=config.get("fit_polarity","positive"),use_current_amp=config["hardware"]["use_current_amp"])
    out = []
    replay_cache = {}
    for variant in VARIANTS:
        case = dict(common, variant=variant,
                    capture_views=["charge","hold"] if variant == VARIANTS[0] else ["reset"])
        if res["status"] == "validated":
            peak = max(abs(vf),abs(float(res["actual_peak_voltage_v"])))
            hold = dwell_seconds(config,policy,abs(vf),"hold_s")
            if variant == VARIANTS[0]:
                hold = max(hold, config.get("observe_hold_s", 0.))
            if variant == VARIANTS[2]:
                hold = 0.
            reset = dwell_seconds(config,policy,peak,"reset_s")
            initial_hold, initial_reset = hold, reset
            target_reset = dwell_seconds(config,policy,abs(vf),"reset_s")
            case["planned_dwell"] = dict(hold_s=hold,reset_s=reset,hold_voltage_v=vf,reset_voltage_v=peak)
            maximum = config["max_hold_s"]
            case["status"] = "compiled_replay_failed"
            while max(hold,reset) <= maximum:
                recipe = dict(t=res["times_s"], v=res["voltage_v"], bound_v=cap, rate=rate,
                              hold_s=hold, reset_s=reset, discharge_s=max(hw["min_charge_time_s"],27/rate),
                              max_points=config["max_points"])
                try:
                    cp = w.build_compiled(recipe)
                    key = compiled_digest(cp)
                    if key not in replay_cache:
                        replay_cache[key] = w.replay_compiled(config, model, cp, vf, T, n-1)
                    review = replay_cache[key]
                except (ValueError, ArithmeticError, RuntimeError) as exc:
                    case["compiled_replay_reason"] = str(exc)
                    break
                case.update(recipe=recipe, compiled_replay=review, compiled_sha256=key,
                            compiled_sample_rate_hz=cp.sample_rate_hz)
                case["dwell_extension"] = dict(hold_s=hold-initial_hold,reset_s=reset-initial_reset,
                                               actual_hold_s=cp.actual_hold_s,actual_reset_s=cp.actual_reset_s)
                # Replay can reveal a larger actual terminal excursion than the charge-only prediction.
                if review["peak_v"] > cap+1e-10:
                    case["compiled_replay_reason"] = "Replayed terminal voltage exceeds specimen limit"
                    break
                peak = min(cap,max(peak,float(review["peak_v"])))
                needed_reset = dwell_seconds(config,policy,peak,"reset_s")
                case["dwell_extension"].update(replayed_peak_v=peak,reset_above_target_plan_s=reset-target_reset)
                if reset < needed_reset:
                    reset = needed_reset
                    continue
                if review["validated"]:
                    case["status"] = "validated"
                    break
                old = (hold,reset)
                if hold and review["hold_error"] > config["reset_tolerance"]:
                    hold = min(maximum,hold*2)
                if review["reset_error"] > config["reset_tolerance"]:
                    reset = min(maximum,reset*2)
                if old == (hold,reset):
                    break
            else:
                case["compiled_replay_reason"] = "Required measured dwell exceeds configured maximum"
        case["case_sha256"] = w.digest(case)
        out.append(case)
    return json.loads(json.dumps(out, default=w.serial, allow_nan=False))


def bounded_voltage_driver(t, voltage, noise_v, extra_knots=()):
    """Split at worst observed voltage error; retain extrema, endpoints and phase boundaries."""
    import numpy as np
    t, v = np.asarray(t,float), np.asarray(voltage,float)
    tolerance = min(3*noise_v, .001*np.ptp(v))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Unresolved voltage waveform or zero voltage-noise estimate")
    keep = {0,len(v)-1,int(np.argmax(v)),int(np.argmin(v))}
    for at in extra_knots:
        keep.add(int(np.argmin(abs(t-at))))
    edges = sorted(keep)
    stack = list(zip(edges[:-1],edges[1:]))
    while stack:
        lo, hi = stack.pop()
        if hi <= lo+1:
            continue
        trial = np.interp(t[lo+1:hi],t[[lo,hi]],v[[lo,hi]])
        residual = abs(v[lo+1:hi]-trial)
        ix = int(np.argmax(residual))
        if residual[ix] > tolerance:
            at = lo+1+ix
            keep.add(at)
            stack.extend([(lo,at),(at,hi)])
    indices = sorted(keep)
    approx = np.interp(t,t[indices],v[indices])
    return approx,dict(indices=indices,original_points=len(t),retained_points=len(indices),
                       max_error_v=float(np.max(abs(approx-v))),tolerance_v=float(tolerance))


def prepare_sample_records(config, captures, out, *, previous_manifest=None,check=lambda:None):
    """Confirm electrical baselines, retain raw data and prepare independent fit records."""
    from pathlib import Path
    import numpy as np
    from .calibration import read_calibration_csv
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    evidence = config["hardware"]["calibration_evidence"]
    inoise = float(evidence["baseline_current_noise_a"])
    vnoise = float(evidence["baseline_voltage_noise_v"])
    if not np.isfinite([inoise,vnoise]).all() or min(inoise,vnoise) <= 0:
        raise ValueError("Require measured positive instrument voltage and current noise")
    baseline = config.get("baseline_current_a",0.)
    manifest = dict(calibration_id=config["calibration_id"], traces=[])
    settling, seen = {}, set()
    signed_settling = {"positive":{},"negative":{}}

    if previous_manifest is not None:
        from os.path import relpath
        previous = Path(previous_manifest)
        prior = w.read(previous)
        if prior["calibration_id"]!=config["calibration_id"]:
            raise ValueError("Cumulative specimen records require a shared hardware calibration")
        for row in prior["traces"]:
            check()
            manifest["traces"].append(dict(row,path=Path(relpath(previous.parent/row["path"],out)).as_posix()))
            seen.add(row["raw_sha256"])
        settling = dict(config["settling_by_voltage_s"])
        signed_settling = {side:dict(table) for side,table in config["signed_settling_by_voltage_s"].items()}
    if "campaign_peak_limit_v" in config:
        manifest["campaign_peak_limit_v"] = config["campaign_peak_limit_v"]

    def dwell(t, v, i, lo, hi, target):
        mask = (t >= lo) & (t <= hi)
        tt, vv, ii = t[mask], v[mask], i[mask]
        bad = (abs(vv-target) > 5*vnoise) | (abs(ii-baseline-evidence.get("open_parallel_conductance_s",0.)*target) > 5*inoise)
        first = int(np.flatnonzero(bad)[-1]+1) if bad.any() else 0
        if len(tt)-first < 5:
            raise ValueError("Calibration plateau lacks a resolved stable electrical tail")
        # Upper bound on first sustained within-band sample, including acquisition resolution.
        return float(max(tt[first]-lo, np.max(np.diff(tt))))

    for idx,row in enumerate(captures):
        check()
        path = Path(row["path"]).resolve()
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint in seen:
            raise ValueError("Duplicate physical capture in calibration")
        seen.add(fingerprint)
        tr = read_calibration_csv(path, row)
        t, v, i = tr.times_s, tr.voltage_v.copy(), tr.current_a
        before = t < 0
        if before.sum() < 5 or np.any(abs(v[before]) > 5*vnoise) or np.any(abs(i[before]-baseline) > 5*inoise):
            raise ValueError("Pretrigger zero is outside independently measured electrical baseline")
        offset = float(np.median(v[before]))
        v -= offset
        v[before] = 0.
        charge = row["actual_charge_s"]
        start_reset = charge + row["actual_hold_s"] + row["actual_discharge_s"]
        end = start_reset + row["actual_reset_s"]
        if t[-1] < end-np.max(np.diff(t)):
            raise ValueError("Calibration did not capture entire reset")
        reset = dwell(t,v,i,start_reset,end,0.)
        polarity = row.get("polarity")
        if polarity is not None and polarity not in ("positive","negative","mixed"):
            raise ValueError("Invalid specimen trace polarity")
        if polarity in ("positive","negative"):
            direction = 1. if polarity=="positive" else -1.
            if np.min(direction*v)<-5*vnoise or np.max(direction*v)<=5*vnoise:
                raise ValueError("Measured excitation does not agree with declared trace polarity")
        if row["stage"] in ("sample_positive_steps","sample_negative_steps"):
            target = float(row["target_v"])
            hold = dwell(t,v,i,row["edge_s"],charge,target)
            key = str(abs(target))
            side = "positive" if target>0 else "negative"
            if polarity is not None and polarity != side:
                raise ValueError("Signed step and trace polarity differ")
            signed_settling[side][key] = dict(hold_s=hold,reset_s=reset)
            prev = settling.get(key,dict(hold_s=0.,reset_s=0.))
            settling[key] = dict(hold_s=max(hold,prev["hold_s"]),reset_s=max(reset,prev["reset_s"]))
        knots = [0.,row["edge_s"],charge,start_reset,end]
        if before.any():
            knots.append(t[before][-1])
        v, driver = bounded_voltage_driver(t,v,vnoise,knots)
        prepared = out / f"fit_trace_{idx:03}.csv"
        np.savetxt(prepared,np.column_stack((t,i,v)),delimiter=",",
                   header="Time (s),Current (A),Voltage (V)",comments="")
        command_metadata = {}
        if row["split"] == "validation":
            if "compiled" in row:
                cp = row["compiled"]
                times, volts = cp.control_knots()
                command_metadata.update(command_times_s=times, command_voltage_v=volts,
                                        sample_rate_hz=cp.sample_rate_hz)
            else:
                command_metadata.update({k:row[k] for k in
                                        ("command_times_s","command_voltage_v","sample_rate_hz")})
            command_metadata["programmed_voltage_gain"] = row["programmed_voltage_gain"]
        if polarity is not None:
            command_metadata.update(polarity=polarity,target_v=row.get("target_v"),bias_v=row.get("bias_v"),frequency_hz=row.get("frequency_hz"))
        manifest["traces"].append(dict(**command_metadata, path=prepared.name,raw_path=str(path),raw_sha256=fingerprint,
            split=row["split"],stage=row["stage"],edge_s=row["edge_s"],voltage_driver=driver,noise_a=max(inoise,float(np.std(i[before],ddof=1))),
            reset_verified=True, baseline_voltage_offset_v=offset, measured_reset_settling_s=reset,
            initial_condition="Zero equilibrium assumed from independently referenced stable pretrigger. Fitted model reset checked separately."))
    if {r["split"] for r in manifest["traces"]} != {"train","validation"} or not settling:
        raise ValueError("Require positive steps and independent train and validation records")
    observed = max(max(x.values()) for x in settling.values())
    observed = max(observed,max(r["measured_reset_settling_s"] for r in manifest["traces"]))
    return dict(config,observed_settling_s=observed,settling_by_voltage_s=settling,signed_settling_by_voltage_s=signed_settling), str(w.save(out/"manifest.json",manifest))


def sample_response_evidence(config, manifest_path):
    """Predict held-out CH2 on its recorded time axis without fitting gain, lag or offset."""
    from pathlib import Path
    import numpy as np
    from .analysis import instrument_current
    from .calibration import read_calibration_csv
    from .awg import MAX_RATE
    tol = config["qualification"].get("max_sample_voltage_error_fraction")
    if tol is None or not 0 < tol < 1:
        raise ValueError("Require independently specified max_sample_voltage_error_fraction in (0,1)")
    hw = config["hardware"]
    measured_rise = float(hw["measured_step_rise_s"])
    if not np.isfinite(measured_rise) or measured_rise <= 0:
        raise ValueError("Missing independently measured voltage rise time")
    rows = []
    for row in w.read(manifest_path)["traces"]:
        if row["split"] != "validation":
            continue
        rate = float(row["sample_rate_hz"])
        gain = float(row["programmed_voltage_gain"])
        times = np.asarray(row["command_times_s"],dtype=float)
        volts = np.asarray(row["command_voltage_v"],dtype=float)
        if (not 0 < rate <= MAX_RATE or not np.isfinite(gain) or gain <= 0 or
            not math.isclose(gain, hw["voltage_gain"], rel_tol=1e-10) or times.ndim != 1 or
            times.shape != volts.shape or len(times) < 2 or not np.isfinite([times,volts]).all() or
            np.any(np.diff(times) <= 0) or times[0] != 0 or max(abs(volts[[0,-1]])) > 1e-10):
            raise ValueError("Invalid held-out DAC command, rate or programmed voltage gain")
        if "fit_polarity" in config:
            sign = 1. if config["fit_polarity"]=="positive" else -1.
            if np.min(sign*volts)<-1e-12 or np.max(sign*volts)<=0:
                raise ValueError("Held-out physical command and fit polarity differ")
        raw = Path(row["raw_path"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != row["raw_sha256"]:
            raise ValueError("Held-out raw voltage capture has changed")
        tr = read_calibration_csv(raw,row)
        t = tr.times_s
        rise = max(measured_rise,.35/(.13*rate))
        grid = np.union1d(times,t)
        cmd = np.interp(grid,times,volts,left=0.,right=0.)
        pred = instrument_current(grid,cmd,rise_s=rise,delay_s=0.,pretrigger_current=0.)
        sampled = pred[np.searchsorted(grid,t)]
        scale = float(np.max(abs(volts)))
        if scale <= 0 or t[0] >= 0 or t[-1] < times[-1]-np.max(np.diff(t)):
            raise ValueError("Need held-out nonzero full-cycle command and captured pretrigger")
        error = float(np.max(abs(sampled-tr.voltage_v))/scale)
        edge = float(row["edge_s"])
        if not np.isfinite(edge) or edge <= 0:
            raise ValueError("Missing measured sample holdout edge duration")
        # Only credit the waveform if its transition is independently time-resolved.
        dt = float(np.max(np.diff(t)))
        floor = max(edge,10*dt,10*rise)
        rows.append(dict(qualified_min_interval_s=floor,edge_s=edge,raw_sha256=row["raw_sha256"],command_sha256=w.digest(dict(t=times,v=volts)),
                         stage=row["stage"],sample_rate_hz=rate,scope_max_dt_s=float(np.max(np.diff(t))),
                         effective_rise_s=rise,programmed_voltage_gain=gain,peak_command_v=scale,
                         max_error_fraction=error,passed=error <= tol))
    if not rows:
        raise ValueError("No held-out sample voltage records")
    edge_floor = min(r["qualified_min_interval_s"] for r in rows)
    return dict(passed=all(row["passed"] for row in rows),qualified_min_interval_s=edge_floor,
                qualified_min_charge_s=edge_floor*(config["optimization_knots"]-1),max_error_fraction_allowed=tol,
                calibration_id=config["calibration_id"],sample_name=config["sample_name"],records=rows)


def _fit_records(config, manifest_path):
    from pathlib import Path
    import numpy as np
    from .calibration import read_calibration_csv
    evidence = sample_response_evidence(config,manifest_path)
    if not evidence["passed"]:
        return dict(config,model_calibrated=False,fit_report=dict(sample_response_evidence=evidence))
    result = w.fit(config,manifest_path)
    result["fit_report"]["sample_response_evidence"] = evidence
    result["fit_report"]["voltage_drivers"] = [dict(path=r["path"],**{k:v for k,v in r["voltage_driver"].items() if k!="indices"}) for r in w.read(manifest_path)["traces"]]
    if result["model_calibrated"]:
        cell = w.model_for(result).cell_model
        states = []
        for row in w.read(manifest_path)["traces"]:
            tr = read_calibration_csv(Path(manifest_path).parent/row["path"],row)
            v = tr.voltage_v
            peak = float(v[np.argmax(abs(v))])
            ix = row["voltage_driver"]["indices"]
            pred = w.simulate(cell,tr.times_s[ix],v[ix],points_per_interval=2)
            err = float(cell.normalized_state_error(pred["final_state"],cell.equilibrium(0.),cell.equilibrium(peak)))
            states.append(dict(path=row["path"], normalized_reset_error=err))
        result["fit_report"]["calibration_reset_replay"] = states
        result["model_calibrated"] = all(r["normalized_reset_error"] <= config["reset_tolerance"] for r in states)
    return result









def fit_sample_records(config,manifest_path):
    """Fit two independent parameter sets using only measured records of the same polarity."""
    from pathlib import Path
    manifest = w.read(manifest_path)
    records = manifest["traces"]
    if not any("polarity" in r for r in records):
        return _fit_records(config,manifest_path)
    if not all(r.get("polarity") in ("positive","negative","mixed") for r in records):
        raise ValueError("All signed calibration records require explicit polarity")
    sources = [r["raw_sha256"] for r in records]
    if len(sources)!=len(set(sources)):
        raise ValueError("Duplicate physical waveform across polarity or data split")
    fits = {}
    for side in ("positive","negative"):
        rows = [r for r in records if r["polarity"]==side]
        if {r["split"] for r in rows}!={"train","validation"}:
            raise ValueError("Require independent training and validation on the "+side+" polarity")
        branch_manifest = Path(manifest_path).with_name("fit_manifest_"+side+".json")
        w.save(branch_manifest,dict(manifest,traces=rows))
        branch = _fit_records(dict(config,fit_polarity=side),branch_manifest)
        branch["fit_report"]["fit_polarity"] = side
        fits[side] = branch
    passed = all(branch["model_calibrated"] for branch in fits.values())
    responses = [branch["fit_report"]["sample_response_evidence"] for branch in fits.values()]
    evidence = dict(passed=all(r["passed"] for r in responses),
                    qualified_min_charge_s=max(r["qualified_min_charge_s"] for r in responses))
    return dict(config,polarity_fits=fits,model_calibrated=passed,
                fit_report=dict(sample_response_evidence=evidence,
                                mixed_diagnostic_sha256=[r["raw_sha256"] for r in records if r["polarity"]=="mixed"]))

