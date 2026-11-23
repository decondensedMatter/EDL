"""Compose the model, waveform compiler and acquisition tools. File paths are local."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .actuator import ActuatedModel
from .awg import compile_protocol, compiled_control
from .calibration import fit_parameters, read_calibration_csv, replay_current, validate_parameters
from .optimization import optimize_charge, simulate
from .planning import (VARIANTS, build_candidate_plan, calibration_protocols, get_profile,
                       hardware_profile, load_profiles)
from .stern_pnp import SternPNPModel
from .tf_model import TFModel


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value {type(value)}")


def digest(data):
    raw = json.dumps(data, sort_keys=True, allow_nan=False, default=serial).encode()
    return hashlib.sha256(raw).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=serial, allow_nan=False), encoding="utf-8")
    return path


def make_config(sample, amp):
    profile = get_profile(sample)
    hw = hardware_profile(amp)
    hw.update(ch1_input="DC50", current_bandwidth_hz=hw["bandwidth_hz"],
              current_sign=1, ch1_scale_v_per_div=.1, ch2_scale_v_per_div=.2,
              voltage_gain=1., voltage_offset_v=0., current_offset_v=0.,
              voltage_measurement_location="across the two cell electrodes",
              first_order_independently_qualified=False, measured_step_rise_s=None,
              current_response_rise_s=None, current_delay_s=None,
              calibration_evidence=None)
    return dict(sample_name=profile["sample_name"], model=profile["model"],
                profile=profile, hardware=hw, model_parameters=profile["model_parameters"].copy(),
                initial_guess=profile["initial_guess"], parameter_bounds={}, linear_parameters=[],
                model_calibrated=False, fit_report=None, calibration_id=None,
                qualification=dict(max_validation_noise_rmse=None, max_condition_number=None),
                observed_settling_s=None, reset_tolerance=.01, max_hold_s=60.,
                charge_tolerance=.01, voltage_tolerance=.005, max_points=1_000_000,
                optimization_knots=7, starts=3, maxiter=50, capture_points=2000)


def plan(sample, amp):
    names = [sample] if sample else load_profiles()
    return {key: dict(cases=build_candidate_plan(key, amp),
                      calibration=calibration_protocols(key, amp)) for key in names}


def factory(config):
    expected = get_profile(config["sample_name"])["model"]
    if config["model"] != expected:
        raise ValueError("Sample and model family differ")
    return TFModel if expected == "tf" else SternPNPModel


def require_hardware(config):
    hw = config["hardware"]
    if hw.get("data_hardware_calibrated") is not True or not hw.get("calibration_evidence"):
        raise ValueError("Verified hardware calibration and evidence are required")
    if not config.get("calibration_id"):
        raise ValueError("Missing calibration session identifier")


def require_response(config):
    require_hardware(config)
    hw = config["hardware"]
    if hw.get("first_order_independently_qualified") is not True:
        raise ValueError("Command-to-cell first-order response has not passed independent qualification")
    for key in ("measured_step_rise_s", "current_response_rise_s", "current_delay_s"):
        if hw.get(key) is None or not math.isfinite(hw[key]) or hw[key] < 0:
            raise ValueError(f"Missing measured hardware response: {key}")
    if hw["measured_step_rise_s"] <= 0:
        raise ValueError("Positive measured command-path rise time required")


def fit(config, manifest_path):
    require_hardware(config)
    from .analysis import instrument_current, parallel_fixture_current
    source = Path(manifest_path)
    man = read(source)
    if not config.get("fit_polarity") and any("polarity" in row for row in man["traces"]):
        from .session_models import fit_sample_records
        return fit_sample_records(config,source)
    if man.get("calibration_id") != config["calibration_id"]:
        raise ValueError("Trace manifest and hardware calibration session differ")
    bounds = config["parameter_bounds"]
    if not bounds:
        raise ValueError("Explicit independently justified parameter bounds required")
    traces = []
    seen = set()
    for row in man["traces"]:
        path = (source.parent / row["path"]).resolve()
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint in seen:
            raise ValueError("Same waveform data supplied twice in calibration manifest")
        seen.add(fingerprint)
        traces.append(read_calibration_csv(path, row, initial_state=row.get("initial_state")))
    params = dict(config["model_parameters"])
    for key in bounds:
        if params.get(key) is None:
            params[key] = config["initial_guess"].get(key)
    make = factory(config)
    make(params)
    current_rise = config["hardware"].get("current_response_rise_s")
    current_delay = config["hardware"].get("current_delay_s")
    if any(x is None or not math.isfinite(x) or x < 0 for x in (current_rise,current_delay)):
        raise ValueError("Current channel rise time and delay are required for waveform fitting")
    driver_indices = {id(tr.voltage_v):tr.metadata["voltage_driver"]["indices"] for tr in traces if "voltage_driver" in tr.metadata}
    fixture_cache = {}
    def observed_current(cell,t,v,y0):
        key = id(v)
        if key not in fixture_cache:
            fixture_cache[key] = (v,parallel_fixture_current(t,v,config["hardware"]["calibration_evidence"],
                                  rise_s=current_rise,delay_s=current_delay))
        raw = replay_current(cell,t,v,y0,driver_indices=driver_indices.get(id(v)))
        return instrument_current(t,raw,rise_s=current_rise,delay_s=current_delay,
                                  pretrigger_current=raw[0])+fixture_cache[key][1]+config.get("baseline_current_a",0.)
    train = [trace for trace in traces if trace.split=="train"]
    held_out = [trace for trace in traces if trace.split=="validation"]
    if not train or not held_out:
        raise ValueError("Supply distinct training and validation waveform records")
    result = fit_parameters(make, params, train, bounds, instrument_calibrated=True,
                            forward=observed_current,linear_parameters=config["linear_parameters"])
    validation = validate_parameters(make, result["parameters"], held_out, instrument_calibrated=True,
                                     forward=observed_current)
    qual = config["qualification"]
    ok = (result["success"] and result["identifiability"] == "locally_full_rank" and
          qual.get("max_validation_noise_rmse") is not None and qual.get("max_condition_number") is not None and
          result["condition_number"] <= qual["max_condition_number"] and
          all(row["noise_normalized_rmse"] <= qual["max_validation_noise_rmse"]
              for row in validation["validation_metrics"]))
    report = dict(training=result, validation=validation, trace_sha256=sorted(seen),
                  hardware_digest=digest(config["hardware"]), calibration_id=config["calibration_id"])
    return dict(config, model_parameters=result["parameters"], fit_report=report, model_calibrated=bool(ok))


def model_for(config):
    require_response(config)
    report = config.get("fit_report") or {}
    if (config.get("model_calibrated") is not True or report.get("hardware_digest") != digest(config["hardware"])
            or report.get("calibration_id") != config["calibration_id"]
            or report.get("training", {}).get("parameters") != config["model_parameters"]):
        raise ValueError("Qualified specimen fit matching this configuration is required")
    return ActuatedModel(factory(config)(config["model_parameters"]), config["hardware"]["measured_step_rise_s"])


def timing(config):
    hw = config["hardware"]
    loaded = ((config.get("fit_report") or {}).get("sample_response_evidence") or {}).get("qualified_min_charge_s",0.)
    return hardware_profile(hw["use_current_amp"], {
        "measured_bandwidth_hz": hw["current_bandwidth_hz"],
        "min_charge_time_s": max(loaded, hw["min_charge_time_s"], 10*hw["measured_step_rise_s"])})


def build_compiled(recipe):
    v = np.asarray(recipe["v"],float)
    if v[-1] == 0 or np.any(np.sign(v[-1])*v<0):
        raise ValueError("Optimized excursions must match target polarity")
    return compile_protocol(recipe["t"], recipe["v"], recipe["hold_s"], [0.,recipe["discharge_s"]],
                            [recipe["v"][-1],0.], recipe["reset_s"], voltage_limit_v=recipe["bound_v"],
                            sample_rate_hz=recipe["rate"], max_points=recipe["max_points"], _signed_calibration=bool(v[-1]<0))


def peak_limit(config):
    cap = get_profile(config["sample_name"])["peak_bound_v"]
    limit = config.get("campaign_peak_limit_v",cap)
    if not math.isfinite(limit) or not 0 < limit <= cap:
        raise ValueError("Campaign peak limit outside sample rating")
    return limit


def replay_compiled(config, model, compiled, vf, T, intervals):
    ct, cv = compiled_control(compiled)
    if np.any(np.diff(ct) <= 0):
        raise ValueError("Nonmonotonic compiled control")
    cap = peak_limit(config)
    sign = np.sign(vf)
    if not 0<abs(vf)<=cap or (sign*cv).min() < -1e-12 or (sign*cv).max() > cap+1e-12:
        raise ValueError("Compiled DAC waveform outside sample bound")
    if abs(compiled.actual_charge_s-T) > intervals/compiled.sample_rate_hz+1e-14:
        raise ValueError("Compiled duration exceeds per-control-interval rounding allowance")
    # Retain every quantized change in slope. Never mark a skipped replay as qualified.
    if len(ct) > 4096:
        raise ValueError("Compiled replay exceeds 4096 exact knots; use a lower sampling rate or offline replay")
    eq0, eqf = model.equilibrium(0.), model.equilibrium(vf)
    q0, qf = model.charge(eq0,0.), model.charge(eqf,vf)
    def state_check(state, reference, initial):
        return float(model.normalized_state_error(state,reference,initial))
    actual_end = compiled.actual_charge_s
    command_t = np.unique(np.r_[ct, T, actual_end])
    command_v = np.interp(command_t, ct, cv)
    trace = simulate(model,command_t,command_v,rtol=5e-8,atol=2e-11,points_per_interval=10)
    def endpoint(at):
        index = int(np.argmin(abs(trace["times_s"]-at)))
        if abs(trace["times_s"][index]-at) > 8*np.spacing(at):
            raise ArithmeticError("Charge boundary missing from transient replay")
        q_error = abs(trace["charge_c"][index]-qf)/abs(qf-q0)
        v_error = abs(trace["voltage_v"][index]-vf)/abs(vf)
        return index, float(q_error), float(v_error)
    _, requested_q, requested_v = endpoint(T)
    stop, actual_q, actual_v = endpoint(actual_end)
    qerr, verr = max(requested_q,actual_q), max(requested_v,actual_v)
    charge_volts = trace["voltage_v"][:stop+1]
    increments = .5*(charge_volts[:-1]+charge_volts[1:])*np.diff(trace["charge_c"][:stop+1])
    reset = state_check(trace["final_state"],eq0,eqf)
    peak = float(np.max(sign*trace["voltage_v"]))
    has_hold = compiled.actual_hold_s > 0
    held = int(np.argmin(abs(trace["times_s"]-(compiled.actual_charge_s+compiled.actual_hold_s))))
    holderr = state_check(trace["state"][held],eqf,eq0) if has_hold else None
    physical = all(model.validate_state(y) for y in trace["state"])
    ok = (qerr <= config["charge_tolerance"] and verr <= config["voltage_tolerance"] and
          reset <= config["reset_tolerance"] and (not has_hold or holderr <= config["reset_tolerance"])
          and physical and peak <= cap+1e-10 and np.min(sign*trace["voltage_v"]) >= -1e-10)
    return dict(validated=bool(ok), requested_charge_s=T, actual_charge_s=actual_end, exact_knots=len(ct),
                replay_model="Linear interpolation of uploaded DAC codes followed by the qualified first-order voltage model",
                requested_charge_error=requested_q, requested_terminal_voltage_error=requested_v,
                actual_charge_error=actual_q, actual_terminal_voltage_error=actual_v,
                compiled_net_charge_work_j=float(np.sum(increments)),
                compiled_delivered_charge_work_j=float(np.maximum(increments,0.).sum()),
                compiled_returned_charge_work_j=float(-np.minimum(increments,0.).sum()),
                charge_error=float(qerr), terminal_voltage_error=float(verr), peak_v=peak, signed_peak_v=sign*peak,
                reset_error=reset, hold_error=holderr, physical=physical)


def optimize(config, out, limit=None):
    from .session_models import optimize_one, signed_config
    if "polarity_fits" in config:
        for branch in config["polarity_fits"].values():
            model_for(branch)
    else:
        model_for(config)
    hw = timing(config)
    rows = build_candidate_plan(config["sample_name"],hw["use_current_amp"],{
        "min_charge_time_s":hw["min_charge_time_s"]})
    pairs = list(dict.fromkeys((r["final_voltage_v"],r["charge_time_s"]) for r in rows))
    if limit is not None and limit<1:
        raise ValueError("limit must be positive")
    results = []
    for index,(vf,T) in enumerate(pairs[:limit]):
        try:
            branch = signed_config(config,vf)
        except ValueError as exc:
            cases = [dict(status="missing_polarity_fit",variant=v,reason=str(exc),sample_name=config["sample_name"],
                          final_voltage_v=vf,charging_time_s=T) for v in VARIANTS]
        else:
            cases = optimize_one(branch,vf,T)
        for case in cases:
            name = f"case_{index:04d}_{case['variant']}.json"
            save(Path(out)/name,case)
            results.append(dict(file=name,status=case["status"]))
    save(Path(out)/"index.json",results)
    return results


def validate_case(config,case):
    if "polarity_fits" in config or case.get("final_voltage_v"):
        from .session_models import signed_config
        config = signed_config(config,case["final_voltage_v"])
    model_for(config)
    claim = dict(case)
    signed = claim.pop("case_sha256",None)
    if (case.get("config_sha256")!=digest(config) or signed!=digest(claim) or
            case.get("status")!="validated" or case.get("compiled_replay",{}).get("validated") is not True):
        raise ValueError("Hardware output requires a matching configuration and validated compiled waveform")
    cp = build_compiled(case["recipe"])
    if max(abs(cp.voltage_min),abs(cp.voltage_max))>peak_limit(config)+1e-12:
        raise ValueError("Compiled waveform exceeds approved campaign peak")
    return cp


def capture_view(compiled, view, points):
    intervals = {"charge":(0., compiled.actual_charge_s),
                 "hold":(compiled.actual_charge_s,compiled.actual_hold_s),
                 "discharge":(compiled.actual_charge_s+compiled.actual_hold_s,compiled.actual_discharge_s),
                 "reset":(compiled.duration_s-compiled.actual_reset_s,compiled.actual_reset_s),
                 "full":(0.,compiled.duration_s)}
    start,dt = intervals[view]
    if dt<=0 or points<20:
        raise ValueError("No such nonempty capture phase or too few capture samples")
    pad = .05*dt
    return dict(name=view,start_s=start-pad,duration_s=dt+2*pad,sample_rate_hz=min(20e9,points/(dt+2*pad)))

def calibration_recipe(config, stage_name, *, target_v=None, bias_v=None, frequency_hz=None,
                       edge_s=None, hold_s=None, reset_s=None, standard_resistance_ohm=None, standard_capacitance_f=None,
                       standard_excitation_v=None):
    """Construct a single, explicitly selected calibration acquisition for one terminal wiring."""
    from .awg import compile_calibration_protocol
    hw = config["hardware"]
    prof = get_profile(config["sample_name"])
    stages = calibration_protocols(prof, hw["use_current_amp"],
                                  {"measured_bandwidth_hz":hw["current_bandwidth_hz"]})["stages"]
    stage = next((s for s in stages if s["name"]==stage_name),None)
    if stage is None:
        raise ValueError("Unknown calibration stage")
    spec = dict(stage)
    cap = prof["peak_bound_v"]
    if spec["connection_required"]=="sample" and "campaign_peak_limit_v" in config:
        cap = peak_limit(config)
        spec["campaign_peak_limit_v"] = cap
        amp = min(.01,cap/10)
        prior = config.get("previous_campaign_peak_v",0.)
        if not math.isfinite(prior) or not 0<=prior<cap:
            raise ValueError("Previous specimen peak must be below the current level")
        if stage_name in ("sample_positive_steps","sample_negative_steps"):
            spec["levels_v"] = [cap if stage_name=="sample_positive_steps" else -cap]
        elif stage_name=="sample_bipolar_steps":
            peak = min(.1,cap)
            spec["levels_v"] = [0.,peak,-peak,0.]
        elif stage_name=="sample_small_signal":
            spec.update(amplitude_v=amp,bias_voltages_v=[0.,cap/2,-cap/2,cap-amp,-(cap-amp)])
        elif stage_name in ("positive_step_holdout","negative_step_holdout"):
            spec["levels_v"] = [0.,(1 if stage_name=="positive_step_holdout" else -1)*(prior+.63*(cap-prior)),0.]
        elif stage_name in ("ramp_holdout","negative_ramp_holdout"):
            sign = -1 if stage_name=="negative_ramp_holdout" else 1
            spec["knots_v"] = [0.,sign*(prior+.75*(cap-prior)),sign*.25*cap,0.]
    if standard_excitation_v is not None:
        if (stage_name not in ("gain_polarity_offset_delay","known_rc_bandwidth","open_input_parasitics")
                or not math.isfinite(standard_excitation_v) or not 0<standard_excitation_v<=prof["peak_bound_v"]):
            raise ValueError("Standard excitation requires an active physical standard and in-range peak voltage")
        spec["standard_excitation_v"] = standard_excitation_v
        if stage_name=="known_rc_bandwidth":
            spec["amplitude_v"] = standard_excitation_v
        elif stage_name=="gain_polarity_offset_delay":
            spec["levels_v"] = [0.,standard_excitation_v,-standard_excitation_v,0.]
        else:
            spec["levels_v"] = [0.,standard_excitation_v,0.]
    if stage_name in ("open_circuit_voltage","initial_zero","shorted_zero"):
        return spec, None
    if spec["connection_required"]=="sample":
        require_response(config)
    if not hold_s or not edge_s or min(hold_s,edge_s)<=0:
        raise ValueError("Supply measured acquisition hold and finite calibration edge durations")
    if stage_name in ("known_rc_bandwidth","gain_polarity_offset_delay"):
        if not standard_resistance_ohm or standard_resistance_ohm<=0:
            raise ValueError("Enter actual standard resistance in ohms")
        spec["resistance_ohm"] = standard_resistance_ohm
    if stage_name=="known_rc_bandwidth":
        if not standard_capacitance_f or standard_capacitance_f<=0:
            raise ValueError("Enter actual standard capacitance in farads")
        spec["capacitance_f"] = standard_capacitance_f
    t, v = [0.], [0.]
    def extend(value,dt):
        t.append(t[-1]+dt)
        v.append(float(value))
    if "frequencies_hz" in spec:
        if frequency_hz not in spec["frequencies_hz"]:
            raise ValueError("Select one frequency from the generated calibration manifest")
        bias = 0. if bias_v is None else bias_v
        if bias not in spec.get("bias_voltages_v",[0.]):
            raise ValueError("Select a listed calibration bias")
        if bias:
            extend(bias,edge_s)
            extend(bias,hold_s)
        excursion = spec["amplitude_v"]*(-1. if bias<0 else 1.)
        for k in range(1,3*16+1):
            extend(bias+excursion*math.sin(2*math.pi*k/16),1/(16*frequency_hz))
        # The signed compiler needs a nonzero endpoint before its return-to-zero segment.
        extend(bias+excursion,1/(4*frequency_hz))
    else:
        levels = spec.get("levels_v",spec.get("knots_v",[]))
        if stage_name in ("sample_positive_steps","sample_negative_steps"):
            if target_v not in levels:
                raise ValueError("Select a target listed for this signed step protocol")
            levels = [target_v]
        if stage_name=="open_input_parasitics":
            levels = [spec["levels_v"][1]]
        # The final transition back to zero is supplied by the discharge segment.
        if levels and levels[-1]==0:
            levels = levels[:-1]
        for level in levels:
            if level==v[-1]:
                continue
            extend(level,edge_s)
            if stage_name not in ("ramp_holdout","negative_ramp_holdout"):
                extend(level,hold_s)
    reset_s = hold_s if reset_s is None else reset_s
    if not math.isfinite(reset_s) or reset_s<=0:
        raise ValueError("Supply a finite positive calibration reset duration")
    spec.update(target_v=target_v,bias_v=bias_v,frequency_hz=frequency_hz,edge_s=edge_s,hold_s=hold_s,reset_s=reset_s)
    cp = compile_calibration_protocol(t,v,0.,[0.,edge_s],[v[-1],0.],reset_s,
                                      voltage_limit_v=cap,max_points=config["max_points"])
    return spec, cp


def demo(out):
    """Historical 0.1 M illustrative candidate on ideal terminal voltage, offline only."""
    model = TFModel(TFModel.historical_seed())
    cap = get_profile("0.1M")["peak_bound_v"]
    result = optimize_charge(model,.1,.1,cap,minimum_transition_s=.001,knots=3,
                             starts=1,maxiter=8,charge_tolerance=.025)
    data = {k:v for k,v in result.items() if k!="prediction"}
    data.update(status="offline_demo_only", optimizer_status=result["status"],
                sample_name="NaClO4_0.1M", hardware_output_allowed=False,
                provenance="Archived 0.1 M TF parameters and ideal terminal voltage; current cell unmeasured.")
    if "prediction" in result:
        trace = result["prediction"]
        data["illustrative_prediction"] = {k:trace[k] for k in ("times_s","voltage_v","current_a","charge_c")}
    return save(Path(out)/"historical_demo.json",data)









