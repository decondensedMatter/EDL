"""Offline waveform selection for the original Agilent scan loop.

Native and stored sweeps use existing Keysight sweep definitions. Validated
charging waveforms are read from disk. This module never opens an instrument.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from keysight_33511b_chirp_generator import arb_point_count
from keysight_33511b_chirp_plan import make_plan
from edl_charge.planning import VARIANTS, calibration_protocols, get_profile


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _name(*pieces):
    return "_".join(re.sub(r"[^A-Za-z0-9.+-]","-",str(x)) for x in pieces)


def _unavailable(base,label,reason):
    return dict(base, test_name=label,ready=False,reason=reason,voltage_peak_v=0.,
                total_duration_s=0.,waveform_kind="unavailable")


def _sweeps(base,profile,mode,selected,leg,rate,memory):
    modes = ("native","stored_arb") if mode=="both" else (mode,)
    for kind in modes:
        for shape in ("sine","square","triangle"):
            if not selected.intersection(("chirp",shape)):
                continue
            for vpp in profile["chirp_vpp_levels"]:
                p = make_plan(shape,vpp,leg,cycles=2,mode="native" if kind=="native" else "arb")
                result = dict(base,test_name=_name("chirp",kind,shape,f"{vpp:g}Vpp",f"{leg:g}s-leg"),
                              waveform_kind=kind,plan=p,ready=True,voltage_peak_v=vpp/2.,
                              total_duration_s=p.total_duration_s,sample_rate_hz=rate,max_points=memory)
                if kind=="stored_arb":
                    try:
                        result["arb_points"]=arb_point_count(p,rate,memory)
                    except ValueError as exc:
                        result.update(ready=False,reason=str(exc))
                yield result


def _compiled(case):
    from edl_charge.workflow import build_compiled
    return build_compiled(case['recipe'])


def _optimized(base,profile,directory,selected,config):
    if directory is None:
        yield _unavailable(base,"optimized","Set optimized_directory to a validated waveform export")
        return
    directory=Path(directory)
    paths=sorted(directory.glob("case_*.json"))
    if not paths:
        yield _unavailable(base,"optimized","No case_*.json in "+str(directory))
        return
    if config is None and (directory/"fitted_config.json").is_file():
        config=_load(directory/"fitted_config.json")
    matched=False
    for path in paths:
        case=_load(path)
        if case.get("sample_name")!=profile["sample_name"]:
            continue
        variant=case.get("variant")
        if variant not in VARIANTS or not selected.intersection(("optimized",variant)):
            continue
        matched=True
        name=_name(variant,f'{case["final_voltage_v"]:.9g}V',f'{case["charging_time_s"]:.12g}s',path.stem)
        try:
            unsigned=dict(case)
            signature=unsigned.pop("case_sha256",None)
            if signature!=_digest(unsigned):
                raise ValueError("Stored waveform case checksum mismatch")
            if case.get("status")!="validated" or case.get("compiled_replay",{}).get("validated") is not True:
                raise ValueError(f'Waveform unavailable: {case.get("status")}')
            if config is not None:
                from edl_charge.session_models import signed_config
                fitted=signed_config(config,case["final_voltage_v"])
                if fitted.get("sample_name")!=profile["sample_name"] or case.get("config_sha256")!=_digest(fitted):
                    raise ValueError("Fitted specimen configuration checksum or sample differs")
                amp_mode=config["hardware"]["use_current_amp"]
            else:
                amp_mode=case.get("use_current_amp")
            if amp_mode is not base["use_current_amp"]:
                raise ValueError("Missing or different current-path provenance; supply matching fitted_config.json")
            cap=profile["peak_bound_v"]
            if not 0<abs(case["final_voltage_v"])<=case["recipe"]["bound_v"]<=cap:
                raise ValueError("Stored waveform exceeds selected specimen voltage cap")
            seq=_compiled(case)
            if seq.actual_reset_s<=0 or seq.actual_discharge_s<=0:
                raise ValueError("A complete discharge and finite zero reset are required")
            yield dict(base,test_name=name,waveform_kind="compiled",compiled=seq,ready=True,
                       voltage_peak_v=seq.voltage_peak,total_duration_s=seq.duration_s,variant=variant,
                       optimized_case_file=str(path),case_sha256=signature,
                       final_voltage_v=case["final_voltage_v"],
                       charging_time_s=case["charging_time_s"])
        except (ValueError,KeyError,TypeError) as exc:
            yield _unavailable(base,name,str(exc))
    if not matched:
        yield _unavailable(base,"optimized","No stored waveforms match this specimen and requested variants")


def _calibration(base,profile,requests,config):
    from edl_charge.workflow import calibration_recipe,make_config
    if config is None:
        config=make_config(profile["sample_name"],base["use_current_amp"])
    if (config["sample_name"]!=profile["sample_name"] or
            config["hardware"]["use_current_amp"] is not base["use_current_amp"]):
        raise ValueError("Calibration waveform settings differ from the selected specimen/current path")
    stages=calibration_protocols(profile,base["use_current_amp"])["stages"]
    if not requests:
        # Expose connections and planned excitation levels without inventing standard
        # values, recording a passive stage, or applying uncalibrated specimen bias.
        for stage in stages:
            result=_unavailable(base,_name("calibration",stage["name"]),
                    "Select this recipe and supply waveform settings and actual standard values")
            result.update(stage=stage,connection_required=stage["connection_required"])
            yield result
        return
    for n,request in enumerate(requests):
        settings=dict(request)
        stage=settings.pop("stage")
        connection=settings.pop("connection",None)
        name=_name("calibration",stage,n)
        try:
            spec,seq=calibration_recipe(config,stage,**settings)
            if connection!=spec["connection_required"]:
                raise ValueError("Requested connection differs from required "+spec["connection_required"])
            if seq is None:
                raise ValueError("Passive baseline has no repeated generator Sync edge for the original acquisition")
            if seq.actual_reset_s<=0:
                raise ValueError("Calibration waveform must include zero reset")
            yield dict(base,test_name=name,waveform_kind="compiled",compiled=seq,ready=True,
                       voltage_peak_v=seq.voltage_peak,total_duration_s=seq.duration_s,stage=spec,
                       connection_required=connection)
        except (ValueError,TypeError,KeyError) as exc:
            yield _unavailable(base,name,str(exc))


def iter_waveform_cases(sample,use_current_amp,sweep_mode="native",optimized_directory=None,
                        selected_tests=("chirp",),*,calibration_config=None,calibration_requests=(),
                        arb_sample_rate_hz=160e6,arb_max_points=1_000_000,leg_duration_s=1.0):
    """Yield repeatable waveform cases. Check ready before invoking any generator output.

    selected_tests accepts chirp, sine, square, triangle, optimized, individual
    variant names, calibration or all. An unfit specimen/capacity failure produces
    an explicit unavailable case; waveform durations and amplitudes are preserved.
    calibration_config can be a loaded offline config or a JSON file path.
    """
    if not isinstance(use_current_amp,bool):
        raise ValueError("Current mode must be True or False")
    profile=get_profile("LLZO" if sample=="Ta_LLZO" else sample)
    if sweep_mode=="arb":
        sweep_mode="stored_arb"
    if sweep_mode not in ("native","stored_arb","both"):
        raise ValueError("sweep_mode must be native, stored_arb or both")
    if not math.isfinite(leg_duration_s) or leg_duration_s<=0:
        raise ValueError("Sweep leg must be positive and finite")
    if isinstance(selected_tests,str):
        selected_tests=(selected_tests,)
    selected=set(selected_tests)
    if "all" in selected:
        selected.update(("chirp","optimized","calibration"))
    allowed={"all","chirp","sine","square","triangle","optimized","calibration",*VARIANTS}
    if not selected or selected-allowed:
        raise ValueError("Specify known waveform tests: "+", ".join(sorted(allowed)))
    base=dict(sample_name=profile["sample_name"],use_current_amp=use_current_amp,repeat=True)
    config=(_load(calibration_config) if isinstance(calibration_config,(str,Path)) else calibration_config)
    cases=list(_sweeps(base,profile,sweep_mode,selected,leg_duration_s,arb_sample_rate_hz,arb_max_points))
    if selected.intersection(("optimized",*VARIANTS)):
        cases.extend(_optimized(base,profile,optimized_directory,selected,config))
    if "calibration" in selected:
        cases.extend(_calibration(base,profile,calibration_requests,config))
    # This explicit manual path keeps the original readout. Order the available
    # specimen waveforms by final voltage (chirps by Vpp/2); overshoot retains the
    # sample-wide limit. Standalone cases do not provide
    # the cumulative fitting and shared clock of the integrated campaign.
    levels=sorted(set(profile["chirp_vpp_levels"][i]/2.
                      for i in range(len(profile["chirp_vpp_levels"]))))
    def key(case):
        if not case["ready"]:
            return (2,0.,0,0.,0)
        connection=case.get("connection_required")
        if connection and connection not in ("sample","sample_open_circuit"):
            return (0,0.,0,0.,0)
        peak=abs(case.get("final_voltage_v",case["voltage_peak_v"]))
        level=next((v for v in levels if peak<=v+1e-12),peak)
        kind=0 if "stage" in case else 1 if "plan" in case else 2
        return (1,level,kind,abs(case.get("final_voltage_v",0.)),
                case.get("charging_time_s",0.))
    yield from sorted(cases,key=key)


