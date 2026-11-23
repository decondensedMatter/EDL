"""EDL bench CLI. Use <entry> COMMAND --help to inspect the required inputs."""
import argparse
from pathlib import Path
from uuid import uuid4

from . import workflow as w
from .planning import get_profile


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command",required=True)
    def add(name,help_text,*keys):
        sub = subs.add_parser(name,help=help_text)
        for key in keys:
            sub.add_argument("--"+key,required=True)
        return sub
    q=add("plan","Write voltage/time grid and calibration recipes.","out")
    q.add_argument("--sample")
    add("init-config","Write editable specimen and hardware configuration.","sample","out")
    add("fit","Fit training data and check held-out waveform predictions.","config","manifest","out")
    q=add("optimize","Optimize and independently replay local minimum-work controls.","out")
    q.add_argument("--config")
    q.add_argument("--limit",type=int)
    q.add_argument("--demo",action="store_true",help="Historical 0.1 M ideal-terminal demonstration only")
    q=add("run","Acquire one view from a complete validated protocol.","config","case","out")
    q.add_argument("--view",choices=("charge","hold","discharge","reset","full"),required=True)
    q=add("run-campaign","Acquire selected views from each validated case with a persistent index.",
          "config","cases","out")
    q.add_argument("--views",nargs="+",choices=("charge","hold","discharge","reset","full"),
                   default=["charge","hold","reset"])
    q.add_argument("--limit",type=int,help="Maximum number of validated case files, before selecting views")
    add("analyze","Compare measured current to a replay driven by measured terminal voltage.",
        "config","case","capture","out")
    q=add("acquire-calibration","Capture a single named calibration recipe on the selected wiring.",
          "config","recipe","connection","out")
    for k in ("target-v","bias-v","frequency-hz","edge-s","hold-s","reset-s","standard-excitation-v","standard-resistance-ohm","standard-capacitance-f"):
        q.add_argument("--"+k,type=float)
    return p


def run_campaign(config, cases, views, out, limit=None):
    from .measure import acquire_protocol
    if limit is not None and limit < 1:
        raise ValueError("Campaign case limit must be positive")
    if not views or len(views)!=len(set(views)):
        raise ValueError("Specify distinct capture views")
    cases=Path(cases)
    entries=w.read(cases/"index.json")
    folder=Path(out)/("campaign_"+uuid4().hex)
    folder.mkdir(parents=True)
    index=dict(sample_name=config["sample_name"],config_sha256=w.digest(config),
               requested_views=views,status="running",records=[],skipped=[])
    def persist():
        temporary=w.save(folder/"campaign_index.tmp",index)
        temporary.replace(folder/"campaign_index.json")
    print("Campaign index:",folder/"campaign_index.json",flush=True)
    count=0
    try:
        persist()
        for row in entries:
            if limit is not None and count>=limit:
                break
            name=row["file"]
            if Path(name).name!=name or not name.endswith(".json"):
                raise ValueError("Invalid case filename in index")
            if row["status"]!="validated":
                index["skipped"].append(dict(file=name,reason=row["status"]))
                continue
            case=w.read(cases/name)
            cp=w.validate_case(config,case)
            count+=1
            for view in views:
                if view=="hold" and cp.actual_hold_s==0:
                    index["skipped"].append(dict(file=name,view=view,reason="no hold phase"))
                    continue
                record=dict(file=name,case_sha256=case["case_sha256"],view=view,status="started")
                index["records"].append(record)
                persist()
                view_spec=w.capture_view(cp,view,config["capture_points"])
                target=folder/("capture_"+str(len(index["records"])).zfill(5))
                record["output_directory"]=target.name
                persist()
                run=Path(acquire_protocol(cp,config["hardware"],target,dict(case,test_name=case["variant"]),view_spec))
                record["capture_directory"]=str(run.relative_to(folder))
                record["status"]=w.read(run/"metadata.json")["status"]
                if record["status"]!="captured":
                    raise RuntimeError(f"Campaign stopped at {name} / {view}: {record['status']}")
                persist()
        index["status"]="completed" if count else "no_validated_cases"
    except BaseException as exc:
        index.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        persist()
    return folder


def main(use_current_amp, argv=None, *, fixed_sample=None):
    p = parser()
    a = p.parse_args(argv)
    if fixed_sample and a.command in ("plan","init-config"):
        if a.sample and get_profile(a.sample)["sample_name"]!=fixed_sample:
            p.error("Selected sample differs from this entry script")
        a.sample=fixed_sample
    out = Path(a.out)
    if a.command=="plan":
        w.save(out,w.plan(a.sample,use_current_amp))
        return 0
    if a.command=="init-config":
        from .model_defaults import model_fit_defaults
        w.save(out,model_fit_defaults(w.make_config(a.sample,use_current_amp)))
        return 0
    if a.command=="optimize" and a.demo:
        if fixed_sample and fixed_sample!="NaClO4_0.1M":
            p.error("The historical demonstration belongs to the 0.1 M NaClO4 entry")
        if a.config:
            p.error("--demo has its own historical 0.1 M configuration")
        w.demo(out)
        return 0
    if not a.config:
        p.error("--config is required")
    config = w.read(a.config)
    if fixed_sample and config.get("sample_name")!=fixed_sample:
        p.error("Configuration belongs to another specimen")
    hw = config["hardware"]
    if hw["use_current_amp"] is not use_current_amp:
        p.error("Entry script and configured current path differ")
    if a.command=="fit":
        w.save(out,w.fit(config,a.manifest))
    elif a.command=="optimize":
        w.optimize(config,out,a.limit)
    elif a.command=="run-campaign":
        run_campaign(config,a.cases,a.views,out,a.limit)
    elif a.command=="acquire-calibration":
        spec, cp = w.calibration_recipe(config,a.recipe,target_v=a.target_v,bias_v=a.bias_v,
                   frequency_hz=a.frequency_hz,edge_s=a.edge_s,hold_s=a.hold_s,reset_s=a.reset_s,standard_excitation_v=a.standard_excitation_v,
                   standard_resistance_ohm=a.standard_resistance_ohm,standard_capacitance_f=a.standard_capacitance_f)
        if a.connection!=spec["connection_required"]:
            p.error("Manual physical connection must match recipe connection_required")
        meta=dict(status="validated",purpose="calibration",calibration_recipe=spec,
                  test_name={"open_circuit_voltage":"ocv","shorted_zero":"zero_short"}.get(a.recipe,a.recipe),
                  current_conversion_status="calibrated" if hw["data_hardware_calibrated"] else "nominal_uncalibrated",
                  sample_name=config["sample_name"],voltage_limit_v=get_profile(config["sample_name"])["peak_bound_v"],
                  calibration_id=config.get("calibration_id"),config_sha256=w.digest(config))
        if cp is None:
            if not a.hold_s or a.hold_s<=0:
                p.error("Specify a positive --hold-s for the passive baseline capture")
            from .measure import acquire_baseline
            view=dict(name=a.recipe,start_s=0.,duration_s=a.hold_s,
                      sample_rate_hz=min(20e9,config["capture_points"]/a.hold_s))
            acquire_baseline(hw,out,meta,view)
        else:
            from .measure import acquire_protocol
            acquire_protocol(cp,hw,out,meta,w.capture_view(cp,"full",config["capture_points"]))
    else:
        case=w.read(a.case)
        cp=w.validate_case(config,case)
        if a.command=="run":
            from .measure import acquire_protocol
            view=w.capture_view(cp,a.view,config["capture_points"])
            acquire_protocol(cp,hw,out,dict(case,test_name=case["variant"]),view)
        else:
            from .analysis import compare_capture,write_analysis
            meta=w.read(Path(a.capture)/"metadata.json")
            if meta.get("protocol",{}).get("case_sha256")!=case["case_sha256"]:
                raise ValueError("Capture metadata identifies a different protocol")
            ct,cv=cp.control_knots()
            from .session_models import signed_config
            measured_model=w.model_for(signed_config(config,case["final_voltage_v"]))
            case_model=w.ActuatedModel(measured_model.cell_model,case["effective_command_rise_s"])
            report=compare_capture(case_model,a.capture,ct,cv,current_response_rise_s=hw["current_response_rise_s"],
                         delay_s=hw["current_delay_s"],baseline_current_a=config.get("baseline_current_a",0.),
                         settling_noise_a=config.get("noise_a"),settling_window_s=config.get("settling_window_s"))
            write_analysis(report,out)
    return 0


