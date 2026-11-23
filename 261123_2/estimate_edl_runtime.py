"""Offline timing estimates for the six samples. Does not access instruments."""
import argparse
import json
import math
from pathlib import Path
from edl_charge.planning import build_candidate_plan, hardware_profile, load_profiles


def chirp(profile,leg,overhead,ascii_mib_s,bytes_per_sample,scope_depth):
    levels = profile["chirp_vpp_levels"]
    count = len(levels)*3
    period = 4*leg
    duration = period+.01
    points = math.ceil(duration/1e-8)+1
    transfer_s = (2*points*bytes_per_sample/(2**20*ascii_mib_s) if ascii_mib_s else None)
    return dict(voltage_levels=len(levels),shapes=3,acquisitions=count,round_trips_per_trace=2,
                native_stimulus_s=count*period,native_scope_points_per_channel=points,
                native_depth_status=("unverified" if scope_depth is None else
                                     "sufficient" if scope_depth>=points else "insufficient"),
                native_ascii_samples_transferred=2*count*points,
                native_ascii_transfer_s=transfer_s*count if transfer_s is not None else None,
                native_scenario_s=(count*(duration+overhead+(transfer_s or 0))
                                   if scope_depth is None or scope_depth>=points else None),
                legacy_nonoverlap_acquisition_only_s=count*256*3*period,
                legacy_ideal_min_digitize_s_per_trace=255*(period/2)+3*period,
                legacy_timeout_unavoidable=255*(period/2)+3*period>=100,
                stored_arb_required_samples=math.ceil(period*160e6),
                stored_arb_fits_standard_1M=period*160e6<=1_000_000,
                stored_arb_fits_optional_16M=period*160e6<=16_000_000)


def duration_rows(rows,views,overhead,ascii_mib_s,bytes_per_sample,points):
    shots,stim,bench,legacy,slow,pre,scopewait,guard_total,transfer_total = 0,0.,0.,0.,0,0.,0.,0.,0.
    for T,H,D,Z in rows:
        P = T+H+D+Z
        legacy += 256*3*P
        slow += int(255*P>=100)
        for view in views:
            offset,span = {"charge":(0,T),"hold":(T,H),"discharge":(T+H,D),
                           "reset":(P-Z,Z),"full":(0,P)}[view]
            if span<=0:
                continue
            # workflow.capture_view gives ±5% padding. ADER then triggers the
            # full-period guard; _save runs during that guard, not after it.
            view_end = offset+1.05*span
            sample_count = math.ceil(min(points,20e9*1.1*span))+1
            transfer = 3*sample_count*bytes_per_sample/(2**20*ascii_mib_s) if ascii_mib_s else 0.
            shots += 1
            stim += P
            pre += Z
            scopewait += view_end
            guard_total += P
            transfer_total += transfer
            bench += Z+view_end+max(P,transfer)+overhead
    return dict(protocols=len(rows),single_shot_captures=shots,
                stimulus_including_reset_s=stim,additional_pretrigger_reset_s=pre,
                trigger_to_view_end_s=scopewait,post_ader_guard_s=guard_total,
                ascii_transfer_s=transfer_total if ascii_mib_s else None,
                single_shot_scenario_s=bench,legacy_nonoverlap_acquisition_only_s=legacy,
                legacy_unavoidable_timeout_protocols=slow,
                legacy_full_scan_completion_status=("unavailable_at_100s_timeout" if slow else "unverified"))


def accepted(folder):
    from edl_charge.workflow import build_compiled
    groups = {}
    skipped = 0
    for entry in json.loads((folder/"index.json").read_text(encoding="utf8")):
        if entry["status"]!="validated":
            skipped += 1
            continue
        case = json.loads((folder/entry["file"]).read_text(encoding="utf8"))
        if case.get("status")!="validated" or not case.get("compiled_replay",{}).get("validated"):
            raise ValueError("index entry disagrees with compiled replay")
        c = build_compiled(case["recipe"])
        groups.setdefault(case["sample_name"],[]).append((c.actual_charge_s,c.actual_hold_s,
                                                           c.actual_discharge_s,c.actual_reset_s))
    return groups,skipped


def estimate(args):
    holds = args.hold_s
    if (not holds or any(not math.isfinite(x) or x<=0 for x in holds)
        or not math.isfinite(args.overhead_s) or args.overhead_s<0
        or not math.isfinite(args.leg_s) or args.leg_s<=0 or args.points<20
        or args.ascii_bytes_per_sample<=0 or (args.ascii_throughput_mib_s is not None and args.ascii_throughput_mib_s<=0)
        or (args.scope_depth is not None and args.scope_depth<=0)):
        raise ValueError("invalid timing assumptions")
    if len(set(args.views))!=len(args.views):
        raise ValueError("specify each capture view once")
    profiles = load_profiles()
    output = dict(assumptions=dict(hold_s=holds,zero_reset_equals_hold=True,
        per_shot_additional_overhead_s=args.overhead_s,views=args.views,scope_depth_per_channel=args.scope_depth,
        ascii_throughput_mib_s=args.ascii_throughput_mib_s,ascii_bytes_per_channel_sample=args.ascii_bytes_per_sample,
        chirp_leg_s=args.leg_s,calibration_reconnections_optimization_cpu_excluded=True,
        optimized_readout_points_requested=args.points,chirp_memory_readout_unverified_without_scope_depth=True,
        candidate_counts_assume_every_case_passes=True,hold_values_are_scenarios_not_material_measurements=True,
        legacy_time_basis="256 nonoverlapping three-period records per protocol, before overhead or retries",
        ascii_transfer_missing_when_no_throughput=True),samples=[])
    for name,p in profiles.items():
        row = dict(sample=name,chirp=chirp(p,args.leg_s,args.overhead_s,args.ascii_throughput_mib_s,
                                          args.ascii_bytes_per_sample,args.scope_depth),optimized={})
        for amp in (False,True):
            hw = hardware_profile(amp)
            cases = build_candidate_plan(name,amp)
            scenarios = []
            for hold in holds:
                periods = []
                for c in cases:
                    t = c["charge_time_s"]
                    rate = min(160e6,1200/t,(1_000_000-8192)/(1.2*t))
                    discharge = max(hw["min_charge_time_s"],27/rate)
                    h = 0 if c["variant"]=="charge_immediate_discharge" else hold
                    periods.append((t,h,discharge,hold))
                scenarios.append(dict(hold_s=hold,**duration_rows(periods,args.views,args.overhead_s,
                    args.ascii_throughput_mib_s,args.ascii_bytes_per_sample,args.points)))
            row["optimized"]["amp" if amp else "no_amp"] = dict(
                targets=len({c["final_voltage_v"] for c in cases}),durations=len({c["charge_time_s"] for c in cases}),
                minimum_charge_time_s=hw["min_charge_time_s"],cases=len(cases),scenarios=scenarios)
        output["samples"].append(row)
    if args.case_directory:
        groups,skipped = accepted(args.case_directory)
        output["validated_export"] = dict(source=str(args.case_directory),skipped_unvalidated=skipped,
            samples={k:duration_rows(v,args.views,args.overhead_s,args.ascii_throughput_mib_s,
                                     args.ascii_bytes_per_sample,args.points) for k,v in groups.items()})
    return output


def as_text(data):
    lines = ["Times in minutes unless marked s; candidate counts assume all controls pass.",
             "Optimized bench scenarios include pre-reset and full post-ADER guard. ASCII transfer is excluded unless throughput is supplied.",
             "Chirp time is conditional on scope record depth and throughput. Legacy 256-average full grids exceed the 100 s acquisition timeout.",
             "Sample | chirp traces / native stimulus / native bench | amp | candidates | hold(s): optimized bench"]
    for row in data["samples"]:
        c=row["chirp"]
        for label,opt in row["optimized"].items():
            dur="  ".join(f'{r["hold_s"]:g}s={r["single_shot_scenario_s"]/60:.2f}' for r in opt["scenarios"])
            bench=c["native_scenario_s"]
            text="blocked" if bench is None else f'{bench/60:.2f}'
            lines.append(f'{row["sample"]} | {c["acquisitions"]} / {c["native_stimulus_s"]/60:.2f} / {text} | {label} | {opt["cases"]} | {dur}')
    if "validated_export" in data:
        lines.append("Validated export: "+json.dumps(data["validated_export"]))
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hold-s",type=float,nargs="+",default=[.1,1.,10.])
    p.add_argument("--overhead-s",type=float,default=2.)
    p.add_argument("--leg-s",type=float,default=1.)
    p.add_argument("--views",nargs="+",choices=["charge","hold","discharge","reset","full"],default=["charge"])
    p.add_argument("--points",type=int,default=2000)
    p.add_argument("--ascii-throughput-mib-s",type=float)
    p.add_argument("--ascii-bytes-per-sample",type=float,default=16.)
    p.add_argument("--scope-depth",type=int)
    p.add_argument("--case-directory",type=Path,help="workflow index and validated case JSON files")
    p.add_argument("--json",action="store_true")
    args=p.parse_args()
    result=estimate(args)
    print(json.dumps(result,indent=2,allow_nan=False) if args.json else as_text(result))


if __name__=="__main__":
    main()
