"""Offline all-six-session acquisition estimate. No instruments are opened."""
import argparse
import json
import math

from edl_charge.planning import duration_grid, final_voltage_grid, hardware_profile, load_profiles
from edl_charge.session_chirp import plan_windows
from estimate_edl_runtime import duration_rows


def estimate(overhead_s=2., settling_floor_s=0., leg_s=1., min_charge_s=0., use_current_amp=False):
    for v in (overhead_s, settling_floor_s, leg_s, min_charge_s):
        if not math.isfinite(v) or v < 0:
            raise ValueError("Specify finite nonnegative scenario parameters")
    if not .001 <= leg_s <= 250:
        raise ValueError("Native sweep legs must be between .001 and 250 seconds")
    minimum = max(hardware_profile(use_current_amp)["min_charge_time_s"], min_charge_s)
    grid = duration_grid(minimum) if minimum <= 1. else []
    times = [grid[i] for i in sorted({round(j*(len(grid)-1)/5) for j in range(6)})] if grid else []
    shape_views = sum(len(plan_windows(shape, leg_duration_s=leg_s))
                      for shape in ("sine", "square", "triangle"))
    samples = {}
    for name, profile in load_profiles().items():
        cap = profile["peak_bound_v"]
        targets = [sign*v for v in final_voltage_grid(cap) for sign in (1,-1)]
        levels = profile["chirp_vpp_levels"]
        optimized_s = 0.
        captures = 0
        for target in targets:
            fraction = (target/cap)**2
            hold = max(settling_floor_s, .1+.9*fraction)
            reset = max(settling_floor_s, .2+1.8*fraction)
            for charge in times:
                rate = min(160e6, 1200/charge, (1_000_000-8192)/(1.2*charge))
                discharge = max(minimum, 27/rate)
                for hold_s, views in ((hold, ["charge","hold"]), (hold, ["reset"]), (0., ["reset"])):
                    row = duration_rows([(charge,hold_s,discharge,reset)], views, overhead_s, None, 16, 2000)
                    optimized_s += row["single_shot_scenario_s"]
                    captures += row["single_shot_captures"]
        chirp_s = 0.
        for vpp in levels:
            reset = max(settling_floor_s, .2+1.8*(vpp/(2*cap))**2)
            chirp_s += shape_views*(4*leg_s+2*reset+overhead_s)
        samples[name] = dict(charge_targets_v=targets, chirp_vpp_levels=levels,
                             optimized_captures=captures, optimized_s=optimized_s,
                             chirp_captures=shape_views*len(levels), chirp_s=chirp_s,
                             scan_s=optimized_s+chirp_s)
    total = sum(r["scan_s"] for r in samples.values())
    return dict(active_budget_s=7200.,use_current_amp=use_current_amp,scan_s=total, remaining_for_other_work_s=7200.-total,
                selected_charge_times_s=times, samples=samples,
                assumed_extra_s_per_capture=overhead_s, assumed_settling_floor_s=settling_floor_s,
                chirp_leg_s=leg_s, assumed_positive_charge_peak_equals_target=True,
                minimum_charge_time_s=minimum, assumed_loaded_min_charge_s=min_charge_s,
                charge_grid_basis=("scenario using specified sample-loaded minimum" if min_charge_s
                                   else "nominal fixture before independent sample-loaded qualification"),
                calibration_cpu_and_extra_io_in_scan_estimate=False)


def adaptive_estimate(*, overhead_s=2., settling_floor_s=.05, leg_s=1., min_charge_s=.02,
                      standards_s=240., calibration_s_per_trace=12., fit_s_per_level=90.,
                      cpu_s_per_pair=2.,use_current_amp=False,small_signal_frequency_count=1):
    """Forecast from explicit assumptions only. No calibration evidence is produced."""
    from edl_charge.session_models import balanced_pairs
    from edl_charge.session_timing import allocate_sample_share, default_timing_policy, schedule_sample

    if not math.isfinite(standards_s) or not 0 <= standards_s < 7200:
        raise ValueError("Standards overhead must be finite and within the overall active budget")
    if not math.isfinite(settling_floor_s) or settling_floor_s <= 0:
        raise ValueError("Supply a positive illustrative settling floor for the adaptive forecast")
    names = list(load_profiles())
    policy = dict(default_timing_policy(),io_s_per_capture=overhead_s,cpu_s_per_pair=cpu_s_per_pair)
    elapsed = standards_s
    results = {}
    for i,name in enumerate(names):
        remaining = max(0.,7200.-elapsed)
        share = allocate_sample_share(names[i:],remaining,policy=policy,chirp_leg_s=leg_s,
                     planning_min_charge_s=min_charge_s,calibration_s_per_trace=calibration_s_per_trace,
                     fit_s_per_level=fit_s_per_level,use_current_amp=use_current_amp,
                     small_signal_frequency_count=small_signal_frequency_count)["by_sample"][name]
        # The estimator uses only these scenario inputs. They never enter a run configuration.
        config = dict(sample_name=name,hardware=dict(use_current_amp=use_current_amp,
                      current_bandwidth_hz=400e3 if use_current_amp else 6e9,
                      measured_step_rise_s=1e-8,min_charge_time_s=min_charge_s),
                      max_hold_s=60.,observed_settling_s=settling_floor_s)
        pairs = balanced_pairs(config)
        overhead = share["calibration_s"]+share["fit_s"]
        result = schedule_sample(config,pairs,remaining_s=max(0.,share["share_s"]-overhead),
                                 policy=policy,chirp_leg_s=leg_s)
        all_times=sorted({t for _,t in pairs})
        selected={t for _,t in result["selected_pairs"]}
        if not result["feasible"] and all_times:
            selected={all_times[0],all_times[-1]}
        fixed=dict(policy,minimum_scale=result["dwell_policy"]["scale"],maximum_scale=result["dwell_policy"]["scale"])
        level_rows=[]
        spent=0.
        selected_pairs=[]
        completed_chirps=0
        setup_spent=0.
        for level,traces in zip(share["voltage_levels_v"],share["calibration_traces_by_level"]):
            level_pairs=[(v,t) for v,t in pairs if abs(v)==level and t in selected]
            sweep_levels=[vpp for vpp in load_profiles()[name]["chirp_vpp_levels"] if vpp/2==level]
            setup=traces*calibration_s_per_trace+(fit_s_per_level if level in final_voltage_grid(load_profiles()[name]["peak_bound_v"]) else 0.)
            available=max(0.,share["share_s"]-spent)
            if available<=setup+policy["shutdown_reserve_s"]:
                setup_spent+=min(available,setup)
                spent+=min(available,setup)
                level_rows.append(dict(peak_v=level,status="incomplete_calibration_or_fit"))
                break
            setup_spent+=setup
            spent+=setup
            plan=schedule_sample(config,level_pairs,remaining_s=available-setup,policy=fixed,
                                 chirp_leg_s=leg_s,chirp_levels=sweep_levels)
            if not plan["feasible"]:
                plan=schedule_sample(config,[],remaining_s=available-setup,policy=fixed,
                                     chirp_leg_s=leg_s,chirp_levels=sweep_levels)
                if plan["feasible"]:
                    spent+=plan["cost_breakdown_s"]["total_s"]
                    completed_chirps+=plan["planned_chirp_captures"]
                level_rows.append(dict(peak_v=level,status="partial"))
                break
            spent+=plan["cost_breakdown_s"]["total_s"]
            selected_pairs.extend(tuple(pair) for pair in plan["selected_pairs"])
            completed_chirps+=plan["planned_chirp_captures"]
            level_rows.append(dict(peak_v=level,status="completed_selected_grid"))
        elapsed+=spent
        feasible=(len(level_rows)==len(share["voltage_levels_v"]) and
                  all(row["status"]=="completed_selected_grid" for row in level_rows))
        results[name] = dict(allocated_s=share["share_s"],estimated_calibration_fit_s=setup_spent,
                             full_calibration_fit_s=overhead,voltage_levels=level_rows,
                             estimate_s=spent,full_grid=feasible and len(selected_pairs)==len(pairs),feasible=feasible,
                             requested_pairs=len(pairs),selected_pairs=len(selected_pairs),
                             omitted_pairs=[pair for pair in pairs if pair not in selected_pairs],
                             scale=fixed["minimum_scale"],chirp_captures=completed_chirps,
                             cost_breakdown_s=dict(calibration_fit_s=setup_spent,scan_and_optimization_s=spent-setup_spent))
    return dict(forecast_only=True,use_current_amp=use_current_amp,estimated_active_s=elapsed,remaining_s=7200.-elapsed,
                all_grids_fit=all(row["full_grid"] for row in results.values()),
                assumed_standards_s=standards_s,assumed_fit_s_per_level=fit_s_per_level,
                assumed_calibration_s_per_trace=calibration_s_per_trace,assumed_min_charge_s=min_charge_s,
                assumed_settling_s=settling_floor_s,policy=policy,samples=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overhead-s", type=float, default=2.,
                        help="Extra setup/transfer/write seconds assumed per scan capture")
    parser.add_argument("--settling-floor-s", type=float,
                        help="Illustrative minimum active hold and zero reset in seconds")
    parser.add_argument("--leg-s", type=float, default=1.)
    parser.add_argument("--min-charge-s", type=float,
                        help="Assumed loaded-sample charge-time floor in seconds, applied to all specimens")
    parser.add_argument("--amp",action="store_true",help="Use the amplified current path (nominal 400 kHz limit)")
    parser.add_argument("--small-signal-frequency-count",type=int,default=1)
    parser.add_argument("--adaptive",action="store_true",help="Forecast the measured-floor scheduler using stated assumptions")
    parser.add_argument("--standards-s",type=float,default=240.)
    parser.add_argument("--calibration-s-per-trace",type=float,default=12.)
    parser.add_argument("--fit-s-per-level","--fit-s-per-sample",type=float,default=90.,help="Total fit time across both polarities per charging voltage level")
    parser.add_argument("--cpu-s-per-pair",type=float,default=2.)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.adaptive:
        result = adaptive_estimate(overhead_s=args.overhead_s,leg_s=args.leg_s,
                    min_charge_s=.02 if args.min_charge_s is None else args.min_charge_s,
                    settling_floor_s=.05 if args.settling_floor_s is None else args.settling_floor_s,
                    standards_s=args.standards_s,calibration_s_per_trace=args.calibration_s_per_trace,
                    fit_s_per_level=args.fit_s_per_level,cpu_s_per_pair=args.cpu_s_per_pair,
                    use_current_amp=args.amp,small_signal_frequency_count=args.small_signal_frequency_count)
        if args.json:
            print(json.dumps(result,indent=2))
        else:
            print("Adaptive forecast only. Time includes assumed standards, calibration, fitting, compute and scan I/O.")
            for name,row in result["samples"].items():
                print(f"{name:16} {row['selected_pairs']}/{row['requested_pairs']} pairs; "
                      f"{row['chirp_captures']} chirps; scale {row['scale']:.3f}; "
                      f"{row['estimate_s']/60:.2f} min")
            print(f"Estimated active time: {result['estimated_active_s']/60:.2f} min. "
                  f"All planned grids fit this scenario: {result['all_grids_fit']}.")
        return
    result = estimate(args.overhead_s,args.settling_floor_s or 0.,args.leg_s,args.min_charge_s or 0.,args.amp)
    if args.json:
        print(json.dumps(result,indent=2))
        return
    print("Sample             charge shots   chirp shots   conditional scan min")
    for name,row in result["samples"].items():
        print(f"{name:18} {row['optimized_captures']:10} {row['chirp_captures']:13} {row['scan_s']/60:22.2f}")
    print(f"All scan acquisitions: {result['scan_s']/60:.4f} min")
    print(f"Time left for standards, calibration, fitting, optimization and extra I/O: "
          f"{result['remaining_for_other_work_s']/60:.4f} min")


if __name__ == "__main__":
    main()
