"""One active-runtime budget for standards, all six specimens, computation and scans."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from uuid import uuid4

from . import workflow as w
from .planning import load_profiles,final_voltage_grid


class BudgetExpired(TimeoutError):
    pass


class ActiveBudget:
    """Monotonic wall time; only explicitly marked manual sample changes are excluded."""
    def __init__(self, seconds=7200., *, clock=time.monotonic, cleanup_s=5.):
        if not 0 < seconds <= 7200 or not 0 <= cleanup_s < seconds:
            raise ValueError("The combined run budget must be positive and at most 7200 seconds")
        self.seconds, self.clock, self.cleanup_s = float(seconds), clock, cleanup_s
        self.started, self.paused_s, self.scope_end = clock(), 0., None

    def elapsed(self):
        return self.clock()-self.started-self.paused_s

    def remaining(self):
        end = self.seconds-self.cleanup_s
        if self.scope_end is not None:
            end = min(end,self.scope_end)
        return max(0.,end-self.elapsed())

    def check(self, required_s=0.):
        if self.remaining() <= max(0.,float(required_s)):
            raise BudgetExpired("Insufficient active runtime remaining; unfinished tests are retained in the journal")

    @contextmanager
    def scoped(self, seconds):
        old = self.scope_end
        self.scope_end = self.elapsed()+min(float(seconds),self.remaining())
        try:
            self.check()
            yield self
        finally:
            self.scope_end = old

    @contextmanager
    def sample_change(self):
        self.check()
        start = self.clock()
        try:
            yield
        finally:
            self.paused_s += self.clock()-start

    def prompt(self, message, sample_change=False):
        """Standards/operator checks count; sample changes alone pause the timer."""
        suffix = "\nPress Enter when connected, or type ABORT: "
        if sample_change:
            with self.sample_change():
                answer = input(message+suffix)
        else:
            self.check()
            replies = queue.Queue()
            def read():
                try:
                    replies.put((True,input(message+suffix)))
                except BaseException as exc:
                    replies.put((False,exc))
            threading.Thread(target=read,daemon=True).start()
            try:
                ok,answer = replies.get(timeout=self.remaining())
            except queue.Empty as exc:
                raise BudgetExpired("Runtime expired while waiting for a calibration connection") from exc
            if not ok:
                raise answer
        if str(answer).strip().lower() in ("abort","stop","quit"):
            raise KeyboardInterrupt("Operator stopped the session")
        self.check()


def template(use_current_amp=False,readout_mode="updated"):
    from .session_backend import sample_calibration_template,analysis_qualification_defaults
    from .session_calibration import standard_settings_template
    from .session_timing import default_timing_policy
    from .model_defaults import model_fit_defaults
    if type(use_current_amp) is not bool or readout_mode not in ("updated","legacy"):
        raise ValueError("Choose a current path and updated or legacy readout")
    samples = {name:model_fit_defaults(w.make_config(name,use_current_amp)) for name in load_profiles()}
    for cfg in samples.values():
        cfg["qualification"].update(analysis_qualification_defaults())
    return dict(schema_version=2,budget_seconds=7200.,use_current_amp=use_current_amp,readout_mode=readout_mode,
                voltage_stage_peak_policy='sample',
                charging_polarities=["positive","negative"],
                sample_change_time_excluded=True,standards=standard_settings_template(use_current_amp),
                samples=samples,hardware_settings=dict(ch2_scale_v_per_div=1.),sample_calibration=sample_calibration_template(),
                timing_policy=default_timing_policy(),
                shared_calibration_max_s=900.,fit_max_s=120.,optimize_pair_max_s=30.,
                chirp_fraction_of_sample_time=.40,chirp_leg_s=1.,
                hardware_current_limit_a=None,
                policy="All six samples share one active-time budget. Voltage caps only; no CV current-density cutoff.")


def readout_issues(config):
    mode=config.get("readout_mode","updated")
    if mode=="updated":
        return []
    if mode!="legacy":
        return ["readout_mode must be updated or legacy"]
    leg=float(config.get("chirp_leg_s",1.))
    return ["Unchanged readout cannot execute this integrated campaign: passive OCV/zero reference records lack the CH3 trigger required by its retained acquisition.",
            f"Its 256 averages and 100 s timeout are retained. Default native Sync intervals require at least {256*2*leg:g} s per chirp acquisition, before transfers; no readout settings were changed.",
            "The 120-minute legacy campaign is blocked before instrument access. --legacy-manual is explicitly outside the combined budget."]


def compute(operation, payload, folder, budget, maximum_s):
    """Only offline fitting/control calculation runs in a terminable subprocess."""
    budget.check()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    token=uuid4().hex
    source=w.save(folder/(token+"_input.json"),dict(operation=operation,payload=payload))
    target=folder/(token+"_result.json")
    log=folder/(token+"_worker.log")
    limit=min(float(maximum_s),budget.remaining())
    flags=getattr(subprocess,"CREATE_NO_WINDOW",0) if os.name=="nt" else 0
    package_root=Path(__file__).resolve().parents[1]
    with log.open("w",encoding="utf8") as stream:
        process=subprocess.Popen([sys.executable,"-B","-m","edl_charge.session_worker",str(source),str(target)],
            cwd=package_root,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags)
        started=time.monotonic()
        try:
            while process.poll() is None:
                budget.check()
                if time.monotonic()-started>=limit:
                    raise BudgetExpired(f"Offline {operation} exceeded its allocated computation time")
                time.sleep(min(.1,max(.001,budget.remaining())))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.)
    budget.check()
    if time.monotonic()-started > limit:
        raise BudgetExpired(f"Offline {operation} exceeded its allocated computation time")
    if not target.is_file():
        raise RuntimeError(f"Offline {operation} produced no result; see {log}")
    result=w.read(target)
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["result"]


def _capture_seconds(compiled,view):
    z=compiled.actual_reset_s
    spec=w.capture_view(compiled,view,2000)
    return z+max(0.,spec["start_s"]+spec["duration_s"])+compiled.duration_s+5.



def run(config, output, *, backend_factory=None, compute_fn=compute, clock=time.monotonic,
        prompt=None, preflight_fn=None, model_issues_fn=None):
    """One shared active clock; measured calibration and explicit incomplete-work records."""
    from .session_backend import BenchBackend,preflight,chirp_reset_seconds,model_input_issues
    from .session_chirp import plan_windows
    from .session_models import balanced_pairs,signed_config
    from .session_timing import default_timing_policy,schedule_sample,allocate_sample_share
    budget=ActiveBudget(config['budget_seconds'],clock=clock)
    folder=Path(output)/('all_samples_'+uuid4().hex)
    folder.mkdir(parents=True)
    journal=dict(status='preflight',budget_seconds=budget.seconds,sample_change_time_excluded=True,
                 use_current_amp=config['use_current_amp'],readout_mode=config.get('readout_mode','updated'),
                 configuration_sha256=w.digest(config),hardware_current_limit_a=None,
                 stages=[],samples={},schema_version=2)
    w.save(folder/'session_configuration.json',config)
    policy=dict(default_timing_policy(),**config.get('timing_policy',{}))
    def save():
        journal.update(active_elapsed_s=budget.elapsed(),excluded_sample_change_s=budget.paused_s,
                       remaining_s=budget.remaining())
        w.save(folder/'session_journal.tmp',journal).replace(folder/'session_journal.json')
    def record(kind,**fields):
        row=dict(kind=kind,started_active_s=budget.elapsed(),status='started',**fields)
        journal['stages'].append(row)
        save()
        return row
    try:
        issues=readout_issues(config)+(preflight_fn or preflight)(config)
    except Exception as exc:
        issues=[f'Invalid session configuration: {type(exc).__name__}: {exc}']
    if config.get('voltage_stage_peak_policy','sample') not in ('stage','sample'):
        issues.append('Invalid voltage stage peak policy')
    if issues:
        journal.update(status='preflight_failed',missing_or_invalid=issues)
        save()
        return folder
    sample_map=config['samples']
    names=list(sample_map)
    try:
        backend=(backend_factory or BenchBackend)(config,folder,budget,prompt or budget.prompt)
        with budget.scoped(config['shared_calibration_max_s']):
            item=record('shared_instrument_calibration')
            calibrated=backend.calibrate_hardware()
            w.require_response(calibrated)
            item.update(status='qualified',completed_active_s=budget.elapsed())
            w.save(folder/'instrument_calibration.json',calibrated)
            save()
        journal['status']='scanning'
        for idx,name in enumerate(names):
            budget.check()
            allocation=allocate_sample_share(names[idx:],budget.remaining(),policy=policy,
                chirp_leg_s=config['chirp_leg_s'],use_current_amp=config['use_current_amp'],
                small_signal_frequency_count=len(config['sample_calibration']['small_signal_frequencies_hz']))
            sample=dict(status='started',planned_pairs=None,completed_pairs=0,completed_polarity_pairs=0,polarity_pairs=[],captures=[],chirps=[],skipped=[],
                        allocation=allocation,timing_plans=[],dwell_extensions=[],computation_seconds=[])
            journal['samples'][name]=sample
            try:
                with budget.scoped(allocation['share_s']):
                    profile=load_profiles()[name]
                    charge_levels=final_voltage_grid(profile['peak_bound_v'])
                    levels=sorted(set(charge_levels) |
                                  {vpp/2 for vpp in profile['chirp_vpp_levels']})
                    sample.update(planned_pairs=[],planned_voltage_levels_v=levels,voltage_levels=[],planned_chirp_captures=sum(
                        len(plan_windows(shape,leg_duration_s=config['chirp_leg_s']))
                        for vpp in profile['chirp_vpp_levels'] for shape in ('sine','square','triangle')))
                    selected_indices=None
                    runtime_policy=dict(policy)
                    for index,level in enumerate(levels):
                        ceiling=level if config.get('voltage_stage_peak_policy','sample')=='stage' else profile['peak_bound_v']
                        level_folder=folder/name/'voltage_levels'/f'peak_{level:g}V'
                        level_row=dict(peak_v=level,optimized_limit_v=ceiling,status='started')
                        inventory=folder/name/'sample_calibration'/f'peak_{level:g}V'/'calibration_inventory.json'
                        level_row['inventory_path']=str(inventory.relative_to(folder))
                        sample['voltage_levels'].append(level_row)
                        skips_before=len(sample['skipped'])
                        level_budget_short=False
                        item=record('sample_calibration',sample=name,level_v=level)
                        before_cal=budget.elapsed()
                        specimen,manifest=backend.calibrate_sample(deepcopy(sample_map[name]),deepcopy(calibrated),level_v=level)
                        calibration_cost=budget.elapsed()-before_cal
                        fit_cost=0.
                        if Path(manifest).is_file():
                            sample['calibration_manifest']=str(manifest)
                        inventory=Path(manifest).parent/'calibration_inventory.json'
                        if inventory.is_file():
                            sample['calibration_inventory']=str(inventory)
                        item.update(status='captured',completed_active_s=budget.elapsed())
                        save()
                        problems=(model_issues_fn or model_input_issues)(specimen)
                        pairs=[]
                        branches={}
                        branch_paths={}
                        fit_status='missing_model_inputs' if problems else 'not_qualified'
                        loaded_evidence=dict(status='not_tested')
                        report_path=level_folder/(name+'_fit_report.json')
                        if level not in charge_levels:
                            fit_status='no_charge_target_at_level'
                            w.save(report_path,dict(status=fit_status))
                        elif problems:
                            sample['skipped'].append(dict(stage='optimization',reason='missing_model_inputs',details=problems))
                            w.save(report_path,dict(status=fit_status,details=problems))
                        else:
                            item=record('sample_fit_and_independent_validation',sample=name,level_v=level)
                            fitted=None
                            try:
                                before_fit=budget.elapsed()
                                fitted=compute_fn('fit',dict(config=specimen,manifest_path=manifest),folder/'compute',
                                                  budget,config['fit_max_s'])
                                fit_cost=budget.elapsed()-before_fit
                                evidence=fitted.get('fit_report',{}).get('sample_response_evidence',{})
                                loaded_evidence=dict(evidence,status=('passed' if evidence.get('passed') is True
                                                     else 'failed' if evidence.get('passed') is False else 'not_tested'))
                                w.save(level_folder/(name+'_fitted.json'),fitted)
                                if not fitted.get('model_calibrated') or evidence.get('passed') is not True:
                                    raise ValueError('Specimen model or loaded voltage response did not pass independent validation')
                                for polarity,v in (('positive',1.),('negative',-1.)):
                                    branch=signed_config(fitted,v)
                                    if (not branch.get('model_calibrated') or
                                        branch.get('fit_report',{}).get('sample_response_evidence',{}).get('passed') is not True):
                                        raise ValueError('Unqualified '+polarity+' specimen fit')
                                    w.model_for(branch)
                                    branch['campaign_peak_limit_v']=ceiling
                                    branches[polarity]=branch
                                    branch_paths[polarity]=str(w.save(level_folder/(name+'_'+polarity+'_fitted.json'),branch))
                                specimen=fitted
                                pairs=balanced_pairs(specimen)
                                fit_status='qualified'
                                item.update(status=fit_status,completed_active_s=budget.elapsed())
                                w.save(report_path,dict(status=fit_status,fit_report=fitted['fit_report'],fitted_configs=branch_paths))
                                if not pairs:
                                    sample['skipped'].append(dict(stage='optimized_grid',reason='No charge time within qualified sample range',
                                        qualified_min_charge_s=w.timing(specimen)['min_charge_time_s']))
                            except (BudgetExpired,ValueError,KeyError,RuntimeError,ArithmeticError) as exc:
                                level_budget_short=isinstance(exc,BudgetExpired)
                                item.update(status=fit_status,reason=str(exc))
                                w.save(report_path,dict(status=fit_status,reason=str(exc),
                                       fit_report=fitted.get('fit_report') if isinstance(fitted,dict) else None))
                                sample['skipped'].append(dict(stage='optimization',reason=str(exc)))
                                if isinstance(exc,RuntimeError) or loaded_evidence.get('passed') is False:
                                    level_row.update(fit_report=str(report_path),loaded_voltage_qualification=loaded_evidence)
                                    raise
                            save()
                        level_row.update(fit_report=str(report_path),calibration_manifest=str(manifest),fit_status=fit_status)
                        sample.update(sample_fit_status=fit_status,loaded_voltage_qualification=loaded_evidence,
                                      fitted_configs=branch_paths,fit_report=str(report_path))
                        available_chirps=[vpp for vpp in profile['chirp_vpp_levels'] if vpp/2>=level]
                        current_chirps=[vpp for vpp in available_chirps if vpp/2==level]
                        if selected_indices is None and pairs:
                            trace_counts=allocation['by_sample'][name]['calibration_traces_by_level']
                            future_reserve=(calibration_cost*sum(trace_counts[index+1:])/trace_counts[index]
                                            +fit_cost*sum(v>level for v in charge_levels))
                            initial=schedule_sample(specimen,pairs,remaining_s=budget.remaining(),policy=runtime_policy,
                                reserved_s=future_reserve,chirp_leg_s=config['chirp_leg_s'],chirp_levels=available_chirps)
                            sample['timing_plans'].append(initial)
                            selected_times={t for _,t in initial['selected_pairs']}
                            if not initial['feasible']:
                                # Preserve the endpoint duration groups at the lowest levels that fit.
                                all_times=sorted({t for _,t in pairs})
                                selected_times={all_times[0],all_times[-1]}
                            initial_times=sorted({t for _,t in pairs})
                            selected_indices=[initial_times.index(t)/(len(initial_times)-1 or 1) for t in sorted(selected_times)]
                            selected_scale=initial['dwell_policy']['scale']
                            runtime_policy.update(minimum_scale=selected_scale,maximum_scale=selected_scale)
                        level_pairs=[(v,t) for v,t in pairs if abs(v)==level]
                        sample['planned_pairs'].extend(dict(final_v=v,charge_s=t) for v,t in level_pairs)
                        level_times=sorted({t for _,t in level_pairs})
                        selected_times={level_times[round(q*(len(level_times)-1))] for q in selected_indices or []} if level_times else set()
                        pending=[(v,t) for v,t in level_pairs if t in selected_times]
                        for v,t in level_pairs:
                            if (v,t) not in pending:
                                sample['skipped'].append(dict(final_v=v,charge_s=t,reason='Budgeted charging-time coverage reduction'))
                        plan=schedule_sample(specimen,pending,remaining_s=budget.remaining(),policy=runtime_policy,
                            chirp_leg_s=config['chirp_leg_s'],chirp_levels=current_chirps)
                        sample['timing_plans'].append(plan)
                        active_policy=plan['dwell_policy']
                        save()
                        while pending:
                            budget.check(5.)
                            plan=schedule_sample(specimen,pending,remaining_s=budget.remaining(),policy=runtime_policy,
                                                 chirp_leg_s=config['chirp_leg_s'],chirp_levels=current_chirps)
                            sample['timing_plans'].append(plan)
                            omitted={tuple(x) for x in plan['omitted_pairs']}
                            if omitted:
                                level_budget_short=True
                            for v,t in omitted:
                                sample['skipped'].append(dict(final_v=v,charge_s=t,reason='Measured runtime required coverage reduction'))
                            pending=[tuple(p) for p in plan['selected_pairs']]
                            active_policy=plan['dwell_policy']
                            if not plan['feasible'] or not pending:
                                if not plan['feasible']:
                                    level_budget_short=True
                                    sample['skipped'].append(dict(stage='remaining_optimized_pairs',reason=plan['reason']))
                                save()
                                break
                            v,t=pending.pop(0)
                            magnitude=abs(v)
                            if (-v,t) not in pending:
                                raise ValueError('Timing plan lacks the opposite-polarity partner')
                            pending.remove((-v,t))
                            group=dict(magnitude_v=magnitude,charge_s=t,status='optimizing',captures_completed=0,
                                       signed_targets_v=[magnitude,-magnitude],case_files=[])
                            sample['polarity_pairs'].append(group)
                            chirp_config=dict(specimen,session_dwell_policy=active_policy)
                            chirp_reserve=sum(9*(4*config['chirp_leg_s']+2*chirp_reset_seconds(chirp_config,vpp/2)+
                                                runtime_policy['io_s_per_capture']) for vpp in current_chirps)+5.
                            jobs=[]
                            try:
                                budget.check(chirp_reserve+5.)
                                from .planning import VARIANTS
                                expected={VARIANTS[0]:['charge','hold'],VARIANTS[1]:['reset'],VARIANTS[2]:['reset']}
                                for target,sign in ((magnitude,'positive'),(-magnitude,'negative')):
                                    branch=branches[sign]
                                    path_to_fit=branch_paths[sign]
                                    item=record('optimize',sample=name,final_v=target,charge_s=t,fitted_config=path_to_fit)
                                    before=budget.elapsed()
                                    try:
                                        with budget.scoped(budget.remaining()-chirp_reserve):
                                            cases=compute_fn('optimize',dict(config=branch,final_v=target,charge_s=t,
                                                holds_policy=active_policy),folder/'compute',budget,config['optimize_pair_max_s'])
                                    except (BudgetExpired,ValueError,RuntimeError,ArithmeticError) as exc:
                                        item.update(status='rejected_or_timed_out',reason=str(exc))
                                        raise
                                    finally:
                                        sample['computation_seconds'].append(budget.elapsed()-before)
                                        runtime_policy['cpu_s_per_pair']=max(policy['cpu_s_per_pair'],
                                            sum(sample['computation_seconds'])/len(sample['computation_seconds']))
                                    item.update(status='computed',completed_active_s=budget.elapsed())
                                    if (len(cases)!=3 or {c.get('variant') for c in cases}!=set(expected) or
                                        any(c.get('capture_views')!=expected[c['variant']] or
                                            c.get('final_voltage_v')!=target or c.get('fit_polarity')!=sign or
                                            c.get('charging_time_s',t)!=t for c in cases)):
                                        raise ValueError('Incomplete three-variant/four-view group for '+sign)
                                    for case in cases:
                                        path=w.save(folder/'cases'/name/f'{len(sample["polarity_pairs"]):03}_{sign}_{case["variant"]}.json',case)
                                        group['case_files'].append(str(path))
                                        jobs.append((branch,path_to_fit,target,case,path))
                                if any(case.get('status')!='validated' for _,_,_,case,_ in jobs):
                                    raise ValueError('Matched polarity group not fully validated')
                                compiled_jobs=[(w.validate_case(branch,case),branch,fit,target,case,path)
                                               for branch,fit,target,case,path in jobs]
                                for cp,_,_,_,_,_ in compiled_jobs:
                                    if max(abs(cp.voltage_min),abs(cp.voltage_max))>ceiling+1e-10:
                                        raise ValueError('Compiled waveform exceeds campaign voltage ceiling')
                                # Distinct evidence: replay extensions are never labelled as measurements.
                                extended=[case['dwell_extension'] for _,_,_,case,_ in jobs if case.get('dwell_extension')]
                                if extended:
                                    sample['dwell_extensions'].append(dict(final_v_abs=magnitude,charge_s=t,extensions=extended))
                                    table=runtime_policy.setdefault('model_required_dwell_by_voltage_s',{})
                                    floor=table.setdefault(str(magnitude),{})
                                    for phase in ('hold_s','reset_s'):
                                        key='actual_'+phase
                                        values=[e[key] for e in extended if e.get(phase,0.)>0 or
                                                (phase=='reset_s' and e.get('reset_above_target_plan_s',0.)>0)]
                                        if values:
                                            floor[phase]=max(floor.get(phase,0.),max(values))
                                if extended:
                                    chirp_config=dict(specimen,session_dwell_policy=dict(active_policy,
                                        model_required_dwell_by_voltage_s=runtime_policy['model_required_dwell_by_voltage_s']))
                                    chirp_reserve=sum(9*(4*config['chirp_leg_s']+2*chirp_reset_seconds(chirp_config,vpp/2)+
                                        runtime_policy['io_s_per_capture']) for vpp in current_chirps)+5.
                                # Admission is atomic across both signs. Faults during actual capture
                                # retain completed records and leave the group explicitly incomplete.
                                group_cost=sum(_capture_seconds(cp,view)+max(0.,runtime_policy['io_s_per_capture']-5.)
                                    for cp,_,_,_,case,_ in compiled_jobs for view in case['capture_views'])
                                budget.check(group_cost+chirp_reserve)
                            except (BudgetExpired,ValueError,RuntimeError,ArithmeticError) as exc:
                                level_budget_short=level_budget_short or isinstance(exc,BudgetExpired)
                                group.update(status='skipped_before_output',reason=str(exc))
                                if isinstance(exc,RuntimeError):
                                    raise
                                sample['skipped'].append(dict(final_v_abs=magnitude,charge_s=t,reason=str(exc)))
                                save()
                                continue
                            group['status']='acquiring'
                            save()
                            for cp,branch,fit,target,case,path in compiled_jobs:
                                for view in case['capture_views']:
                                    budget.check(_capture_seconds(cp,view))
                                    item=record('optimized_capture',sample=name,case=str(path),final_v=target,
                                                fitted_config=fit,view=view)
                                    before=budget.elapsed()
                                    capture=Path(backend.capture_optimized(branch,case,view))
                                    spent=budget.elapsed()-before
                                    runtime_policy['io_s_per_capture']=max(runtime_policy['io_s_per_capture'],
                                                                             spent-max(0.,_capture_seconds(cp,view)-5.))
                                    status=w.read(capture/'metadata.json')['status']
                                    item.update(status=status,capture_directory=str(capture),completed_active_s=budget.elapsed())
                                    sample['captures'].append(dict(case=str(path),final_v=target,view=view,folder=str(capture),
                                                                   fitted_config=fit,status=status))
                                    if status!='captured':
                                        raise ValueError('Observed waveform or readout failed qualification')
                                    group['captures_completed']+=1
                                    if group['captures_completed'] in (4,8):
                                        sample['completed_pairs']+=1
                                    save()
                            sample['completed_polarity_pairs']+=1
                            group['status']='captured'
                            save()
                        # Sweep at all prescribed levels; every view repeats both round trips.
                        chirps=[(vpp,shape,view) for vpp in current_chirps
                                for shape in ('sine','square','triangle')
                                for view in plan_windows(shape,leg_duration_s=config['chirp_leg_s'])]
                    
                        active_policy=dict(active_policy,model_required_dwell_by_voltage_s=runtime_policy.get('model_required_dwell_by_voltage_s',{}))
                        chirp_config=dict(specimen,session_dwell_policy=active_policy,sample_fit_status=fit_status,
                                          loaded_voltage_qualification=loaded_evidence)
                        for vpp,shape,view in chirps:
                            peak=vpp/2
                            reset=chirp_reset_seconds(chirp_config,peak)
                            if reset>specimen['max_hold_s']:
                                sample['skipped'].append(dict(stage='chirp',vpp=vpp,shape=shape,view=view['name'],
                                            reason='Required reset exceeds the configured maximum',required_reset_s=reset))
                                raise ValueError('Required chirp reset exceeds the configured maximum')
                            budget.check(4*config['chirp_leg_s']+2*reset+5.)
                            item=record('chirp',sample=name,vpp=vpp,shape=shape,view=view['name'])
                            capture=Path(backend.capture_chirp(chirp_config,shape,peak,view))
                            status=w.read(capture/'metadata.json')['status']
                            item.update(status=status,capture_directory=str(capture),completed_active_s=budget.elapsed())
                            sample['chirps'].append(dict(vpp=vpp,shape=shape,view=view['name'],folder=str(capture),status=status))
                            save()
                            if status!='captured':
                                raise ValueError('Chirp voltage or acquisition validation failed')
                        for skipped in sample['skipped'][skips_before:]:
                            skipped['campaign_level_v']=level
                        if level_budget_short:
                            raise BudgetExpired('Current voltage level could not fit its selected work')
                        level_row['status']='completed' if len(sample['skipped'])==skips_before else 'partial'
                        save()
                    sample['status']='completed' if not sample['skipped'] else 'partial'
            except (BudgetExpired,TimeoutError) as exc:
                sample.update(status='partial_budget',reason=str(exc))
            except (ValueError,RuntimeError,ArithmeticError) as exc:
                sample.update(status='failed_qualification',reason=str(exc))
            for level_row in sample.get('voltage_levels',[]):
                if level_row['status']=='started':
                    level_row.update(status=sample['status'],reason=sample.get('reason'))
            for group in sample['polarity_pairs']:
                if group['status'] in ('acquiring','optimizing'):
                    group.update(status='unfinished',reason=sample.get('reason','Group interrupted'))
            save()
        journal['status']=('completed' if all(s['status']=='completed' for s in journal['samples'].values()) else 'partial')
    except (BudgetExpired,TimeoutError) as exc:
        journal.update(status='partial_budget',reason=str(exc))
    except KeyboardInterrupt:
        journal.update(status='interrupted')
    except Exception as exc:
        journal.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for item in journal['stages']:
            if item['status']=='started':
                item.update(status='unfinished',stopped_active_s=budget.elapsed())
        for name in names:
            journal['samples'].setdefault(name,dict(status='not_reached'))
        for row in journal['samples'].values():
            if row.get('status')=='started':
                row['status']='unfinished'
            for level_row in row.get('voltage_levels',[]):
                if level_row['status']=='started':
                    level_row['status']='unfinished'
            for group in row.get('polarity_pairs',[]):
                if group['status'] in ('acquiring','optimizing'):
                    group['status']='unfinished'
        standards=folder/'standards_inventory.json'
        inventories={name:[level['inventory_path'] for level in journal['samples'][name].get('voltage_levels',[])
                           if (folder/level['inventory_path']).is_file()] for name in names}
        manifest=dict(session_journal='session_journal.json',configuration='session_configuration.json',
                      standards_inventory=standards.name if standards.is_file() else None,
                      standards_inventory_status='present' if standards.is_file() else 'missing',
                      use_current_amp=config['use_current_amp'],readout_mode=config.get('readout_mode','updated'),
                      sample_calibrations={name:paths[-1] for name,paths in inventories.items() if paths},
                      sample_calibrations_by_voltage=inventories,
                      sample_inventory_status={name:'present' if paths else 'missing' for name,paths in inventories.items()},
                      status=journal['status'])
        w.save(folder/'analysis_manifest.json',manifest)
        save()
    return folder
