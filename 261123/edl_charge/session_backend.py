"""Physical acquisition backend for a single deadline-controlled six-sample session."""
from copy import deepcopy
from hashlib import sha256
from os.path import relpath
from pathlib import Path

import numpy as np

from . import measure, session_calibration as standards, session_chirp, session_models
from . import workflow as w
from .model_defaults import model_fit_defaults

from .session_timing import dwell_seconds, proposed_dwell_seconds
from .planning import get_profile, load_profiles


def analysis_qualification_defaults():
    """Analysis acceptance policy, independent of measured specimen properties."""
    return dict(max_validation_noise_rmse=3.,max_condition_number=1e6,max_sample_voltage_error_fraction=.02)


def sample_calibration_template():
    return dict(initial_hold_s=.1, max_hold_s=8., edge_s=.002, capture_points=2000,
                small_signal_frequencies_hz=[16.], max_abs_ocv_v=.005)


def _samples(session):
    samples = session.get("samples") or {}
    return list(samples.values()) if isinstance(samples,dict) else list(samples)


def model_input_issues(cfg):
    """Return specimen-specific fit blockers; measured specimen data can still be collected."""
    issues = []
    def issue(message):
        issues.append(f"{cfg.get('sample_name')}: {message}")
    try:
        bounds = cfg.get("parameter_bounds") or {}
        if not bounds:
            issue("parameter_bounds must name the parameters to fit and their independent bounds")
        params = dict(cfg["model_parameters"])
        for name,limits in bounds.items():
            if name not in params or name in ("cells","mesh_grading"):
                issue(f"Unsupported fit parameter {name}")
                continue
            if params[name] is None:
                params[name] = (cfg.get("initial_guess") or {}).get(name)
            if len(limits)!=2 or not np.isfinite(limits).all() or limits[0]>=limits[1]:
                issue(f"Invalid bounds for {name}")
            elif params[name] is not None and not limits[0]<=params[name]<=limits[1]:
                issue(f"Initial {name} outside fit bounds")
            elif name not in cfg.get("linear_parameters",[]) and limits[0]<=0:
                issue(f"Log-space lower bound for {name} must be positive")
        missing = [k for k,val in params.items() if val is None and k!="site_concentration_mol_m3"]
        if missing:
            issue("Missing physical values or bounded fit seeds: "+", ".join(missing))
        else:
            w.factory(cfg)(params)
        for name in ("max_validation_noise_rmse","max_condition_number","max_sample_voltage_error_fraction"):
            val = (cfg.get("qualification") or {}).get(name)
            if val is None or not np.isfinite(val) or val<=0:
                issue(f"Supply qualification.{name}")
            elif name=="max_sample_voltage_error_fraction" and val>=1:
                issue("max_sample_voltage_error_fraction must be below one")
    except (KeyError, TypeError, ValueError) as exc:
        issue(str(exc))
    return issues


def preflight(sessionconfig, *, include_model_inputs=False):
    """Return all actionable pre-bench input issues without contacting equipment."""
    issues = []
    def finite(value):
        return type(value) in (int,float) and np.isfinite(value)
    for key in ("budget_seconds","shared_calibration_max_s","fit_max_s","optimize_pair_max_s","chirp_leg_s"):
        value = sessionconfig.get(key)
        if not finite(value) or value <= 0 or (key=="budget_seconds" and value>7200):
            issues.append(f"Invalid {key}, require finite positive seconds (combined budget <=7200)")
    fraction = sessionconfig.get("chirp_fraction_of_sample_time",.4)
    if not finite(fraction) or not 0<=fraction<=1:
        issues.append("chirp_fraction_of_sample_time must be in [0,1]")
    if sessionconfig.get("sample_change_time_excluded") is not True:
        issues.append("sample_change_time_excluded must be true")
    try:
        timing_policy = sessionconfig.get("timing_policy") or {}
        if not isinstance(timing_policy,dict):
            raise ValueError("timing_policy must be a mapping")
        # Exercise the same public validator and curve used by actual acquisitions.
        for phase in ("hold_s","reset_s"):
            proposed = proposed_dwell_seconds(1.,1.,phase,timing_policy)
            limit = (sessionconfig.get("sample_calibration") or {}).get("max_hold_s")
            if finite(limit) and proposed>limit:
                raise ValueError(f"{phase} proposal exceeds the calibration retry ceiling")
    except (TypeError,ValueError,KeyError) as exc:
        issues.append(f"Invalid timing_policy: {exc}")
    if finite(sessionconfig.get("chirp_leg_s")) and not .001<=sessionconfig["chirp_leg_s"]<=250:
        issues.append("chirp_leg_s outside the native sweep range [0.001,250] seconds")
    amp = sessionconfig.get("use_current_amp")
    if type(amp) is not bool:
        issues.append("Choose a boolean use_current_amp mode for this session")
    if sessionconfig.get("hardware_current_limit_a") is not None:
        issues.append("hardware_current_limit_a must be null")
    mapping = sessionconfig.get("samples",{})
    if not isinstance(mapping,dict) or any(not isinstance(v,dict) or k!=v.get("sample_name")
                                           for k,v in mapping.items()):
        issues.append("samples must be keyed by the matching canonical sample_name")
    st = sessionconfig.get("standards") or {}
    if st.get("use_current_amp",False) is not amp:
        issues.append("Standards use_current_amp must match the selected measurement path")
    required = {
        "resistor": ("ohm","source"), "holdout_resistor": ("ohm","source"),
        "rc": ("ohm","farad","source","topology"),
        "buffer": ("source","series_resistance_ohm","minimum_input_resistance_ohm"),
        "adc_rails_v": ("CH1","CH2")}
    for group,fields in required.items():
        for field in fields:
            value = (st.get(group) or {}).get(field)
            if value is None or value=="" or (isinstance(value,list) and None in value):
                issues.append(f"Supply standards.{group}.{field}")
    for key in ("adc_rails_source","max_reference_error_v"):
        if not st.get(key):
            issues.append(f"Supply standards.{key}")
    for j,ref in enumerate(st.get("voltage_references") or []):
        if ref.get("value_v") is None or not ref.get("source"):
            issues.append(f"Supply standards.voltage_references[{j}] independently measured value and source")
    try:
        standards.standard_requests(st)
        reference_v = [r["value_v"] for r in st["voltage_references"][:-1]]
        cap = max(p["peak_bound_v"] for p in load_profiles().values())
        if min(reference_v)>-cap or max(reference_v)<cap:
            issues.append(f"DC training references must span at least -{cap:g} to +{cap:g} V for all six specimens")
    except (TypeError, ValueError, KeyError) as exc:
        issues.append(f"Instrument standards: {exc}")
    samples = [cfg for cfg in _samples(sessionconfig) if isinstance(cfg,dict)]
    names = [c.get("sample_name") for c in samples]
    if len(names)!=6 or set(names)!=set(load_profiles()):
        issues.append("Supply one configuration for each of the six named specimens")
    for cfg in samples:
        sample = str(cfg.get("sample_name"))
        def issue(message):
            issues.append(f"{sample}: {message}")
        if cfg.get("hardware",{}).get("use_current_amp") is not amp:
            issue("hardware use_current_amp differs from the selected measurement path")
        try:
            current_profile = get_profile(sample)
            if cfg.get("profile",{}).get("peak_bound_v") != current_profile["peak_bound_v"]:
                issue("Update the sample profile peak_bound_v to the current session voltage limit")
            hw = dict(transfer_chunk_points=8192,voltage_tolerance_v=0.)
            hw.update(cfg.get("hardware") or {})
            hw.update(sessionconfig.get("hardware_settings") or {})
            measure._validate(None,hw,{},dict(name="preflight",start_s=0.,duration_s=.002,sample_rate_hz=1e6))
        except (KeyError,TypeError,ValueError) as exc:
            issue(f"Hardware or profile settings: {exc}")
        if include_model_inputs:
            issues.extend(model_input_issues(cfg))
    policy = sessionconfig.get("sample_calibration") or {}
    for key in ("max_abs_ocv_v","initial_hold_s","max_hold_s","edge_s","capture_points"):
        v = policy.get(key)
        if v is None or not np.isfinite(v) or v<=0:
            issues.append(f"Supply positive sample_calibration.{key}")
    if (policy.get("max_hold_s") or 0)<(policy.get("initial_hold_s") or 0):
        issues.append("max_hold_s must cover initial_hold_s")
    if (policy.get("capture_points") or 0)<256:
        issues.append("Sample calibration requires at least 256 points")
    freqs = policy.get("small_signal_frequencies_hz") or []
    if not freqs or any(f not in {4.**j for j in range(11)} for f in freqs):
        issues.append("Choose sample small-signal frequencies from 1,4,...,1048576 Hz")
    if sessionconfig.get("hardware_settings",{}).get("use_current_amp",amp) is not amp:
        issues.append("Hardware use_current_amp must match the selected measurement path")
    return issues


def chirp_reset_seconds(config,cell_peak):
    cap = get_profile(config["sample_name"])["peak_bound_v"]
    if not np.isfinite(cell_peak) or not 0<cell_peak<=cap:
        raise ValueError("Chirp peak outside the specimen operating range")
    if config.get("session_dwell_policy") is not None:
        return dwell_seconds(config,config["session_dwell_policy"],cell_peak,"chirp_reset_s")
    return max(.2+1.8*(cell_peak/cap)**2,config["observed_settling_s"],
               session_models._floor(config,{},cell_peak,"reset_s"))


class _Inventory:
    """Incremental, relocatable map from experiment roles to original scope files."""
    purposes = dict(short="instrument zero and baseline noise",resistor="signed current transfer",
        holdout_resistor="independent transfer validation",open="open-fixture leakage and capacitance",
        buffer_loaded="OCV buffer input loading",buffered_ocv="sample open-circuit voltage",
        driven_zero="sample baseline under driven zero",sample_positive_steps="target-specific hold and reset settling",
        sample_negative_steps="negative target-specific hold and reset settling",
        sample_bipolar_steps="mixed-polarity diagnostic",sample_small_signal="bias-dependent frequency response",
        negative_step_holdout="independent negative-step test",positive_step_holdout="independent positive-step test",
        negative_ramp_holdout="independent negative-ramp test",ramp_holdout="independent positive-ramp test")

    def __init__(self,path,**header):
        self.path = Path(path)
        self.data = dict(schema_version=1,status="collecting",records=[],**header)
        self.write()

    def relative(self,path):
        return Path(relpath(Path(path).resolve(), self.path.parent.resolve())).as_posix()

    def write(self):
        temp = w.save(self.path.with_suffix(".tmp"),self.data)
        temp.replace(self.path)

    def acquire(self,folder,meta):
        protocol = meta.get("protocol",{})
        name = protocol.get("test_name","unspecified")
        stage = protocol.get("calibration_stage") or protocol.get("standard") or {}
        role = self.purposes.get(name)
        if role is None:
            if name.startswith("reference_"):
                role = "external voltage calibration"
            elif name.startswith("rc_"):
                role = "known series RC response"
        row = dict(name=name,purpose=role or name,status=meta.get("status"),recipe=stage,polarity=stage.get("polarity"),
                   metadata=self.relative(folder/"metadata.json"),started_utc=meta.get("started_utc"),
                   capture_view=meta.get("capture_view"),observed=meta.get("observed"),
                   sample_rate_hz=meta.get("scope_sample_rate_hz"),
                   timing={k:meta[k] for k in ("actual_charge_s","actual_hold_s","actual_discharge_s",
                       "actual_reset_s","protocol_duration_s","zero_reset_duration_s") if k in meta})
        for key in ("capture_csv","raw_channels_csv"):
            if meta.get(key):
                path = folder/meta[key]
                row[key] = self.relative(path)
                if path.is_file() and key=="raw_channels_csv":
                    row["raw_sha256"] = sha256(path.read_bytes()).hexdigest()
        self.data["records"].append(row)
        self.write()

    def annotate(self,**values):
        self.data["records"][-1].update(values)
        self.write()


class BenchBackend:
    def __init__(self,sessionconfig,outputroot,budget,prompt):
        self.session = sessionconfig
        self.out = Path(outputroot)
        self.budget = budget
        self.prompt = prompt
        self.inventory = None
        self.sample_state = {}

    def _accepted(self,folder):
        self.budget.check()
        folder = Path(folder)
        meta = w.read(folder/"metadata.json")
        if self.inventory is not None:
            self.inventory.acquire(folder,meta)
        if meta.get("status")!="captured":
            raise ValueError(f"Acquisition failed qualification in {folder}: {meta.get('status')}")
        return folder, meta

    def _raw(self,folder):
        self.budget.check()
        return standards._load(folder,self.session["standards"])[0]

    def _collect(self,inventory,acquire,*args):
        self.inventory = inventory
        try:
            result = acquire(*args)
            inventory.data["status"] = "completed"
            return result
        except BaseException as exc:
            inventory.data.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                                  error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.inventory = None
            inventory.write()

    def calibrate_hardware(self):
        inventory = _Inventory(self.out/"standards_inventory.json",scope="shared instrument transfer",use_current_amp=self.session["use_current_amp"],
                               standards=self.session["standards"])
        return self._collect(inventory,self._calibrate_hardware)

    def _calibrate_hardware(self):
        cfg = deepcopy(_samples(self.session)[0])
        hw = cfg["hardware"]
        hw.update(self.session.get("hardware_settings",{}))
        if hw["use_current_amp"] is not self.session["use_current_amp"]:
            raise ValueError("Specimen current readout and requested hardware mode differ")
        st = self.session["standards"]
        hw.setdefault("voltage_tolerance_v",st["max_reference_error_v"])
        captures = {}
        for req in standards.standard_requests(st):
            self.budget.check()
            self.prompt("Connect "+req["connection"]+". Confirm wiring and standard identity.")
            self.budget.check()
            key = req["key"]
            meta = dict(sample_name="instrument_standards",test_name=key,standard=req,
                        voltage_limit_v=get_profile(cfg["sample_name"])["peak_bound_v"],status="validated")
            cp = None
            if not req["passive"]:
                _,cp = w.calibration_recipe(cfg,req["stage"],edge_s=req["edge_s"],hold_s=req["hold_s"],
                        standard_resistance_ohm=req.get("resistance_ohm"),
                        standard_capacitance_f=req.get("capacitance_f"),frequency_hz=req.get("frequency_hz"),
                        standard_excitation_v=req["standard_excitation_v"])
            view = standards.standard_view(req,cp,st)
            opts = dict(deadline_remaining_s=self.budget.remaining)
            if cp is None:
                folder = measure.acquire_baseline(hw,self.out,meta,view,**opts)
            else:
                folder = measure.acquire_protocol(cp,hw,self.out,meta,view,**opts)
            self._accepted(folder)
            captures[key] = folder if cp is None else (folder,cp)
        cfg = standards.process_standards(cfg,st,captures,check=self.budget.check)
        lo,hi = st["adc_rails_v"]["CH2"]
        gain = cfg["hardware"]["voltage_scale_factor"]
        offset = cfg["hardware"]["voltage_offset_v"]
        cap = max(p["peak_bound_v"] for p in load_profiles().values())
        if not (lo<offset-cap/gain and offset+cap/gain<hi):
            raise ValueError("CH2 raw ADC range does not cover the full calibrated specimen voltage window")
        w.save(self.out/"hardware_calibration.json",cfg)
        self.inventory.data.update(calibration_id=cfg["calibration_id"],calibration_result="hardware_calibration.json")
        return cfg

    def _sample_trace(self,cfg,stage,kwargs,polarity,hold,reset,policy):
        """Choose a view resolving the stimulus edges and complete zero-reset tail."""
        hw = cfg["hardware"]
        points = int(policy["capture_points"])
        edge = policy["edge_s"]
        for _ in range(8):
            spec,cp = w.calibration_recipe(cfg,stage,hold_s=hold,reset_s=reset,edge_s=edge,**kwargs)
            rate = max((points-16)/cp.duration_s,32*kwargs.get("frequency_hz",0))
            if np.ceil(rate*cp.duration_s)+16>hw.get("max_capture_points",100000):
                raise ValueError("Sample calibration record cannot resolve excitation within scope point limit")
            if rate*edge>=16:
                break
            edge = max(edge*1.1,18/rate)
        else:
            raise ValueError(f"Cannot resolve {stage} at selected capture depth")
        actual = ("mixed" if cp.voltage_min<0<cp.voltage_max else
                  "negative" if cp.voltage_min<0 else "positive")
        if actual!=polarity:
            raise ValueError(f"Calibration waveform {stage} has {actual} voltage, expected {polarity}")
        spec = dict(spec,polarity=polarity,actual_peak_min_v=cp.voltage_min,actual_peak_max_v=cp.voltage_max)
        view = dict(name=stage,start_s=-12/rate,duration_s=cp.duration_s+14/rate,sample_rate_hz=rate)
        meta = dict(sample_name=cfg["sample_name"],test_name=stage,status="validated",
                    voltage_limit_v=w.peak_limit(cfg),calibration_stage=spec)
        self.budget.check(cp.duration_s)
        folder = measure.acquire_protocol(cp,hw,self.out,meta,view,deadline_remaining_s=self.budget.remaining)
        folder,record = self._accepted(folder)
        self._raw(folder)
        path = folder/record["capture_csv"]
        return dict(path=str(path.resolve()),stage=stage,split=spec["split"],polarity=polarity,
                    target_v=kwargs.get("target_v"),bias_v=kwargs.get("bias_v"),frequency_hz=kwargs.get("frequency_hz"),
                    edge_s=edge,compiled=cp,programmed_voltage_gain=record["hardware_settings"]["voltage_gain"],
                    **{k:record[k] for k in ("actual_charge_s","actual_hold_s","actual_discharge_s","actual_reset_s")})

    @staticmethod
    def _settled(row,cfg):
        from .calibration import read_calibration_csv
        trace = read_calibration_csv(row["path"],row)
        t,v,i = trace.times_s,trace.voltage_v,trace.current_a
        evidence = cfg["hardware"]["calibration_evidence"]
        base = cfg["baseline_current_a"]
        def tail(lo,hi,target):
            idx = np.flatnonzero((t>=lo)&(t<=hi))
            last = idx[-max(5,len(idx)//10):]
            return (len(last)>=5 and np.max(abs(v[last]-target))<=5*evidence["baseline_voltage_noise_v"]
                    and np.max(abs(i[last]-base-evidence.get("open_parallel_conductance_s",0.)*target))<=5*evidence["baseline_current_noise_a"])
        reset = row["actual_charge_s"]+row["actual_hold_s"]+row["actual_discharge_s"]
        ok = tail(reset,reset+row["actual_reset_s"],0.)
        if row["stage"] in ("sample_positive_steps","sample_negative_steps"):
            ok = ok and tail(row["edge_s"],row["actual_charge_s"],row["target_v"])
        return ok

    def calibrate_sample(self,sampleconfig,hardwareconfig,*,level_v):
        sample = sampleconfig["sample_name"]
        cfg = dict(sampleconfig,campaign_peak_limit_v=level_v)
        level = w.peak_limit(cfg)
        previous = self.sample_state.get(sample)
        if previous and level<=previous[0]:
            raise ValueError("Specimen calibration levels must increase")
        out = self.out/sample/"sample_calibration"/f"peak_{level:g}V"
        inventory = _Inventory(out/"calibration_inventory.json",
                   sample_name=sample,calibration_id=hardwareconfig["calibration_id"],
                   use_current_amp=hardwareconfig["hardware"]["use_current_amp"],
                   sample_policy=self.session["sample_calibration"],analysis_acceptance=cfg["qualification"],
                   fit_input_issues=model_input_issues(cfg),campaign_peak_limit_v=level,
                   requested_positive_targets_v=[level],requested_negative_targets_v=[-level])
        inventory.data["shared_hardware_result"] = inventory.relative(self.out/"hardware_calibration.json")
        inventory.data["shared_standards_inventory"] = inventory.relative(self.out/"standards_inventory.json")
        return self._collect(inventory,self._calibrate_sample,cfg,hardwareconfig,previous,out)

    def _baseline(self,cfg):
        hw = cfg["hardware"]
        sample = cfg["sample_name"]
        policy = self.session["sample_calibration"]
        self.budget.check()
        self.prompt(f"Change specimen to {sample} and connect the verified OCV buffer with generator disconnected.",
                    sample_change=True)
        self.budget.check()
        view = dict(name="buffered_ocv",start_s=0.,duration_s=.05,sample_rate_hz=20000.)
        meta = dict(sample_name=sample,test_name="buffered_ocv")
        folder = measure.acquire_baseline(hw,self.out,meta,view,deadline_remaining_s=self.budget.remaining)
        ocv_folder,ocv_meta = self._accepted(folder)
        self._raw(folder)
        a = np.loadtxt(ocv_folder/ocv_meta["capture_csv"],delimiter=",",skiprows=1,ndmin=2)
        evidence = hw["calibration_evidence"]
        vnoise = evidence["baseline_voltage_noise_v"]
        block_means = np.array([np.mean(block) for block in np.array_split(a[:,2],8)])
        mean_ocv = float(np.mean(a[:,2])-evidence.get("shorted_voltage_mean_v",0.))
        spread = float(np.ptp(block_means))
        uncertainty = max(float(3*np.std(block_means,ddof=1)),
                          evidence.get("shorted_block_mean_spread_v",0.))
        self.inventory.annotate(mean_ocv_v=mean_ocv,block_mean_range_v=spread,
                                ocv_uncertainty_v=uncertainty,absolute_limit_v=policy["max_abs_ocv_v"])
        if abs(mean_ocv)+uncertainty>policy["max_abs_ocv_v"] or spread>5*vnoise:
            raise ValueError(f"{sample} OCV outside declared symmetric zero-voltage tolerance")
        path = "current amplifier" if hw["use_current_amp"] else "direct 50-ohm sense path"
        self.prompt(f"Reconnect generator and the calibrated {path} for the installed specimen.")
        self.budget.check()
        zero_hold = policy["initial_hold_s"]
        evidence = hw["calibration_evidence"]
        inoise,vnoise = (evidence[k] for k in ("baseline_current_noise_a","baseline_voltage_noise_v"))
        if min(inoise,vnoise)<=0:
            raise ValueError("Driven zero requires resolved standard noise")
        while True:
            folder = measure.acquire_zero_reset(hw,self.out,dict(sample_name=sample,test_name="driven_zero"),
                    zero_hold,deadline_remaining_s=self.budget.remaining)
            zero_folder,zero_meta = self._accepted(folder)
            self._raw(folder)
            a = np.loadtxt(zero_folder/zero_meta["capture_csv"],delimiter=",",skiprows=1,ndmin=2)
            i,v = a[:,1],a[:,2]
            if (np.max(abs(v))<=5*vnoise and np.max(abs(i))<=5*inoise and
                    abs(np.mean(i[:len(i)//2])-np.mean(i[len(i)//2:]))<=inoise):
                self.inventory.annotate(electrically_settled=True)
                break
            self.inventory.annotate(electrically_settled=False)
            if zero_hold>=policy["max_hold_s"]:
                raise ValueError("Driven zero does not agree with independently measured baseline")
            zero_hold = min(2*zero_hold,policy["max_hold_s"])
        cfg.update(baseline_current_a=float(np.median(i)),noise_a=max(float(np.std(i,ddof=1)),inoise),
                   sample_baseline_evidence=dict(ocv_folder=str(ocv_folder),zero_folder=str(zero_folder),
                                driven_zero_s=zero_hold,measured_ocv_v=mean_ocv))
    def _calibrate_sample(self,sampleconfig,hardwareconfig,previous,out):
        level = sampleconfig["campaign_peak_limit_v"]
        cfg = deepcopy(previous[1] if previous else sampleconfig)
        cfg["campaign_peak_limit_v"] = level
        cfg["previous_campaign_peak_v"] = previous[0] if previous else 0.
        cfg["hardware"] = deepcopy(hardwareconfig["hardware"])
        cfg["calibration_id"] = hardwareconfig["calibration_id"]
        sample = cfg["sample_name"]
        cap = get_profile(sample)["peak_bound_v"]
        policy = self.session["sample_calibration"]
        if previous is None:
            self._baseline(cfg)
        else:
            self.inventory.data["previous_manifest"] = self.inventory.relative(previous[2])
        amp = min(.01,level/10)
        jobs = [("sample_positive_steps",dict(target_v=level),"positive",level),
                ("sample_negative_steps",dict(target_v=-level),"negative",level)]
        biases = []
        if previous is None:
            jobs.append(("sample_bipolar_steps",{},"mixed",min(.1,level)))
            biases.extend((0.,level/2,-level/2))
        if level==cap:
            biases.extend((level-amp,-(level-amp)))
        for bias in biases:
            polarity = "positive" if bias>0 else "negative" if bias<0 else "mixed"
            for f in policy["small_signal_frequencies_hz"]:
                jobs.append(("sample_small_signal",dict(bias_v=bias,frequency_hz=f),polarity,abs(bias)+amp))
        # Both independent holdout families are established on each sign at the first level.
        # Add new held-out excitations at each higher level without repeating earlier evidence.
        step_tests = (("positive_step_holdout","positive",.63),("negative_step_holdout","negative",.63))
        ramp_tests = (("ramp_holdout","positive",.75),("negative_ramp_holdout","negative",.75))
        prior_levels = len(cfg["settling_by_voltage_s"]) if previous else 0
        holdouts = step_tests+ramp_tests if not prior_levels else (step_tests if prior_levels%2 else ramp_tests)
        for stage,polarity,fraction in holdouts:
            lower = cfg["previous_campaign_peak_v"]
            jobs.append((stage,{},polarity,lower+fraction*(level-lower)))
        rows = []
        timing = self.session.get("timing_policy")
        for stage,kwargs,polarity,peak in jobs:
            hold = max(policy["initial_hold_s"],proposed_dwell_seconds(peak,cap,"hold_s",timing))
            reset = proposed_dwell_seconds(peak,cap,"reset_s",timing)
            if max(hold,reset)>policy["max_hold_s"]:
                raise ValueError("Voltage-dependent calibration dwell exceeds the configured retry ceiling")
            while True:
                row = self._sample_trace(cfg,stage,kwargs,polarity,hold,reset,policy)
                settled = self._settled(row,cfg)
                self.inventory.annotate(electrically_settled=bool(settled),excursion_v=peak,
                                        commanded_plateau_s=hold,commanded_reset_s=reset)
                if settled:
                    rows.append(row)
                    break
                if min(hold,reset)>=policy["max_hold_s"]:
                    raise ValueError(f"{sample} {stage} failed electrical settling within maximum hold")
                hold = min(hold*2,policy["max_hold_s"])
                reset = min(reset*2,policy["max_hold_s"])
        self.budget.check()
        result = session_models.prepare_sample_records(cfg,rows,out,previous_manifest=previous[2] if previous else None,check=self.budget.check)
        self.sample_state[sample] = (level,*result)
        self.inventory.data.update(fit_manifest=self.inventory.relative(result[1]),
                                   settling_by_voltage_s=result[0]["settling_by_voltage_s"],
                                   signed_settling_by_voltage_s=result[0]["signed_settling_by_voltage_s"],
                                   observed_settling_s=result[0]["observed_settling_s"])
        self.budget.check()
        return result

    def capture_optimized(self,config,case,view):
        self.budget.check()
        cp = w.validate_case(config,case)
        spec = w.capture_view(cp,view,config["capture_points"]) if isinstance(view,str) else view
        folder = measure.acquire_protocol(cp,config["hardware"],self.out,dict(case,test_name=case["variant"]),
                                         spec,deadline_remaining_s=self.budget.remaining)
        self._accepted(folder)
        self._raw(folder)
        return folder

    def capture_chirp(self,config,shape,cell_peak,window):
        self.budget.check()
        w.require_hardware(config)
        hw = dict(config["hardware"],chirp_reset_s=chirp_reset_seconds(config,cell_peak))
        fit = config.get("fit_report") or {}
        voltage = fit.get("sample_response_evidence") or {}
        voltage_status = ("passed" if voltage.get("passed") is True else
                          "failed" if voltage.get("passed") is False else "not_tested")
        profile = get_profile(config["sample_name"])
        limit = min(w.peak_limit(config),max(profile["chirp_vpp_levels"])/2)
        if not 0 < cell_peak <= limit:
            raise ValueError("Chirp peak exceeds current campaign limit")
        meta = dict(sample_name=config["sample_name"],test_name="chirp_"+shape,
                    voltage_limit_v=limit,
                    calibration_id=config["calibration_id"],model_prediction_used=False,
                    sample_fit_status=config.get("sample_fit_status","qualified" if config.get("model_calibrated") and
                                                 voltage_status=="passed" else "not_qualified"),
                    loaded_voltage_qualification=config.get("loaded_voltage_qualification",dict(status=voltage_status)))
        folder = session_chirp.acquire_chirp_window(shape,cell_peak,window,hw,self.out,meta,
                                                   deadline_remaining_s=self.budget.remaining)
        self._accepted(folder)
        self._raw(folder)
        return folder















