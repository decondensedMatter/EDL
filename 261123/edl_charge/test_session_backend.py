from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from . import session_backend as backend
from . import workflow as w


class BenchTest(unittest.TestCase):
    def test_sample_capture_sequence_evidence_and_charged_prompts(self):
        cfg = w.make_config("NaClO4_0.1M",False)
        cfg["hardware"].update(data_hardware_calibrated=True,ocv_buffer_verified=True,
                first_order_independently_qualified=True,measured_step_rise_s=1e-8,
                current_response_rise_s=1e-8,current_delay_s=0.,current_bandwidth_hz=1e6,
                calibration_evidence=dict(baseline_current_noise_a=1e-6,baseline_voltage_noise_v=1e-5))
        cfg["calibration_id"]="mock-session"
        policy = dict(backend.sample_calibration_template(),max_abs_ocv_v=.01)
        session = dict(sample_calibration=policy,standards=dict(adc_rails_v=dict(CH1=[-1,1],CH2=[-3,3])))
        budget,prompt = Mock(),Mock()
        budget.remaining.return_value=7200.
        with TemporaryDirectory() as tmp:
            bench = backend.BenchBackend(session,tmp,budget,prompt)
            folders = []
            def capture(cp,hw,root,meta,view,**options):
                self.assertIs(options["deadline_remaining_s"],budget.remaining)
                folder = Path(root)/str(len(folders))
                folders.append(folder)
                folder.mkdir()
                record = dict(status="captured",capture_csv="physical.csv",raw_channels_csv="raw.csv",
                              hardware_settings=hw,protocol=meta)
                n = int(np.ceil(view["duration_s"]*view["sample_rate_hz"]))+1
                t = view["start_s"]+np.arange(n)/view["sample_rate_hz"]
                v = np.zeros(n)
                if cp is not None:
                    xt,u = cp.control_knots()
                    v = np.interp(t,xt,u,left=0.,right=0.)
                    record.update({k:getattr(cp,k) for k in ("actual_charge_s","actual_hold_s",
                                  "actual_discharge_s","actual_reset_s")})
                i = 2e-8*np.sin(np.arange(n))
                if cp is not None and meta["test_name"]=="sample_negative_steps":
                    stage = meta["calibration_stage"]
                    edge = stage["edge_s"]
                    i[(t>edge)&(t<edge+.2*stage["hold_s"])] -= 8e-6
                    reset_start = cp.duration_s-cp.actual_reset_s
                    i[(t>reset_start)&(t<reset_start+.3*cp.actual_reset_s)] += 8e-6
                for f,cols,header in (("physical.csv",(t,i,v),"Time (s),Current (A),Voltage (V)"),
                                        ("raw.csv",(t,i*50,v,v*0),"t,i,v,sync")):
                    np.savetxt(folder/f,np.column_stack(cols),delimiter=",",header=header,comments="")
                w.save(folder/"metadata.json",record)
                return folder
            def baseline(hw,root,meta,view,**opts):
                return capture(None,hw,root,meta,view,**opts)
            def zero(hw,root,meta,duration_s,**opts):
                return baseline(hw,root,meta,dict(start_s=0.,duration_s=.05,sample_rate_hz=20000),**opts)
            with (patch.object(backend.measure,"acquire_baseline",side_effect=baseline),
                  patch.object(backend.measure,"acquire_zero_reset",side_effect=zero),
                  patch.object(backend.measure,"acquire_protocol",side_effect=capture)):
                updated,low_manifest = bench.calibrate_sample(cfg,cfg,level_v=.05)
                low = w.read(low_manifest)
                low_inventory = w.read(Path(low_manifest).parent/"calibration_inventory.json")
                for record in low_inventory["records"]:
                    recipe = record["recipe"]
                    self.assertLessEqual(max(abs(recipe.get(k,0.)) for k in ("actual_peak_min_v","actual_peak_max_v")),.05+1e-12)
                updated,manifest = bench.calibrate_sample(cfg,cfg,level_v=.1)
                self.assertEqual(low,w.read(low_manifest))
            traces = w.read(manifest)["traces"]
            self.assertEqual(len(traces),14)
            self.assertEqual({r["split"] for r in traces},{"train","validation"})
            self.assertEqual(set(updated["settling_by_voltage_s"]),{"0.05","0.1"})
            inventory=w.read(Path(manifest).parent/"calibration_inventory.json")
            self.assertEqual(inventory["status"],"completed")
            for row in inventory["records"]:
                if row["recipe"].get("split")=="validation":
                    self.assertGreater(row["excursion_v"],.05)
                    self.assertLess(row["excursion_v"],.1)

            self.assertTrue(inventory["fit_input_issues"])
            self.assertEqual(inventory["requested_negative_targets_v"],[-.1])
            self.assertEqual({r["polarity"] for r in traces if r["split"]=="validation"},{"positive","negative"})
            self.assertEqual(len([r for r in traces if r["split"]=="validation"]),6)
            signed=inventory["signed_settling_by_voltage_s"]
            for v in (.05,.1):
                for phase in ("hold_s","reset_s"):
                    self.assertGreater(signed["negative"][str(v)][phase],signed["positive"][str(v)][phase])
                    self.assertEqual(inventory["settling_by_voltage_s"][str(v)][phase],signed["negative"][str(v)][phase])
            self.assertEqual(len(inventory["records"])+len(low_inventory["records"]),len(traces)+2)
            step_rows=[r for r in inventory["records"] if r["name"]=="sample_positive_steps"]
            negative=[r for r in inventory["records"] if r["name"]=="sample_negative_steps"]
            self.assertEqual([r["commanded_reset_s"] for r in step_rows],[r["commanded_reset_s"] for r in negative])
            self.assertEqual([r["commanded_plateau_s"] for r in step_rows],[r["commanded_plateau_s"] for r in negative])
            self.assertTrue(all(r["recipe"]["actual_peak_min_v"]<0 and r["recipe"]["actual_peak_max_v"]==0 for r in negative))
            self.assertTrue(np.all(np.diff([r["commanded_reset_s"] for r in step_rows])>0))
            self.assertTrue(np.all(np.diff([r["commanded_plateau_s"] for r in step_rows])>0))
            for row in inventory["records"]:
                self.assertTrue((Path(manifest).parent/row["raw_channels_csv"]).is_file())
                self.assertEqual(len(row["raw_sha256"]),64)
            self.assertEqual(prompt.call_count,2)
            self.assertEqual([c.kwargs.get("sample_change",False) for c in prompt.call_args_list],[True,False])
            self.assertTrue(all((Path(manifest).parent/r["path"]).is_file() for r in traces))
            self.assertTrue(all(Path(r["raw_path"]).exists() for r in traces))
            self.assertTrue(all("command_times_s" in r for r in traces if r["split"]=="validation"))

    def test_preflight_names_missing_inputs_and_never_contacts_instruments(self):
        cfg=w.make_config("LLZO",False)
        issues=backend.preflight(dict(samples=[cfg],sample_calibration=backend.sample_calibration_template()))
        report=" ".join(issues)
        self.assertIn("standards",report)
        self.assertIn("six",report)
        self.assertNotIn("length_m",report)
        fit_only = " ".join(backend.model_input_issues(cfg))
        for name in ("length_m","area_m2","parameter_bounds","max_sample_voltage_error_fraction"):
            self.assertIn(name,fit_only)
        with_model = backend.preflight(dict(samples=[cfg]),include_model_inputs=True)
        self.assertIn("length_m"," ".join(with_model))

    def test_invalid_runtime_settings_rejected_in_one_preflight(self):
        from .session import template
        cfg=template()
        names=list(cfg["samples"])
        cfg["samples"][names[0]],cfg["samples"][names[1]]=cfg["samples"][names[1]],cfg["samples"][names[0]]
        cfg.update(budget_seconds=7201,shared_calibration_max_s=float("inf"),fit_max_s=0,
                   optimize_pair_max_s=-2,chirp_leg_s="1",chirp_fraction_of_sample_time=1.01,
                   hardware_current_limit_a=.01,use_current_amp=True,sample_change_time_excluded=False)
        issues=" ".join(backend.preflight(cfg))
        for field in ("budget_seconds","shared_calibration_max_s","fit_max_s","optimize_pair_max_s",
                      "chirp_leg_s","chirp_fraction_of_sample_time","hardware_current_limit_a",
                      "use_current_amp","sample_change_time_excluded","canonical sample_name"):
            self.assertIn(field,issues)

    def test_chirp_voltage_and_observed_reset_floors(self):
        cfg=dict(sample_name="NaClO4_0.1M",observed_settling_s=.3,
                 settling_by_voltage_s={"0.2":{"reset_s":.75},"0.8":{"reset_s":1.1}})
        self.assertEqual(backend.chirp_reset_seconds(cfg,.1),.75)
        self.assertEqual(backend.chirp_reset_seconds(cfg,.8),2.)
        cfg["observed_settling_s"]=3.
        self.assertEqual(backend.chirp_reset_seconds(cfg,.1),3.)



    def test_missing_fit_geometry_allows_measured_calibration_for_both_readouts(self):
        from .session import template
        for amp in (False,True):
            session=template()
            session["use_current_amp"]=amp
            for name in session["samples"]:
                cfg=w.make_config(name,amp)
                cfg["qualification"].update(backend.analysis_qualification_defaults())
                session["samples"][name]=cfg
            st=backend.standards.standard_settings_template(amp)
            st.update(voltage_references=[dict(value_v=x,source="external DC source") for x in (-2.,2.,1.)],
                      resistor=dict(ohm=1000.,source="standard A"),
                      holdout_resistor=dict(ohm=2000.,source="standard B"),
                      rc=dict(topology="series",ohm=120.,farad=1e-8,source="standard C"),
                      adc_rails_source="measured acquisition range",adc_rails_v=dict(CH1=[-4,4],CH2=[-4,4]),
                      buffer=dict(source="series loading resistor",series_resistance_ohm=1e6,
                                  minimum_input_resistance_ohm=1e8))
            session["standards"]=st
            self.assertEqual(backend.preflight(session),[])
            self.assertIn("length_m"," ".join(backend.model_input_issues(session["samples"]["LLZO"])))
    def test_raw_only_schedule_and_chirp_use_the_same_selected_dwell(self):
        from .session_timing import schedule_sample, dwell_seconds
        cfg=w.make_config("LLZO",False)
        cfg.update(observed_settling_s=.7,settling_by_voltage_s={"1.0":dict(hold_s=.6,reset_s=.5)},
                   calibration_id="session",sample_fit_status="missing_inputs")
        cfg["hardware"].update(data_hardware_calibrated=True,calibration_evidence={"test":"mock"})
        scheduled=schedule_sample(cfg,[],remaining_s=1200,policy=dict(maximum_scale=.4))
        self.assertTrue(scheduled["feasible"])
        cfg["session_dwell_policy"]=scheduled["dwell_policy"]
        reset=dwell_seconds(cfg,cfg["session_dwell_policy"],.5,"chirp_reset_s")
        self.assertEqual(backend.chirp_reset_seconds(cfg,.5),reset)
        bench=backend.BenchBackend({},Path("."),Mock(),Mock())
        with (patch.object(backend.session_chirp,"acquire_chirp_window",return_value=Path("mock")) as acquire,
              patch.object(bench,"_accepted"),patch.object(bench,"_raw")):
            bench.capture_chirp(cfg,"sine",.5,{"name":"mock"})
        self.assertEqual(acquire.call_args.args[3]["chirp_reset_s"],reset)
        meta=acquire.call_args.args[5]
        self.assertFalse(meta["model_prediction_used"])
        self.assertEqual(meta["loaded_voltage_qualification"]["status"],"not_tested")
        self.assertEqual(meta["sample_fit_status"],"missing_inputs")

    def test_preflight_accepts_absent_deprecated_fraction_and_rejects_invalid_dwell(self):
        from .session import template
        cfg=template()
        cfg.pop("chirp_fraction_of_sample_time",None)
        cfg["timing_policy"]=dict(reset_target_s=-1.)
        issues=" ".join(backend.preflight(cfg))
        self.assertIn("timing_policy",issues)
        self.assertNotIn("chirp_fraction_of_sample_time",issues)


    def test_tf_prior_initializes_each_nominal_concentration_without_calibration(self):
        from .planning import load_profiles
        for name in load_profiles():
            cfg=w.make_config(name,False)
            if cfg["model"]!="tf":
                continue
            cfg["qualification"].update(backend.analysis_qualification_defaults())
            initialized=backend.model_fit_defaults(cfg)
            self.assertEqual(backend.model_input_issues(initialized),[])
            self.assertFalse(initialized["model_calibrated"])
            self.assertFalse(initialized["hardware"]["data_hardware_calibrated"])
            self.assertEqual(initialized["model_parameters"]["concentration_mol_m3"],
                             cfg["model_parameters"]["concentration_mol_m3"])
            for key,(lo,hi) in initialized["parameter_bounds"].items():
                self.assertLess(lo,initialized["initial_guess"][key])
                self.assertLess(initialized["initial_guess"][key],hi)
                self.assertIsNone(initialized["model_parameters"][key])
            self.assertEqual(initialized["fit_initialization"]["status"],"unverified_initialization")
            self.assertIsNone(cfg["model_parameters"]["Gamma"])

    def test_solid_search_seeds_leave_only_actual_geometry_for_fit(self):
        for name in ("LLZO","LaF3"):
            cfg=backend.model_fit_defaults(w.make_config(name,True))
            cfg["qualification"].update(backend.analysis_qualification_defaults())
            blockers=backend.model_input_issues(cfg)
            self.assertEqual(len(blockers),1)
            self.assertIn("area_m2",blockers[0])
            self.assertNotIn("D_m2_s",blockers[0])
            self.assertFalse(cfg["model_calibrated"])
            cfg["model_parameters"]["area_m2"]=2e-6
            if name=="LLZO":
                self.assertIn("length_m",blockers[0])
                cfg["model_parameters"]["length_m"]=.0005
            self.assertEqual(backend.model_input_issues(cfg),[])
