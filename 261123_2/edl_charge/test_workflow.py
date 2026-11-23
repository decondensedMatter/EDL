"""Campaign integrity and compiled-grid endpoint tests; all hardware operations offline."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import workflow as w
from .cli import main
from .actuator import ActuatedModel


class FastRC:
    characteristic_time_s = .002
    def __init__(self, params=None):
        if params:
            self.characteristic_time_s=float(params["tau"])
    def equilibrium(self, v):
        return np.array([float(v)])
    def rhs(self,t,y,v):
        return (v-y)/self.characteristic_time_s
    def current(self,y,v,voltage_rate_v_s=0):
        return float((v-y[0])/self.characteristic_time_s)
    def charge(self,y,v):
        return float(y[0])
    def normalized_state_error(self,y,target,initial):
        return float(np.max(abs(y-target))/max(1e-12,np.max(abs(target-initial))))
    def validate_state(self,y):
        return bool(np.isfinite(y).all())


class WorkflowTests(unittest.TestCase):
    def test_sample_only_mode_requires_consistent_nominal_hardware_and_qualified_fit(self):
        cfg = w.make_config("NaClO4_9M",False)
        cfg.update(calibration_mode="sample_only_nominal",calibration_id="offline")
        hw = cfg["hardware"]
        hw.update(calibration_mode="sample_only_nominal",data_hardware_calibrated=False,
                  calibration_evidence=dict(status="nominal_unverified"),measured_step_rise_s=1e-6,
                  current_delay_s=0.,current_response_rise_s=0.)
        w.require_response(cfg)
        with self.assertRaisesRegex(ValueError,"Qualified specimen fit"):
            w.model_for(cfg)
        with self.assertRaisesRegex(ValueError,"modes differ"):
            w.require_hardware(dict(cfg,calibration_mode="standards"))
        for flag in ("data_hardware_calibrated","first_order_independently_qualified"):
            with self.assertRaisesRegex(ValueError,"unverified nominal"):
                w.require_response(dict(cfg,hardware=dict(hw,**{flag:True})))
        with self.assertRaisesRegex(ValueError,"rise time"):
            w.require_response(dict(cfg,hardware=dict(hw,measured_step_rise_s=0.)))
    def test_campaign_views_and_failure_index(self):
        from .cli import run_campaign
        from types import SimpleNamespace
        from .planning import VARIANTS
        with tempfile.TemporaryDirectory() as temp:
            tmp=Path(temp)
            cfg=w.make_config("LLZO",True)
            entries=[dict(file="failed.json",status="numerical_failure")]
            for n,variant in enumerate(VARIANTS):
                name=f"case_{n}.json"
                entries.append(dict(file=name,status="validated"))
                w.save(tmp/name,dict(variant=variant,case_sha256=str(n)))
            w.save(tmp/"index.json",entries)
            def validate(_,case):
                return SimpleNamespace(actual_hold_s=0. if case["variant"]==VARIANTS[-1] else 1.)
            limit_hit=[None]
            def capture(compiled,hw,out,case,view):
                status="bound_exceeded" if view==limit_hit[0] else "captured"
                folder=out/"mock_run"
                w.save(folder/"metadata.json",dict(status=status))
                return folder
            with patch.object(w,"validate_case",side_effect=validate), \
                    patch.object(w,"capture_view",side_effect=lambda cp,view,n:view), \
                    patch("edl_charge.measure.acquire_protocol",side_effect=capture) as acquire:
                folder=run_campaign(cfg,tmp,["charge","hold","reset"],tmp/"ok")
                report=w.read(folder/"campaign_index.json")
                self.assertEqual(report["status"],"completed")
                self.assertEqual(len(report["records"]),8)
                self.assertEqual(acquire.call_count,8)
                self.assertEqual(len({r["capture_directory"] for r in report["records"]}),8)
                limit_hit[0]="hold"
                with self.assertRaisesRegex(RuntimeError,"bound_exceeded"):
                    run_campaign(cfg,tmp,["charge","hold","reset"],tmp/"fail")
                failed=w.read(next((tmp/"fail").glob("campaign_*/campaign_index.json")))
                self.assertEqual(failed["status"],"failed")
                self.assertEqual([r["status"] for r in failed["records"]],["captured","bound_exceeded"])
    def test_updated_entry_binds_specimen_and_current_path(self):
        from edl_updated_entry import main as updated
        from contextlib import redirect_stderr
        from io import StringIO
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"cfg.json"
            updated("Ta_LLZO",False,["init-config","--out",str(path)])
            data=w.read(path)
            self.assertEqual(data["sample_name"],"LLZO")
            self.assertFalse(data["hardware"]["use_current_amp"])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                updated("LaF3",False,["optimize","--config",str(path),"--out",folder])
            with patch("keysight_33511b_chirp_scan.run_scan") as run:
                updated("LLZO",False,["chirp","--mode","arb"])
                settings=run.call_args.args[0]
                self.assertIsNone(settings.electrode_area_m2)
                self.assertEqual(settings.current_preamplifier_gain_a_per_v,.02)
                self.assertEqual(settings.sample_name,"Ta_LLZO")
                updated("LaF3",True,["chirp"])
                settings=run.call_args.args[0]
                self.assertEqual(2*settings.current_preamplifier_gain_a_per_v,.002)
            import keysight_33511b_chirp_scan as scan
            settings.base_save_dir=folder
            plan=scan.scan_plans(settings)[0]
            scan.validate_settings(settings,[plan])
            settings.target_time_step_s=1.
            with patch.object(scan,"waveform_metadata",return_value=(1.,0.,5)), \
                    patch.object(scan,"read_chunk",return_value=[0.]*5):
                saved=scan.save_capture(None,plan,settings,"test")
            self.assertIn("area_unmeasured",Path(saved).name)

    def test_synthetic_fit_optimize_reload(self):
        from .analysis import instrument_current, parallel_fixture_current
        from .calibration import replay_current
        with tempfile.TemporaryDirectory() as folder, patch.object(w,"factory",return_value=FastRC):
            out=Path(folder)
            cfg=w.make_config("0.1M",False)
            cfg.update(model_parameters={"tau":.008},parameter_bounds={"tau":[.001,.02]},
                       calibration_id="synthetic-fixture",observed_settling_s=.12,
                       optimization_knots=3,starts=1,maxiter=25,max_points=8336,
                       charge_tolerance=.03,voltage_tolerance=.03)
            cfg["hardware"].update(data_hardware_calibrated=True,calibration_evidence={"open_parallel_conductance_s":.13,"open_parallel_capacitance_f":.017},
                                   first_order_independently_qualified=True,measured_step_rise_s=.002,
                                   current_response_rise_s=.007,current_delay_s=.0015)
            cfg["qualification"].update(max_validation_noise_rmse=.25,max_condition_number=5.)
            w.save(out/"config.json",cfg)
            manifest=dict(calibration_id=cfg["calibration_id"],traces=[])
            t=np.linspace(0,.08,41)
            for split,knots in (("train",[0,0,.12,.04,0]),("validation",[0,0,-.07,-.03,0])):
                v=np.interp(t,[0,.01,.02,.05,.08],knots)
                raw=replay_current(FastRC({"tau":.004}),t,v,np.zeros(1))
                observed=instrument_current(t,raw,rise_s=.007,delay_s=.0015)
                observed += parallel_fixture_current(t,v,cfg["hardware"]["calibration_evidence"],rise_s=.007,delay_s=.0015)
                path=out/(split+".csv")
                np.savetxt(path,np.column_stack((t,observed,v)),delimiter=",",
                           header="Time (s),Current (A),Voltage (V)",comments="")
                manifest["traces"].append(dict(path=path.name,split=split,noise_a=1e-3,reset_verified=True))
            w.save(out/"manifest.json",manifest)
            main(False,["fit","--config",str(out/"config.json"),"--manifest",str(out/"manifest.json"),
                        "--out",str(out/"fitted.json")])
            fitted=w.read(out/"fitted.json")
            self.assertTrue(fitted["model_calibrated"])
            self.assertAlmostEqual(fitted["model_parameters"]["tau"],.004,places=7)
            main(False,["optimize","--config",str(out/"fitted.json"),"--out",str(out/"cases"),"--limit","1"])
            cases=w.read(out/"cases"/"index.json")
            self.assertEqual(len(cases),3)
            for row in cases:
                case=w.read(out/"cases"/row["file"])
                self.assertEqual(case["status"],"validated",case.get("compiled_replay",case.get("reason")))
                self.assertGreater(w.validate_case(fitted,case).actual_reset_s,0.)
                case["recipe"]["rate"]+=1
                with self.assertRaisesRegex(ValueError,"matching configuration"):
                    w.validate_case(fitted,case)

    def test_plan_and_uncalibrated_output(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp=Path(folder)
            for amp in (True,False):
                main(amp,["plan","--sample","LLZO","--out",str(tmp/"plan")])
                rows=w.read(tmp/"plan")["LLZO"]["cases"]
                self.assertEqual(max(x["final_voltage_v"] for x in rows),1.0)
                main(amp,["init-config","--sample","LLZO","--out",str(tmp/"cfg")])
                cfg=w.read(tmp/"cfg")
                with self.assertRaisesRegex(ValueError,"hardware calibration"):
                    w.validate_case(cfg,{})
                with self.assertRaisesRegex(ValueError,"hardware calibration"):
                    w.optimize(cfg,tmp,1)

    def test_standards(self):
        cfg=w.make_config("0.1M",True)
        spec,cp=w.calibration_recipe(cfg,"gain_polarity_offset_delay",edge_s=.001,
                                     hold_s=.005,standard_resistance_ohm=1000.)
        self.assertEqual(cp.voltage_min,-.01)
        self.assertEqual(cp.voltage_max,.01)
        self.assertEqual(spec["resistance_ohm"],1000.)
        with self.assertRaisesRegex(ValueError,"hardware calibration"):
            w.calibration_recipe(cfg,"sample_positive_steps",target_v=.1,edge_s=.001,hold_s=.005)

    def test_acquisition_headers_feed_calibration(self):
        from .calibration import read_calibration_csv
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"capture.csv"
            path.write_text("Time (s),Current (A),Voltage (V),Marker (V)\n"
                            "-0.1,0,0,0\n0,0,0,1\n0.1,0.05,0.1,1\n",encoding="utf-8")
            trace=read_calibration_csv(path,dict(split="train",reset_verified=True,noise_a=1e-6))
            np.testing.assert_allclose(trace.voltage_v,[0,0,.1])
            np.testing.assert_allclose(trace.current_a,[0,0,.05])

    def test_compiled_replay(self):
        cfg=w.make_config("0.1M",False)
        m=ActuatedModel(FastRC(),.0001)
        t=np.array([0.,.04,.08,.12])
        # Settled holds on both sides establish the endpoint and subsequent reset.
        recipe=dict(t=t,v=[0.,.05,.1,.1],bound_v=.6895,rate=1000.,hold_s=.04,
                    reset_s=.05,discharge_s=.03,max_points=10000)
        cp=w.build_compiled(recipe)
        report=w.replay_compiled(cfg,m,cp,.1,.12,3)
        self.assertTrue(report["validated"])
        self.assertLess(report["charge_error"],.001)

    def test_compiled_tick_rounding_checks_both_boundaries_and_charge_only_work(self):
        cfg=w.make_config("0.1M",False)
        model=ActuatedModel(FastRC(),.0001)
        requested=.1022
        recipe=dict(t=[0.,.0511,requested],v=[0.,.05,.1],bound_v=.6895,rate=1000.,
                    hold_s=.05,reset_s=.05,discharge_s=.04,max_points=10000)
        compiled=w.build_compiled(recipe)
        self.assertAlmostEqual(compiled.actual_charge_s,.104)
        report=w.replay_compiled(cfg,model,compiled,.1,requested,2)
        self.assertGreater(report["requested_charge_error"],report["actual_charge_error"])
        self.assertGreater(report["requested_terminal_voltage_error"],report["actual_terminal_voltage_error"])
        self.assertEqual(report["charge_error"],report["requested_charge_error"])
        self.assertGreater(report["compiled_net_charge_work_j"], .004)
        self.assertAlmostEqual(report["compiled_net_charge_work_j"],
                               report["compiled_delivered_charge_work_j"]-report["compiled_returned_charge_work_j"])
        self.assertFalse(report["validated"])


if __name__=="__main__":
    unittest.main()

