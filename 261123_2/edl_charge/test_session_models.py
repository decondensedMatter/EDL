"""Offline session-grid, settling and compiled qualification regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from . import session_models as sm
from . import workflow as w
from .actuator import ActuatedModel
from .test_workflow import FastRC


class SessionModelTests(unittest.TestCase):
    def test_balanced_grid_and_compiled_cases(self):
        cfg = w.make_config("0.1M",False)
        cfg["hardware"]["measured_step_rise_s"] = .0001
        cfg.update(max_points=8336,optimization_knots=3,starts=1,maxiter=25,
                   charge_tolerance=.03,voltage_tolerance=.03,observed_settling_s=.12)
        grid = sm.balanced_pairs(cfg)
        self.assertEqual(len({t for _,t in grid}),6)
        self.assertEqual(len(set(grid)), len(grid))
        self.assertIn(w.timing(cfg)["min_charge_time_s"],[t for _,t in grid])
        with patch.object(w,"model_for",return_value=ActuatedModel(FastRC(),.0001)):
            cases = sm.optimize_one(cfg,.1,1.)
            self.assertEqual([c["status"] for c in cases],["validated"]*3)
            for case in cases:
                cp = w.validate_case(cfg,case)
                self.assertEqual(sm.compiled_digest(cp),case["compiled_sha256"])
                self.assertEqual(cp.sample_rate_hz,case["compiled_sample_rate_hz"])
            self.assertEqual(cases[-1]["recipe"]["hold_s"],0.)
            self.assertEqual(sum(len(c["capture_views"]) for c in cases),4)

    def test_independent_signed_fits_negative_optimization_and_replay(self):
        from .analysis import instrument_current
        from .calibration import replay_current
        cfg = w.make_config("0.1M",False)
        cfg.update(model_parameters={"tau":.006},parameter_bounds={"tau":[.001,.02]},
                   calibration_id="asymmetric_fixture",campaign_peak_limit_v=.1,optimization_knots=3,starts=1,maxiter=25,
                   max_points=8336,charge_tolerance=.03,voltage_tolerance=.03)
        cfg["hardware"].update(data_hardware_calibrated=True,first_order_independently_qualified=True,
            measured_step_rise_s=.00001,current_response_rise_s=.001,current_delay_s=.0001,
            calibration_evidence=dict(baseline_current_noise_a=.001,baseline_voltage_noise_v=.00001))
        cfg["qualification"].update(max_sample_voltage_error_fraction=.02,max_validation_noise_rmse=.2,
                                    max_condition_number=5.)
        t = np.linspace(-.05,.6,131)
        knots = [0,.025,.275,.3,.55]
        rows = []
        with TemporaryDirectory() as tmp, patch.object(w,"factory",return_value=FastRC):
            tmp = Path(tmp)
            for side,sign,tau in (("positive",1.,.003),("negative",-1.,.009)):
                for split,amp in (("train",.1),("validation",.07)):
                    target = sign*amp
                    levels = [0,target,target,0,0]
                    v = np.interp(t,knots,levels,left=0,right=0)
                    current = replay_current(FastRC({"tau":tau}),t,v,np.zeros(1))
                    measured = instrument_current(t,current,rise_s=.001,delay_s=.0001)
                    path = tmp/f"{side}_{split}.csv"
                    np.savetxt(path,np.c_[t,measured,v],delimiter=",",comments="",header="Time(s),Current(A),Voltage(V)")
                    rows.append(dict(path=path,stage=f"sample_{side}_steps" if split=="train" else f"{side}_step_holdout",
                        target_v=target,split=split,polarity=side,actual_charge_s=.275,actual_discharge_s=.025,
                        actual_hold_s=0,actual_reset_s=.25,edge_s=.025,command_times_s=knots,
                        command_voltage_v=levels,sample_rate_hz=1e6,programmed_voltage_gain=1.))
            prepared, manifest = sm.prepare_sample_records(cfg,rows,tmp/"prepared")
            result = sm.fit_sample_records(prepared,manifest)
            self.assertTrue(result["model_calibrated"])
            self.assertAlmostEqual(sm.signed_config(result,.1)["model_parameters"]["tau"],.003,places=5)
            negative = sm.signed_config(result,-.1)
            self.assertAlmostEqual(negative["model_parameters"]["tau"],.009,places=5)
            with self.assertRaisesRegex(ValueError,"matching independent"):
                sm.optimize_one(negative,.1,1.)
            self.assertEqual(sm.balanced_pairs(result)[0][0],-sm.balanced_pairs(result)[1][0])
            cases = sm.optimize_one(result,-.1,1.)
            for case in cases:
                self.assertEqual(case["status"],"validated",case.get("reason"))
                self.assertLess(case["actual_peak_voltage_v"],0)
                self.assertLess(case["compiled_replay"]["signed_peak_v"],0)
                self.assertGreater(case["compiled_replay"]["compiled_net_charge_work_j"],0)
                self.assertGreaterEqual(case["compiled_replay"]["actual_terminal_voltage_error"],0)
                cp = w.validate_case(result,case)
                self.assertLess(cp.voltage_min,0.)
                self.assertGreaterEqual(cp.voltage_min,-.1)
                self.assertEqual(case["voltage_limit_v"],.1)
                self.assertEqual(cp.voltage_max,0.)

    def test_selected_standard_excitation_and_independent_reset(self):
        cfg = w.make_config("0.1M",False)
        for stage,args in (("gain_polarity_offset_delay",dict(standard_resistance_ohm=1000.)),
                           ("known_rc_bandwidth",dict(standard_resistance_ohm=1000.,standard_capacitance_f=1e-9,
                                                      frequency_hz=65536.)),
                           ("open_input_parasitics",{})):
            spec,compiled = w.calibration_recipe(cfg,stage,edge_s=1e-6,hold_s=1e-4,reset_s=2e-4,
                                                  standard_excitation_v=.25,**args)
            self.assertAlmostEqual(compiled.voltage_max,.25)
            self.assertAlmostEqual(compiled.voltage_min,0. if stage=="open_input_parasitics" else -.25)
            self.assertGreaterEqual(compiled.actual_reset_s,2e-4)
            self.assertEqual(spec["reset_s"],2e-4)
        with self.assertRaises(ValueError):
            w.calibration_recipe(cfg,"sample_positive_steps",standard_excitation_v=.25)

    def test_dwell_policy_replayed_peak_and_state_extensions(self):
        cfg = w.make_config("0.1M",False)
        cfg["hardware"]["measured_step_rise_s"] = .0001
        cfg.update(observed_settling_s=.1,max_points=10000)
        result = dict(status="validated",actual_peak_voltage_v=.15,times_s=[0.,.5,1.],
                      voltage_v=[0.,.16,.1])
        def dwell(config,policy,v,phase):
            self.assertEqual(policy,{"scale":.5})
            return v+(.1 if phase=="hold_s" else .2)
        def review(config,model,cp,vf,T,intervals):
            ok = cp.actual_reset_s >= .6
            return dict(validated=ok,peak_v=.18,reset_error=.001 if ok else .1,hold_error=0.)
        cfg["session_dwell_policy"] = {"scale":.5}
        with patch.object(w,"model_for",return_value=ActuatedModel(FastRC(),.0001)),                 patch.object(w,"optimize_charge",return_value=result),                 patch.object(w,"replay_compiled",side_effect=review),                 patch("edl_charge.session_timing.dwell_seconds",side_effect=dwell):
            cases = sm.optimize_one(cfg,.1,1.)
        for case in cases:
            self.assertEqual(case["status"],"validated")
            self.assertEqual(case["dwell_policy"],cfg["session_dwell_policy"])
            self.assertAlmostEqual(case["planned_dwell"]["reset_s"],.35)
            self.assertAlmostEqual(case["recipe"]["reset_s"],.76)
            self.assertAlmostEqual(case["dwell_extension"]["reset_s"],.41)
            self.assertAlmostEqual(case["dwell_extension"]["reset_above_target_plan_s"],.46)
        self.assertEqual(cases[-1]["recipe"]["hold_s"],0.)

    def test_compression_preserves_voltage_and_cell_response(self):
        from .calibration import replay_current
        from .analysis import parallel_fixture_current
        t = np.linspace(0,.1,201)
        base = np.interp(t,[0,.02,.08,.1],[0,.15,.15,0])
        measured = base + np.random.default_rng(807).normal(0,1e-5,len(t))
        v, info = sm.bounded_voltage_driver(t,measured,1e-5,[.02,.08])
        self.assertLess(info["retained_points"],len(t)//4)
        self.assertLessEqual(info["max_error_v"],info["tolerance_v"])
        self.assertEqual([min(v),max(v)],[min(measured),max(measured)])
        m = FastRC()
        raw_i = replay_current(m,t,measured,np.zeros(1))
        reduced_i = replay_current(m,t,v,np.zeros(1),driver_indices=info["indices"])
        self.assertLess(max(abs(raw_i-reduced_i)),.02)
        # Analytical current on a linear ramp after a first-order measuring channel.
        ev = dict(open_parallel_capacitance_f=.01,open_parallel_conductance_s=.2)
        tau = .003
        actual = parallel_fixture_current(t,t,ev,rise_s=tau*np.log(9.),delay_s=0.)
        expected = .2*(t-tau*(1-np.exp(-t/tau)))+.01*(1-np.exp(-t/tau))
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_trace_preparation_requires_measured_reset(self):
        cfg = w.make_config("0.1M",False)
        cfg["hardware"]["calibration_evidence"] = dict(baseline_current_noise_a=.001,baseline_voltage_noise_v=.001)
        t = np.linspace(-.12,1.2,133)
        v = np.interp(t,[0,.1,.5,.6,1.2],[0,.1,.1,0,0])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for k,split in enumerate(["train","validation"]):
                path = root/f"raw{k}.csv"
                np.savetxt(path,np.c_[t,t*0,v+.0001*k],delimiter=",",comments="",
                           header="Time(s),Current(A),Voltage(V)")
                rows.append(dict(path=path,stage="sample_positive_steps",split=split,target_v=.1,
                            edge_s=.1,actual_charge_s=.5,actual_hold_s=0.,actual_discharge_s=.1,actual_reset_s=.6))
            rows[-1].update(command_times_s=[0,.1,.5,.6,1.2],command_voltage_v=[0,.1,.1,0,0],
                            sample_rate_hz=1e6,programmed_voltage_gain=1.)
            cfg["hardware"]["measured_step_rise_s"] = 1e-5
            cfg["qualification"]["max_sample_voltage_error_fraction"] = .02
            updated, manifest = sm.prepare_sample_records(cfg,rows,root/"derived")
            self.assertGreater(updated["settling_by_voltage_s"]["0.1"]["hold_s"],0.)
            self.assertEqual(len(w.read(manifest)["traces"]),2)
            response = sm.sample_response_evidence(updated,manifest)
            self.assertTrue(response["passed"])
            self.assertGreaterEqual(response["qualified_min_interval_s"],.1)
            fitted = dict(updated,fit_report=dict(sample_response_evidence=response))
            self.assertGreaterEqual(min(t for _,t in sm.balanced_pairs(fitted)),.6)
            response["qualified_min_charge_s"] = 1.1
            self.assertEqual(sm.balanced_pairs(fitted),[])
            changed = w.read(manifest)
            changed["traces"][-1]["command_voltage_v"] = [0,.2,.2,0,0]
            w.save(manifest,changed)
            with patch.object(w,"fit",side_effect=AssertionError("must reject before fitting")):
                failed = sm.fit_sample_records(updated,manifest)
            self.assertFalse(failed["model_calibrated"])
            self.assertFalse(failed["fit_report"]["sample_response_evidence"]["passed"])
            np.savetxt(path,np.c_[t,t*0,v+.03],delimiter=",",comments="",header="Time(s),Current(A),Voltage(V)")
            with self.assertRaisesRegex(ValueError,"Pretrigger"):
                sm.prepare_sample_records(cfg,rows,root/"bad")






