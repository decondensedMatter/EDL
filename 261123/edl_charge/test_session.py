"""Offline controller tests with a deterministic wall clock and no VISA resources."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import session as s
from . import workflow as w
from .planning import VARIANTS, load_profiles


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class SessionTests(unittest.TestCase):
    def test_incomplete_preflight_never_connects(self):
        backend = Mock(side_effect=AssertionError("preflight must precede instrument access"))
        with TemporaryDirectory() as tmp:
            folder = s.run(s.template(), tmp, backend_factory=backend)
            result = w.read(folder/"session_journal.json")
            self.assertEqual(result["status"],"preflight_failed")
            self.assertTrue(result["missing_or_invalid"])
            backend.assert_not_called()

    def test_unchanged_readout_rejects_session_before_access(self):
        backend = Mock(side_effect=AssertionError("legacy budget check must precede hardware"))
        with TemporaryDirectory() as tmp:
            for amp in (True,False):
                folder = s.run(s.template(amp,"legacy"),tmp,backend_factory=backend,preflight_fn=lambda cfg: [])
                journal = w.read(folder/"session_journal.json")
                self.assertEqual(journal["status"],"preflight_failed")
                self.assertTrue(any("256 averages" in issue for issue in journal["missing_or_invalid"]))
                self.assertEqual(journal["stages"],[])
            backend.assert_not_called()

    def simulate(self, shortage=False, missing=False, reject_negative=False,fail_high=False,fit_timeout=False,overshoot=False,policy=None):
        clock = Clock()
        profiles = load_profiles()
        cfg = s.template()
        if policy is not None:
            cfg["voltage_stage_peak_policy"]=policy
        if fit_timeout or overshoot:
            name=next(iter(profiles))
            profiles={name:profiles[name]}
            cfg["samples"]={name:cfg["samples"][name]}
        if shortage:
            cfg["budget_seconds"] = 95.
        unique = 0
        active = []
        def consume(seconds):
            active.append(seconds)
            clock.advance(seconds)

        class Backend:
            def __init__(self, config, folder, budget, prompt):
                self.folder, self.budget = folder, budget

            def calibrate_hardware(self):
                def connect(message):
                    consume(4.)
                    return ""
                with patch("builtins.input",side_effect=connect):
                    self.budget.prompt("Connect independent reference")
                consume(1.)
                return {"instrument":True}

            def calibrate_sample(self, specimen, calibrated, *, level_v):
                first=not hasattr(self,"last_sample") or self.last_sample!=specimen["sample_name"]
                if first:
                    with self.budget.sample_change():
                        clock.advance(120.)
                self.last_sample=specimen["sample_name"]
                if fail_high and level_v>=.2:
                    raise ValueError("Observed clipping in calibration")
                consume(3.)
                specimen["observed_settling_s"] = .1
                return specimen, "mock-manifest"

            def capture(self, seconds):
                nonlocal unique
                unique += 1
                capture = self.folder/f"capture_{unique}"
                w.save(capture/"metadata.json",dict(status="captured"))
                consume(seconds)
                return capture

            def capture_optimized(self,specimen,case,view):
                return self.capture(.2)

            def capture_chirp(self,specimen,shape,peak,view):
                return self.capture(.1)

        def compute(op,payload,folder,budget,limit):
            budget.check()
            consume(2. if op=="fit" else 3.)
            if op == "fit":
                if fit_timeout:
                    raise s.BudgetExpired("fit worker timed out")
                fitted = dict(payload["config"],model_calibrated=True,observed_settling_s=.1,
                              fit_report=dict(sample_response_evidence=dict(passed=True,qualified_min_charge_s=.001)))
                return dict(fitted,polarity_fits={side:dict(fitted,fit_polarity=side,fit_report=dict(fitted["fit_report"],fit_polarity=side)) for side in ("positive","negative")})
            self.assertEqual(payload["holds_policy"]["scale"],payload["holds_policy"]["maximum_scale"])
            side = "positive" if payload["final_v"]>0 else "negative"
            self.assertEqual(payload["config"]["fit_polarity"],side)
            self.assertNotIn("session_dwell_policy",payload["config"])
            status = "compiled_replay_failed" if reject_negative and side=="negative" else "validated"
            return [dict(variant=v,status=status,capture_views=views,final_voltage_v=payload["final_v"],
                         fit_polarity=side,config_sha256=w.digest(payload["config"]),
                         dwell_extension=dict(actual_hold_s=.25,actual_reset_s=.35,hold_s=.05,reset_s=.05))
                    for v,views in zip(VARIANTS,(["charge","hold"],["reset"],["reset"]))]

        comp = SimpleNamespace(voltage_min=0.,voltage_max=.11 if overshoot else .1,actual_reset_s=.1,actual_hold_s=.2,actual_charge_s=.2,
                               actual_discharge_s=.1,duration_s=.6)
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            for fn in ["require_response","model_for"]:
                stack.enter_context(patch.object(w,fn))
            stack.enter_context(patch.object(w,"validate_case",return_value=comp))
            stack.enter_context(patch.object(w,"timing",return_value=dict(min_charge_time_s=.001)))
            stack.enter_context(patch("edl_charge.session_models.balanced_pairs",
                                      return_value=[(.1,.1),(-.1,.1),(.2,.1),(-.2,.1)]))
            stack.enter_context(patch("edl_charge.session_chirp.plan_windows",
                                      return_value=[dict(name="sampled_window")]))
            folder = s.run(cfg,tmp,backend_factory=Backend,compute_fn=compute,clock=clock,
                           preflight_fn=lambda config: [],model_issues_fn=lambda cfg: ["unknown geometry"] if missing else [])
            report = w.read(folder/"session_journal.json")
        self.assertAlmostEqual(report["active_elapsed_s"],sum(active),places=7)
        self.assertEqual(report["excluded_sample_change_s"],120.*len(profiles),report.get("error",report.get("missing_or_invalid")))
        self.assertEqual(set(report["samples"]),set(profiles))
        return report

    def test_all_six_share_one_clock(self):
        report = self.simulate()
        self.assertEqual(report["status"],"completed")
        self.assertEqual(report["stages"][0]["completed_active_s"],5.)
        self.assertEqual(len([r for r in report["stages"] if r["kind"]=="optimize"]),24)
        for name,row in report["samples"].items():
            self.assertEqual(row["completed_pairs"],4)
            self.assertEqual(row["completed_polarity_pairs"],2)
            self.assertTrue(all(c["fitted_config"] for c in row["captures"]))
            self.assertEqual([c["view"] for c in row["captures"]],["charge","hold","reset","reset"]*4)
            self.assertEqual(len(row["chirps"]),len(load_profiles()[name]["chirp_vpp_levels"])*3)
            self.assertEqual(row["skipped"],[])
            plans = row["timing_plans"]
            self.assertGreaterEqual(len(plans),3)
            self.assertEqual(len({p["dwell_policy"]["scale"] for p in plans}),1)
            self.assertGreaterEqual(plans[-1]["dwell_policy"]["cpu_s_per_pair"],3.)
            self.assertTrue(plans[-1]["dwell_policy"]["model_required_dwell_by_voltage_s"])

    def test_missing_model_inputs_preserve_calibrations_and_sweeps(self):
        report = self.simulate(missing=True)
        self.assertEqual(report["status"],"partial")
        self.assertEqual(sum(r["kind"]=="sample_calibration" for r in report["stages"]),sum(len(row["voltage_levels"]) for row in report["samples"].values()))
        self.assertFalse(any(r["kind"] in ("optimize","sample_fit_and_independent_validation")
                             for r in report["stages"]))
        for name,row in report["samples"].items():
            self.assertEqual(row["status"],"partial")
            self.assertEqual(row["completed_pairs"],0)
            self.assertEqual(row["skipped"][0]["reason"],"missing_model_inputs")
            self.assertEqual(len(row["chirps"]),len(load_profiles()[name]["chirp_vpp_levels"])*3)

    def test_high_level_failure_preserves_lower_voltages(self):
        report=self.simulate(fail_high=True)
        for name,row in report["samples"].items():
            stages=[item for item in report["stages"] if item.get("sample")==name]
            magnitudes=[item["level_v"] if "level_v" in item else abs(item["final_v"]) if "final_v" in item
                        else item["vpp"]/2 for item in stages]
            self.assertEqual(magnitudes,sorted(magnitudes))
            self.assertEqual(row["completed_polarity_pairs"],1)
            self.assertEqual(row["voltage_levels"][-1]["status"],"failed_qualification")
            self.assertTrue(all(level["status"]=="completed" for level in row["voltage_levels"][:-1]))
            self.assertTrue(all(c["status"]=="captured" for c in row["captures"]+row["chirps"]))

    def test_rejected_negative_polarity_prevents_both_acquisitions(self):
        report = self.simulate(reject_negative=True)
        self.assertEqual(report["status"],"partial")
        for row in report["samples"].values():
            self.assertEqual(row["captures"],[])
            self.assertEqual(row["completed_polarity_pairs"],0)
            self.assertTrue(all(group["status"]=="skipped_before_output" and group["captures_completed"]==0
                                for group in row["polarity_pairs"]))

    def test_fit_timeout_fences_next_voltage(self):
        row=next(iter(self.simulate(fit_timeout=True)["samples"].values()))
        self.assertEqual(row["status"],"partial_budget")
        self.assertEqual(row["voltage_levels"][-1]["peak_v"],.1)
        self.assertEqual(row["captures"],[])

    def test_default_accepts_overshoot_and_orders_signed_final_voltages(self):
        self.assertEqual(s.template()["voltage_stage_peak_policy"],"sample")
        report=self.simulate(overshoot=True)
        name,row=next(iter(report["samples"].items()))
        self.assertEqual(row["status"],"completed")
        self.assertEqual([r["final_v"] for r in row["captures"]],[.1]*4+[-.1]*4+[.2]*4+[-.2]*4)
        self.assertTrue(all(level["optimized_limit_v"]==load_profiles()[name]["peak_bound_v"]
                            for level in row["voltage_levels"]))

    def test_compiled_peak_checked_before_any_output(self):
        row=next(iter(self.simulate(overshoot=True,policy="stage")["samples"].values()))
        groups=[group for group in row["polarity_pairs"] if group["magnitude_v"]==.1]
        self.assertTrue(groups)
        self.assertTrue(all(group["status"]=="skipped_before_output" and "ceiling" in group["reason"] for group in groups))
        self.assertFalse(any(abs(c["final_v"])==.1 for c in row["captures"]))

    def test_scoped_time_shortage_is_partial(self):
        report = self.simulate(shortage=True)
        self.assertIn(report["status"],("partial","partial_budget"))
        self.assertLessEqual(report["active_elapsed_s"],report["budget_seconds"])
        self.assertTrue(any(s["status"]!="completed" for s in report["samples"].values()))

    def test_hung_numerical_worker_is_terminated(self):
        process = Mock()
        process.poll.return_value = None
        with TemporaryDirectory() as tmp, patch.object(s.subprocess,"Popen",return_value=process),                 patch.object(s.time,"monotonic",side_effect=[0.,1.]):
            with self.assertRaises(s.BudgetExpired):
                s.compute("optimize",{},Path(tmp),s.ActiveBudget(100.,clock=lambda:0.),.2)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()






