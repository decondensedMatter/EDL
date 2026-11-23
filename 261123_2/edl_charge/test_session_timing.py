import unittest

from . import session_timing as timing


class SessionTimingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(sample_name="NaClO4_1M",observed_settling_s=.05,max_hold_s=30.,
                        hardware=dict(use_current_amp=False,min_charge_time_s=.001,
                                      current_bandwidth_hz=1e6,measured_step_rise_s=1e-5),
                        settling_by_voltage_s={".2":{"hold_s":.08,"reset_s":.07},
                                               ".8":{"hold_s":.03,"reset_s":.03}})
        self.pairs = [(v,t) for t in (.02,.04,.08,.16,.32,1.) for v in (.1,.2,.4,.8)]

    def schedule(self,seconds,**opts):
        return timing.schedule_sample(self.cfg,self.pairs,remaining_s=seconds,**opts)

    def test_dwell_measurement_and_ordering(self):
        self.cfg["settling_by_voltage_s"][".2"]["hold_s"]=2.
        levels=[timing.dwell_seconds(self.cfg,{},v,"hold_s") for v in (.1,.2,.4,.8)]
        self.assertGreaterEqual(levels[0],2.)
        self.assertTrue(all(b>a for a,b in zip(levels,levels[1:])))
        self.assertEqual(timing.proposed_dwell_seconds(.8,.8,"reset_s"),2.)
        self.cfg["observed_settling_s"]=4.
        self.assertGreaterEqual(timing.dwell_seconds(self.cfg,{},.1,"chirp_reset_s"),4.)

    def test_bisect_then_uniformly_reduce(self):
        full=self.schedule(100000)
        self.assertTrue(full["full_grid"])
        self.assertEqual(full["dwell_policy"]["scale"],1.)
        cost=full["cost_breakdown_s"]["total_s"]
        tight=self.schedule(cost)
        self.assertTrue(tight["full_grid"])
        self.assertLess(tight["dwell_policy"]["scale"],1.)
        self.assertAlmostEqual(tight["cost_breakdown_s"]["total_s"],cost-5.,places=5)
        reduced=self.schedule(cost*.65)
        self.assertTrue(reduced["feasible"])
        self.assertFalse(reduced["full_grid"])
        for v in (.1,.2,.4,.8):
            times={t for x,t in reduced["selected_pairs"] if x==v}
            self.assertTrue({.02,1.}<=times)
        self.assertEqual(reduced["planned_chirp_captures"],45)
        self.assertLessEqual(reduced["cost_breakdown_s"]["total_s"],cost*.65-5)

    def test_no_budget_or_overlarge_floor(self):
        blocked=self.schedule(1.,reserved_s=5.)
        self.assertFalse(blocked["feasible"])
        self.assertEqual(len(blocked["omitted_pairs"]),len(self.pairs))
        self.cfg["observed_settling_s"]=40.
        self.assertFalse(self.schedule(100000)["feasible"])

    def test_no_qualified_times_still_costs_chirps(self):
        result=timing.schedule_sample(self.cfg,[],remaining_s=100000.)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["selected_pairs"],[])
        self.assertGreater(result["cost_breakdown_s"]["chirp_and_reset_s"],0.)

    def test_weighted_share_and_no_double_allocation(self):
        names=["NaClO4_1M","LLZO","LaF3"]
        report=timing.allocate_sample_share(names,3000.)
        rows=report["by_sample"]
        self.assertAlmostEqual(sum(row["share_s"] for row in rows.values()),3000.)
        self.assertLess(rows[names[0]]["share_s"],rows[names[-1]]["share_s"])
        self.assertEqual(rows[names[0]]["calibration_traces"],10+4*len(rows[names[0]]["voltage_levels_v"]))
        tiny=timing.allocate_sample_share(names,12.)
        self.assertAlmostEqual(sum(row["share_s"] for row in tiny["by_sample"].values()),12.)

    def test_stale_measurement_cannot_replace_new_floor(self):
        result=self.schedule(100000.)
        self.cfg["settling_by_voltage_s"][".2"]["hold_s"]=4.
        with self.assertRaisesRegex(ValueError,"current measured settling"):
            timing.dwell_seconds(self.cfg,result["dwell_policy"],.2,"hold_s")

    def test_fixed_scale_replay_floor_and_replanning(self):
        policy=dict(minimum_scale=.4,maximum_scale=.4,
                    model_required_dwell_by_voltage_s={".2":{"hold_s":3.,"reset_s":4.}})
        result=self.schedule(100000.,policy=policy)
        dwell=result["dwell_policy"]
        self.assertEqual(dwell["scale"],.4)
        self.assertEqual(timing.dwell_seconds(self.cfg,dwell,.2,"hold_s"),3.)
        self.assertLess(timing.dwell_seconds(self.cfg,dwell,.1,"hold_s"),3.)
        self.assertGreater(timing.dwell_seconds(self.cfg,dwell,.4,"hold_s"),3.)
        self.assertGreater(timing.dwell_seconds(self.cfg,dwell,.8,"chirp_reset_s"),4.)
        self.assertNotEqual(dwell["settling_by_voltage_s"],dwell["model_required_dwell_by_voltage_s"])
        smaller=self.schedule(result["cost_breakdown_s"]["total_s"]*.85,policy=dwell)
        self.assertTrue(smaller["feasible"])
        self.assertLess(len(smaller["selected_pairs"]),len(self.pairs))
        self.assertEqual(smaller["dwell_policy"]["scale"],.4)
        with self.assertRaises(ValueError):
            timing.proposed_dwell_seconds(.2,.8,"hold_s",policy,scale=.8)
        with self.assertRaises(ValueError):
            timing.dwell_seconds(self.cfg,dict(dwell,scale=.2),.2,"hold_s")

    def test_both_polarities_kept_together_and_dwell_uses_magnitude(self):
        both=[(sign*v,t) for v,t in self.pairs for sign in (1,-1)]
        self.cfg["settling_by_voltage_s"]["-.2"]={"hold_s":.6,"reset_s":.9}
        policy=dict(model_required_dwell_by_voltage_s={"-.4":{"hold_s":2.}})
        self.assertEqual(timing.dwell_seconds(self.cfg,policy,-.4,"hold_s"),2.)
        self.assertEqual(timing.dwell_seconds(self.cfg,policy,.4,"hold_s"),2.)
        self.assertGreater(timing.dwell_seconds(self.cfg,policy,-.1,"reset_s"),.9)
        self.assertEqual(timing.proposed_dwell_seconds(-.1,.8,"reset_s"),
                         timing.proposed_dwell_seconds(.1,.8,"reset_s"))
        full=timing.schedule_sample(self.cfg,both,remaining_s=99999,policy=policy)
        smaller=timing.schedule_sample(self.cfg,both,remaining_s=.7*full["cost_breakdown_s"]["total_s"],
                                       policy=policy)
        self.assertTrue(smaller["feasible"])
        kept={tuple(pair) for pair in smaller["selected_pairs"]}
        self.assertTrue(len(kept)<len(both))
        self.assertEqual({(-v,t) for v,t in kept},kept)
        for v in (.1,.2,.4,.8):
            self.assertTrue({(v,.02),(-v,.02),(v,1.),(-v,1.)}<=kept)
