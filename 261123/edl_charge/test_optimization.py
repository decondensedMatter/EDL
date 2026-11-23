"""Synthetic RC and finite-volume controls; durations and parameters are test fixtures."""
import unittest
import numpy as np
try:
    from .optimization import simulate,optimize_charge,replay_variants
    from .stern_pnp import SternPNPModel
    from .actuator import ActuatedModel
except ImportError:
    from optimization import simulate,optimize_charge,replay_variants
    from stern_pnp import SternPNPModel
    from actuator import ActuatedModel


class RC:
    def equilibrium(self,v): return np.array([float(v)])
    def rhs(self,t,y,v): return v-y
    def charge(self,y,v=0.): return float(y[0])
    def current(self,y,v,voltage_rate_v_s=0.): return v-y[0]
    def normalized_state_error(self,y,target,initial): return abs((y-target)/(target-initial))[0]


class Tests(unittest.TestCase):
    def test_rc_variants_and_lower_bound(self):
        m = RC()
        result = optimize_charge(m,.1,2.,.3,minimum_transition_s=.04,knots=4,starts=2)
        self.assertEqual(result["status"],"validated",result.get("reason"))
        p = result["prediction"]
        q = p["charge_c"][-1]
        # For unit R,C: W = q^2/2 + integral(I²) >= q²/2 + q²/T.
        self.assertGreaterEqual(p["work_j"]+1e-7,q*q*(.5+1/2))
        self.assertLessEqual(result["terminal_charge_error"],.01)
        variants = replay_variants(result,m,2,1,4)
        for v in variants.values():
            self.assertEqual(v["voltage_v"][-1],0.)
            self.assertAlmostEqual(np.interp(2.,v["times_s"],v["charge_c"]),q,places=7)
        unreachable = optimize_charge(m,.1,.01,.1,minimum_transition_s=.1)
        self.assertEqual(unreachable["status"],"no_feasible_control_found")

    def test_filtered_rc_physical_work_and_deadline(self):
        m = ActuatedModel(RC(),.2)
        p = simulate(m,[0.,.2,.3],[0.,.1,.1],points_per_interval=120)
        q = p["charge_c"]
        diss = np.trapz(p["current_a"]**2,p["times_s"])
        self.assertAlmostEqual(p["work_j"],q[-1]**2/2+diss,places=7)
        self.assertGreater(np.sum(np.diff(q)*(p["command_voltage_v"][1:]+p["command_voltage_v"][:-1])/2),
                           1.05*p["work_j"])
        self.assertLess(p["voltage_v"][-1],.0995)
        bad = optimize_charge(m,.1,.1,.1,minimum_transition_s=.001,knots=3,
                              starts=1,maxiter=4,charge_tolerance=.99)
        self.assertEqual(bad["status"],"no_feasible_control_found")

    def test_nonlinear_closed_cell(self):
        m = SternPNPModel(dict(area_m2=1e-6,length_m=10e-9,D_m2_s=1e-11,
            concentration_mol_m3=1.,eps_r=30.,stern_capacitance_f_m2=.08,temp_K=300.,
            site_concentration_mol_m3=15.,cells=8))
        m = ActuatedModel(m,.0004*m.characteristic_time_s)
        result = optimize_charge(m,.003,5*m.characteristic_time_s,.012,
                                 minimum_transition_s=.05*m.characteristic_time_s,
                                 knots=3,starts=1,maxiter=15)
        self.assertEqual(result["status"],"validated",str(result.get("validation_checks",result.get("reason"))))
        self.assertTrue(result["validation_checks"]["mesh_checked"])


if __name__ == "__main__":
    unittest.main()

