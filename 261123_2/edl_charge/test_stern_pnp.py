"""Numerical checks using a synthetic nanocell; no specimen parameters are implied."""
import unittest
import numpy as np
from scipy.integrate import solve_ivp
try:
    from .stern_pnp import SternPNPModel, F
except ImportError:
    from stern_pnp import SternPNPModel, F


P = dict(area_m2=1e-6, length_m=50e-9, D_m2_s=1e-11,
         concentration_mol_m3=1., eps_r=30., stern_capacitance_f_m2=.08,
         temp_K=300., site_concentration_mol_m3=15., cells=32)


class ClosedCellTest(unittest.TestCase):
    def test_stationarity_and_capacitance(self):
        for site in (15., None):
            m = SternPNPModel(dict(P, site_concentration_mol_m3=site))
            v = 1e-4
            c = m.equilibrium(v)
            self.assertTrue(m.validate_state(c))
            self.assertLess(np.max(abs(m.rhs(0, c, v))) * m.characteristic_time_s / m.c0, 1e-6)
            self.assertLess(abs(m.current(c, v)), m.charge(c, v) / m.characteristic_time_s * 1e-6)
            ld = m.debye_length_m
            cap = m.area / (2 / m.cs + 2 * ld / m.epsilon * np.tanh(m.length / (2 * ld)))
            self.assertAlmostEqual(m.charge(c, v) / (cap * v), 1., delta=.006)
            np.testing.assert_allclose(m.equilibrium(-v), c[::-1], rtol=1e-10)
            fine = SternPNPModel(dict(m.parameters, cells=64))
            self.assertTrue(fine.validate_state(m.remap_state(c, fine)))
            qf = fine.charge(fine.equilibrium(v), v)
            self.assertLess(abs(qf - v * cap), abs(m.charge(c, v) - v * cap))

    def test_ramp_work_heat_and_displacement(self):
        m = SternPNPModel(P)
        T = m.characteristic_time_s
        amplitude = .007
        rate = amplitude / T
        initial = m.equilibrium(0)
        qscale = abs(m.charge(m.equilibrium(amplitude), amplitude))
        escale = qscale * amplitude
        # Last three integration variables are normalized charge, work, and heat.
        def rhs(t, y):
            v = rate * t
            c = y[:-3]
            dc = m.rhs(t, c, v)
            j = -np.cumsum(m.dx * dc)[:-1]
            mu = m.rt * (np.log(c / (m.site - c)) + m._potential(c, v) / m.vt)
            heat_rate = -m.area * (j @ np.diff(mu))
            current = m.current(c, v, rate)
            return np.r_[dc, current / qscale, v * current / escale, heat_rate / escale]
        sol = solve_ivp(rhs, (0, T), np.r_[initial, 0, 0, 0], method="BDF",
                        rtol=1e-8, atol=1e-11)
        self.assertTrue(sol.success)
        self.assertTrue(all(m.validate_state(y[:-3]) for y in sol.y.T))
        c = sol.y[:-3, -1]
        q, w, h = sol.y[-3:, -1]
        self.assertAlmostEqual(q, m.charge(c, amplitude) / qscale, places=7)
        self.assertGreater(h, 0)
        self.assertAlmostEqual(w - h, m.energy(c, amplitude) / escale, places=6)
        self.assertGreater(m.current(initial, 0, rate), 0)


if __name__ == "__main__":
    unittest.main()

