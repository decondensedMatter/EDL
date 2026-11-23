"""TF equilibrium, charge conservation and frozen 0.1 M reference regressions."""
import unittest

import numpy as np

if __package__:
    from .tf_model import TFModel
else:
    from tf_model import TFModel


class TFModelTest(unittest.TestCase):
    def setUp(self):
        self.params = TFModel.historical_seed()
        self.model = TFModel(self.params)

    def test_equilibrium(self):
        for voltage in (0., .1, -.4, 1.):
            y = self.model.equilibrium(voltage)
            self.assertTrue(self.model.validate_state(y))
            np.testing.assert_allclose(self.model.rhs(0, y, voltage), 0, atol=1e-8)
            self.assertAlmostEqual(self.model.current(y, voltage), 0.)
            self.assertEqual(self.model.normalized_state_error(y, y, self.model.equilibrium(0)), 0.)

    def test_current_and_charge(self):
        model = self.model
        y = model.equilibrium(0)
        rates = model.rhs(0, y, .1)
        current = model.current(y, .1)
        self.assertAlmostEqual(current, .1/self.params["R"], places=13)
        self.assertAlmostEqual(model.charge(rates), current, places=13)
        np.testing.assert_allclose(model.charge(np.r_[1., np.zeros(8)]),
                                   np.pi*(.0008**2)*.01, rtol=1e-14)

    def test_independent_conditions(self):
        y = self.model.equilibrium(.3)
        other = TFModel(dict(self.params, concentration_mol_m3=10))
        self.assertFalse(np.allclose(y, other.equilibrium(.3)))
        np.testing.assert_array_equal(y, self.model.equilibrium(.3))
        y[1:3] = .7
        self.assertFalse(self.model.validate_state(y))
        y[1] = -.1
        self.assertFalse(self.model.validate_state(y))


if __name__ == "__main__":
    unittest.main()
