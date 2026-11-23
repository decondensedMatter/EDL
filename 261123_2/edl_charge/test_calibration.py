"""Offline calibration checks with synthetic data, separate train and validation records."""
import unittest
import numpy as np
try:
    from . import calibration as c
    from .stern_pnp import SternPNPModel
except ImportError:
    import calibration as c
    from stern_pnp import SternPNPModel


class CalibrationTests(unittest.TestCase):
    def test_resistor_delay_gain_offset(self):
        t = np.linspace(0,.1,251)
        v = np.interp(t,[0,.015,.03,.05,.065,.085,.1],[0,0,.4,.4,-.2,0,0])
        delay, gain, offset, resistance = .0023,-2.4,1e-6,10000
        raw = (np.interp(t-delay,t,v)/resistance-offset)/gain
        fit = c.fit_resistor(t,v,raw,resistance,max_delay_s=.005)
        self.assertAlmostEqual(fit["delay_s"],delay,places=7)
        self.assertAlmostEqual(fit["gain"],gain,places=6)
        self.assertAlmostEqual(fit["offset_a"],offset,places=11)
        self.assertEqual(c.clipping_diagnostic(raw)["status"],"unknown_missing_adc_range")
        self.assertGreater(c.clipping_diagnostic(raw,adc_min=min(raw),adc_max=max(raw))["samples_at_rail"],0)

    def test_hold_and_bandwidth(self):
        t = np.linspace(0,1,1001)
        y = 1-np.exp(-np.maximum(t-.2,0)/.03)
        s = c.step_response(t,y,plateau_samples=100,minimum_rise_samples=10)
        self.assertTrue(s["sampling_qualified"])
        self.assertAlmostEqual(s["rise_time_s"],.03*np.log(9),places=4)
        kw = dict(baseline_a=2e-8,noise_a=1e-9,window_s=.2,max_hold_s=1,tolerance_sigma=3)
        self.assertTrue(c.equilibrium_tail(t,np.full_like(t,2e-8),**kw)["settled"])
        self.assertEqual(c.equilibrium_tail(t,np.full_like(t,5e-8),**kw)["status"],"max_hold_reached")

    def test_specimen_fit_and_holdout(self):
        params = dict(area_m2=1e-6,length_m=50e-9,D_m2_s=1e-11,concentration_mol_m3=1.,
                      eps_r=30.,stern_capacitance_f_m2=.08,temp_K=300.,site_concentration_mol_m3=15.,cells=8)
        model = SternPNPModel(params)
        t = np.linspace(0,model.characteristic_time_s*2,15)
        v = .002*(t/t[-1])
        def trace(w,split):
            i = c.replay_current(model,t,w,model.equilibrium(0))
            return c.CalibrationTrace(t,w,i,split=split,reset_verified=True,noise_a=max(abs(i))*1e-3)
        train, holdout = trace(v,"train"),trace(-v*.7,"validation")
        bounds = {"D_m2_s":(3e-12,4e-11)}
        seed = dict(params,D_m2_s=7e-12)
        fit = c.fit_parameters(SternPNPModel,seed,[train],bounds,instrument_calibrated=True)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(fit["parameters"]["D_m2_s"]/params["D_m2_s"],1.,places=4)
        self.assertEqual(fit["rank"],1)
        val = c.validate_parameters(SternPNPModel,fit["parameters"],[holdout],instrument_calibrated=True)
        self.assertLess(val["noise_normalized_rmse"],.01)
        basis = dict(instrument_calibrated=False,calibration_mode="sample_only_nominal")
        nominal_validation = c.validate_parameters(SternPNPModel,fit["parameters"],[holdout],**basis)
        self.assertEqual(nominal_validation,val)
        with self.assertRaises(ValueError):
            c.fit_parameters(SternPNPModel,seed,[holdout],bounds,**basis)
        with self.assertRaises(ValueError):
            c.validate_parameters(SternPNPModel,fit["parameters"],[train],**basis)
        with self.assertRaises(ValueError):
            c.fit_parameters(SternPNPModel,seed,[holdout],bounds,instrument_calibrated=True)
        with self.assertRaises(ValueError):
            c.fit_parameters(SternPNPModel,seed,[train],bounds,instrument_calibrated=False)


if __name__ == "__main__":
    unittest.main()
