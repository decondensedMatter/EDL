"""Synthetic partial captures: independent shots and known RC/filter limits."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from edl_charge.analysis import (compare_capture,instrument_current,settling_diagnostic,
                                 fit_decay,compare_discharge_decay,write_analysis)


class RC:
    def equilibrium(self,v): return np.array([v],dtype=float)
    def rhs(self,t,y,v): return v-y
    def charge(self,y,v): return y[0]
    def current(self,y,v,r=0.): return v-y[0]


class AnalysisTests(unittest.TestCase):
    def test_delayed_measured_hold(self):
        # Drive ramps linearly 0->1 in 1 s then holds. Hold I = (1-e^-1)e^-(t-1).
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d/"metadata.json").write_text(json.dumps(dict(status="captured",actual_charge_s=1,
                actual_hold_s=3,actual_discharge_s=1,actual_reset_s=3)))
            t = np.linspace(1.5,3,81)
            i = (1-np.exp(-1))*np.exp(-(t-1))
            with (d/"capture.csv").open("w",newline="") as handle:
                w=csv.writer(handle)
                w.writerow(["Time (s)","Current (A)","Voltage (V)","Sync (V)"])
                w.writerows(zip(t,i,np.ones_like(t),np.zeros_like(t)))
            result = compare_capture(RC(),d,[0,1,4,5,8],[0,1,1,0,0])
            m = result["metrics"]
            self.assertTrue(m["partial_protocol"])
            self.assertLess(m["rmse_a"],1e-6)
            self.assertAlmostEqual(m["measured_window_charge_c"],i[0]-i[-1],places=4)
            # A named out-of-bound capture remains available for diagnostics only.
            filename = "2026-11-23_synthetic_hold.csv"
            (d/"capture.csv").rename(d/filename)
            md = json.loads((d/"metadata.json").read_text())
            md.update(capture_csv=filename,status="bound_exceeded")
            (d/"metadata.json").write_text(json.dumps(md))
            diagnostic = compare_capture(RC(),d,[0,1,4,5,8],[0,1,1,0,0])
            self.assertTrue(diagnostic["metrics"]["diagnostic_only"])
            self.assertFalse(result["metrics"]["diagnostic_only"])
            write_analysis(diagnostic,d/"out")
            for f in ("analysis.json","overlay.png","prediction.csv"):
                self.assertGreater((d/"out"/f).stat().st_size,0)

    def test_filter_decay_and_censor(self):
        t = np.linspace(0,3,601)
        y = instrument_current(t,np.ones_like(t),rise_s=.2*np.log(9),delay_s=.1)
        np.testing.assert_allclose(y,1-np.exp(-np.maximum(t-.1,0)/.2),atol=1e-12)
        i = 2.3+np.exp(-t/.35)
        tau = fit_decay(t,i)
        self.assertAlmostEqual(tau["tau_s"],.35,places=6)
        d = {"metrics":{"discharge_decay":tau}}
        self.assertEqual(compare_discharge_decay(d,d)["held_minus_immediate_tau_s"],0.)
        self.assertEqual(settling_diagnostic(t,i,target_a=2.3,noise_a=.004,window_s=.2)["status"],
                         "sustained_electrical_plateau")
        self.assertEqual(settling_diagnostic(t,i,target_a=0,noise_a=.004,window_s=.2)["status"],"right_censored")


if __name__=="__main__":
    unittest.main()

