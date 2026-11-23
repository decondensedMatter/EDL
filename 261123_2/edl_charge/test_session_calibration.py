import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .awg import compiled_control
from .session_calibration import (_lowpass, process_standards, standard_requests,
                                  standard_settings_template, standard_view)
from .workflow import calibration_recipe, make_config


class StandardCalibrationTest(unittest.TestCase):
    def test_physical_standards_and_independent_rejection(self):
        self._fixture_scenario(2e-5,1e-10)

    def test_noise_only_fixture_is_not_subtracted(self):
        self._fixture_scenario(0.,0.)

    def test_negative_fixture_conductance_fails(self):
        with self.assertRaisesRegex(ValueError,"negative open-fixture conductance"):
            self._fixture_scenario(-2e-5,0.)

    def test_negative_fixture_capacitance_fails(self):
        with self.assertRaisesRegex(ValueError,"negative open-fixture capacitance"):
            self._fixture_scenario(2e-5,-1e-11)

    def test_measured_amplified_path_and_bandwidth_ceiling(self):
        self._fixture_scenario(2e-5,1e-10,amp=True)

    def _fixture_scenario(self, g, c, amp=False):
        settings = standard_settings_template(amp)
        tau_i,delay,ig = (0.45e-6,80e-9,-.0019) if amp else (10e-9,3e-9,-.021)
        settings.update(voltage_references=[dict(value_v=x,source="external calibrator",uncertainty_v=1e-8)
                                            for x in (-.4,.4,.1)],
                        resistor=dict(ohm=1000. if amp else 100.,source="resistor A"),
                        holdout_resistor=dict(ohm=2000. if amp else 200.,source="resistor B"),
                        rc=dict(ohm=1230. if amp else 123.,farad=1e-9,source="series standard C",topology="series"),
                        adc_rails_source="observed rails",adc_rails_v=dict(CH1=[-1,1],CH2=[-1,1]),
                        max_reference_error_v=1e-6,max_relative_error=.03,
                        buffer=dict(source="measured series resistor",series_resistance_ohm=1e6,
                                    minimum_input_resistance_ohm=1e8))
        config = make_config("NaClO4_0.1M",amp)
        captures = {}
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as tmp:
            for req in standard_requests(settings):
                key = req["key"]
                folder = Path(tmp)/key
                folder.mkdir()
                if req["passive"]:
                    t = np.arange(100)/1e6
                    level = -.4*.9995 if key=="buffer_loaded" else req.get("reference_v",0.)
                    v,i = np.full(len(t),level),np.zeros(len(t))
                    captures[key] = folder
                else:
                    stage, cp = calibration_recipe(config,req["stage"],edge_s=req["edge_s"],hold_s=req["hold_s"],
                                                   standard_resistance_ohm=req.get("resistance_ohm"),
                                                   standard_capacitance_f=req.get("capacitance_f"),
                                                   frequency_hz=req.get("frequency_hz"),
                                                   standard_excitation_v=req["standard_excitation_v"])
                    view = standard_view(req,cp,settings)
                    dt = 1/view["sample_rate_hz"]
                    t = np.arange(view["start_s"], cp.duration_s+dt, dt)
                    knots, command = compiled_control(cp)
                    v = _lowpass(np.interp(t,knots,command,left=0,right=0)*.98,dt,8e-9)
                    if req["stage"]=="known_rc_bandwidth":
                        r,rc_cap = req["resistance_ohm"],req["capacitance_f"]
                        ideal = (v-_lowpass(v,dt,r*rc_cap))/r
                    else:
                        ideal = v/req["resistance_ohm"] if "resistance_ohm" in req else np.zeros_like(v)
                    filtered_v = _lowpass(v,dt,tau_i)
                    expected = _lowpass(ideal,dt,tau_i)+g*filtered_v+c*(v-filtered_v)/tau_i
                    i = np.interp(t-delay,t,expected)
                    captures[key] = folder,cp
                raw = np.column_stack((t,i/ig+.0005+rng.normal(0,1e-10,len(t)),
                                       v/1.02+.0001+rng.normal(0,1e-10,len(t)),np.zeros(len(t))))
                np.savetxt(folder/"raw.csv",raw,delimiter=",",header="t,i,v,s")
                (folder/"metadata.json").write_text(json.dumps(dict(status="captured",raw_channels_csv="raw.csv",
                    hardware_settings=config["hardware"])))
            result = process_standards(config,settings,captures)
            hw = result["hardware"]
            fixture = hw["calibration_evidence"]
            self.assertAlmostEqual(fixture["open_parallel_conductance_s"],g,delta=abs(g)*.005)
            self.assertAlmostEqual(fixture["open_parallel_capacitance_f"],c,delta=abs(c)*.005)
            detail = fixture["open_fixture_measurement"]
            self.assertEqual(detail["raw_sha256"],fixture["raw_sha256"]["open"])
            self.assertGreater(detail["conductance_uncertainty_s"],0.)
            self.assertGreater(detail["capacitance_uncertainty_f"],0.)
            self.assertEqual(detail["conductance_resolved"],g>0.)
            self.assertEqual(detail["capacitance_resolved"],c>0.)
            self.assertTrue(hw["ocv_buffer_verified"])
            self.assertTrue(hw["first_order_independently_qualified"])
            self.assertEqual(hw["current_sign"],-1)
            self.assertAlmostEqual(hw["current_gain_a_per_v"],abs(ig),places=5)
            self.assertEqual(hw["use_current_amp"],amp)
            self.assertEqual(fixture["use_current_amp"],amp)
            if amp:
                self.assertLessEqual(hw["current_bandwidth_hz"],400000.)
            self.assertAlmostEqual(hw["voltage_scale_factor"],1.02,places=5)
            self.assertAlmostEqual(hw["current_delay_s"],delay,delta=delay*.08)
            settings["holdout_resistor"]["ohm"]=300
            with self.assertRaisesRegex(ValueError,"qualification failed"):
                process_standards(config,settings,captures)
            settings["adc_rails_v"]["CH2"]=[-.0001,.0001]
            with self.assertRaisesRegex(ValueError,"ADC rail"):
                process_standards(config,settings,captures)

    def test_missing_physical_values(self):
        with self.assertRaises(ValueError):
            standard_requests(standard_settings_template())





