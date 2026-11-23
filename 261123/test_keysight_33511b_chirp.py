import math
import unittest
from unittest.mock import Mock, call

from keysight_33511b_chirp_scan import ScanSettings, scan_plans
import keysight_33511b_chirp_generator as generator
import keysight_33511b_chirp_plan as plan


class SweepPlanningTests(unittest.TestCase):
    def test_material_endpoints(self):
        endpoints = {"0.01M NaClO4": 1.6, "0.1M NaClO4": 1.6,
                     "1M NaClO4": 1.6, "9M NaClO4": 1.6,
                     "Ta_LLZO": 2.0, "LaF3": 4.0}
        self.assertEqual(endpoints.keys(), plan.WINDOW_WIDTHS_V.keys())
        for material, target in endpoints.items():
            with self.subTest(material=material):
                width = plan.WINDOW_WIDTHS_V[material]
                margin = plan.WINDOW_MARGINS_V[material]
                levels = list(plan.CHIRP_VPP_LEVELS[material])
                scans = scan_plans(ScanSettings(sample_name=material, waveform_types=("sine",)))
                self.assertEqual([s.voltage_vpp for s in scans], levels)
                if "NaClO4" in material:
                    self.assertEqual(levels, [.1, .2, .4, .8, 1.6])
                self.assertEqual(levels[-1], target)
                self.assertEqual(levels[0], 0.1)
                self.assertEqual(levels, sorted(set(levels)))
                self.assertTrue(all(b == 2 * a for a, b in zip(levels, levels[1:-1])))
                self.assertLessEqual(levels[-1], 2 * levels[-2])
                self.assertEqual(plan.peak_voltage(target), target / 2)
                # Check physical bounds independently of the voltage-list implementation.
                self.assertGreaterEqual(width / 2 - target / 2, margin - 1e-12)
                for vpp in levels:
                    self.assertGreaterEqual(-vpp / 2, -width / 2 + margin - 1e-12)
                    self.assertLessEqual(vpp / 2, width / 2 - margin + 1e-12)
        self.assertEqual(plan.voltage_levels(1.0), [0.1, 0.2, 0.4, 0.8])
        self.assertEqual(plan.voltage_levels(1.1, margin_v=0.15), [0.1, 0.2, 0.4, 0.8])

    def test_frequency_limits(self):
        expected = {"sine": 2e6, "square": 2e6, "triangle": 20e3}
        for waveform, top in expected.items():
            self.assertEqual(plan.make_plan(waveform, 1, 1, max_fraction=1).f_max,
                             plan.WAVEFORM_LIMITS_HZ[waveform])
            self.assertEqual(plan.make_plan(waveform, 1, 1).f_max, top)
            self.assertEqual(plan.make_plan(waveform, 1, 1, mode="arb").f_max, 2e6)

    def test_round_trips_and_phase(self):
        sweep = plan.make_plan("sine", 0.1, 2)
        self.assertEqual(sweep.total_duration_s, 8)
        self.assertEqual([sweep.frequency_at(t) for t in range(0, 9, 2)],
                         [2e6, 10, 2e6, 10, 2e6])
        leg_phase = 2 * (2e6 - 10) / math.log(2e6 / 10)
        for k in range(5):
            self.assertAlmostEqual(sweep.phase_cycles_at(k * 2), k * leg_phase, places=8)
        for t in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
            h = 1e-5
            derivative = (sweep.phase_cycles_at(t + h) - sweep.phase_cycles_at(t - h)) / (2 * h)
            self.assertTrue(math.isclose(derivative, sweep.frequency_at(t), rel_tol=1e-6))

    def test_invalid_inputs(self):
        for value in (None, 0, -1, 0.1, math.inf, math.nan):
            with self.subTest(window=value), self.assertRaises(ValueError):
                plan.voltage_levels(value)
        for margin in (-1, 1, math.nan, math.inf):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                plan.voltage_levels(1, margin)
        for kwargs in ({"leg_duration_s": 0}, {"f_min": -1}, {"f_max": 21e6},
                       {"f_max": math.inf}, {"voltage_vpp": math.nan}, {"cycles": -1},
                       {"waveform": "bad"}, {"mode": "bad"}, {"max_fraction": 0},
                       {"max_fraction": -0.5}, {"max_fraction": 1.1},
                       {"max_fraction": math.nan}, {"max_fraction": math.inf}):
            options = dict(waveform="sine", voltage_vpp=1, leg_duration_s=2)
            options.update(kwargs)
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan.make_plan(**options)
        sweep = plan.make_plan("square", 1, 1)
        for t in (-1, 5, math.nan, math.inf):
            with self.subTest(t=t), self.assertRaises(ValueError):
                sweep.frequency_at(t)


class GeneratorFailureTests(unittest.TestCase):
    def test_native_rejects_unsupported_firmware_with_output_off(self):
        device = Mock()
        device.query.return_value = "Keysight,33511B,SERIAL,3.99"
        with self.assertRaisesRegex(RuntimeError, "firmware"):
            generator.configure_native(device, plan.make_plan("sine", 2.2, 1))
        self.assertIn(call("OUTP1 OFF"), device.write.call_args_list)
        self.assertNotIn(call("OUTP1 ON"), device.write.call_args_list)

    def test_arb_capacity_failure_precedes_device_access(self):
        device = Mock()
        with self.assertRaisesRegex(ValueError, "capacity"):
            generator.configure_arb(device, plan.make_plan("sine", 2.2, 1, mode="arb"))
        self.assertEqual(device.mock_calls, [])

    def test_start_error_disables_output_in_both_modes(self):
        for mode in ("native", "arb"):
            with self.subTest(mode=mode):
                device = Mock()
                device.query.side_effect = ['-200,"Execution error"', '+0,"No error"']
                with self.assertRaisesRegex(RuntimeError, "Execution error"):
                    generator.start_output(device, mode)
                writes = device.write.call_args_list
                self.assertEqual(writes[0], call("OUTP1 ON"))
                self.assertEqual(writes[-1], call("OUTP1 OFF"))
                self.assertEqual(call("*TRG") in writes, mode == "arb")


if __name__ == "__main__":
    unittest.main()
