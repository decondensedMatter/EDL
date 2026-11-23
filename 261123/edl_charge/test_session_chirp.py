import json
import math
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from edl_charge import session_chirp as chirp
from edl_charge.test_measure import Sim


class TestSessionChirp(unittest.TestCase):
    def test_windows_cover_three_bands_with_small_records(self):
        for shape in ("sine", "square", "triangle"):
            fast, mid, slow = chirp.plan_windows(shape)
            self.assertGreater(fast["sample_rate_hz"], mid["sample_rate_hz"])
            self.assertGreater(mid["sample_rate_hz"], slow["sample_rate_hz"])
            for view in (fast, mid, slow):
                self.assertLess(math.ceil(view["duration_s"] * view["sample_rate_hz"]), 100000)

    def test_finite_run_calibrated_gain_and_reset(self):
        self.run_chirp(False)
        self.run_chirp(True)

    def run_chirp(self,use_amp):
        sim = Sim()
        hw = dict(current_gain_a_per_v=.02, current_sign=1, use_current_amp=use_amp,
                  current_bandwidth_hz=2e6, ch1_scale_v_per_div=.2, ch2_scale_v_per_div=1,
                  ch1_input="DC50", voltage_measurement_location="cell", voltage_gain=2)
        query = sim.fg.query
        def read(cmd):
            return {"SOUR1:VOLT:OFFS?": "0", "SOUR1:FUNC?": "DC"}.get(cmd) or query(cmd)
        sim.fg.query = read
        with TemporaryDirectory() as folder, patch.object(chirp, "configure_native") as setup:
            run = chirp.acquire_chirp_window("sine", .2, chirp.plan_windows("sine")[0],
                      hw, folder, dict(sample_name="sample", voltage_limit_v=.3),
                      resource_manager_factory=lambda:sim, sleep=sim.sleep, clock=lambda:sim.now,
                      deadline_remaining_s=lambda:100-sim.now)
            record = json.loads((run/"metadata.json").read_text())
            self.assertEqual(record["status"], "captured")
            self.assertTrue(record["partial_record"])
            self.assertEqual(record["effective_current_bandwidth_hz"],400e3 if use_amp else 2e6)
            self.assertEqual(record["window_all_fundamentals_within_current_bandwidth"],not use_amp)
            self.assertFalse(record["whole_protocol_energy_available"])
            self.assertEqual(setup.call_args.args[1].voltage_vpp, .2)
            self.assertEqual(sim.now, 6.)
            self.assertEqual(sum(c == "INIT1" for _,c,_ in sim.log), 1)
