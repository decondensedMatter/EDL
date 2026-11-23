import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from edl_charge.awg import compile_protocol
from edl_charge.measure import acquire_baseline, acquire_protocol, acquire_zero_reset


class Sim:
    def __init__(self):
        self.now = 0
        self.log = []
        self.fg = Instrument(self, "fg")
        self.scope = Instrument(self, "scope")

    def sleep(self, t):
        self.now += t

    def open_resource(self, address, **options):
        return self.fg if "33511" in address or "MY57301577" in address else self.scope

    def close(self):
        pass


class Instrument:
    def __init__(self, sim, name):
        self.sim, self.name = sim, name
        self.values = {}

    def write(self, cmd):
        self.sim.log.append((self.name, cmd, self.sim.now))
        key, _, value = cmd.partition(" ")
        self.values[key] = value

    def query(self, cmd):
        self.sim.log.append((self.name, cmd, self.sim.now))
        if cmd == "*IDN?":
            return "Keysight,33511B,MY57301577,1" if self.name == "fg" else "Agilent,DSA90604A,MY50140103,1"
        if cmd == ":WAVEFORM:XORIGIN?":
            return self.values[":TIMEBASE:POSITION"]
        if cmd == ":WAVEFORM:XINCREMENT?":
            return str(1/float(self.values[":ACQUIRE:SRATE"]))
        if cmd == ":WAVEFORM:POINTS?":
            return self.values[":ACQUIRE:POINTS"]
        if cmd.startswith(":WAVEFORM:DATA?"):
            return ",".join(["0.1"] * int(cmd.split(",")[-1]))
        if cmd == "SOUR1:DATA:VOL:CAT?":
            return '"FOREIGN.ARB"'
        if cmd == "OUTP?":
            return "0"
        if cmd in (":SYSTem:ERRor?", "SYST:ERR?"):
            return '0,"No error"'
        if cmd[:-1] in self.values:
            return self.values[cmd[:-1]]
        return "1"

    def close(self):
        pass


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.sim = Sim()
        self.p = compile_protocol([0, .1], [0, .1], hold_s=1, reset_s=.2, voltage_limit_v=.2)
        self.h = dict(current_gain_a_per_v=.02, current_sign=-1, use_current_amp=False,
                      current_bandwidth_hz=2e6, ch1_scale_v_per_div=.2, ch2_scale_v_per_div=1,
                      ch1_input="DC50", voltage_measurement_location="buffered cell terminals")
        self.meta = dict(status="validated", voltage_limit_v=.2)
        self.view = dict(start_s=.02, duration_s=.01, sample_rate_hz=1000, name="charge")

    def run_view(self, folder):
        return acquire_protocol(self.p, self.h, folder, self.meta, self.view,
                                resource_manager_factory=lambda: self.sim,
                                sleep=self.sim.sleep, clock=lambda: self.sim.now)

    def test_partial_view_does_not_cut_protocol(self):
        with TemporaryDirectory() as folder, patch("edl_charge.measure.configure_sequence",return_value="E12345678S"):
            run = self.run_view(folder)
            meta = json.loads((run/"metadata.json").read_text())
            self.assertEqual(meta["status"], "captured")
            self.assertEqual(meta["observed"]["points"], 11)
            with (run/meta["capture_csv"]).open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(float(rows[1][0]), .02)
            self.assertEqual(float(rows[1][1]), -.002)
            with (run/meta["raw_channels_csv"]).open() as stream:
                raw = list(csv.reader(stream))
            self.assertEqual(raw[0][1], "raw_CH1_V")
            self.assertEqual(float(raw[1][1]), .1)
            self.assertAlmostEqual(self.sim.now, self.p.actual_reset_s+self.p.duration_s)
            self.assertFalse(meta["owned_memory_released"])
            commands = [(s,c) for s,c,_ in self.sim.log]
            self.assertLess(commands.index(("scope", ":SINGLE")), commands.index(("fg", "*TRG")))
            self.assertLess(commands.index(("fg", "*TRG")), commands.index(("fg", "OUTP OFF")))

    def test_remaining_budget_prevents_output(self):
        with TemporaryDirectory() as folder, self.assertRaises(TimeoutError), patch(
                "edl_charge.measure.configure_sequence", return_value="E12345678S"):
            acquire_protocol(self.p, self.h, folder, self.meta, self.view,
                             resource_manager_factory=lambda: self.sim,
                             deadline_remaining_s=lambda: 3.1)
        self.assertNotIn(("fg", "OUTP ON", 0), self.sim.log)
        self.assertIn(("fg", "OUTP OFF", 0), self.sim.log)

    def test_driven_zero_is_recorded_after_dwell_before_off(self):
        query = self.sim.fg.query
        def dc_query(cmd):
            return {"SOUR1:FUNC?":"DC", "SOUR1:VOLT:OFFS?":"0"}.get(cmd) or query(cmd)
        self.sim.fg.query = dc_query
        with TemporaryDirectory() as folder:
            run = acquire_zero_reset(self.h, folder, {"test_name":"baseline"}, 2.,
                                     resource_manager_factory=lambda:self.sim,
                                     clock=lambda:self.sim.now, sleep=self.sim.sleep,
                                     deadline_remaining_s=lambda:100-self.sim.now)
            record = json.loads((run/"metadata.json").read_text())
            self.assertTrue(record["generator_connected"])
            self.assertEqual(record["status"], "captured")
            self.assertEqual(record["zero_reset_duration_s"], 2.)
            scope_run = self.sim.log.index(("scope", ":RUN", 2.))
            self.assertGreater(self.sim.log.index(("fg", "OUTP1 OFF", 2.)), scope_run)

    def test_invalid_input_never_opens_instruments(self):
        self.h["ch1_input"] = "DC"
        with TemporaryDirectory() as folder, self.assertRaises(ValueError):
            self.run_view(Path(folder))
        self.assertFalse(self.sim.log)

    def test_ocv_scope_only_and_actual_samplerate(self):
        self.h["ocv_buffer_verified"] = True
        self.view["sample_rate_hz"] = 900
        base_write = self.sim.scope.write
        def rounded_write(cmd):
            if cmd.startswith(":ACQUIRE:SRATE "):
                cmd = ":ACQUIRE:SRATE 1000"
            base_write(cmd)
        self.sim.scope.write = rounded_write
        with TemporaryDirectory() as folder:
            out = acquire_baseline(self.h, folder, dict(test_name="ocv", sample_name="NaCl 1M"), self.view,
                                   resource_manager_factory=lambda:self.sim, clock=lambda:self.sim.now, sleep=self.sim.sleep)
            meta = json.loads((out/"metadata.json").read_text())
            self.assertIn("NaCl_1M_ocv", meta["capture_csv"])
            self.assertEqual(meta["scope_sample_rate_hz"], 1000)
            self.assertEqual(meta["observed"]["points"], 11)
            self.assertTrue(all(name != "fg" for name, cmd, t in self.sim.log))
            self.assertNotIn("generator_id", meta)

    def test_bound_violation(self):
        self.h["voltage_scale_factor"] = 4
        with TemporaryDirectory() as folder, patch("edl_charge.measure.configure_sequence",return_value="E12345678S"):
            out = self.run_view(folder)
            record = json.loads((out/"metadata.json").read_text())
            self.assertFalse(record["voltage_bound_ok"])
            self.assertEqual(record["status"], "bound_exceeded")


if __name__ == "__main__":
    unittest.main()
