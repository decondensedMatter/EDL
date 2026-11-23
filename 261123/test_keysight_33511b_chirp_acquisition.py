"""Offline checks for single-shot scope capture and calibrated CSV transfer."""

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import keysight_33511b_chirp_scan as scan


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.settings = scan.ScanSettings()
        self.plan = scan.scan_plans(self.settings)[0]
        self.scope = Mock()
        self.fg = Mock()

    def setup_replies(self, depth):
        self.scope.query.side_effect = lambda command: {
            ":STOP;*OPC?": "1", ":SYSTem:ERRor?": '0,"No error"',
            ":ACQUIRE:SRATE?": "1e8", ":ACQUIRE:POINTS?": str(depth),
        }[command]

    def test_scope_setup_and_short_memory(self):
        self.setup_replies(500_000_000)
        scan.configure_scope(self.scope, self.plan, self.settings)
        writes = [call.args[0] for call in self.scope.write.call_args_list]
        for command in (":TIMEBASE:REFERENCE LEFT", ":TIMEBASE:POSITION 0",
                        ":ACQUIRE:MODE HRESOLUTION", ":ACQUIRE:AVERAGE OFF",
                        ":TRIGGER:EDGE:SOURCE CHANNEL3", ":TRIGGER:LEVEL CHANNEL3,0.2",
                        ":WAVEFORM:FORMAT ASCII"):
            self.assertIn(command, writes)
        self.assertFalse(any(command.startswith(":WAVEFORM:POINTS ") for command in writes))
        self.setup_replies(100_000_000)
        with self.assertRaisesRegex(RuntimeError, "acquisition points"):
            scan.acquire(self.scope, self.fg, self.plan, self.settings)
        self.fg.write.assert_called_once_with("OUTP1 OFF")

    def test_arm_before_output_then_done_and_output_off(self):
        events = []
        def wait(scope, query, timeout):
            events.append(query)
        self.scope.write.side_effect = events.append
        self.fg.write.side_effect = events.append
        self.fg.query.return_value = '0,"No error"'
        for mode in ("native", "arb"):
            self.settings.sweep_mode = mode
            plan = scan.scan_plans(self.settings)[0]
            events.clear()
            with patch.object(scan, "configure_scope"), patch.object(scan, "wait_for_scope", wait):
                scan.acquire(self.scope, self.fg, plan, self.settings)
            expected = [":SINGLE", ":AER?", "OUTP1 ON", "INIT1"]
            if mode == "arb":
                expected.append("*TRG")
            self.assertEqual(events, expected + [":ADER?", "OUTP1 OFF"])

    def test_output_off_on_completion_timeout(self):
        self.fg.query.return_value = '0,"No error"'
        with patch.object(scan, "configure_scope"), patch.object(
                scan, "wait_for_scope", side_effect=[None, TimeoutError("capture")]):
            with self.assertRaises(TimeoutError):
                scan.acquire(self.scope, self.fg, self.plan, self.settings)
        self.assertEqual(self.fg.write.call_args_list[-1].args, ("OUTP1 OFF",))

    def test_actual_waveform_must_cover_both_cycles(self):
        with patch.object(scan, "waveform_metadata", return_value=(1e-8, 0., 100)):
            with self.assertRaisesRegex(RuntimeError, "complete chirp"):
                scan.save_capture(self.scope, self.plan, self.settings, "test")

    def test_oversized_arb_rejected_before_visa_connection(self):
        self.settings.sweep_mode = "arb"
        self.settings.arb_max_points = 8
        visa = Mock()
        with patch.dict(sys.modules, pyvisa=visa), patch("builtins.print"):
            with self.assertRaises(ValueError):
                scan.run_scan(self.settings)
        visa.ResourceManager.assert_not_called()

    def test_configuration_failure_releases_instruments(self):
        visa = Mock()
        manager = visa.ResourceManager.return_value
        manager.open_resource.side_effect = [self.fg, self.scope]
        with patch.dict(sys.modules, pyvisa=visa), patch("builtins.print"), patch.object(
                scan, "configure_native", side_effect=RuntimeError("generator error")):
            with self.assertRaisesRegex(RuntimeError, "generator error"):
                scan.run_scan(self.settings)
        self.fg.write.assert_called_with("OUTP1 OFF")
        for resource in (self.fg, self.scope, manager):
            resource.close.assert_called_once_with()

    def test_ascii_chunk_pairing_and_current_conversion(self):
        with TemporaryDirectory() as temp:
            self.settings.base_save_dir = temp
            self.settings.target_time_step_s = self.plan.total_duration_s / 4
            self.settings.transfer_chunk_points = 2
            self.settings.voltage_scale_factor = 0.5
            channel = None
            ranges = []
            def write(command):
                nonlocal channel
                if command.startswith(":WAVEFORM:SOURCE CHANNEL"):
                    channel = int(command[-1])
            def query(command):
                if command.startswith(":WAVEFORM:DATA? "):
                    start, count = map(int, command.split()[1].split(","))
                    ranges.append((channel, start, count))
                    return ",".join(str(index + channel) for index in range(start, start + count))
                return {":WAVEFORM:XINCREMENT?": str(self.settings.target_time_step_s),
                        ":WAVEFORM:XORIGIN?": "0", ":WAVEFORM:POINTS?": "5"}[command]
            self.scope.write.side_effect = write
            self.scope.query.side_effect = query
            for flag in (True, False):
                self.settings.use_current_amp = flag
                ranges.clear()
                path = scan.save_capture(self.scope, self.plan, self.settings, "test")
                with Path(path).open(newline="") as handle:
                    columns, *rows = list(csv.reader(handle))
                self.assertEqual(columns, ["Time (s)", "Current (A)", "Voltage (V)"])
                self.assertEqual(ranges, [(ch, start, n) for start, n in [(0, 2), (2, 2), (4, 1)]
                                          for ch in (1, 2)])
                for index, row in enumerate(rows):
                    t, current, voltage = map(float, row)
                    self.assertAlmostEqual(t, index * self.settings.target_time_step_s)
                    self.assertAlmostEqual(current, (1 + index) * 1e-3 * (2 if flag else 1))
                    self.assertAlmostEqual(voltage, (2 + index) * 0.5)
                self.assertEqual(len(rows), 5)


if __name__ == "__main__":
    unittest.main()
