import unittest

import numpy as np

from edl_charge.awg import compile_protocol, compile_calibration_protocol, configure_sequence, start_output


class Generator:
    def __init__(self):
        self.commands = []
        self.arrays = []
        self.payload = None

    def write(self, s):
        self.commands.append(s)

    def write_raw(self, s):
        self.payload = s

    def write_binary_values(self, cmd, values, **kwargs):
        self.arrays.append(values)

    def query(self, cmd):
        return {"*OPC?": "1", "SYST:ERR?": '0,"No error"', "DATA:VOL:FREE?": "1000000",
                "FUNC:ARB:FILT?": "STEP", "FUNC:ARB:SRAT?": "160000000",
                "VOLT?": ".95", "VOLT:OFFS?": ".475"}[cmd]


class TestAwg(unittest.TestCase):
    def test_sequence(self):
        p = compile_protocol([0, 1e-6, 3e-6], [0, .95, .8], 1, [0, 2e-6],
                             [.8, 0], 0.5, voltage_limit_v=1, sample_rate_hz=160e6)
        self.assertLess(p.allocated_points, 10000)
        self.assertAlmostEqual(p.duration_s, 1.500005)
        self.assertEqual(p.segments[0].control, "repeatTilTrig")
        self.assertEqual(p.segments[-1].control, "repeatInf")
        self.assertEqual(sum(s.marker == "highAtStart" for s in p.segments), 1)
        finite = sum(len(s.samples_volts)*s.repeat_count for s in p.segments[1:-1])
        self.assertEqual(finite, round(p.duration_s*p.sample_rate_hz))
        self.assertTrue(all(8 <= len(s.samples_volts) and s.repeat_count <= 1000000 for s in p.segments))
        g = Generator()
        configure_sequence(g, p, 1)
        self.assertEqual(g.commands[0], "OUTP OFF")
        self.assertNotIn("OUTP ON", g.commands)
        self.assertIn("VOLT 0.95", g.commands)
        self.assertIn("VOLT:OFFS 0.475", g.commands)
        self.assertTrue(np.all(g.arrays[0] == -32767))
        descriptor = g.payload.split(b"#", 1)[1]
        n = int(descriptor[:1])
        self.assertEqual(int(descriptor[1:1+n]), len(descriptor[1+n:-1]))
        start_output(g)
        self.assertEqual(g.commands[-2:], ["OUTP ON", "*TRG"])

    def test_short_holds_and_grid(self):
        p = compile_protocol([0, 168.75e-9], [0, 1], 1e-9, reset_s=1e-9,
                             voltage_limit_v=1, sample_rate_hz=160e6)
        self.assertAlmostEqual(p.actual_charge_s, 168.75e-9)
        self.assertEqual(p.actual_hold_s, 6.25e-9)
        self.assertEqual(len(p.segments[1].samples_volts), 28)
        self.assertEqual(p.segments[1].samples_volts[-1], 1)
        self.assertEqual(p.segments[-2].samples_volts[-1], 0)

    def test_long_ramp_and_limits(self):
        p = compile_protocol([0, 1], [0, 1], voltage_limit_v=1)
        self.assertLess(p.sample_rate_hz, 100)
        for t, v, kwargs in ((1, 1, {"sample_rate_hz": 160e6}),
                             (1, 1.01, {}), (5e-9, 1, {})):
            with self.subTest(t=t, v=v, kwargs=kwargs), self.assertRaises(ValueError):
                compile_protocol([0, t], [0, v], voltage_limit_v=1, **kwargs)

    def test_negative_calibration_and_dac_replay(self):
        p = compile_calibration_protocol([0, 1], [0, -.37], hold_s=10000,
                                         reset_s=.1, voltage_limit_v=.4, sample_rate_hz=100)
        self.assertEqual((p.voltage_min, p.voltage_max, p.voltage_peak), (-.37, 0, .37))
        t, v = p.control_knots()
        self.assertLess(len(t), 300)
        self.assertEqual(v[0], 0)
        self.assertEqual(v[-1], 0)
        self.assertEqual(v[t == p.actual_charge_s][0], -.37)
        charge = p.segments[1].samples_volts
        self.assertLessEqual(np.max(np.abs(np.interp(np.arange(100)/100, t, v)-charge)),
                             .37/65534/2 + 1e-12)


if __name__ == "__main__":
    unittest.main()
