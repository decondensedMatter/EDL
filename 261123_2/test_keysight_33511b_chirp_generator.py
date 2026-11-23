"""Offline checks of command order and the uploaded DAC record."""

import math
import unittest

import keysight_33511b_chirp_generator as generator
from keysight_33511b_chirp_plan import make_plan


class GeneratorResource:
    def __init__(self, free=1_000_000):
        self.free = free
        self.writes = []
        self.settings = {}
        self.uploads = {}
        self.upload_settings = {}

    def write(self, command):
        self.writes.append(command)
        if command == "SOUR1:DATA:VOL:CLE":
            self.free += sum(len(values) for values in self.uploads.values())
            self.uploads.clear()
        key, _, value = command.partition(" ")
        self.settings[key] = {"ON": "1", "OFF": "0"}.get(value, value)

    def query(self, command):
        replies = {
            "*IDN?": "Keysight Technologies,33511B,MY00000001,5.03-1.00",
            "SYST:ERR?": '+0,"No error"',
            "SOUR1:DATA:VOL:CAT?": ",".join(self.uploads),
            "SOUR1:DATA:VOL:FREE?": str(self.free),
        }
        if command in replies:
            return replies[command]
        if command == "SOUR1:FUNC:ARB:POIN?":
            return str(len(self.uploads[self.settings["SOUR1:FUNC:ARB"]]))
        return self.settings[command.removesuffix("?")]

    def write_binary_values(self, command, values, **settings):
        name = command.split()[1].rstrip(",")
        self.uploads[name] = values
        self.upload_settings = settings
        self.free -= len(values)


class GeneratorConfigurationTests(unittest.TestCase):
    def assert_common(self, device, amplitude):
        self.assertEqual(device.writes[:3], ["OUTP1 OFF", "INIT1:CONT OFF", "ABOR"])
        self.assertNotIn("*RST", device.writes)
        self.assertNotIn("OUTP1 ON", device.writes)
        self.assertEqual(device.settings["OUTP1:LOAD"], "INF")
        self.assertEqual(device.settings["SOUR1:VOLT:UNIT"], "VPP")
        self.assertEqual(float(device.settings["SOUR1:VOLT:OFFS"]), 0)
        self.assertEqual(float(device.settings["SOUR1:VOLT"]), amplitude)
        self.assertEqual(device.settings["OUTP:SYNC"], "1")

    def test_native_all_shapes(self):
        for shape, scpi, top in [("square", "SQU", 2e6), ("sine", "SIN", 2e6),
                                ("triangle", "RAMP", 20e3)]:
            with self.subTest(shape=shape):
                device = GeneratorResource()
                plan = make_plan(shape, 2.2, 1)
                generator.configure_native(device, plan)
                self.assert_common(device, 2.2)
                self.assertEqual(device.settings["SOUR1:FUNC"], scpi)
                if shape == "triangle":
                    self.assertEqual(device.settings["SOUR1:FUNC:RAMP:SYMM"], "50")
                    self.assertLess(device.writes.index("SOUR1:FREQ 100"),
                                    device.writes.index("SOUR1:FUNC RAMP"))
                if shape == "square":
                    self.assertEqual(device.settings["SOUR1:FUNC:SQU:DCYC"], "50")
                self.assertEqual(float(device.settings["SOUR1:FREQ:STAR"]), top)
                self.assertEqual(float(device.settings["SOUR1:FREQ:STOP"]), 10)
                self.assertEqual(device.settings["SOUR1:SWE:SPAC"], "LOG")
                self.assertEqual(device.settings["SOUR1:SWE:TIME"], "1")
                self.assertEqual(device.settings["SOUR1:SWE:RTIM"], "1")
                self.assertEqual(device.settings["TRIG1:COUN"], "2")
                self.assertEqual(device.settings["TRIG1:SOUR"], "IMM")
                generator.start_output(device, "native")
                self.assertEqual(device.writes[-2:], ["OUTP1 ON", "INIT1"])

    def test_arb_all_shapes_and_normalized_samples(self):
        rate = 8000
        for shape in ("sine", "triangle", "square"):
            with self.subTest(shape=shape):
                # Nonintegral record duration tests ceil sizing and the last sample timestamp.
                plan = make_plan(shape, 2.2, 0.010013, f_max=100, mode="arb")
                device = GeneratorResource()
                generator.configure_arb(device, plan, rate)
                self.assert_common(device, 2.2)
                self.assertEqual(device.upload_settings, {"datatype": "h", "is_big_endian": False})
                self.assertEqual(device.settings["FORM:BORD"], "SWAP")
                self.assertEqual(device.settings["SOUR1:BURS:NCYC"], "1")
                self.assertEqual(device.settings["TRIG1:SOUR"], "BUS")
                self.assertEqual(device.settings["SOUR1:FUNC:ARB:FILT"],
                                 "STEP" if shape == "square" else "NORM")
                data = next(iter(device.uploads.values()))
                self.assertEqual(len(data), math.ceil(4 * plan.leg_duration_s * rate))
                for i, sample in enumerate(data):
                    phase = plan.phase_cycles_at(i / rate) % 1
                    if shape == "square":
                        expected = 1 if phase < 0.5 else -1
                    elif shape == "triangle":
                        expected = 1 - 4 * abs((phase + 0.25) % 1 - 0.5)
                    else:
                        expected = math.sin(math.tau * phase)
                    self.assertAlmostEqual(sample / 32767, expected, delta=0.51 / 32767)
                self.assertLessEqual(max(abs(x / 32767 * 1.1) for x in data), 1.1)
                generator.start_output(device, "arb")
                self.assertEqual(device.writes[-3:], ["OUTP1 ON", "INIT1", "*TRG"])

    def test_refuse_insufficient_device_memory_and_preserve_loaded_waveform(self):
        plan = make_plan("sine", 0.1, 0.01, f_max=100, mode="arb")
        device = GeneratorResource(free=319)
        with self.assertRaisesRegex(ValueError, "319 free"):
            generator.configure_arb(device, plan, 8000)
        self.assertEqual(device.uploads, {})
        device.free = 320
        generator.configure_arb(device, plan, 8000)
        generator.configure_arb(device, plan, 8000)
        self.assertEqual(device.free, 0)
        self.assertEqual(len(device.uploads), 1)
        self.assertFalse(any("CLE" in command for command in device.writes))

    def test_release_owned_buffer_between_shapes(self):
        device = GeneratorResource(free=320)
        for shape in ("sine", "square", "triangle"):
            plan = make_plan(shape, 0.1, 0.01, f_max=100, mode="arb")
            owned = generator.configure_arb(device, plan, 8000)
            self.assertIsNotNone(owned)
            self.assertIsNone(generator.configure_arb(device, plan, 8000))
            self.assertTrue(generator.release_arb(device, owned))
            self.assertEqual(device.free, 320)

    def test_preserve_other_waveforms_even_if_loaded_after_our_upload(self):
        device = GeneratorResource()
        plan = make_plan("sine", 0.1, 0.01, f_max=100, mode="arb")
        owned = generator.configure_arb(device, plan, 8000)
        device.uploads["OTHER.ARB"] = [1, 2]
        self.assertFalse(generator.release_arb(device, owned))
        self.assertIn("OTHER.ARB", device.uploads)
        second = make_plan("square", 0.1, 0.01, f_max=100, mode="arb")
        self.assertIsNone(generator.configure_arb(device, second, 8000))
        self.assertFalse(generator.release_arb(device, None))


if __name__ == "__main__":
    unittest.main()
