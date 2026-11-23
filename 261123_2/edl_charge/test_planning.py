import unittest

if __package__:
    from . import planning as p
else:
    import planning as p


class PlanningTest(unittest.TestCase):
    def test_all_bounds_and_endpoints(self):
        caps = [.8, .8, .8, .8, 1., 2.]
        profiles = p.load_profiles()
        self.assertEqual([s["peak_bound_v"] for s in profiles.values()], caps)
        from keysight_33511b_chirp_plan import CHIRP_VPP_LEVELS
        for name, profile in profiles.items():
            chirp_name = (name.split("_")[1] + " NaClO4" if name.startswith("NaClO4")
                          else "Ta_LLZO" if name == "LLZO" else name)
            self.assertEqual(profile["chirp_vpp_levels"], list(CHIRP_VPP_LEVELS[chirp_name]))
            self.assertLessEqual(max(profile["chirp_vpp_levels"]) / 2, profile["peak_bound_v"])
            if name.startswith("NaClO4"):
                self.assertEqual(profile["chirp_vpp_levels"], [.1, .2, .4, .8, 1.6])
                self.assertEqual((profile["window_width_v"], profile["margin_v"]), (1.6, 0.))
                self.assertEqual(profile["electrode_window_width_v"], 1.6)
                self.assertEqual(p.final_voltage_grid(profile["peak_bound_v"]), [.1, .2, .4, .8])
                self.assertAlmostEqual(profile["model_parameters"]["area_m2"], 3.141592653589793 * (.0016 / 2)**2)
                self.assertEqual(profile["electrode_gap_m"], .017)
            for mode, floor in ((True, 8.75e-6), (False, 168.75e-9)):
                rows = p.build_candidate_plan(name, mode)
                self.assertEqual({r["variant"] for r in rows}, set(p.VARIANTS))
                self.assertEqual(min(r["charge_time_s"] for r in rows), floor)
                self.assertEqual(max(r["final_voltage_v"] for r in rows), profile["peak_bound_v"])
                self.assertEqual(max(r["charge_time_s"] for r in rows), 1.)
                self.assertAlmostEqual(profile["window_width_v"]/2-profile["margin_v"], profile["peak_bound_v"])
        self.assertEqual(p.final_voltage_grid(.8), [.1, .2, .4, .8])
        self.assertEqual(p.duration_grid(.25), [1., .5, .25])

    def test_measured_cutoff_and_calibration(self):
        self.assertGreaterEqual(p.hardware_profile(False, {"measured_bandwidth_hz": 1e4})
                                ["min_charge_time_s"], 350e-6-1e-15)
        self.assertEqual(p.hardware_profile(True, {"measured_bandwidth_hz": 1e9})
                         ["min_charge_time_s"], 8.75e-6)
        prof = p.get_profile("9M NaClO₄")
        self.assertIsNone(prof["initial_guess"]["eps_r"])
        self.assertIsNone(prof["model_parameters"]["R"])
        cal = p.calibration_protocols(prof, True)
        self.assertEqual(cal["hardware"]["data_hardware_calibrated"], False)
        self.assertTrue(any(s["split"]=="validation" for s in cal["stages"]))
        sine = next(s for s in cal["stages"] if s["name"]=="sample_small_signal")
        self.assertLessEqual(max(sine["bias_voltages_v"])+sine["amplitude_v"], prof["peak_bound_v"])


if __name__ == "__main__":
    unittest.main()
