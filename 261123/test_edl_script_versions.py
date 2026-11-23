import ast
from pathlib import Path
import unittest

from build_edl_scan_versions import SOURCE, SAMPLES, make_legacy
from edl_legacy_readout_check import check
from edl_waveform_library import iter_waveform_cases


class ScriptVersions(unittest.TestCase):
    def test_all_shapes_and_modes_finish_before_next_voltage(self):
        from keysight_33511b_chirp_scan import ScanSettings, scan_plans
        for sample in SAMPLES:
            cases=list(iter_waveform_cases(sample,False,'both'))
            ready=[c for c in cases if c['ready']]
            peaks=[c['voltage_peak_v'] for c in ready]
            self.assertEqual(peaks,sorted(peaks))
            for peak in sorted(set(peaks)):
                group=[c for c in ready if c['voltage_peak_v']==peak]
                self.assertEqual({c['plan'].waveform for c in group},{'sine','square','triangle'})
        plans=scan_plans(ScanSettings(sample_name='0.1M NaClO4'))
        self.assertEqual([p.voltage_vpp for p in plans],sorted(p.voltage_vpp for p in plans))

    def test_manual_cases_interleave_tests_at_each_physical_level(self):
        from unittest.mock import patch
        def case(peak,kind,**extra):
            return dict(ready=True,voltage_peak_v=peak,waveform_kind=kind,**extra)
        sweeps=[case(.4,'native',plan=object()),case(.1,'native',plan=object())]
        charging=[case(.4,'compiled',final_voltage_v=-.4),
                  case(.3,'compiled',final_voltage_v=.1),
                  case(.2,'compiled',final_voltage_v=-.1)]
        calibration=[case(.4,'compiled',stage={},connection_required='sample'),
                     case(.1,'compiled',stage={},connection_required='sample')]
        with patch('edl_waveform_library._sweeps',return_value=sweeps), \
             patch('edl_waveform_library._optimized',return_value=charging), \
             patch('edl_waveform_library._calibration',return_value=calibration):
            cases=list(iter_waveform_cases('0.1M',False,selected_tests='all'))
        self.assertEqual([abs(c.get('final_voltage_v',c['voltage_peak_v'])) for c in cases],[.1]*4+[.4]*3)
        self.assertIn('stage',cases[0])
        self.assertIn('stage',cases[4])

    def test_unchanged_legacy_readout(self):
        original = SOURCE.read_text(encoding="utf-8-sig")
        def functions(text):
            return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body
                    if isinstance(n,ast.FunctionDef)}
        before = functions(original)
        acquisition = original[original.index('            time.sleep(2)'):original.index('\n\n\nif __name__')]
        for sample in SAMPLES:
            for amp in (True,False):
                changed = make_legacy(sample,amp)
                after = functions(changed)
                for name in ('capture_waveform','configure_scope','turn_off_function_generator'):
                    self.assertEqual(before[name],after[name],(sample,amp,name))
                self.assertIn(acquisition,changed)
                assignments = {n.targets[0].id:n.value for n in ast.parse(changed).body
                    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
                self.assertEqual(ast.literal_eval(assignments['use_current_amp']),amp)
        files = list((SOURCE.parent/'sample_scripts').glob('*.py'))
        self.assertEqual(len(files),24)
        for path in files:
            ast.parse(path.read_text(encoding='utf-8'))

    def test_all_entries_bind_integrated_campaign_without_visa(self):
        import runpy
        import sys
        from types import ModuleType
        from unittest.mock import Mock, patch
        master = ModuleType("run_all_samples")
        master.main = Mock(return_value=2)
        with patch.dict(sys.modules,{"run_all_samples":master,"pyvisa":None}):
            for sample in SAMPLES:
                for amp in (True,False):
                    path_tag = "with_current_amp" if amp else "without_current_amp"
                    mode_tag = "with_amp" if amp else "without_amp"
                    for readout,suffix in (("updated","updated_readout"),("legacy","waveform_only")):
                        path = SOURCE.parent/"sample_scripts"/f"{sample.replace('.','p')}_{path_tag}_{suffix}.py"
                        argv = ["check","--config","profile.json"]
                        with patch.object(sys,"argv",[str(path),"--campaign-six",*argv]),                                 patch.object(sys,"path",list(sys.path)), self.assertRaises(SystemExit) as exc:
                            runpy.run_path(str(path),run_name="__main__")
                        self.assertEqual(exc.exception.code,2)
                        master.main.assert_called_with(argv,use_current_amp=amp,readout_mode=readout)
                        if readout == "legacy":
                            with patch.object(sys,"argv",[str(path)]), patch.object(sys,"path",list(sys.path)),                                     self.assertRaises(SystemExit):
                                runpy.run_path(str(path),run_name="__main__")
                            cfg = str(SOURCE.parent/f"all_samples_120min_{mode_tag}_legacy.json")
                            master.main.assert_called_with(["run","--config",cfg,"--out","session_data"],
                                                           use_current_amp=amp,readout_mode="legacy")

    def test_all_profiles_and_limits(self):
        for sample in SAMPLES:
            for amp in (True,False):
                cases=list(iter_waveform_cases(sample,amp,'both'))
                self.assertEqual(len({c['test_name'] for c in cases}),len(cases))
                native=[c for c in cases if c['waveform_kind']=='native']
                stored=[c for c in cases if c['waveform_kind']=='stored_arb']
                self.assertTrue(native)
                self.assertTrue(all(c['ready'] for c in native))
                self.assertTrue(all(not c['ready'] for c in stored))
                self.assertTrue(all(not check(c)[0] for c in native))
        self.assertTrue(check(dict(ready=True,waveform_kind='compiled',total_duration_s=.02))[0])


if __name__=='__main__':
    unittest.main()

