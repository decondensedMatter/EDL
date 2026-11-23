"""Create offline configs, manifests and the deliverable; never imports PyVISA."""
import hashlib
import json
from pathlib import Path
import platform
import time
import unittest
import zipfile
from build_edl_scan_versions import main as build_entries
from edl_charge.workflow import make_config,plan,save
from edl_charge.planning import load_profiles
from edl_charge.session import template
from edl_charge.model_defaults import model_fit_defaults

ROOT=Path(__file__).parent

def main():
    previous=ROOT/'EDL_PACKAGE_MANIFEST.json'
    if previous.is_file():
        for name,expected in json.loads(previous.read_text(encoding='utf8'))['files'].items():
            if name.startswith('sample_configs/') or (name.startswith('all_samples_120min_') and name.endswith('.json')):
                if (ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Preserve locally edited configuration before regenerating templates: '+name)
    original_hashes={
        'Agilent_infiniium_BTO_regular_waveform_scan_separate_triggered.py':
            'c1162d8b3ebf63318ea9b9f1da45d324b11b710c41e419951ef6d4402a82633e',
        'Agilent_infiniium_BTO_PUND_scan_separate_triggered_volt_amp.py':
            '6cf883942b3aa14f3e1c9a36cc34fe1806e5dee5480719d74658227659163d71'}
    for name,expected in original_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Original source changed: '+name)
    build_entries()
    configdir=ROOT/'sample_configs'
    configdir.mkdir(exist_ok=True)
    for sample in load_profiles():
        for amp in (True,False):
            suffix='with_current_amp' if amp else 'without_current_amp'
            save(configdir/(sample.replace('.','p')+'_'+suffix+'.json'),model_fit_defaults(make_config(sample,amp)))
    save(ROOT/'waveform_plan_with_current_amp.json',plan(None,True))
    save(ROOT/'waveform_plan_without_current_amp.json',plan(None,False))
    save(ROOT/'all_samples_120min_without_amp.json',template())
    campaign_configs=[]
    for amp in (False,True):
        for readout in ('updated','legacy'):
            from run_all_samples import config_filename
            filename=config_filename(amp,readout)
            save(ROOT/filename,template(amp,readout))
            campaign_configs.append(filename)
    started=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT),top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Offline tests failed; no new delivery archive was created')
    verification=dict(offline_tests_passed=result.testsRun,elapsed_s=time.monotonic()-started,
                      python=platform.python_version(),hardware_operated=False,
                      original_source_hashes_verified=original_hashes,
                      scope='Numerical, file-generation and mocked acquisition tests; no bench validation')
    save(ROOT/'OFFLINE_VERIFICATION.json',verification)
    selected = [
        'EDL_SCRIPTS_README.txt','ENVIRONMENT.txt','RUNTIME_ESTIMATES.txt','HOLD_RESET_PLAN.txt','requirements.txt',
        'edl_waveform_library.py','edl_waveform_output.py','edl_legacy_readout_check.py',
        'edl_updated_entry.py','prepare_edl_waveforms.py','estimate_edl_runtime.py',
        'estimate_session_runtime.py','run_all_samples.py','run_all_samples_without_amp.py','run_all_samples_with_amp.py',
        'all_samples_120min_without_amp.json','OFFLINE_VERIFICATION.json',
        'build_edl_scan_versions.py','make_edl_delivery.py','test_edl_script_versions.py',
        'keysight_33511b_chirp_plan.py','keysight_33511b_chirp_generator.py','keysight_33511b_chirp_scan.py',
        'Agilent_infiniium_BTO_regular_waveform_scan_separate_triggered.py',
        'Agilent_infiniium_BTO_PUND_scan_separate_triggered_volt_amp.py',
        'waveform_plan_with_current_amp.json','waveform_plan_without_current_amp.json',
    ]
    files=[ROOT/n for n in selected+campaign_configs]
    files+=list(ROOT.glob('test_keysight_33511b_chirp*.py'))
    files+=list((ROOT/'edl_charge').glob('*.py'))+list((ROOT/'edl_charge').glob('*.json'))
    files+=list((ROOT/'sample_scripts').glob('*.py'))+list(configdir.glob('*.json'))
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    save(ROOT/'EDL_PACKAGE_MANIFEST.json',dict(files=hashes,offline_tests_passed=result.testsRun,
         hardware_operated=False,entry_scripts=24,configuration_profiles=12,
         integrated_session='run_all_samples.py',campaign_templates=4,
         charging_polarities=['positive','negative'],active_budget_seconds=7200,
         run_order='specimen, ascending final-voltage magnitude (chirp Vpp/2), tests and both polarities',
         voltage_stage_peak_policy='sample',calibration_order='progressive cumulative per voltage level'))
    files.append(ROOT/'EDL_PACKAGE_MANIFEST.json')
    archive=ROOT/'EDL_all_samples_120min_bipolar_calibration.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith('sample_scripts/')])==24
        for name,expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    print(f'{archive}: {len(files)} files, {archive.stat().st_size:,} bytes; contents verified')

if __name__=='__main__':
    main()
