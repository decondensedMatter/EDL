"""Reproducibly create six samples x two current paths x two readout versions.

The legacy source's configure_scope, capture_waveform, turn_off_function_generator,
and the complete acquisition/conversion block are retained verbatim.
"""
import ast
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / "Agilent_infiniium_BTO_regular_waveform_scan_separate_triggered.py"
SAMPLES = ("NaClO4_0.01M", "NaClO4_0.1M", "NaClO4_1M", "NaClO4_9M", "LLZO", "LaF3")


def replace_function(text, name, replacement):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno-1])+replacement.rstrip()+"\n"+"".join(lines[node.end_lineno:])


SETUP = '''def setup_function_generator(waveform_type, frequency_khz, amplitude_v):
    # Waveform-only replacement. Resource identities and readout are unchanged.
    from edl_waveform_output import configure_output
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    fg.write_termination = "\\n"
    fg.read_termination = "\\n"
    try:
        fg.write("*RST")
        fg.write("*CLS")
        # Zero-bias waveform between tests; charging recipes also contain their
        # model-validated zero-reset segment on every repetition.
        reset_s = max(zero_reset_between_tests_s,
                      getattr(active_waveform_case.get("compiled"), "actual_reset_s", 0.))
        fg.write("FUNC DC")
        fg.write("VOLT:OFFS 0")
        fg.write("OUTP ON")
        time.sleep(reset_s)
        fg.write("OUTP OFF")
        configure_output(fg, active_waveform_case)
        fg.write("OUTP ON")
        time.sleep(0.5)
        print(f"Uploaded {waveform_type}; full waveform period {active_waveform_case['total_duration_s']:g} s")
    finally:
        fg.close()
        rm.close()
'''

SAVE = '''def save_data(device_name, waveform_type, frequency_khz, amplitude_v, time_values, voltage_values, current_values):
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    test = "".join(c if c.isalnum() or c in "._-" else "-" for c in waveform_type)
    filename = f"{timestamp}_{device_name}_{test}.csv"
    csv_path = os.path.join(SAVE_PATH, filename)
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time (s)", "Current (A)", "Voltage (V)"])
        writer.writerows(zip(time_values, current_values, voltage_values))
    print(f"Data saved to {csv_path} successfully!")
'''


def make_legacy(sample, amp):
    text = SOURCE.read_text(encoding="utf-8-sig")
    begin = text.index('device_name = ')
    end = text.index('number_of_periods = ', begin)
    text = text[:begin]+f'''# WAVEFORM SELECTION (all six samples have a matching pair of files).
from edl_waveform_library import iter_waveform_cases
from edl_legacy_readout_check import check as check_legacy_readout
from edl_charge.planning import get_profile

device_name = {sample!r}
waveform_type = "EDL"
sweep_mode = "both"  # native and stored_arb; infeasible records are reported.
selected_tests = ("chirp", "optimized", "calibration")
leg_duration_s = 1.0
optimized_directory = None  # Folder made by prepare_edl_waveforms.py optimize.
calibration_config = None   # Matching fitted/config JSON path, when available.
calibration_requests = ()   # Explicit standards/recipe selections; see README.
zero_reset_between_tests_s = 1.0  # Preliminary chirp reset; qualify experimentally.
scope_record_limit_points = None  # Optional known installed-depth check; no scope change.
active_waveform_case = None

'''+text[end:]
    text = text.replace('use_current_amp = True  # Set to True if using a current amplifier with a gain of 2x',
                        f'use_current_amp = {amp}  # Requested current-readout version')
    if not amp:
        text = text.replace('current_preamplifier_gain_a_per_v = 1e-3',
                            'current_preamplifier_gain_a_per_v = 1/50')
    text = text.replace('max_allowed_amplitude_v = 4.0',
                        'max_allowed_amplitude_v = get_profile(device_name)["peak_bound_v"]')
    text = replace_function(text, "build_save_path", '''def build_save_path():
    """Date and specimen directory; each filename identifies the test."""
    return os.path.join(BASE_SAVE_DIR, datetime.date.today().strftime("%Y%m%d"), device_name)
''')
    text = replace_function(text, "setup_function_generator", SETUP)
    text = replace_function(text, "save_data", SAVE)
    old = '    for frequency_khz in frequencies_khz:\n        for amplitude_v in amplitudes_v:\n'
    new = '''    global active_waveform_case, waveform_type
    cases = iter_waveform_cases(device_name, use_current_amp, sweep_mode, optimized_directory,
        selected_tests, calibration_config=calibration_config,
        calibration_requests=calibration_requests, leg_duration_s=leg_duration_s)
    for active_waveform_case in cases:
        ready, detail = check_legacy_readout(active_waveform_case, number_of_periods,
            target_time_step_s, 100., scope_record_limit_points)
        print(detail)
        if not ready:
            continue
        waveform_type = active_waveform_case["test_name"]
        frequency_khz = 1.0 / active_waveform_case["total_duration_s"] / 1e3
        # Retain the original inner acquisition block and current formula.
        for amplitude_v in [active_waveform_case["voltage_peak_v"]]:
'''
    if old not in text:
        raise ValueError("Original scan layout changed; review before regenerating")
    text = text.replace(old,new)
    return text


def main():
    out = HERE / "sample_scripts"
    out.mkdir(exist_ok=True)
    for sample in SAMPLES:
        for amp in (True,False):
            stem = sample.replace(".","p")+("_with_current_amp" if amp else "_without_current_amp")
            # All imports resolve to the shared package one directory above.
            prefix = ('# Generated sample entry; original acquisition retained in this version.\n'
                      'import sys\nfrom pathlib import Path\n'
                      'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n\n')
            route = "with_amp" if amp else "without_amp"
            dispatch = (
                'if __name__ == "__main__":\n'
                '    from edl_updated_entry import CAMPAIGN_FLAGS, campaign_from_entry\n'
                '    if len(sys.argv)==1:\n'
                f'        config = Path(__file__).resolve().parents[1] / "all_samples_120min_{route}_legacy.json"\n'
                f'        raise SystemExit(campaign_from_entry({sample!r},{amp},"legacy",["run","--config",str(config),"--out","session_data"]))\n'
                '    if sys.argv[1] in CAMPAIGN_FLAGS:\n'
                f'        raise SystemExit(campaign_from_entry({sample!r},{amp},"legacy",sys.argv[2:]))\n'
                '    if sys.argv[1] in ("--help","-h"):\n'
                '        print("Six-specimen session: --campaign-six {init-session,check,run}.")\n'
                '        print("No arguments checks the default 120-minute legacy campaign. --legacy-manual selects the historical unbudgeted acquisition.")\n'
                '        raise SystemExit(0)\n'
                '    if sys.argv[1:] != ["--legacy-manual"]:\n'
                '        raise SystemExit("Unknown option. Use --help to see supported entry modes.")\n\n')
            (out/(stem+"_waveform_only.py")).write_text(prefix+dispatch+make_legacy(sample,amp),encoding="utf-8")
            updated = (f'"""{sample}: updated readout, {"with" if amp else "without"} current amplifier."""\n'
                       'import sys\nfrom pathlib import Path\n'
                       'sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n'
                       'from edl_updated_entry import main\n\n'
                       f'if __name__ == "__main__":\n    raise SystemExit(main({sample!r}, {amp}))\n')
            (out/(stem+"_updated_readout.py")).write_text(updated,encoding="utf-8")
    print(f"Created {len(SAMPLES)*4} sample scripts in {out}")


if __name__ == "__main__":
    main()


