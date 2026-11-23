# Generated sample entry; original acquisition retained in this version.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from edl_updated_entry import CAMPAIGN_FLAGS, campaign_from_entry
    if len(sys.argv)==1:
        config = Path(__file__).resolve().parents[1] / "all_samples_120min_without_amp_legacy.json"
        raise SystemExit(campaign_from_entry('NaClO4_9M',False,"legacy",["run","--config",str(config),"--out","session_data"]))
    if sys.argv[1] in CAMPAIGN_FLAGS:
        raise SystemExit(campaign_from_entry('NaClO4_9M',False,"legacy",sys.argv[2:]))
    if sys.argv[1] in ("--help","-h"):
        print("Six-specimen session: --campaign-six {init-session,check,run}.")
        print("No arguments checks the default 120-minute legacy campaign. --legacy-manual selects the historical unbudgeted acquisition.")
        raise SystemExit(0)
    if sys.argv[1:] != ["--legacy-manual"]:
        raise SystemExit("Unknown option. Use --help to see supported entry modes.")

import os
import time
import datetime
import csv

import numpy as np
import pyvisa


# -----------------------------
# User-editable parameters
# -----------------------------
# WAVEFORM SELECTION (all six samples have a matching pair of files).
from edl_waveform_library import iter_waveform_cases
from edl_legacy_readout_check import check as check_legacy_readout
from edl_charge.planning import get_profile

device_name = 'NaClO4_9M'
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

number_of_periods = 3  # Number of waveform periods captured per measurement
use_current_amp = False  # Requested current-readout version
current_preamplifier_gain_a_per_v = 1/50  # Conversion from scope channel 1 voltage to current (A/V)
electrode_area_m2 = (35.2e-6)**2  # Active electrode area in square metres
voltage_scale_factor = 1.0  # Optional scaling for channel 2 if needed
max_allowed_amplitude_v = get_profile(device_name)["peak_bound_v"]  # Do not run if the requested amplitude exceeds this value

target_time_step_s = 1e-8

function_generator_address = "USB0::0x0957::0x2707::MY57301577::INSTR"
oscope_address = "USB0::0x0957::0x9005::MY50140103::INSTR"
# Data are saved to: BASE_SAVE_DIR\<YYYYMMDD>\<device_name>_<waveform tag>
# e.g. D:\Ferroelectric_BTO_optimization\20261106\PLD220_6_Tri
# The date and subfolder are generated automatically; only edit the base directory.
BASE_SAVE_DIR = r"D:\Ferroelectric_BTO_optimization"

WAVEFORM_FOLDER_TAGS = {
    "sine": "Sin",
    "square": "Squ",
    "triangle": "Tri",
    "ramp": "Ramp",
    "pulse": "Puls",
}


def build_save_path():
    """Date and specimen directory; each filename identifies the test."""
    return os.path.join(BASE_SAVE_DIR, datetime.date.today().strftime("%Y%m%d"), device_name)


# Fixed once at start-up so a run that crosses midnight stays in one folder.
SAVE_PATH = build_save_path()


def setup_function_generator(waveform_type, frequency_khz, amplitude_v):
    # Waveform-only replacement. Resource identities and readout are unchanged.
    from edl_waveform_output import configure_output
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    fg.write_termination = "\n"
    fg.read_termination = "\n"
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


def turn_off_function_generator():
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    try:
        fg.write("OUTP OFF")
        print("Function generator output turned OFF.")
    finally:
        fg.close()
        rm.close()


def capture_waveform(scope, channel, capture_duration_s):
    scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
    scope.write(":WAVEFORM:FORMAT ASCII")

    x_increment = float(scope.query(":WAVeform:XINCrement?"))
    if not np.isclose(x_increment, target_time_step_s, rtol=0.02, atol=1e-12):
        print(f"Warning: scope returned {x_increment:g} s/point, requested {target_time_step_s:g}")
    x_origin = float(scope.query(":WAVeform:XORigin?"))
    raw_data = scope.query(":WAVEFORM:DATA?").strip()

    try:
        raw_values = np.array([float(i) for i in raw_data.split(",") if i.strip()], dtype=np.float64)
    except ValueError as exc:
        print(f"Error converting waveform data for Channel {channel}: {exc}")
        return None, None

    # Infiniium ASCII waveform samples are already returned in calibrated
    # engineering units. YINCREMENT/YORIGIN conversion is only required when
    # reading integer BYTE/WORD waveform data.
    voltage_values = raw_values
    time_values = np.arange(len(voltage_values)) * x_increment + x_origin
    print(f"Channel {channel}: received {len(voltage_values)} points at {x_increment:.3g} s/point")
    if len(time_values) == 0:
        return None, None

    end_time = time_values[0] + capture_duration_s
    mask = (time_values >= time_values[0]) & (time_values <= end_time)
    if np.count_nonzero(mask) < 10:
        return time_values, voltage_values

    return time_values[mask], voltage_values[mask]


def save_data(device_name, waveform_type, frequency_khz, amplitude_v, time_values, voltage_values, current_values):
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


def configure_scope(scope, frequency_hz, amplitude_v):
    period_s = 1.0 / frequency_hz
    if number_of_periods <= 0:
        raise ValueError("number_of_periods must be greater than zero")
    capture_duration_s = float(number_of_periods) * period_s
    requested_points = int(round(capture_duration_s / target_time_step_s)) + 1
    time_div_s = capture_duration_s / 10.0
    time_div_s = max(time_div_s, 1e-6)

    trigger_level = 0.2

    scope.write("*RST")
    scope.write("*CLS")
    scope.write(":CHANNEL1:DISPLAY ON")
    scope.write(":CHANNEL2:DISPLAY ON")
    scope.write(":CHANNEL3:DISPLAY ON")
    scope.write(f":TIMEBASE:SCALE {time_div_s}")
    scope.write(":TIMEBASE:POSITION 0")
    scope.write(":CHANNEL1:SCALE 0.1")
    scope.write(":CHANNEL2:SCALE 1")
    scope.write(":CHANNEL3:SCALE 1")
    scope.write(":CHANNEL1:OFFSET 0")
    scope.write(":CHANNEL2:OFFSET 0")
    scope.write(":CHANNEL3:OFFSET 0")
    scope.write(":WAVEFORM:FORMAT ASCII")
    scope.write(":ACQUIRE:SRATE:AUTO OFF")
    scope.write(f":ACQUIRE:SRATE {1.0 / target_time_step_s}")
    scope.write(f":ACQUIRE:POINTS {requested_points}")
    scope.write(f":WAVEFORM:POINTS {requested_points}")

    scope.write(":ACQUIRE:MODE HRESOLUTION")
    scope.write(":ACQuire:AVERage ON")
    scope.write(":ACQuire:AVERage:COUNt 256") 
    scope.write(":TRIGGER:MODE EDGE")
    scope.write(":TRIGGER:EDGE:SOURCE CHANNEL3")
    scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
    scope.write(":TRIGGER:HYSTERESIS NREJECT")
    scope.write(":TRIGGER:SWEEP TRIGGERED")
    scope.write(f":TRIGger:LEVel CHANnel3,{trigger_level}")
    print(f"Single acquisition; requesting {requested_points} points at {target_time_step_s:g} s/point")

    return capture_duration_s


def main():
    if electrode_area_m2 <= 0 or target_time_step_s <= 0:
        raise ValueError("electrode_area_m2 and target_time_step_s must be greater than zero")
    os.makedirs(SAVE_PATH, exist_ok=True)
    print(f"Saving data to: {SAVE_PATH}")
    global active_waveform_case, waveform_type
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
            amplitude_v = float(amplitude_v)
            if amplitude_v > max_allowed_amplitude_v:
                print(
                    f"Skipping measurement: requested amplitude {amplitude_v} V exceeds the limit of {max_allowed_amplitude_v} V."
                )
                continue

            print(f"\nRunning measurement: {waveform_type} @ {frequency_khz} kHz, amplitude {amplitude_v} V")
            setup_function_generator(waveform_type, frequency_khz, amplitude_v)
            time.sleep(2)  # Wait for the system to stabilize after uploading a new waveform

            rm = pyvisa.ResourceManager()
            scope = rm.open_resource(oscope_address)
            scope.write_termination = "\n"
            scope.read_termination = "\n"
            scope.timeout = 100000
            print("Oscilloscope ID:", scope.query("*IDN?"))

            frequency_hz = float(frequency_khz) * 1e3
            capture_duration_s = configure_scope(scope, frequency_hz, amplitude_v)

            # Perform one acquisition and explicitly read the completion reply.
            # This prevents a pending/unterminated query in the scope I/O queue.
            scope.query(":DIGITIZE CHANNEL1,CHANNEL2;*OPC?")

            time_ch1, current_raw = capture_waveform(scope, 1, capture_duration_s)
            time_ch2, voltage_raw = capture_waveform(scope, 2, capture_duration_s)

            if time_ch1 is None or time_ch2 is None or current_raw is None or voltage_raw is None:
                print("Failed to capture waveform data. Skipping save.")
                scope.write(":RUN")
                scope.close()
                rm.close()
                continue

            if len(time_ch1) != len(current_raw) or len(time_ch2) != len(voltage_raw):
                print("Captured waveform lengths do not match. Skipping save.")
                scope.write(":RUN")
                scope.close()
                rm.close()
                continue
            if use_current_amp==True:
                current_values = 2*current_raw * current_preamplifier_gain_a_per_v
            else:
                current_values = current_raw * current_preamplifier_gain_a_per_v
            voltage_values = voltage_raw * voltage_scale_factor

            save_data(
                device_name=device_name,
                waveform_type=waveform_type,
                frequency_khz=frequency_khz,
                amplitude_v=amplitude_v,
                time_values=time_ch1,
                voltage_values=voltage_values,
                current_values=current_values,
            )

            scope.write(":RUN")
            scope.close()
            rm.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        turn_off_function_generator()
