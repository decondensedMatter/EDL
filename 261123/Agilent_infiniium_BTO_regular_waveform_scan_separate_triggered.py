import os
import time
import datetime
import csv

import numpy as np
import pyvisa


# -----------------------------
# User-editable parameters
# -----------------------------
device_name = "Xin_0p1M_NaClO4_electrolyte"  # Added to the output filename
waveform_type = "triangle"  # Options: sine, square, triangle, ramp
# frequencies_khz = [1,2,3,5,7,10]  # Frequencies in kHz
# amplitudes_v = [0.1,0.24,0.3,0.5,0.8,1]  # Output amplitudes in volts
frequencies_khz = [1,5,10,20,50,100]  # Frequencies in kHz
amplitudes_v = [1,2,3,4]
number_of_periods = 3  # Number of waveform periods captured per measurement
use_current_amp = True  # Set to True if using a current amplifier with a gain of 2x
current_preamplifier_gain_a_per_v = 1e-3  # Conversion from scope channel 1 voltage to current (A/V)
electrode_area_m2 = (35.2e-6)**2  # Active electrode area in square metres
voltage_scale_factor = 1.0  # Optional scaling for channel 2 if needed
max_allowed_amplitude_v = 4.0  # Do not run if the requested amplitude exceeds this value

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
    """Return BASE_SAVE_DIR\\<today YYYYMMDD>\\<device_name>_<waveform tag>."""
    date_folder = datetime.date.today().strftime("%Y%m%d")
    waveform_tag = WAVEFORM_FOLDER_TAGS.get(waveform_type.lower(), waveform_type.capitalize())
    subfolder = f"{device_name}_{waveform_tag}"
    return os.path.join(BASE_SAVE_DIR, date_folder, subfolder)


# Fixed once at start-up so a run that crosses midnight stays in one folder.
SAVE_PATH = build_save_path()


def setup_function_generator(waveform_type, frequency_khz, amplitude_v):
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    fg.write_termination = "\n"
    fg.read_termination = "\n"
    fg.write("*RST")
    fg.write("*CLS")

    waveform_map = {
        "sine": "SIN",
        "square": "SQU",
        "triangle": "TRI",
        "ramp": "RAMP",
        "pulse": "PULS",
    }

    if waveform_type.lower() not in waveform_map:
        raise ValueError(f"Unsupported waveform type: {waveform_type}. Choose from {list(waveform_map.keys())}")

    freq_hz = float(frequency_khz) * 1e3
    amplitude_v = float(amplitude_v)
    amplitude_vpp = 2.0 * amplitude_v

    fg.write(f"FUNC {waveform_map[waveform_type.lower()]}")
    fg.write(f"FREQ {freq_hz}")
    fg.write(f"VOLT {amplitude_vpp}")
    fg.write("VOLT:OFFS 0")
    fg.write("OUTP ON")
    time.sleep(0.5)

    print(f"Uploaded {waveform_type} at {frequency_khz} kHz: {amplitude_v} V peak ({amplitude_vpp} Vpp).")

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def sanitize(value):
        return str(value).replace("-", "m")

    filename = (
        f"{device_name}_{waveform_type}_f{sanitize(frequency_khz)}kHz_"
        f"amp{sanitize(amplitude_v)}V_"
        f"area{electrode_area_m2:.6g}m2_"
        f"gain{sanitize(current_preamplifier_gain_a_per_v)}AperV_{timestamp}.csv"
    )
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
    for frequency_khz in frequencies_khz:
        for amplitude_v in amplitudes_v:
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
