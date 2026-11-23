import csv
import datetime
import itertools
import os
import time

import numpy as np
import pyvisa


def long_path(path):
    """Prepend the \\\\?\\ prefix on Windows so writes to paths longer than
    the 260-character MAX_PATH limit succeed without registry changes."""
    if os.name != "nt":
        return path
    path = os.path.abspath(path)
    if path.startswith("\\\\?\\"):
        return path
    return "\\\\?\\" + path


# -----------------------------
# User-editable scan parameters
# -----------------------------
device_name = "Autumn_2D_A4"
# PUND (reduced): one negative reset pulse (2x width, sets a known initial state)
# followed by two positive pulses P (switching) and U (non-switching).
# Pr (switching polarization) is later obtained offline as integral(P) - integral(U).
pulse_widths_s = [100e-6]  # Flat-top duration of each P/U pulse (reset is 2x this)
rise_times_s = [200e-6]  # Shared by the reset, P, and U pulses
gap_reset_to_positive_s = [10e-6]  # Reset-to-P gap duration
gap_positive_to_positive_s = [10e-6]  # P-to-U gap duration
# Requested PEAK voltage at the high-impedance sample (not the generator or
# monitor voltage) for the reset/P/U pulses. For example, 0.8 means +/-0.8 V at the sample.
amplitudes_v = [35]

baseline_before_s = 5e-6
baseline_after_s = 5e-6
arb_sample_rate_hz = 1e7
maximum_arb_points = 1_000_000
target_time_step_s = 1e-8
number_of_periods = 2
use_current_amp = False  # Set to True if using a current amplifier with a gain of 2x
current_preamplifier_gain_a_per_v = 1/50
electrode_area_m2 = (10e-6) ** 2
voltage_amplifier_gain = 20.0  # Magnitude only; X-20 output is inverted relative to its input
# The X-20 inverts the generator waveform, so without compensation the sample
# sees a POSITIVE reset followed by two NEGATIVE pulses. Set True to negate the
# generator waveform so the sample sees a NEGATIVE reset followed by two POSITIVE pulses.
invert_waveform = False
generator_high_z_voltage_factor = 2.0  # 50-ohm-calibrated generator driving the high-Z X-20 input
# The output-monitor waveform is already inverted relative to the generator,
# just like the sample waveform. Use +40 here to preserve its measured sign.
monitor_to_sample_voltage_gain = 40.0
# Safety limit for the PEAK voltage actually applied to the sample.
max_allowed_amplitude_v = 100

function_generator_address = "USB0::0x0957::0x2707::MY57301577::INSTR"
oscope_address = "USB0::0x0957::0x9005::MY50140103::INSTR"
function_generator_timeout_ms = 30000
SAVE_ROOT = r"D:\Ferroelectric_BTO_optimization"
SAVE_PATH = os.path.join(SAVE_ROOT, datetime.datetime.now().strftime("%Y%m%d") + "\\PUND_Autumn_2D_A4")


def segment(value, duration_s):
    if duration_s < 0:
        raise ValueError("Pulse widths and gaps cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(1, int(round(duration_s * arb_sample_rate_hz)))
    return np.full(point_count, value, dtype=float)


def ramp(start_value, stop_value, duration_s):
    if duration_s < 0:
        raise ValueError("Rise time cannot be negative")
    if duration_s == 0:
        return np.empty(0, dtype=float)
    point_count = max(2, int(round(duration_s * arb_sample_rate_hz)))
    return np.linspace(start_value, stop_value, point_count, endpoint=False)


def pulse(polarity, pulse_width_s, rise_time_s):
    return np.concatenate((
        ramp(0.0, polarity, rise_time_s),
        segment(polarity, pulse_width_s),
        ramp(polarity, 0.0, rise_time_s),
    ))


def build_pund_waveform(pulse_width_s, rise_time_s, gap_reset_to_positive_s, gap_positive_to_positive_s):
    if pulse_width_s <= 0:
        raise ValueError("pulse_width_s must be greater than zero")
    waveform = np.concatenate((
        segment(0.0, baseline_before_s),
        pulse(-1.0, 2 * pulse_width_s, rise_time_s),
        segment(0.0, gap_reset_to_positive_s),
        pulse(1.0, pulse_width_s, rise_time_s),
        segment(0.0, gap_positive_to_positive_s),
        pulse(1.0, pulse_width_s, rise_time_s),
        segment(0.0, baseline_after_s),
    ))
    if len(waveform) > maximum_arb_points:
        raise ValueError(f"Waveform has {len(waveform)} points; limit is {maximum_arb_points}")
    total_duration_s = len(waveform) / arb_sample_rate_hz
    return waveform, total_duration_s


def setup_function_generator(waveform, total_duration_s, sample_amplitude_v):
    sample_amplitude_v = float(sample_amplitude_v)
    if sample_amplitude_v <= 0 or sample_amplitude_v > max_allowed_amplitude_v:
        raise ValueError("Pulse amplitude is outside the allowed range")
    # The generator amplitude is specified for a matched 50-ohm load. The X-20
    # input is high impedance, so its input sees twice the generator setting.
    generator_amplitude_v = sample_amplitude_v / (
        generator_high_z_voltage_factor * voltage_amplifier_gain
    )
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    fg.timeout = function_generator_timeout_ms
    fg.write_termination = "\n"
    fg.read_termination = "\n"
    try:
        fg.write("*RST")
        fg.write("*CLS")
        fg.write("DATA VOLATILE, " + ",".join(map(str, waveform)))
        fg.write("FUNC:USER VOLATILE")
        fg.write("FUNC USER")
        fg.write(f"FREQ {1.0 / total_duration_s}")
        fg.write(f"VOLT {2.0 * generator_amplitude_v}")
        fg.write("VOLT:OFFS 0")
        fg.write("OUTP ON")
    finally:
        fg.close()
        rm.close()

    print(
        f"Uploaded PUND waveform: generator 50-ohm setting {generator_amplitude_v:g} V peak "
        f"({2.0 * generator_amplitude_v:g} Vpp), for a requested sample voltage of "
        f"{sample_amplitude_v:g} V peak."
    )


def turn_off_function_generator():
    rm = pyvisa.ResourceManager()
    fg = rm.open_resource(function_generator_address)
    fg.timeout = function_generator_timeout_ms
    try:
        fg.write("OUTP OFF")
        print("Function generator output turned OFF.")
    finally:
        fg.close()
        rm.close()


def capture_waveform(scope, channel):
    scope.write(f":WAVEFORM:SOURCE CHANNEL{channel}")
    scope.write(":WAVEFORM:FORMAT ASCII")
    x_increment = float(scope.query(":WAVEFORM:XINCREMENT?"))
    if not np.isclose(x_increment, target_time_step_s, rtol=0.02, atol=1e-12):
        print(f"Warning: scope returned {x_increment:g} s/point, requested {target_time_step_s:g}")
    x_origin = float(scope.query(":WAVEFORM:XORIGIN?"))
    raw_data = scope.query(":WAVEFORM:DATA?").strip()
    values = np.array([float(item) for item in raw_data.split(",") if item.strip()])
    time_values = np.arange(len(values)) * x_increment + x_origin
    return time_values, values


def configure_scope(scope, total_duration_s, sample_amplitude_v):
    capture_duration_s = number_of_periods * total_duration_s
    requested_points = int(round(capture_duration_s / target_time_step_s)) + 1
    expected_monitor_peak_v = sample_amplitude_v / monitor_to_sample_voltage_gain
    scope.write("*RST")
    scope.write("*CLS")
    scope.write(":CHANNEL1:DISPLAY ON")
    scope.write(":CHANNEL2:DISPLAY ON")
    scope.write(":CHANNEL3:DISPLAY ON")
    # The X-20 monitor conversion factor of 40 is specified for a 50-ohm load.
    scope.write(":CHANNEL2:INPUT DC50")
    scope.write(f":TIMEBASE:SCALE {capture_duration_s / 10.0}")
    scope.write(":TIMEBASE:POSITION 0")
    scope.write(":CHANNEL1:SCALE 0.005")
    scope.write(f":CHANNEL2:SCALE {max(expected_monitor_peak_v / 3.0, 0.02)}")
    scope.write(":CHANNEL3:SCALE 1")
    scope.write(":CHANNEL1:OFFSET 0")
    scope.write(":CHANNEL2:OFFSET 0")
    scope.write(":CHANNEL3:OFFSET 0")
    scope.write(":WAVEFORM:FORMAT ASCII")
    scope.write(":ACQUIRE:SRATE:AUTO OFF")
    scope.write(f":ACQUIRE:SRATE {1.0 / target_time_step_s}")
    scope.write(f":ACQUIRE:POINTS {requested_points}")
    scope.write(f":WAVEFORM:POINTS {requested_points}")
    print(f"Requesting {requested_points} points at {target_time_step_s:g} s/point")
    scope.write(":ACQUIRE:MODE HRESOLUTION")
    scope.write(":ACQuire:AVERage ON")
    scope.write(":ACQuire:AVERage:COUNt 256")
    scope.write(":TRIGGER:MODE EDGE")
    scope.write(":TRIGGER:EDGE:SOURCE CHANNEL3")
    scope.write(":TRIGger:LEVel CHANnel3,0.2")
    scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
    scope.write(":TRIGGER:HYSTERESIS NREJECT")
    scope.write(":TRIGGER:SWEEP TRIGGERED")


def format_value(value):
    return f"{value:.6g}"


def sample_polarity_tag():
    # Polarity sequence actually applied at the sample (after the X-20 inversion).
    return "negreset_pos" if invert_waveform else "posreset_neg"


def save_data(pulse_width_s, rise_time_s, gap_reset_s, gap_positive_s, sample_amplitude_v, time_values, current_values, voltage_values):
    os.makedirs(SAVE_PATH, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{device_name}_PUND_periods{number_of_periods}_pw{format_value(pulse_width_s)}s_"
        f"rise{format_value(rise_time_s)}s_"
        f"pre{format_value(baseline_before_s)}s_post{format_value(baseline_after_s)}s_"
        f"gap1_{format_value(gap_reset_s)}s_gap2_{format_value(gap_positive_s)}s_"
        f"sample_amp{format_value(sample_amplitude_v)}Vpeak_{sample_polarity_tag()}_area{format_value(electrode_area_m2)}m2_"
        f"gain{format_value(current_preamplifier_gain_a_per_v)}AperV_{timestamp}.csv"
    )
    csv_path = os.path.join(SAVE_PATH, filename)
    with open(long_path(csv_path), mode="w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time (s)", "Current (A)", "Voltage (V)"])
        writer.writerows(zip(time_values, current_values, voltage_values))
    print(f"Data saved to {csv_path}")


def main():
    if electrode_area_m2 <= 0 or arb_sample_rate_hz <= 0 or target_time_step_s <= 0:
        raise ValueError("Electrode area, ARB sample rate, and target time step must be positive")
    if (
        voltage_amplifier_gain <= 0
        or generator_high_z_voltage_factor <= 0
        or monitor_to_sample_voltage_gain <= 0
    ):
        raise ValueError("Voltage conversion factors must be greater than zero")
    scan = itertools.product(
        pulse_widths_s,
        rise_times_s,
        gap_reset_to_positive_s,
        gap_positive_to_positive_s,
        amplitudes_v,
    )
    for pulse_width_s, rise_time_s, gap_reset_s, gap_positive_s, sample_amplitude_v in scan:
        waveform, total_duration_s = build_pund_waveform(
            pulse_width_s, rise_time_s, gap_reset_s, gap_positive_s
        )
        if invert_waveform:
            # Pre-invert so the inverting X-20 delivers neg reset + two positive pulses.
            waveform = -waveform
        print(
            f"Running PUND ({sample_polarity_tag()} at sample) width={pulse_width_s:g}, rise={rise_time_s:g}, "
            f"reset-to-positive gap={gap_reset_s:g}, "
            f"positive-to-positive gap={gap_positive_s:g}, "
            f"requested sample amplitude={sample_amplitude_v:g} V peak"
        )
        setup_function_generator(waveform, total_duration_s, sample_amplitude_v)
        time.sleep(2)
        rm = pyvisa.ResourceManager()
        scope = rm.open_resource(oscope_address)
        scope.write_termination = "\n"
        scope.read_termination = "\n"
        scope.timeout = 30000
        try:
            configure_scope(scope, total_duration_s, sample_amplitude_v)
            scope.query(":DIGITIZE CHANNEL1,CHANNEL2;*OPC?")
            time_ch1, current_raw = capture_waveform(scope, 1)
            time_ch2, voltage_raw = capture_waveform(scope, 2)
            if len(time_ch1) != len(time_ch2):
                raise RuntimeError("Current and voltage waveform lengths do not match")
            if use_current_amp:
                current_values = 2 * current_raw * current_preamplifier_gain_a_per_v
            else:
                current_values = current_raw * current_preamplifier_gain_a_per_v

            # Channel 2 is the X-20 output monitor terminated into 50 ohms.
            # It already contains the amplifier inversion, so multiplying by a
            # positive 40 preserves the polarity actually present at the sample.
            voltage_values = voltage_raw * monitor_to_sample_voltage_gain
            save_data(
                pulse_width_s, rise_time_s, gap_reset_s, gap_positive_s, sample_amplitude_v,
                time_ch1, current_values, voltage_values,
            )
        finally:
            scope.write(":RUN")
            scope.close()
            rm.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        turn_off_function_generator()
