"""Offline acquisition planning; no plan or nominal timing constitutes hardware calibration."""
import json
import math
import re
from decimal import Decimal
from pathlib import Path

DIR = Path(__file__).parent
VARIANTS = ("charge_hold_observe", "charge_hold_discharge", "charge_immediate_discharge")


def load_profiles():
    profiles = json.loads((DIR / "profiles.json").read_text(encoding="utf-8"))
    seed = json.loads((DIR / "historical_tf_seed.json").read_text(encoding="utf-8"))["parameters"]
    for profile in profiles.values():
        profile["initial_guess"] = {}
        if profile["model"] == "tf":
            candidate = dict(seed)
            candidate.update({k: v for k, v in profile["model_parameters"].items() if v is not None})
            if candidate["concentration_mol_m3"] == 9000:
                candidate["eps_r"] = None
            profile["initial_guess"] = candidate
    return profiles


def get_profile(sample_name):
    def normalized(name):
        return re.sub(r"[^a-z0-9.]", "", name.lower().replace("₄", "4"))
    key = normalized(sample_name)
    for name, profile in load_profiles().items():
        labels = [name]
        if name.startswith("NaClO4_"):
            molarity = name.split("_")[1]
            labels += [molarity, molarity + " NaClO4"]
        if key in {normalized(label) for label in labels}:
            return profile
    raise ValueError(f"Unknown sample: {sample_name}")


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def duration_grid(min_time_s, max_time_s=1.0):
    minimum = Decimal(str(_positive(min_time_s, "minimum time")))
    t = Decimal(str(_positive(max_time_s, "maximum time")))
    if minimum > t:
        raise ValueError("minimum time exceeds maximum time")
    times = []
    while t > minimum:
        times.append(float(t))
        t /= 2
    return times + [float(minimum)]


def final_voltage_grid(max_voltage):
    maximum = Decimal(str(_positive(max_voltage, "maximum voltage")))
    voltage = Decimal(".1")
    if maximum < voltage:
        raise ValueError("voltage cap below first target (0.1 V)")
    values = []
    while voltage < maximum:
        values.append(float(voltage))
        voltage *= 2
    return values + [float(maximum)]


def hardware_profile(use_current_amp, hardware_calibration=None):
    """Apply measured timing limits conservatively. Frequencies in Hz, times in s."""
    cal = dict(hardware_calibration or {})
    dac = 160e6
    scope_bw = 6e9
    bandwidth = min(scope_bw, 400e3 if use_current_amp else scope_bw)
    if "measured_bandwidth_hz" in cal:
        bandwidth = min(bandwidth, _positive(cal["measured_bandwidth_hz"], "measured bandwidth"))
    rise = max(10e-9, .35/(.13*dac))
    minimum = max(10*rise, 10*.35/bandwidth)
    if "min_charge_time_s" in cal:
        minimum = max(minimum, _positive(cal["min_charge_time_s"], "minimum charge time"))
    # Suppress ulp noise when an analog limit is already an exact DAC tick.
    ticks = math.ceil(math.nextafter(minimum*dac, -math.inf))
    return dict(use_current_amp=bool(use_current_amp), scope_model="DSA90604A",
                scope_serial="MY50140103", generator_model="33511B",
                generator_serial="MY57301577", scope_bandwidth_hz=scope_bw,
                scope_max_sample_rate_hz=20e9, generator_sample_rate_hz=dac,
                current_channel="CHAN1", current_coupling="DC", current_termination_ohm=50.,
                current_gain_a_per_v=cal.get("current_gain_a_per_v", .002 if use_current_amp else .02),
                current_gain_source="historical script, verify with resistor" if use_current_amp
                                    else "nominal 50 ohm termination, verify with resistor",
                voltage_scale_factor=cal.get("voltage_scale_factor", 1.),
                voltage_scale_source="illustrative unity until voltage-path calibration",
                data_hardware_calibrated=bool(cal.get("data_hardware_calibrated", False)),
                bandwidth_hz=bandwidth, generator_rise_time_s=rise, min_charge_time_s=ticks/dac)


def build_candidate_plan(sample_name, use_current_amp, hardware_calibration=None):
    profile = get_profile(sample_name)
    hw = hardware_profile(use_current_amp, hardware_calibration)
    bound = profile["peak_bound_v"]
    return [dict(sample_name=profile["sample_name"], material=profile["sample_name"],
                 model=profile["model"], use_current_amp=bool(use_current_amp),
                 initial_voltage_v=0., final_voltage_v=voltage, peak_bound_v=bound,
                 charge_time_s=duration, variant=variant, reset_voltage_v=0.,
                 reset_time_s=None, reset_policy="derive from fit and observed return to baseline",
                 calibration_status=profile["calibration_status"])
            for magnitude in final_voltage_grid(bound)
            for duration in duration_grid(hw["min_charge_time_s"]) for voltage in (magnitude,-magnitude) for variant in VARIANTS]


def calibration_protocols(profile, use_current_amp=False, hardware_calibration=None):
    """Declarative trace recipes; external standards require physical reconnection."""
    if isinstance(profile, str):
        profile = get_profile(profile)
    hw = hardware_profile(use_current_amp, hardware_calibration)
    cap = profile["peak_bound_v"]
    targets = final_voltage_grid(cap)
    amp = min(.01, cap/10)
    # The direct scope path can observe faster edges than the 20 MHz 33511B can excite.
    excitation_limit = min(20e6, hw["bandwidth_hz"])
    freqs = [min(1., excitation_limit)]
    while freqs[-1]*4 < excitation_limit:
        freqs.append(freqs[-1]*4)
    if freqs[-1] < excitation_limit:
        freqs.append(excitation_limit)
    stages = []
    def add(name, connection, split, **recipe):
        stages.append(dict(name=name, connection_required=connection, split=split, **recipe))
    add("initial_zero", "sample", "baseline", voltage_v=0.)
    add("open_circuit_voltage", "sample_open_circuit", "baseline", generator_output="disconnected")
    add("shorted_zero", "shorted_sense_input", "instrument", generator_output="off")
    add("gain_polarity_offset_delay", "known_resistor", "instrument", resistance_ohm=None,
        levels_v=[0., .01, -.01, 0.], record_standard_value=True)
    add("open_input_parasitics", "open_fixture", "instrument", levels_v=[0., .01, 0.])
    add("known_rc_bandwidth", "known_rc_standard", "instrument", resistance_ohm=None,
        capacitance_f=None, frequencies_hz=freqs, amplitude_v=amp)
    add("sample_positive_steps", "sample", "train", levels_v=targets, start_each_at_v=0.,
        hold_until="measured steady baseline or record unsettled timeout")
    add("sample_negative_steps", "sample", "train", levels_v=[-v for v in targets], start_each_at_v=0.,
        hold_until="measured steady baseline or record unsettled timeout")
    add("sample_bipolar_steps", "sample", "train", levels_v=[0., .1, -.1, 0.])
    add("sample_small_signal", "sample", "train", frequencies_hz=freqs,
        bias_voltages_v=[0., cap/2, cap-amp, -cap/2, -(cap-amp)], amplitude_v=amp)
    add("negative_step_holdout", "sample", "validation", levels_v=[0., -.63*cap, 0.])
    add("positive_step_holdout","sample","validation",levels_v=[0., .63*cap, 0.])
    add("negative_ramp_holdout","sample","validation",knots_v=[0., -cap*.75, -cap*.25, 0.])
    add("ramp_holdout", "sample", "validation", knots_v=[0., cap*.75, cap*.25, 0.])
    return dict(sample_name=profile["sample_name"], hardware=hw, stages=stages,
                initial_voltage_v=0., reset_voltage_v=0.,
                reset_criterion="sustained current return within measured baseline noise and model-state tolerance",
                save_raw_channels=True, separate_validation=True)

