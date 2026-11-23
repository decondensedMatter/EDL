"""Offline planning for continuous logarithmic sweeps. Voltages are generator Vpp."""

from dataclasses import dataclass
from decimal import Decimal
from math import exp, expm1, isfinite, log


MARGIN_V = 0.1
NACLO4_SAMPLES = ("0.01M NaClO4", "0.1M NaClO4", "1M NaClO4", "9M NaClO4")
WINDOW_WIDTHS_V = {
    **dict.fromkeys(NACLO4_SAMPLES, 1.6),
    "Ta_LLZO": 2.4,
    "LaF3": 4.2,
}
WINDOW_MARGINS_V = {
    **dict.fromkeys(NACLO4_SAMPLES, 0.0),
    "Ta_LLZO": MARGIN_V,
    "LaF3": MARGIN_V,
}
# Explicit bipolar sweep schedules. Positive charging uses its own peak bound.
CHIRP_VPP_LEVELS = {
    **dict.fromkeys(NACLO4_SAMPLES, (.1, .2, .4, .8, 1.6)),
    "Ta_LLZO": (.1, .2, .4, .8, 1.6, 2.0),
    "LaF3": (.1, .2, .4, .8, 1.6, 3.2, 4.0),
}
WINDOW_NOTES = {
    **dict.fromkeys(NACLO4_SAMPLES,
                    "User-specified two-electrode operating limit (2026-11-23): "
                    "frequency sweeps -0.8 to +0.8 V (1.6 Vpp), no additional margin. "
                    "Optimized charging, including overshoot, is limited to +0.8 V. Stability is unverified."),
    "Ta_LLZO": "Measured two-terminal range ±1.2 V (2.4 V full width). "
               "Use the full 2.4 V window width.",
    "LaF3": "User-supplied range −2.1 to +2.1 V (4.2 V full width). "
            "Use the full 4.2 V window width.",
}
WAVEFORM_LIMITS_HZ = {"sine": 20e6, "square": 20e6, "triangle": 200e3}
ARB_MAX_SAMPLE_RATE_HZ = 160e6
ARB_MIN_WAVEFORM_POINTS = 8
ARB_RECOMMENDED_SAMPLES_PER_FAST_CYCLE = 80


def _positive(value, name):
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def voltage_levels(window_width_v, margin_v=MARGIN_V):
    """Double from 0.1 Vpp, ending at full width minus a margin at BOTH boundaries."""
    if window_width_v is None:
        raise ValueError("An explicit full window width is required")
    _positive(window_width_v, "window_width_v")
    if not isfinite(margin_v) or margin_v < 0:
        raise ValueError("margin_v must be finite and nonnegative")
    # Preserve exact decimal endpoints before converting to generator units.
    target = float(Decimal(str(window_width_v)) - 2 * Decimal(str(margin_v)))
    _positive(target, "final Vpp")
    levels = []
    level = 0.1
    while level < target:
        levels.append(level)
        level *= 2
    return levels + [target]


def peak_voltage(voltage_vpp):
    """Peak excursion from the DC offset in volts."""
    return _positive(voltage_vpp, "voltage_vpp") / 2


@dataclass(frozen=True)
class SweepPlan:
    waveform: str
    f_min: float
    f_max: float
    leg_duration_s: float
    voltage_vpp: float
    cycles: int = 2
    mode: str = "native"

    def __post_init__(self):
        if self.waveform not in WAVEFORM_LIMITS_HZ:
            raise ValueError("waveform must be sine, square or triangle")
        for name in ("f_min", "f_max", "leg_duration_s", "voltage_vpp"):
            _positive(getattr(self, name), name)
        limit = _frequency_limit(self.waveform, self.mode)
        if not self.f_min < self.f_max <= limit:
            raise ValueError("frequency range exceeds waveform limits or is empty")
        if type(self.cycles) is not int or self.cycles < 1:
            raise ValueError("cycles must be a positive integer")
        _positive(self.total_duration_s, "total_duration_s")

    @property
    def total_duration_s(self):
        return 2 * self.leg_duration_s * self.cycles

    def _leg(self, t):
        if not isfinite(t) or not 0 <= t <= self.total_duration_s:
            raise ValueError("t must be within the planned sweep")
        if t == self.total_duration_s:
            return 2 * self.cycles, 0.0
        index, elapsed = divmod(t, self.leg_duration_s)
        return int(index), elapsed / self.leg_duration_s

    def frequency_at(self, t):
        index, fraction = self._leg(t)
        start, end = (self.f_min, self.f_max) if index % 2 else (self.f_max, self.f_min)
        if fraction == 0:
            return start
        return exp((1 - fraction) * log(start) + fraction * log(end))

    def phase_cycles_at(self, t):
        """Integrated carrier phase in cycles starting at zero."""
        index, fraction = self._leg(t)
        ratio_log = log(self.f_max) - log(self.f_min)
        leg_cycles = self.leg_duration_s * ((self.f_max - self.f_min) / ratio_log)
        # Express the ascending integral in terms of the instantaneous
        # frequency to keep the expm1 argument nonpositive in both legs.
        scale = self.frequency_at(t) if index % 2 else self.f_max
        partial = scale * (-expm1(-ratio_log * fraction) / ratio_log)
        return index * leg_cycles + partial * self.leg_duration_s


def _frequency_limit(waveform, mode):
    if mode == "native":
        return WAVEFORM_LIMITS_HZ[waveform]
    if mode == "arb":
        return ARB_MAX_SAMPLE_RATE_HZ / ARB_MIN_WAVEFORM_POINTS
    raise ValueError("mode must be native or arb")


def make_plan(waveform, voltage_vpp, leg_duration_s, f_min=10.0,
              max_fraction=0.1, cycles=2, mode="native", f_max=None):
    """Build a high-to-low-to-high plan using a fraction of the waveform limit."""
    if not isfinite(max_fraction) or not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be finite and in (0, 1]")
    if waveform not in WAVEFORM_LIMITS_HZ:
        raise ValueError("waveform must be sine, square or triangle")
    if f_max is None:
        f_max = _frequency_limit(waveform, mode) * max_fraction
    return SweepPlan(waveform, f_min, f_max, leg_duration_s, voltage_vpp, cycles, mode)
