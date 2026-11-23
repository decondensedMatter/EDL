"""Read-only feasibility checks for the intentionally unchanged legacy readout."""
import math


def check(case, number_of_periods=3, time_step_s=1e-8, timeout_s=100., record_limit_points=None):
    if not case.get("ready"):
        return False, case.get("reason", "Waveform is unavailable")
    period = float(case["total_duration_s"])
    if not math.isfinite(period) or period <= 0:
        return False, "Invalid waveform period"
    marker = period/2 if case["waveform_kind"] == "native" else period
    record = number_of_periods*period
    minimum_acquisition = 255*marker + record
    if minimum_acquisition >= timeout_s:
        return False, (f"Original 256-average readout needs at least {minimum_acquisition:.3g} s "
                       f"but its timeout is {timeout_s:g} s; use the updated-readout version")
    points = round(record/time_step_s)+1
    if record_limit_points is not None and points > record_limit_points:
        return False, f"Original record requests {points:,} points, above the supplied installed-memory limit"
    message = (f"Legacy record: {points:,} points, {record:g} s, 256 averages. "
               "Actual acquisition can take longer; phase repeatability and installed depth require verification.")
    if record < 10e-6:
        return False, "Original 10 us minimum time window cannot preserve this requested short record"
    return True, message
