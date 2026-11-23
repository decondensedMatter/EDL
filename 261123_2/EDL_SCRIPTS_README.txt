EDL WAVEFORM SCRIPTS — 2026-11-23

UNATTENDED INDIVIDUAL RUNS — LATEST ADDITION

For one sample with no current amplifier and updated readout, download its
self-contained Python file from standalone_scripts and run it with no arguments.
It performs connected-sample checks, sweeps and eligible optimized charging with
no keyboard input and no specimen change. Hardware scaling/response is explicitly
nominal, as requested; external-standard calibration is not performed. Each of
the six default runs has a 20-minute active budget. See INDIVIDUAL_SCAN.txt for
the exact command, software setup, output locations and model-input limitations.

The shared sample_scripts/*_without_current_amp_updated_readout.py entries also
start this individual workflow with no arguments. Existing explicit subcommands
and the full six-specimen external-standards campaign remain available below.

WHAT IS INCLUDED

sample_scripts contains 24 files: six samples x two current paths x two readout
versions. Each file selects that specimen's whole test set; separate Python
files are not generated for thousands of individual voltage/time combinations.
Keep the shared modules beside the sample_scripts folder when moving this package.

  *_waveform_only.py
    Based on Agilent_infiniium_BTO_regular_waveform_scan_separate_triggered.py.
    The scope configuration, capture function, trigger/acquisition block, current
    conversion formula, original resource addresses and shutdown function are
    retained verbatim. Changes are waveform selection/output, date/sample/test
    file naming, and the requested current-path settings (amp switch; 1/50 A/V
    without the amplifier). The original acquisition uses 256 averages and a
    100-second timeout. A pre-output waveform check reports combinations that
    cannot finish under these unchanged settings; it does not silently change
    the readout. The original BTO electrode_area_m2 setting is deliberately
    retained in these files and is NOT used by the EDL prediction model.

  *_updated_readout.py
    Uses the same sample selection, waveform constraints and instrument identities,
    with the separately authorized readout changes: single-shot acquisition,
    explicit arm/trigger/completion order, phase-specific records, measured rate
    and depth checks, chunked transfers, and a timeout based on record duration.
    Data are saved by date, sample, test, and timestamp. See ENVIRONMENT.txt.

The two original files named in the request have not been overwritten. The PUND
file's X-20 VOLTAGE amplifier and monitor scaling are different from the optional
CURRENT amplifier. The X-20 wiring is not silently imported into the new scans.
No reference electrode or temperature-monitor channel is used.

VOLTAGE-FIRST RUN ORDER

Within each specimen, voltage is the outer loop. Levels are ordered by absolute
cell excursion, using Vpp/2 for bipolar chirps and abs(final voltage) for charging.
For example, the 0.1 V level includes the 0.2 Vpp sweeps and both +0.1 and -0.1 V
charging tests. The 0.05 V level contains the initial 0.1 Vpp sweeps only.
At each level the script collects cumulative calibration, fits and independently
validates both signs where charging is requested, executes the selected charging
times and all three variants for both signs, and completes every sweep shape/view
before advancing. Baseline/standard calibration comes first; no future high-voltage
specimen calibration is performed ahead of lower-voltage tests.

The user-selected voltage_stage_peak_policy is "sample": charging is ordered by
final voltage, and optimized overshoot may reach the overall sample limit
(0.8 V NaClO4, 1 V LLZO, 2 V LaF3). Calibration and sweeps remain level-bounded.
Thus final voltages increase in order, but optimized physical peaks need not.
The optional "stage" policy also limits overshoot to the current level and may
reject additional rapid charging cases. Rejected controls are recorded and never
applied; no feasible optimal waveform is promised for every target.
Stage-specific calibration/fit files remain immutable in the output. An unfinished
selected group, timeout or hardware qualification failure stops progression to
higher levels for that specimen; records already saved remain available.

VOLTAGE CONVENTION (TWO ELECTRODES)

Vpp always means maximum minus minimum CELL voltage. It is not the voltage of
one electrode versus Ag/AgCl. Latest explicit user settings, 2026-11-23:

Sample                 absolute charging peak / V   maximum bipolar Vpp / V
All four NaClO4        0.8                          1.6 (-0.8 to +0.8 V)
Ta-doped LLZO          1.0                          2.0 (-1.0 to +1.0 V)
LaF3                  2.0                          4.0 (-2.0 to +2.0 V)

NaClO4 sweep Vpp: 0.1, 0.2, 0.4, 0.8, 1.6 V.
NaClO4 optimized final voltages: +/-0.1, +/-0.2, +/-0.4, +/-0.8 V.
The optimization starts at zero; all charging overshoot must also remain within
the absolute 0.8 V cap. No additional NaClO4 margin or current compliance cutoff
is applied. Noise/model/ADC qualification thresholds are not current compliance.
LLZO sweep Vpp: 0.1, 0.2, 0.4, 0.8, 1.6, 2.0 V.
LaF3 sweep Vpp: 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.0 V.
LLZO optimized final voltages: +/-0.1, +/-0.2, +/-0.4, +/-0.8, +/-1.0 V, with a 1.0 V
overshoot cap. This is an effective 0.2 V margin below the approximate +1.2 V
onset. LaF3 final targets are +/-0.1, +/-0.2, +/-0.4, +/-0.8, +/-1.6, +/-2.0 V.
Its absolute peak cap is 2.0 V. Both signs start from zero and return to zero.
The saved two-electrode LLZO study reports onsets near -1.2 and +1.2 V:
approximately 2.4 V is their full difference, not the positive onset from zero.
For either two- or three-electrode measurements, a full operational window spans
both onsets. Zero to one onset is a one-sided range; the electrode count changes
the voltage reference, not the definition of interval width.

These are operating choices, not confirmed decomposition-free bounds. A 1.6 V
single-electrode stability width can correspond to 3.2 V bipolar cell Vpp only
under ideal equal voltage sharing with the initial electrode potentials centered
in that window. Referenced CV alone does not establish these conditions. The
profiles explicitly store cell limits rather than automatically doubling CV
widths. With no reference electrode, individual interface potentials are not
observed. Zero cell voltage is assumed to be OCV only after the symmetry check.

SWEEPS

Continuous logarithmic fast -> 10 Hz -> fast, twice per waveform. Sine, bipolar
square, triangle. Vpp doubles from 0.1 V, with the exact maximum appended.
Native mode: sine/square 2 MHz maximum, triangle 20 kHz maximum.
Stored ARB: 2 MHz maximum for all three, 160 MSa/s, with capacity checked first.
These retain the requested one-tenth rate margin. The margin helps resolution
but does not itself establish the response of the cell or measurement chain.

Default sweep leg is 1 second, four seconds total. At 160 MSa/s this requires
640 million stored ARB points, exceeding both the 1M standard and 16M option.
It is explicitly unavailable at that duration; no silent frequency or duration
reduction occurs. The continuous native sweep avoids generator storage, but a
full 4-second record at the retained 100 MSa/s needs about 401 million scope
points per channel. Installed scope depth must support it. ASCII transfer can
dominate elapsed time and storage. Use the preview command before any run.

The legacy version repeatedly plays a complete charging protocol because its
readout averages 256 acquisitions. It cannot be interpreted as a single-shot
relaxation/energy measurement. Native Sync is one down/up sweep; stored ARB Sync
is one complete stored record. Native carrier-phase repeatability and continuous
sequence wrap/marker behavior still require bench verification.

OPTIMIZED CHARGING

NaClO4 uses the prior TF model: two kinetic adsorption populations with Frumkin
interactions, not a generic transfer function. In the historical 0.1 M comparison
it was the broad-waveform/BIC choice, though another model performed better on
one AC metric and NTF was close in BIC. Its parameters are starting guesses only
for new specimens, especially 1 M and 9 M. Each concentration must be fitted and
pass independent waveform validation. Relative permittivity is fitted, not treated
as a measured constant. Historical bounds and seeds are tagged unverified.
Each polarity is fitted independently; negative controls evaluate the physical
model at negative cell voltages rather than mirroring positive predictions.
Mixed-polarity diagnostic records are retained but not reused across fit splits.

LaF3 and LLZO use a closed, conservative one-dimensional Stern-PNP model with
blocking interfaces, a mobile positive carrier plus fixed background, and an
optional finite-site exclusion term. Carrier density, diffusivity, permittivity,
Stern capacitance and unknown geometry must be supplied/fitted. This effective
model does not automatically describe electronic leakage, grain boundaries or
irreversible chemistry. The archived scalar LaF3 approximation is not mislabeled
as a full spatial PNP solver.

Charge times: 1, 0.5, 0.25, ... seconds and the exact minimum endpoint. Nominal
minima: 168.75 ns without the current amplifier; 8.75 us with the 400 kHz amplifier.
These are ten times the estimated limiting 10-90% rise, rounded to DAC ticks.
Measured slower responses increase the lower bound. ARB STEP filtering is used;
the pulse/PRBS 8.4 ns specification is not used as the arbitrary-waveform rise.

Final magnitudes: 0.1, 0.2, 0.4, ... V with the exact cap appended, at BOTH signs. The
optimizer minimizes signed electrical work delivered at the CELL terminals over
the charging interval, integral Vcell Icell dt. Later discharge energy is not
subtracted. Positive delivered and returned energy are also reported. The target
is the equilibrium electrode charge at the requested final voltage plus a final
terminal-voltage condition; full internal equilibrium is not forced at that time.
Residual internal relaxation is therefore observable during the hold.

Overshoot is allowed when useful; it is not artificially imposed. All commanded
and modeled voltages remain within the cap. Fast high-voltage targets may be
infeasible, especially at the cap with no overshoot headroom; those are recorded
as rejected rather than output. A finite-basis, multistart local optimizer does
not certify a global optimum. Numerical replays tighten solver/time integration,
refine the PNP mesh, and validate the uploaded DAC codes at both requested and
actual quantized endpoints. Actual physical output still depends on measured
hardware response and is checked in captured windows, not guaranteed by software.

Three variants: charge/observe-hold, charge/long-hold/discharge, and charge/immediate
discharge. Every variant returns to zero and includes a reset. Holds/reset are
constrained by measured settling and full-state replay;
full-state replay verifies the return. Too-short holds fail validation rather
than being called sufficient. The observation hold can be longer. Discharge ramp
duration is matched across variants and respects the selected ARB sample rate.

COMBINED 120-MINUTE CAMPAIGNS: ALL 24 ENTRY FILES

Each of the 24 sample entries accepts --campaign-six (alias --session) and binds
its own current path and readout mode. These are four alternatives of six samples,
not 24 sequential runs sharing one two-hour budget. One complete six-sample run
has a 7200-second active budget including calibration, connections to standards,
model fitting, optimization, uploads, capture, holds/resets and saving. Only the
explicit specimen-change prompt pauses the clock. No reference electrode or
monitored temperature channel is added.

Convenient updated-readout entry points:
  python run_all_samples_without_amp.py check
  python run_all_samples_without_amp.py run --out data
  python run_all_samples_with_amp.py check
  python run_all_samples_with_amp.py run --out data

The matching configuration filenames are:
  all_samples_120min_without_amp_updated.json
  all_samples_120min_with_amp_updated.json
  all_samples_120min_without_amp_legacy.json
  all_samples_120min_with_amp_legacy.json
The older all_samples_120min_without_amp.json is a compatibility template for the
updated direct-current path. Complete actual bench inputs before running; check
is offline. Generate a fresh independent template with init-session --out FILE.

Example routes through an individual entry (the campaign still covers all six):
  python sample_scripts/NaClO4_0p1M_without_current_amp_updated_readout.py --campaign-six check
  python sample_scripts/LLZO_with_current_amp_waveform_only.py --campaign-six check

The twelve unchanged-readout entries retain the original readout functions and
acquisition block verbatim. Their 256 averages and 100-second timeout cannot
execute the requested integrated two-hour protocol: a default native chirp needs
at least 512 seconds of Sync intervals for one averaged acquisition, and passive
OCV/zero records do not provide the original CH3 trigger. Their integrated run
therefore reports incompatibility BEFORE opening instruments. Bare invocation
uses this guarded route. Only explicit --legacy-manual selects the historical
standalone acquisition, which is OUTSIDE the 120-minute campaign contract. The
updated-readout alternatives support the integrated acquisition. No setting is
silently changed to disguise a legacy campaign as an updated-readout run.

CALIBRATION INCLUDED FOR LATER ANALYSIS

The script acquires independent DC references, current gain/sign/offset and
response using known resistors and series-RC standards, a buffer-loading test,
open-fixture leakage/capacitance, baseline noise and ADC clipping evidence. Direct
and amplified readouts are calibrated separately; amplifier bandwidth is capped
at 400 kHz. The 0.25 V reference-fixture excitation is not a sample stimulus.
Standards must be connected when prompted; no switching hardware is assumed.

Each specimen is measured at zero, then calibrated incrementally at ascending
levels. The first level includes two signed training steps, a mixed bipolar
diagnostic, zero/midpoint small signals and independent held-out step/ramp trials.
Each later level adds two signed training steps and one new independent holdout
per sign; the final level also adds near-cap biased tones. New holdouts reach
between the previous and current level, testing the newly entered range.
All earlier training/validation roles and records are retained in cumulative fits.
At one small-signal frequency there are 192 specimen captures if every level is
reached, including OCV/driven-zero baselines, plus standards and any retries.
Calibration holds/resets increase with magnitude and extend if settling fails.

The script supplies analysis acceptance defaults and documented numerical fit
initializations. NaClO4 priors use the historical TF fit, transferred as starting
values only. Stern-PNP uses explicitly exploratory numerical seeds, never fake
material measurements. Fit identifiability and independent validation must pass
at EACH polarity. Actual reference values/identities, resistor and capacitor
values, verified scope rails/ranges, and missing specimen geometry remain physical
inputs: LLZO thickness and overlap area, and LaF3 overlap area. The DC references
must bracket -2 to +2 V and CH2 must cover that range (master default 1 V/div).

A missing geometry or failed fit blocks optimized output for that specimen but
does not discard its calibrated raw traces or otherwise qualified chirps. This is
reported as partial work, not a successful optimized experiment. Current amplitude
is not limited by a CV current-density cutoff; clipping/fit checks are data-quality
criteria. Open-fixture current is kept separate from modeled cell current.

Each output directory includes session_configuration.json, session_journal.json,
analysis_manifest.json, standards_inventory.json and per-specimen calibration
inventories. Records retain raw channels, corrected voltage/current, timing,
calibration identity, sign, split, hashes and immutable polarity-specific fit
configs. Every case/capture links to the exact config used for its validation.
Keep the complete output directory for later analysis.

HOLD/RESET ALLOCATION AND COVERAGE

At the start of a specimen scan the scheduler uses its remaining active share,
measured settling, anticipated setup/transfer/CPU time and required chirps to
select one scale s between 0.1 and 1. It locks that voltage-dependent curve for
the specimen. With x=abs(V)/Vcap, before any additional replay-required extension:
  hold = max(0.1 s, measured hold floor) + s*0.9 s*x^2
  reset = max(0.2 s, measured reset floor) + s*1.8 s*x^2
Hold uses the final magnitude; reset uses the actual excursion including overshoot.
Both signs use the more conservative measured floor at each magnitude. The floor
is a monotone envelope so a higher voltage never receives a smaller planned dwell.
Full-state replay can extend a case and propagates its requirement conservatively
to later higher-voltage cases. Measured and model-required floors remain separate
in the saved evidence. Immediate-discharge tests have zero hold and a finite reset.

The initial matrix includes all final magnitudes and both signs at up to six
logarithmic charge times. To fit the budget, the scheduler removes intermediate
charge durations uniformly across signs and voltages, retaining both qualified
time endpoints where feasible. It checks both polarity optimizations and reserves
all eight phase-specific acquisitions before the first output in a matched pair.
The four views per sign are charge, post-charge hold, held-discharge tail and
immediate-discharge tail. Every view repeats the whole waveform from zero.

Measured CPU/acquisition time updates later coverage planning. Once the selected
durations of a level have begun, an inability to finish them stops higher levels.
Physical reset
requirements are never reduced to make the run appear complete. Sweep views
cover fast/mid/slow windows from separate full two-round-trip native waveforms;
all specified sine/square/triangle Vpp levels stay in the plan. A partial sweep
record cannot establish whole-sweep measured energy. Standalone stored ARB remains
available only where generator memory permits it.

A slower independently measured sample response raises the minimum usable charge
time. Millisecond sample tests do not establish nanosecond response. Invalid or
uncompleted cases and unmatched/interrupted groups remain explicit in the journal.

The full signed matrix at a 20 ms time floor would need about 148.5 minutes for
scan acquisition alone, before calibration and CPU. Progressive calibration and
refitting add overhead. The conservative adaptive example uses about 119.6 active
minutes and leaves some highest-voltage tests unfinished. It is a time-bounded
scenario, not measured throughput or a guarantee of complete coverage.
See RUNTIME_ESTIMATES.txt and HOLD_RESET_PLAN.txt for numbers and assumptions.

GETTING STARTED WITH SEPARATE SAMPLE FILES (Python 3.10+)

All commands below are examples for 0.1 M with amplifier; select the corresponding
sample file for another sample or current path. Offline commands do not open VISA.

python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py --help
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py plan --out plan.json
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py init-config --out cell.json
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py chirp --preview --mode native
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py chirp --preview --mode arb

The sample_configs directory already contains editable configurations for all
twelve specimen/current-path combinations. Do not mark calibration flags true
without the corresponding evidence. Unknown values are null, not invented fits.

Calibration order and connections are in the generated plan. Acquire one physical
connection at a time: disconnected/buffered OCV and zero/short baselines; calibrated
voltage path; known resistor for gain, sign, offset and delay; open fixture;
known RC standards for bandwidth and response; specimen steps, small signals at
multiple biases, and separate negative-step/ramp holdout records. Enter actual
standard R/C values. Sample nonzero tests require hardware-path qualification.

Example known-resistor acquisition (requires that actual resistor connection):
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py acquire-calibration --config cell.json --recipe gain_polarity_offset_delay --connection known_resistor --standard-resistance-ohm 1000 --edge-s 0.001 --hold-s 0.01 --out data

The number 1000 above is an EXAMPLE, not a measurement of a connected resistor.
Use the real standard and appropriate recording times/scales. Calibration helpers
fit_resistor, step_response, clipping_diagnostic and equilibrium_tail are in
edl_charge/calibration.py. A steady current is not proof of full-state equilibrium.
Do not use direct 50-ohm scope termination to claim unloaded OCV; a verified buffer
is required for that passive recipe. Passive baseline recipes cannot use the
legacy CH3-triggered averaging readout without changing that readout.

Fit manifest example (paths relative to manifest; noise below is illustrative):
{
  "calibration_id": "same-session-id-as-cell-config",
  "traces": [
    {"path": "train.csv", "split": "train", "noise_a": 1e-8, "reset_verified": true},
    {"path": "heldout.csv", "split": "validation", "noise_a": 1e-8, "reset_verified": true}
  ]
}
Use independently measured noise; preserve the recorded time axis. Current must
be gain/offset corrected, but not pre-delay corrected: the fitter applies the
qualified channel response and delay to the predicted current. Supply an explicit
initial model state if the initial zero equilibrium was not verified. Bounds and
qualification thresholds must be justified from calibration/independent data.
Rank, conditioning and held-out errors are reported; unsupported fits are rejected.

python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py fit --config cell.json --manifest traces.json --out fitted.json
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py optimize --config fitted.json --out cases
python sample_scripts/NaClO4_0p1M_with_current_amp_updated_readout.py run-campaign --config fitted.json --cases cases --views charge hold reset --out data

Each view repeats the WHOLE protocol from a reset; do not join separate shots into
one measured energy integral. Charge resolves the drive, hold observes equilibration,
and reset observes the current decay after discharge. Optional discharge view
resolves the finite falling edge. Completed captures and failures are indexed;
readout failures or observed bounds violations stop a campaign.

One case can be run/analyzed with run and analyze; use command --help for required
paths. The model is driven by measured cell voltage in the recorded window. A late
window starts from a modeled replay of the earlier full command. Decay fits are
descriptive; different fitted time constants do not alone identify a mechanism.

prepare_edl_waveforms.py offers offline-only plan/init-config/fit/optimize/demo.
The historical demo is explicitly offline and cannot be sent to hardware. For
the legacy variant set optimized_directory to its exported cases and supply the
matching fitted_config.json (or calibration_config). Configure selected_tests
and calibration_requests at the top of that file. Its unchanged readout limitations
remain in force even for otherwise valid waveforms.

RUNTIME AND VERIFICATION

See RUNTIME_ESTIMATES.txt and estimate_edl_runtime.py. Estimates are conditional
on measured hold/reset, accepted cases, capture views and actual transport speed.
The test suite uses numerical models and mocked VISA; no hardware was operated.
test_edl_script_versions proves the legacy readout functions and acquisition block
remain text-identical to the original source. The full suite also checks physical
model invariants, optimization, DAC bounds, calibration/validation separation,
sequence compilation, readout completion, and saved-file interoperability.

SOURCES

Keysight arbitrary-waveform/rise-time specifications:
https://www.keysight.com/content/dam/keysight/en/doc/ungate/data-sheets/5992-2572.pdf
DSA90604A bandwidth and fixed 50-ohm inputs:
https://www.keysight.com/gb/en/assets/7018-01734/data-sheets/5989-7819.pdf
Programming references:
https://www.batronix.com/files/Keysight/Funktionsgeneratoren/33500B%2633600A/33500-33600-Manual.pdf
https://www.keysight.com/sn/en/assets/9018-06975/programming-guides/9018-06975.pdf
User's LLZO material source (PO0106 is powder, not a pellet-dimension specification):
https://www.msesupplies.com/products/ampcera-llzo-nano-powder-ta-doped-lithium-lanthanum-zirconate-garnet-500nm-2
The supplied initialization and voltage-window defaults draw on historical studies.
