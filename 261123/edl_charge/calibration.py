"""Offline resistor calibration, post-hold diagnostics and specimen parameter inference."""
from dataclasses import dataclass, field
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar


def _vector(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("expected a finite 1-D signal")
    return a


def _signals(times, *channels):
    t = _vector(times)
    y = tuple(_vector(ch) for ch in channels)
    if len(t) < 3 or np.any(np.diff(t) <= 0) or any(len(a) != len(t) for a in y):
        raise ValueError("signals need matching lengths and strictly increasing measured times")
    return (t,) + y


def fit_resistor(times_s, voltage_v, measured_current_a, resistance_ohm, *, max_delay_s):
    """Fit I_expected(t-delay) = gain*I_raw(t)+offset. Positive delay means raw current lags."""
    t, v, raw = _signals(times_s, voltage_v, measured_current_a)
    if not np.isfinite(resistance_ohm) or resistance_ohm <= 0 or not 0 <= max_delay_s < (t[-1]-t[0])/2:
        raise ValueError("supply positive resistance and a physical maximum delay")
    keep = (t >= t[0]+max_delay_s) & (t <= t[-1]-max_delay_s)
    if keep.sum() < 5 or np.ptp(raw[keep]) == 0 or np.ptp(v[keep]) == 0:
        raise ValueError("resistor calibration needs distinct current and voltage levels")
    design = np.column_stack((raw[keep] - raw[keep].mean(), np.ones(keep.sum())))
    def fit(delay):
        truth = np.interp(t[keep]-delay, t, v) / resistance_ohm
        beta = np.linalg.lstsq(design, truth, rcond=None)[0]
        return beta, float(np.mean((design @ beta - truth)**2))

    if max_delay_s:
        grid = np.linspace(-max_delay_s, max_delay_s, 65)
        idx = np.argmin([fit(d)[1] for d in grid])
        lo, hi = grid[max(0,idx-1)], grid[min(len(grid)-1,idx+1)]
        trial = minimize_scalar(lambda d: fit(d)[1], bounds=(lo,hi), method="bounded",
                                options={"xatol": max_delay_s*1e-8})
        delay = min((grid[idx], trial.x), key=lambda d: fit(d)[1])
    else:
        delay = 0.
    (gain, intercept), mse = fit(delay)
    return dict(gain=float(gain), polarity=int(np.sign(gain)),
                offset_a=float(intercept-gain*raw[keep].mean()), delay_s=float(delay),
                rms_error_a=np.sqrt(mse), n_samples=int(keep.sum()))


def step_response(times_s, signal, *, plateau_samples, minimum_rise_samples):
    """Single isolated step. Report empirical rise and first-order equivalent bandwidth."""
    t, y = _signals(times_s, signal)
    k = int(plateau_samples)
    if k < 2 or 2*k >= len(t) or minimum_rise_samples < 1:
        raise ValueError("supply measured plateau lengths and required rise samples")
    low, high = np.median(y[:k]), np.median(y[-k:])
    if high == low:
        raise ValueError("no net step between plateaus")
    f = (y-low)/(high-low)
    def cross(level):
        pairs = np.flatnonzero((f[:-1] < level) & (f[1:] >= level))
        if not len(pairs):
            raise ValueError("step crossing unavailable")
        i = pairs[0]
        return t[i]+(level-f[i])*(t[i+1]-t[i])/(f[i+1]-f[i]), i
    a, i = cross(.1)
    b, j = cross(.9)
    if b <= a:
        raise ValueError("invalid 10–90 rise")
    samples = (b-a)/max(np.diff(t)[i:j+1])
    return dict(rise_time_s=float(b-a), first_order_equivalent_bandwidth_hz=float(np.log(9)/(2*np.pi*(b-a))),
                samples_across_rise=float(samples), sampling_qualified=bool(samples>=minimum_rise_samples),
                start_plateau_sd=float(np.std(y[:k], ddof=1)), end_plateau_sd=float(np.std(y[-k:], ddof=1)))


def clipping_diagnostic(signal, *, adc_min=None, adc_max=None, rail_tolerance=0.):
    """Rail proximity requires trace-unit ADC metadata. A flat plateau alone is ambiguous."""
    y = _vector(signal)
    if adc_min is None or adc_max is None:
        return dict(status="unknown_missing_adc_range", samples_at_rail=None)
    if not np.isfinite([adc_min,adc_max,rail_tolerance]).all() or adc_min >= adc_max or not 0 <= 2*rail_tolerance < adc_max-adc_min:
        raise ValueError("invalid ADC limits or tolerance")
    n = np.count_nonzero((y <= adc_min+rail_tolerance) | (y >= adc_max-rail_tolerance))
    return dict(status="at_or_beyond_rail" if n else "within_rails", samples_at_rail=int(n))


def equilibrium_tail(times_s, current_a, *, baseline_a, noise_a, window_s, max_hold_s, tolerance_sigma):
    """Compare final-window level, scatter and linear drift with independent leakage/noise."""
    t,i = _signals(times_s,current_a)
    if not np.isfinite([baseline_a,noise_a,window_s,max_hold_s,tolerance_sigma]).all() or min(noise_a,window_s,tolerance_sigma)<=0 or max_hold_s<window_s:
        raise ValueError("require baseline, positive noise, window, max hold and tolerance")
    duration = t[-1]-t[0]
    if duration < window_s:
        return dict(settled=False, status="insufficient_hold")
    selected = t >= t[-1]-window_s
    if np.count_nonzero(selected)<3:
        return dict(settled=False, status="insufficient_samples")
    u, z = t[selected], i[selected]
    slope = np.polyfit(u-u.mean(),z,1)[0]
    mean, sd = float(z.mean()), float(z.std(ddof=1))
    lim = tolerance_sigma*noise_a
    settled = max(abs(mean-baseline_a),sd,abs(slope)*window_s) <= lim
    return dict(settled=bool(settled), status=("settled" if settled else
                "max_hold_reached" if duration>=max_hold_s else "continue_hold"),
                mean_a=mean, sd_a=sd, slope_a_s=float(slope))


@dataclass
class CalibrationTrace:
    times_s: object
    voltage_v: object
    current_a: object
    split: str = "train"
    initial_state: object = None
    reset_verified: bool = False
    noise_a: object = None
    name: str = ""
    metadata: dict = field(default_factory=dict)


def read_calibration_csv(path, metadata, *, initial_state=None):
    """Load gain- and offset-corrected current on its recorded time axis.

    CSV header names are Time(s), Current(A), Voltage(V); metadata declares split,
    noise_a and reset_verified. Use a separate pre-trigger window when correcting delay.
    """
    src = Path(path)
    with src.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        keys = ("Time(s)","Voltage(V)","Current(A)")
        columns = reader.fieldnames or ()
        normalized = ["".join(name.split()) for name in columns]
        if any(normalized.count(key)!=1 for key in keys):
            raise ValueError("missing Time(s), Current(A) or Voltage(V) column")
        original_keys = [columns[normalized.index(key)] for key in keys]
        rows = [[float(row[k]) for k in original_keys] for row in reader]
    if not rows:
        raise ValueError("empty calibration file")
    t,v,i = _signals(*np.array(rows).T)
    if metadata.get("split") not in ("train","validation"):
        raise ValueError("assign the record to train or validation before fitting")
    return CalibrationTrace(t,v,i,metadata["split"],initial_state,
                            metadata.get("reset_verified") is True,metadata.get("noise_a"),
                            str(src),dict(metadata))


def replay_current(model,times_s,voltage_v,initial_state,*,rtol=2e-7,atol=1e-10,driver_indices=None):
    """Replay measured PWL voltage, retaining each sample. At corners use right-hand slew."""
    t,v = _signals(times_s,voltage_v)
    slopes = np.diff(v)/np.diff(t)
    breaks = np.flatnonzero(~np.isclose(slopes[1:],slopes[:-1],rtol=1e-11,atol=0))+1
    edges = np.r_[0,breaks,len(t)-1]
    if driver_indices is not None:
        edges = np.asarray(driver_indices)
        if (edges.ndim!=1 or edges.size<2 or edges.dtype.kind not in "iu" or
            edges[0]!=0 or edges[-1]!=len(t)-1 or np.any(np.diff(edges)<=0) or
            not np.allclose(np.interp(t,t[edges],v[edges]),v,rtol=1e-12,atol=1e-15)):
            raise ValueError("Invalid compressed voltage driver")
    states = np.empty((len(t),len(initial_state)))
    states[0] = initial_state
    for a,b in zip(edges[:-1],edges[1:]):
        rate = (v[b]-v[a])/(t[b]-t[a])
        def rhs(now,y):
            return model.rhs(now,y,v[a]+rate*(now-t[a]))
        res = solve_ivp(rhs,(t[a],t[b]),states[a],method="BDF",t_eval=t[a+1:b+1],rtol=rtol,atol=atol)
        if not res.success:
            raise ValueError("forward integration failed: "+res.message)
        states[a+1:b+1] = res.y.T
    if hasattr(model,"validate_state") and not all(model.validate_state(s) for s in states):
        raise ValueError("forward trajectory violates physical state bounds")
    rates = np.r_[slopes,slopes[-1]]
    return np.array([model.current(s,x,r) for s,x,r in zip(states,v,rates)])


def _prepare(traces,split):
    traces = list(traces)
    if not traces or any(tr.split != split for tr in traces):
        raise ValueError("require only explicitly assigned "+split+" records")
    prepared = []
    for tr in traces:
        t,v,i = _signals(tr.times_s,tr.voltage_v,tr.current_a)
        if tr.initial_state is None and (not tr.reset_verified or v[0] != 0):
            raise ValueError("need verified zero-bias initial equilibrium or explicit initial state")
        noise = np.broadcast_to(np.asarray(tr.noise_a,dtype=float),i.shape)
        if not np.all(np.isfinite(noise)) or np.any(noise<=0):
            raise ValueError("measured positive current-noise scale required")
        prepared.append((tr,t,v,i,noise))
    return prepared


def _residuals(factory,params,prepared,forward):
    m = factory(params)
    out = []
    for tr,t,v,i,noise in prepared:
        y0 = m.equilibrium(0) if tr.initial_state is None else np.asarray(tr.initial_state,dtype=float)
        if hasattr(m,'validate_state') and not m.validate_state(y0):
            raise ValueError("invalid initial state for candidate parameters")
        pred = _vector(forward(m,t,v,y0))
        if len(pred)!=len(t):
            raise ValueError("forward prediction length mismatch")
        out.append((pred-i)/noise)
    return out


def _stats(prepared,errors):
    return [dict(name=tr.name,split=tr.split,n_samples=len(t),
                 noise_normalized_rmse=float(np.sqrt(np.mean(err**2))),
                 rmse_a=float(np.sqrt(np.mean((err*noise)**2))))
            for (tr,t,v,i,noise),err in zip(prepared,errors)]


def fit_parameters(model_factory,initial_parameters,traces,parameter_bounds,*,instrument_calibrated,
                   forward=None,linear_parameters=(),max_nfev=None,rank_rtol=1e-7):
    """Fit explicitly bounded parameters on training traces; log coordinates by default.

    SVD is local to the noise-weighted residual and chosen parameter coordinates.
    An inactive direction or active bound requires independent physical constraints.
    """
    if not instrument_calibrated:
        raise ValueError("verify instrument gain, polarity, timing and offset before fitting")
    prepared = _prepare(traces,"train")
    params = dict(initial_parameters)
    names = list(parameter_bounds)
    if not names or not set(linear_parameters)<=set(names) or not 0<rank_rtol<1:
        raise ValueError("provide fitted parameters and a valid numerical rank threshold")
    lower, upper = np.array([parameter_bounds[n] for n in names],dtype=float).T
    x = np.array([params[n] for n in names],dtype=float)
    log = np.array([n not in linear_parameters for n in names])
    if (not np.isfinite(np.r_[x,lower,upper]).all() or np.any(lower>=upper)
        or np.any(x<lower) or np.any(x>upper) or np.any(lower[log]<=0)):
        raise ValueError("require finite ordered bounds, positive log parameters and an admissible seed")
    for val in (lower,upper,x):
        val[log] = np.log(val[log])
    def unpack(vals):
        values = vals.copy()
        values[log] = np.exp(values[log])
        return dict(params,**dict(zip(names,values)))
    replay = replay_current if forward is None else forward
    def objective(vals):
        return np.concatenate(_residuals(model_factory,unpack(vals),prepared,replay))
    opt = least_squares(objective,x,bounds=(lower,upper),max_nfev=max_nfev,x_scale="jac")
    s = np.linalg.svd(opt.jac,compute_uv=False)
    rank = int(np.sum(s > s[0]*rank_rtol))
    fitted = unpack(opt.x)
    return dict(parameters=fitted,fit_parameter_names=names,success=bool(opt.success),message=opt.message,
                rank=rank,rank_rtol=rank_rtol,singular_values=s.tolist(),
                condition_number=float(s[0]/s[-1]) if rank==len(names) else None,
                identifiability=("rank_deficient" if rank<len(names) else "bound_active" if
                                 np.any(opt.active_mask) else "locally_full_rank"),
                noise_normalized_rmse=float(np.sqrt(np.mean(opt.fun**2))),
                training_metrics=_stats(prepared,_residuals(model_factory,fitted,prepared,replay)))


def validate_parameters(model_factory,parameters,traces,*,instrument_calibrated,forward=None):
    """Evaluate frozen parameter predictions exclusively on declared validation records."""
    if not instrument_calibrated:
        raise ValueError("instrument calibration required")
    prepared = _prepare(traces,"validation")
    errors = _residuals(model_factory,parameters,prepared,replay_current if forward is None else forward)
    return dict(validation_metrics=_stats(prepared,errors),
                noise_normalized_rmse=float(np.sqrt(np.mean(np.concatenate(errors)**2))))



