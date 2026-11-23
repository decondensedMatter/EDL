"""Single-shot, trigger-relative measured-voltage replay and electrical tail diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter
from scipy.integrate import trapezoid
from .calibration import replay_current
from .optimization import simulate


def load_capture(folder):
    folder = Path(folder)
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("status") not in ("captured","bound_exceeded"):
        raise ValueError("a completed capture is required")
    filename = metadata.get("capture_csv","capture.csv")
    if not isinstance(filename,str) or not filename or Path(filename).name!=filename or filename in (".",".."):
        raise ValueError("capture_csv must name one file in the acquisition folder")
    with (folder/filename).open(encoding="utf-8-sig",newline="") as stream:
        r = csv.reader(stream)
        headers = [h.replace(" ","") for h in next(r)]
        idx = [headers.index(h) for h in ("Time(s)","Current(A)","Voltage(V)")]
        vals = np.array([[float(row[j]) for j in idx] for row in r])
    if vals.ndim!=2 or len(vals)<3 or not np.isfinite(vals).all() or np.any(np.diff(vals[:,0])<=0):
        raise ValueError("need at least three finite samples on an increasing time axis")
    return metadata,*vals.T


def instrument_current(t,i,*,rise_s,delay_s,pretrigger_current=0.):
    """Causal first-order convolution of PWL raw current and nonperiodic time delay."""
    t,i = np.asarray(t),np.asarray(i)
    dt = np.diff(t)
    if (not np.isfinite([rise_s,delay_s,pretrigger_current]).all() or min(rise_s,delay_s)<0
        or len(t)!=len(i) or not len(dt) or dt.min()<=0):
        raise ValueError("invalid channel response or sample axis")
    if rise_s:
        x = dt*np.log(9)/rise_s
        decay = np.exp(-x)
        one = -np.expm1(-x)
        # Stable coefficient for linearly interpolated input.
        w = 1-one/x
        small = x<1e-5
        w[small] = x[small]/2-x[small]**2/6
        drive = one*i[:-1]+w*np.diff(i)
        f = np.empty_like(i,dtype=float)
        f[0] = pretrigger_current
        if np.allclose(x,x[0],rtol=1e-11,atol=0):
            f[1:] = lfilter([1.],[1.,-decay[0]],drive,zi=[decay[0]*f[0]])[0]
        else:
            for k in range(len(dt)):
                f[k+1] = decay[k]*f[k]+drive[k]
    else:
        f = i
    return np.interp(t-delay_s,t,f,left=pretrigger_current)


def settling_diagnostic(t,i,*,target_a,noise_a,window_s,tolerance_sigma=3.):
    """Earliest sustained measured-current plateau through the end of this phase."""
    t,i = np.asarray(t),np.asarray(i)
    if min(noise_a,window_s,tolerance_sigma)<=0 or not np.isfinite([target_a,noise_a,window_s,tolerance_sigma]).all():
        raise ValueError("settling needs a target, positive noise and positive verification window")
    out = dict(status="right_censored",settling_time_s=None,target_current_a=float(target_a),
               noise_a=float(noise_a),verification_window_s=float(window_s))
    if len(t)<3 or t[-1]-t[0]<window_s:
        return out
    # Require all subsequent nonoverlapping windows to pass; search backward
    # from the last complete window so that late rebound revokes early settling.
    end = len(t)-1
    earliest = None
    bound = tolerance_sigma*noise_a
    while end>=2:
        if t[end]-t[0] < window_s*(1-1e-10):
            break
        start = max(0,np.searchsorted(t,t[end]-window_s,side="right")-1)
        if end-start<2:
            break
        z = i[start:end+1]
        u = t[start:end+1]-t[start]
        slope = np.polyfit(u-u.mean(),z,1)[0]
        if max(abs(z.mean()-target_a),z.std(ddof=1),abs(slope)*window_s)>bound:
            break
        earliest = float(t[start])
        end = start
    if earliest is not None:
        out.update(status="sustained_electrical_plateau",settling_time_s=earliest)
    return out


def fit_decay(t,i):
    """Descriptive single exponential + offset on a constant-voltage return/hold."""
    t,i = np.asarray(t),np.asarray(i)
    if len(t)<8 or t[-1]<=t[0] or np.ptp(i)==0:
        return dict(status="insufficient_decay_information")
    span, scale = t[-1]-t[0],np.ptp(i)
    x,y = (t-t[0])/span,(i-i[-1])/scale
    def curve(x,a,tau,b):
        return a*np.exp(-x/tau)+b
    try:
        p,cov = curve_fit(curve,x,y,p0=(y[0],.2,0),bounds=([-np.inf,1e-8,-np.inf],[np.inf,np.inf,np.inf]),maxfev=3000)
    except (RuntimeError,ValueError,FloatingPointError):
        return dict(status="fit_failed")
    se = np.sqrt(max(cov[1,1],0))*span
    return dict(status="fit",tau_s=float(p[1]*span),tau_standard_error_s=float(se) if np.isfinite(se) else None,
                amplitude_a=float(p[0]*scale),offset_a=float(p[2]*scale+i[-1]),
                relative_rmse=float(np.sqrt(np.mean((y-curve(x,*p))**2))),fit_window_s=float(span))


def compare_discharge_decay(held,immediate):
    """Compare independent single-shot constant-voltage decay fits (absolute time constants)."""
    a,b = held["metrics"].get("discharge_decay",{}),immediate["metrics"].get("discharge_decay",{})
    if a.get("status")!="fit" or b.get("status")!="fit":
        return dict(status="unavailable")
    ea,eb = a["tau_standard_error_s"],b["tau_standard_error_s"]
    return dict(status="descriptive_comparison",held_tau_s=a["tau_s"],immediate_tau_s=b["tau_s"],
                held_minus_immediate_tau_s=a["tau_s"]-b["tau_s"],
                independent_fit_difference_standard_error_s=float(np.hypot(ea,eb)) if ea is not None and eb is not None else None)


def parallel_fixture_current(t, voltage, evidence, *, rise_s, delay_s):
    """Scope-channel response to measured parallel G*V and C*dV/dt on a PWL grid."""
    if not isinstance(evidence,dict):
        return np.zeros(len(t))
    g = float(evidence.get("open_parallel_conductance_s",0.))
    c = float(evidence.get("open_parallel_capacitance_f",0.))
    if not np.isfinite([g,c]).all() or min(g,c)<0:
        raise ValueError("Applied parallel fixture parameters must be finite and nonnegative")
    t, voltage = np.asarray(t,float), np.asarray(voltage,float)
    ohmic = instrument_current(t,g*voltage,rise_s=rise_s,delay_s=delay_s,
                               pretrigger_current=g*voltage[0])
    if c == 0:
        return ohmic
    dt = np.diff(t)
    drive = c*np.diff(voltage)/dt
    if rise_s == 0:
        idx = np.searchsorted(t,t-delay_s,side="right")-1
        displacement = np.where(idx<0,0.,drive[np.clip(idx,0,len(drive)-1)])
    else:
        # The capacitor contributes a constant current within each voltage interval.
        tau = rise_s/np.log(9.)
        alpha,step = np.exp(-dt/tau), -np.expm1(-dt/tau)
        filtered = np.zeros(len(t))
        for k in range(len(dt)):
            filtered[k+1] = alpha[k]*filtered[k]+step[k]*drive[k]
        query = t-delay_s
        idx = np.clip(np.searchsorted(t,query,side="right")-1,0,len(drive)-1)
        dt_query = np.maximum(0.,query-t[idx])
        displacement = filtered[idx]+(drive[idx]-filtered[idx])*(-np.expm1(-dt_query/tau))
    return ohmic+displacement


def compare_capture(model,folder,control_times_s,control_voltage_v,*,current_response_rise_s=0,
                    delay_s=0,baseline_current_a=0,settling_noise_a=None,target_current_a=None,
                    settling_window_s=None):
    """Each report describes exactly one capture. Absolute times are marker-relative."""
    meta,t,i,v = load_capture(folder)
    ct,cv = np.asarray(control_times_s,dtype=float),np.asarray(control_voltage_v,dtype=float)
    if (ct.ndim!=1 or ct.shape!=cv.shape or len(ct)<2 or ct[0]!=0 or cv[0]!=0
        or not np.isfinite([ct,cv]).all() or np.any(np.diff(ct)<=0) or not np.isfinite(baseline_current_a)):
        raise ValueError("need full PWL protocol starting from zero voltage at trigger time zero")
    cell = getattr(model,"cell_model",model)
    state = cell.equilibrium(0)
    hist_t,hist_i,hist_v = np.array([]),np.array([]),np.array([])
    expected_v_at_start = 0.
    if t[0]>0:
        knots = np.r_[ct[ct<t[0]],t[0]]
        prior = simulate(model,knots,np.interp(knots,ct,cv),points_per_interval=64)
        state = prior["final_state"][:-1] if cell is not model else prior["final_state"]
        hist_t,hist_i = prior["times_s"][:-1],prior["current_a"][:-1]
        hist_v = prior["voltage_v"][:-1]
        expected_v_at_start = prior["voltage_v"][-1]
    raw = replay_current(cell,t,v,state)
    joined_t,joined_i = np.r_[hist_t,t],np.r_[hist_i,raw]
    predicted = instrument_current(joined_t,joined_i,rise_s=current_response_rise_s,
                                   delay_s=delay_s)[-len(t):]+baseline_current_a
    evidence = meta.get("hardware_settings",{}).get("calibration_evidence")
    fixture = parallel_fixture_current(joined_t,np.r_[hist_v,v],evidence,
                                       rise_s=current_response_rise_s,delay_s=delay_s)[-len(t):]
    predicted += fixture
    err = i-predicted
    rms = float(np.sqrt(np.mean(err**2)))
    measured_span = float(np.ptp(i))
    partial = bool(t[0]>0 or t[-1]<ct[-1])
    metrics = dict(capture_status=meta["status"],diagnostic_only=meta["status"]=="bound_exceeded",
                   window_start_s=float(t[0]),window_end_s=float(t[-1]),partial_protocol=partial,
                   integral_scope="single_captured_window",prior_state_source=(
                   "full_command_replay_to_window_start" if t[0]>0 else "assumed_zero_reset_equilibrium"),
                   first_voltage_mismatch_v=float(v[0]-expected_v_at_start),rmse_a=rms,
                   current_span_normalized_rmse=rms/measured_span if measured_span else None,
                   baseline_current_a=float(baseline_current_a),points=len(t),
                   measured_window_charge_c=float(trapezoid(i,t)),measured_window_work_j=float(trapezoid(i*v,t)),
                   modeled_cell_window_charge_c=float(trapezoid(raw,t)),modeled_cell_window_work_j=float(trapezoid(raw*v,t)),
                   current_response_rise_s=float(current_response_rise_s),delay_s=float(delay_s))
    # Only assess constant-command portions of the post-charge hold and the zero reset.
    # Infer the phases from captured hardware durations, never by combining other shots.
    phases = {}
    timing = [meta.get(k) for k in ("actual_charge_s","actual_hold_s","actual_discharge_s","actual_reset_s")]
    if all(x is not None for x in timing):
        charge,hold,discharge,reset = map(float,timing)
        for name,a,b in (("post_charge_hold",charge,charge+hold),
                         ("zero_after_discharge",charge+hold+discharge,sum(timing))):
            mask = (t>=a)&(t<=b)
            if b<=a or np.count_nonzero(mask)<3:
                continue
            x,z = t[mask],i[mask]
            phase = dict(observed_start_s=float(x[0]),observed_end_s=float(x[-1]),phase_start_s=a,
                         last_observed_current_a=float(z[-1]),mean_observed_current_a=float(np.mean(z)))
            if settling_noise_a is not None and settling_window_s is not None:
                target = target_current_a
                if target is None:
                    g = evidence.get("open_parallel_conductance_s",0.) if isinstance(evidence,dict) else 0.
                    target = baseline_current_a+g*float(np.median(v[mask]))
                settle = settling_diagnostic(x,z,target_a=target,noise_a=settling_noise_a,window_s=settling_window_s)
                if settle["settling_time_s"] is not None:
                    settle["observed_settling_time_from_phase_start_s"] = settle["settling_time_s"]-a
                    settle["earlier_phase_unobserved"] = bool(x[0]-a>max(np.diff(t)))
                phase["settling"] = settle
            phases[name] = phase
            if name=="zero_after_discharge":
                metrics["discharge_decay"] = fit_decay(x,z)
    metrics["phases"] = phases
    return dict(metadata=meta,metrics=metrics,series=dict(times_s=t,voltage_v=v,command_voltage_v=np.interp(t,ct,cv),
                current_a=i,predicted_cell_current_a=raw,predicted_fixture_channel_current_a=fixture,predicted_instrument_current_a=predicted,residual_a=err))


def write_analysis(result,output_dir):
    """Write a measured/predicted overlay, single-shot series and compact JSON report."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    dest = Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    m,s = result["metrics"],result["series"]
    (dest/"analysis.json").write_text(json.dumps(dict(metrics=m,metadata=result["metadata"]),indent=2,allow_nan=False),encoding="utf8")
    with (dest/"prediction.csv").open("w",newline="") as handle:
        w=csv.writer(handle)
        w.writerow(s)
        w.writerows(zip(*s.values()))
    fig = Figure(figsize=(8,5),layout="constrained")
    FigureCanvasAgg(fig)
    ax,bx = fig.subplots(2,1,sharex=True)
    t = s["times_s"]
    ax.plot(t,s["voltage_v"],label="Measured sample voltage")
    ax.plot(t,s["command_voltage_v"],"--",label="Command")
    ax.set_ylabel("Voltage (V)")
    bx.plot(t,s["current_a"],label="Measured current",alpha=.8)
    bx.plot(t,s["predicted_instrument_current_a"],"--",label="Predicted instrument current")
    bx.set_ylabel("Current (A)")
    bx.set_xlabel("Time from trigger (s)")
    for a in (ax,bx):
        a.legend()
    fig.suptitle(("Voltage bound exceeded — diagnostic capture" if m["diagnostic_only"] else "Single-shot comparison")+(" (partial protocol)" if m["partial_protocol"] else ""))
    fig.savefig(dest/"overlay.png",dpi=150)
    return dest




