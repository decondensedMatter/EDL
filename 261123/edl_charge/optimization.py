"""Fixed-time local charging control with closed-cell replay and post-charge branches."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def simulate(model,times_s,voltage_v,initial_state=None,*,rtol=2e-7,atol=1e-10,
             points_per_interval=16):
    """Replay continuous PWL samples. Work uses trapezoidal V dQ on each interval.

    At drive knots the displayed current uses the right-hand slew, except at the
    final sample, which uses the left-hand slew. Charge quadrature preserves the
    geometric displacement contribution across slope switches.
    """
    t = np.asarray(times_s,dtype=float)
    v = np.asarray(voltage_v,dtype=float)
    n = int(points_per_interval)
    if (t.ndim!=1 or t.size<2 or t.shape!=v.shape or not np.all(np.isfinite([t,v]))
        or np.any(np.diff(t)<=0) or n<2 or rtol<=0 or atol<=0):
        raise ValueError("require finite voltages on increasing times and positive solver tolerances")
    y = np.asarray(model.equilibrium(v[0]) if initial_state is None else initial_state,dtype=float)
    ts,vs,ys,rs = [],[],[],[]
    for a in range(len(t)-1):
        dt = t[a+1]-t[a]
        dv = v[a+1]-v[a]
        # Integrating on [0,1] avoids poor time scaling for sub-microsecond waveforms.
        def ode(s,state):
            return dt*model.rhs(t[a]+s*dt,state,v[a]+s*dv)
        s = np.linspace(0,1,n+1)
        res = solve_ivp(ode,(0,1),y,t_eval=s,method="BDF",rtol=rtol,atol=atol)
        if not res.success:
            raise ArithmeticError(res.message)
        y = res.y[:,-1].copy()
        count = n+1 if a==len(t)-2 else n
        ts.extend(t[a]+dt*s[:count])
        vs.extend(v[a]+dv*s[:count])
        ys.extend(res.y[:,:count].T)
        rs.extend([dv/dt]*count)
    ts,vs,ys = np.array(ts),np.array(vs),np.array(ys)
    if hasattr(model,"validate_state") and not all(model.validate_state(state) for state in ys):
        raise ArithmeticError("trajectory leaves model physical domain or violates inventory")
    q = np.array([model.charge(state,u) for state,u in zip(ys,vs)])
    current = np.array([model.current(state,u,r) for state,u,r in zip(ys,vs,rs)])
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(current)):
        raise ArithmeticError("nonfinite terminal charge or current")
    actual = (np.array([model.terminal_voltage(state,u) for state,u in zip(ys,vs)])
              if hasattr(model,"terminal_voltage") else vs.copy())
    dq = np.diff(q)
    dw = (actual[:-1]+actual[1:])*dq/2
    return dict(times_s=ts,command_voltage_v=vs,voltage_v=actual,current_a=current,state=ys,charge_c=q,final_state=y,
                work_j=float(dw.sum()),delivered_work_j=float(np.maximum(dw,0).sum()),
                returned_work_j=float(-np.minimum(dw,0).sum()))


def optimize_charge(model,final_voltage_v,charging_time_s,voltage_limit_v,*,
                    minimum_transition_s,knots=7,charge_tolerance=.01,rtol=2e-7,
                    maxiter=50,starts=3,mesh_check=True,refinement_tolerance=.02,terminal_voltage_tolerance=.005):
    """Find a feasible local minimum of net terminal work on the charge segment.

    The 10–90% rise-time convention implies slew <= 0.8*voltage_limit/minimum_transition.
    Interior voltages have the target polarity and magnitude <= limit. The spatial endpoint is reported as a memory diagnostic.
    """
    vf,T,cap = map(float,(final_voltage_v,charging_time_s,voltage_limit_v))
    if (not np.isfinite([vf,T,cap,minimum_transition_s,rtol,refinement_tolerance]).all() or
        not 0<abs(vf)<=cap or min(T,minimum_transition_s,rtol,refinement_tolerance)<=0 or
        not 0<charge_tolerance<1 or not 0<terminal_voltage_tolerance<1 or int(knots)!=knots or knots<3 or maxiter<1 or starts<1):
        raise ValueError("invalid endpoint, voltage limit, duration, tolerance or discretization")
    sign = 1. if vf>0 else -1.
    t = np.linspace(0,T,int(knots))
    slew = .8*cap/minimum_transition_s
    delta = slew*(t[1]-t[0])/cap
    y0, yf = model.equilibrium(0.), model.equilibrium(vf)
    qi, qf = model.charge(y0,0.), model.charge(yf,vf)
    span = abs(qf-qi)
    if not np.isfinite(span) or span<=0:
        raise ValueError("target has zero or nonfinite equilibrium charge excursion")
    eref = max(span*abs(vf),np.finfo(float).tiny)
    base = dict(times_s=t,final_voltage_v=vf,charging_time_s=T,voltage_limit_v=cap,
                minimum_transition_s=minimum_transition_s,slew_limit_v_s=slew,
                charge_tolerance=charge_tolerance,terminal_voltage_tolerance=terminal_voltage_tolerance,basis_knots=int(knots),
                global_minimum_certified=False,basis_refinement_performed=False)
    if abs(vf)/T>slew*(1+1e-12):
        return dict(base,status="no_feasible_control_found",reason="slew bound precludes endpoint voltage")
    def volts(x):
        return sign*cap*np.r_[0.,x,abs(vf)/cap]
    def voltage_slack(x):
        d = np.diff(volts(x)/cap)
        return np.r_[delta-d,delta+d]

    cache = {}
    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            p = simulate(model,t,volts(x),y0,rtol=rtol,points_per_interval=8)
            cache[key] = (p["work_j"]/eref,(p["charge_c"][-1]-qf)/span,(p["voltage_v"][-1]-vf)/vf)
        return cache[key]
    # Leave room for independent replay of a nearly active charge constraint.
    tol = charge_tolerance*.9
    def charge_slack(x):
        e = evaluate(x)[1]
        ev = evaluate(x)[2]
        return np.array([tol-e,tol+e,terminal_voltage_tolerance*.9-ev,terminal_voltage_tolerance*.9+ev])
    ramp = vf*t/T
    envelope = sign*np.minimum.reduce([np.full_like(t,cap),slew*t,abs(vf)+slew*(T-t)])
    early = sign*np.minimum(abs(vf),slew*t)
    seeds = [ramp,envelope,early]
    candidates = []
    failures = []
    seen = set()
    attempted = 0
    def feasible_candidate(x):
        w,e,ev = evaluate(x)
        u = sign*volts(x)
        return (abs(e)<=tol+2e-7 and abs(ev)<=terminal_voltage_tolerance*.9+2e-7 and voltage_slack(x).min()>=-1e-9 and
                u.min()>=-1e-12*cap and u.max()<=cap*(1+1e-12))
    for seed in seeds[:int(starts)]:
        x0 = sign*seed[1:-1]/cap
        if tuple(x0) in seen:
            continue
        seen.add(tuple(x0))
        try:
            attempted += 1
            opt = minimize(lambda x:evaluate(x)[0],x0,method="SLSQP",bounds=[(0.,1.)]*len(x0),
                constraints=[dict(type="ineq",fun=voltage_slack),dict(type="ineq",fun=charge_slack)],
                options=dict(maxiter=maxiter,ftol=2e-8,eps=2e-5))
            if opt.success and feasible_candidate(opt.x):
                candidates.append((evaluate(opt.x)[0],opt.x.copy(),str(opt.message)))
            else:
                failures.append(str(opt.message))
        except (ValueError,ArithmeticError,RuntimeError) as exc:
            failures.append("forward replay failed: "+str(exc))
    if not candidates:
        status = ("numerical_failure" if not cache or
                  all("forward replay failed:" in msg for msg in failures) else "no_feasible_control_found")
        return dict(base,status=status,reason="; ".join(failures),starts_attempted=attempted)
    _,x,msg = min(candidates,key=lambda item:item[0])
    u = volts(x)
    coarse = simulate(model,t,u,y0,rtol=rtol,points_per_interval=16)
    fine = simulate(model,t,u,y0,rtol=rtol/4,atol=2.5e-11,points_per_interval=40)
    w = fine["work_j"]
    energy_scale = max(abs(w),np.finfo(float).eps*eref)
    end = float(abs(fine["charge_c"][-1]-qf)/span)
    actual = fine["voltage_v"]
    ev = abs(actual[-1]-vf)/abs(vf)
    violation = max(0.,float(np.max(sign*u)-cap),float(-np.min(sign*u)),float(np.max(sign*actual)-cap),float(-np.min(sign*actual)))
    checks = dict(time_work_relative=abs(w-coarse["work_j"])/energy_scale,
                  time_charge_relative=abs(fine["charge_c"][-1]-coarse["charge_c"][-1])/span,
                  mesh_checked=False)
    ok = (checks["time_work_relative"]<=.005 and checks["time_charge_relative"]<=.005
          and end<=charge_tolerance and ev<=terminal_voltage_tolerance and violation<=1e-12*cap
          and np.max(abs(np.diff(u)/np.diff(t)))<=slew*(1+1e-8))
    m = None
    if mesh_check and hasattr(model,"refined"):
        m = model.refined()
    elif mesh_check and hasattr(model,"remap_state") and hasattr(model,"parameters"):
        m = type(model)(dict(model.parameters,cells=2*model.cells))
    if m is not None:
        p = simulate(m,t,u,rtol=rtol/4,atol=2.5e-11,points_per_interval=40)
        q0m = m.charge(m.equilibrium(0),0)
        qfm = m.charge(m.equilibrium(vf),vf)
        em = abs(p["charge_c"][-1]-qfm)/abs(qfm-q0m)
        checks.update(mesh_checked=True,mesh_terminal_charge_error=float(em),
                      mesh_work_relative=abs(p["work_j"]-w)/energy_scale,
                      mesh_charge_relative=abs(p["charge_c"][-1]-fine["charge_c"][-1])/span)
        ok = ok and em<=charge_tolerance and max(checks["mesh_work_relative"],
                                                 checks["mesh_charge_relative"])<=refinement_tolerance
    baseline = simulate(model,t,ramp,y0,rtol=rtol/4,points_per_interval=40)
    baseline_error = abs(baseline["charge_c"][-1]-qf)/span
    baseline_v_error = abs(baseline["voltage_v"][-1]-vf)/abs(vf)
    return dict(base,status="validated" if ok else "numerical_failure",voltage_v=u,
                reason=msg if ok else "independent replay failed one or more tolerances",
                starts_attempted=attempted,prediction=fine,net_charge_work_j=w,
                terminal_charge_error=end,terminal_voltage_error=ev,actual_peak_voltage_v=float(actual[np.argmax(sign*actual)]),full_state_error=model.normalized_state_error(fine["final_state"],yf,y0),
                bound_violation=violation,validation_checks=checks,linear_ramp_terminal_error=baseline_error,
                feasible_linear_ramp_work_j=baseline["work_j"] if baseline_error<=charge_tolerance and baseline_v_error<=terminal_voltage_tolerance else None)


def replay_variants(result,model,hold_s,discharge_s,reset_s):
    """Branch from the same drive endpoint. All records include return to zero and reset."""
    if result.get("status")!="validated":
        raise ValueError("variants require independently validated charge control")
    if not np.isfinite([hold_s,discharge_s,reset_s]).all() or min(hold_s,discharge_s,reset_s)<=0:
        raise ValueError("hold, discharge and reset durations must be positive")
    t = np.asarray(result["times_s"])
    v = np.asarray(result["voltage_v"])
    out = {}
    for name,delay in (("charge_hold_observe",hold_s),("charge_hold_discharge",hold_s),
                       ("charge_immediate_discharge",0.)):
        tt,uu = list(t),list(v)
        phases = [dict(name="charge",start_s=float(t[0]),end_s=float(t[-1]))]
        def extend(label,duration,value):
            start = tt[-1]
            tt.append(start+duration)
            uu.append(value)
            phases.append(dict(name=label,start_s=float(start),end_s=float(tt[-1])))
        if delay:
            extend("observe_hold" if name=="charge_hold_observe" else "hold",delay,v[-1])
        extend("return_after_observation" if name=="charge_hold_observe" else "discharge",discharge_s,0.)
        extend("zero_reset",reset_s,0.)
        trace = simulate(model,tt,uu)
        trace.update(phases=phases,control_times_s=np.asarray(tt),control_voltage_v=np.asarray(uu))
        out[name] = trace
    return out


