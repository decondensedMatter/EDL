"""Estimate the selected current and voltage-path transfer functions from physical standards acquired this session."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .awg import compiled_control
from .calibration import clipping_diagnostic


def _pos(value, label):
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"Supply a positive measured value for {label}")
    return float(value)


def _source(obj):
    if not isinstance(obj.get("source"), str) or not obj["source"].strip():
        raise ValueError("Supply standard identity and measurement provenance in source")


def standard_settings_template(use_current_amp=False):
    """Actual reference values and bench identities must be supplied by the operator."""
    if type(use_current_amp) is not bool:
        raise ValueError("Choose amplified or direct current readout")
    freqs = [4096.,16384.,65536.,262144.] if use_current_amp else [65536.,262144.,1048576.]
    return dict(use_current_amp=use_current_amp,voltage_references=[dict(source=None,value_v=None,uncertainty_v=None) for _ in range(3)],
                resistor=dict(source=None,ohm=None), rc=dict(source=None,ohm=None,farad=None,topology="series"),
                holdout_resistor=dict(source=None,ohm=None), rc_frequencies_hz=freqs,
                adc_rails_v=dict(CH1=[None,None],CH2=[None,None]),adc_rails_source=None,
                max_reference_error_v=.01,max_relative_error=.05,standard_excitation_v=.25,
                buffer=dict(source=None,series_resistance_ohm=None,minimum_input_resistance_ohm=None))


def standard_view(request, compiled, settings):
    if request["passive"]:
        duration, rate, start = .002, 1e6, 0.
    else:
        duration = compiled.duration_s
        rate = (max(request["frequency_hz"]*64,2048/duration) if "frequency_hz" in request
                else max(500e6 if settings.get("use_current_amp",False) else 2e9,32/request["edge_s"]))
        start = -8/rate
        duration += -start
    return dict(name=request["key"],start_s=start,duration_s=duration,sample_rate_hz=rate)


def standard_requests(settings):
    """Validate supplied standards before connecting. Return ordered acquisitions."""
    amp = settings.get("use_current_amp",False)
    if type(amp) is not bool:
        raise ValueError("Invalid current measurement mode")
    refs = settings.get("voltage_references") or []
    if len(refs) < 3:
        raise ValueError("Supply two independent DC levels spanning zero and a held-out third level")
    levels = []
    for ref in refs:
        _source(ref)
        if ref.get("value_v") is None:
            raise ValueError("Supply the independently measured DC reference voltage")
        if ref.get("uncertainty_v") is not None:
            _pos(ref["uncertainty_v"],"actual external DC reference uncertainty in volts")
        levels.append(float(ref["value_v"]))
    if (not np.all(np.isfinite(levels)) or len(set(levels)) != len(levels)
            or not min(levels[:-1]) < 0 < max(levels[:-1])):
        raise ValueError("Voltage reference levels must be distinct, finite and bracket zero in training")
    zero_wiring = ("zero the current amplifier input per its zero-input procedure, keep its output connected to CH1"
                   if amp else "short the direct CH1 sense input")
    result = [dict(key="short",stage="shorted_zero",connection=zero_wiring+
                   "; apply zero at specimen-side CH2 buffer input",passive=True)]
    for j, level in enumerate(levels):
        result.append(dict(key=f"reference_{j}", stage="shorted_zero", passive=True,
                           connection=f"apply independent DC reference {level:g} V to the specimen-side voltage buffer input with generator disconnected",
                           reference_v=level))
    for name, stage in (("resistor","gain_polarity_offset_delay"), ("rc","known_rc_bandwidth"),
                        ("holdout_resistor","gain_polarity_offset_delay")):
        spec = settings.get(name) or {}
        _source(spec)
        base = dict(stage=stage, connection=f"{name} {spec['source']} across the same specimen terminals, leaving the fixture and sense wiring in place", passive=False,
                    resistance_ohm=_pos(spec.get("ohm"), name+" resistance"))
        if name == "rc":
            if spec.get("topology") != "series":
                raise ValueError("Use a characterized series RC standard across measured terminals")
            base["capacitance_f"] = _pos(spec.get("farad"), "RC capacitance")
            freqs = sorted({_pos(f,"frequency") for f in settings.get("rc_frequencies_hz",[])})
            if any(f not in {4.**k for k in range(11)} for f in freqs):
                raise ValueError("Choose RC frequencies from 1, 4, 16, ... 1048576 Hz")
            if amp and max(freqs,default=0)>400e3:
                raise ValueError("Current-amplifier RC standards must remain within the 400 kHz ceiling")
            if len(freqs) < 3:
                raise ValueError("Supply at least three RC test frequencies")
            result.extend(dict(base, key=f"rc_{j}", frequency_hz=f) for j,f in enumerate(freqs))
        else:
            result.append(dict(base,key=name))
    if settings["resistor"]["source"] == settings["holdout_resistor"]["source"]:
        raise ValueError("Supply an independent held-out resistor identity")
    if not str(settings.get("adc_rails_source") or "").strip():
        raise ValueError("Supply actual acquisition-range ADC rail provenance")
    for channel in ("CH1","CH2"):
        rails = np.asarray((settings.get("adc_rails_v") or {}).get(channel),float)
        if rails.shape != (2,) or not np.isfinite(rails).all() or rails[0]>=rails[1]:
            raise ValueError(f"Supply actual {channel} raw ADC voltage rails")
    if _pos(settings.get("max_relative_error"),"maximum relative error") > .2:
        raise ValueError("Relative error threshold exceeds 20%")
    reference_tolerance = _pos(settings.get("max_reference_error_v"), "maximum reference error")
    if max(r.get("uncertainty_v") or 0. for r in refs)>=reference_tolerance:
        raise ValueError("External reference uncertainty must be smaller than the DC calibration error budget")
    amplitude = _pos(settings.get("standard_excitation_v",.25),"standard excitation amplitude")
    buf = settings.get("buffer") or {}
    _source(buf)
    _pos(buf.get("series_resistance_ohm"),"measured series resistance for buffer test")
    _pos(buf.get("minimum_input_resistance_ohm"),"minimum buffer resistance for specimen loading")
    j = int(np.argmax(np.abs(levels)))
    result.append(dict(key="buffer_loaded", stage="shorted_zero", passive=True,
                       connection=f"Apply reference {j} ({levels[j]:g} V) through measured series resistor to buffer input"))
    result.append(dict(key="open",stage="open_input_parasitics",connection="open specimen terminals, keeping fixture and both sense paths in place",passive=False))
    for request in result:
        if not request["passive"]:
            request.update(edge_s=1e-6 if amp else 64e-9, hold_s=12e-6 if amp else 512e-9,
                           standard_excitation_v=amplitude)
        if "frequency_hz" in request and request["frequency_hz"]>1.05e6:
            raise ValueError("RC waveform frequency exceeds finite 16-knot compiler excitation limit")
    return result


def _load(folder, settings):
    folder = Path(folder)
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "captured" or meta["hardware_settings"]["use_current_amp"] is not settings.get("use_current_amp",False):
        raise ValueError(f"Require successful acquisition in the selected current-readout mode in {folder}")
    path = folder/meta["raw_channels_csv"]
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if a.shape[0]<16 or a.shape[1]!=4 or not np.isfinite(a).all():
        raise ValueError(f"Invalid raw capture {path}")
    dt = np.diff(a[:,0])
    if dt[0]<=0 or not np.allclose(dt,dt[0],rtol=1e-6,atol=0):
        raise ValueError("Require uniformly sampled scope data")
    for j, channel in enumerate(("CH1","CH2"),1):
        lo, hi = settings["adc_rails_v"][channel]
        if clipping_diagnostic(a[:,j], adc_min=lo, adc_max=hi)["samples_at_rail"]:
            raise ValueError(f"{channel} reached ADC rail in {path}")
    return a,meta,sha256(path.read_bytes()).hexdigest()


def _lowpass(x, dt, tau):
    """Exact endpoint response to the linearly interpolated input at uniform intervals."""
    ratio = dt/tau
    alpha = np.exp(-ratio)
    step = -np.expm1(-ratio)
    right = 1-step/ratio
    y,_ = lfilter([right,step-right],[1.,-alpha],x,zi=[(1-right)*x[0]])
    return y


def _relative(pred, observed):
    scale = np.ptp(observed)
    if scale <= 0:
        raise ValueError("Insufficient excitation")
    return (pred-observed)/scale


def _fixture_basis(time, volts, tau, delay):
    """CH1 observations for unit parallel conductance and capacitance, including lag."""
    filtered = _lowpass(volts,time[1]-time[0],tau)
    # tau*y' + y = V gives the filtered derivative without differentiating scope noise.
    derivative = (volts-filtered)/tau
    return (np.interp(time-delay,time,filtered),np.interp(time-delay,time,derivative))


def process_standards(config, settings, captures, *, check=lambda:None):
    requests = standard_requests(settings)
    amp = settings.get("use_current_amp",False)
    if config["hardware"]["use_current_amp"] is not amp:
        raise ValueError("Standard settings and actual hardware current-readout modes disagree")
    arrays, commands, hashes, folders, command_gains = {}, {}, {}, {}, {}
    for req in requests:
        check()
        key = req["key"]
        if req["passive"]:
            folder = captures[key]
        else:
            folder, compiled = captures[key]
        a, meta, digest = _load(folder,settings)
        hw_capture = meta["hardware_settings"]
        for field in ("ch1_scale_v_per_div","ch2_scale_v_per_div","ch1_input","voltage_measurement_location"):
            if hw_capture.get(field)!=config["hardware"].get(field):
                raise ValueError(f"Standard {key} acquired with incompatible {field}; repeat at the qualified range")
        if digest in hashes.values():
            raise ValueError("The same raw capture was supplied for multiple standards")
        if not req["passive"]:
            t,u = compiled_control(compiled)
            if np.count_nonzero(a[:,0]<0)<5 or a[-1,0]<compiled.duration_s-(a[1,0]-a[0,0]):
                raise ValueError(f"Standard view must cover trigger and excitation for {key}")
            commands[key] = np.interp(a[:,0],t,u,left=0,right=0)
            command_gains[key] = _pos(meta["hardware_settings"]["voltage_gain"],"acquisition generator gain")
        arrays[key], hashes[key], folders[key] = a, digest, str(Path(folder).resolve())
    refs = settings["voltage_references"]
    x = np.array([np.mean(arrays[f"reference_{j}"][:,2]) for j in range(len(refs))])
    levels = np.array([ref["value_v"] for ref in refs],float)
    if np.ptp(x[:-1]) <= 0:
        raise ValueError("No DC voltage reference sensitivity")
    gain, intercept = np.polyfit(x[:-1],levels[:-1],1)
    error = np.max(np.abs(gain*x+intercept-levels))
    worst_case_error = max(abs(gain*x+intercept-levels)+np.array([r.get("uncertainty_v") or 0. for r in refs]))
    if gain<=0 or worst_case_error > settings["max_reference_error_v"]:
        raise ValueError("Independent voltage reference qualification failed")
    for j in range(len(refs)):
        if np.std(arrays[f"reference_{j}"][:,2],ddof=1)*gain > settings["max_reference_error_v"]:
            raise ValueError("Unstable DC reference exceeds voltage-error budget")
    zero = arrays["short"]
    offset = np.mean(zero[:,1])
    def voltage(key):
        return arrays[key][:,2]*gain+intercept
    if abs(np.mean(voltage("short"))) > settings["max_reference_error_v"]:
        raise ValueError("Shorted CH2 zero disagrees with voltage calibration")
    open_time = arrays["open"][:,0]
    before = open_time < 0
    open_raw = arrays["open"][:,1]-np.mean(arrays["open"][before,1])
    open_v = voltage("open")-np.mean(voltage("open")[before])
    open_command = commands["open"]
    plateau_idx = np.flatnonzero(np.isclose(open_command,open_command.max(),rtol=1e-5,atol=1e-12))
    if len(plateau_idx)<8 or np.any(np.diff(plateau_idx)!=1):
        raise ValueError("Open fixture needs a complete positive plateau")
    last = plateau_idx[-1]
    tail = plateau_idx[len(plateau_idx)//2:]
    delta_v = float(np.mean(open_v[tail]))
    # The same CH2 offset cancels in the open-fixture plateau difference.
    span_uncertainty = abs(delta_v)*2*settings["max_reference_error_v"]/np.ptp(levels[:-1])
    voltage_uncertainty = max(span_uncertainty,
                              5*np.hypot(np.std(open_v[before],ddof=1),np.std(open_v[tail],ddof=1)))
    if delta_v <= voltage_uncertainty:
        raise ValueError("Open fixture excitation is unresolved")
    raw_leak = float(np.mean(open_raw[tail]))
    raw_noise = max(np.std(zero[:,1],ddof=1),np.std(open_raw[before],ddof=1),np.std(open_raw[tail],ddof=1))
    raw_leak_error = 5*np.sqrt(2.)*raw_noise
    raw_estimate = raw_leak/delta_v
    raw_slope_error = (raw_leak_error+abs(raw_estimate)*voltage_uncertainty)/(delta_v-voltage_uncertainty)
    raw_slope = raw_estimate if abs(raw_estimate)>raw_slope_error else 0.
    key = "resistor"
    t = arrays[key][:,0]
    dt = t[1]-t[0]
    u = commands[key]
    v = voltage(key)
    # Use the late parts of the two nonzero constant-command plateaus.
    plateau = np.zeros(len(t),bool)
    for level in (min(u),max(u)):
        idx = np.flatnonzero(np.isclose(u,level,rtol=1e-5,atol=1e-12))
        if len(idx)<8 or level==0 or np.any(np.diff(idx)!=1):
            raise ValueError("Need resolved bipolar training resistor plateaus")
        plateau[idx[len(idx)//2:]] = True
    raw = arrays[key][:,1]-offset
    if not np.any(raw[plateau]):
        raise ValueError("Resistor current is unresolved")
    resistor_only = raw[plateau]-raw_slope*v[plateau]
    if np.ptp(resistor_only)<=raw_leak_error:
        raise ValueError("Resistor current is unresolved after fixture leakage correction")
    ig = np.linalg.lstsq(resistor_only[:,None],v[plateau]/settings[key]["ohm"],rcond=None)[0][0]
    actgain = np.linalg.lstsq(u[plateau,None],v[plateau],rcond=None)[0][0]
    if actgain<=0 or ig==0:
        raise ValueError("Invalid transfer gain")
    tol = settings["max_relative_error"]
    open_i = open_raw*ig
    open_leak = raw_leak*ig
    signed_g = open_leak/delta_v
    g_uncertainty = (abs(ig)*raw_leak_error+abs(signed_g)*voltage_uncertainty)/(delta_v-voltage_uncertainty)
    if signed_g < -g_uncertainty:
        raise ValueError("Significantly negative open-fixture conductance; check wiring and offset")
    g = signed_g if signed_g>g_uncertainty else 0.
    def open_capacitance(tau,delay):
        conductance_basis,cap_basis = _fixture_basis(open_time,open_v,tau,delay)
        slope = cap_basis[:last+1]
        norm = np.dot(slope,slope)
        if norm<=0:
            raise ValueError("Open fixture capacitive sensitivity is unresolved")
        projected = open_i[:last+1]-g*conductance_basis[:last+1]
        signed_c = float(np.dot(slope,projected)/norm)
        # Bound the projection of baseline/plateau noise and leakage uncertainty
        # without assuming independent samples of filtered oscilloscope noise.
        uncertainty = float(np.dot(abs(slope),abs(ig)*raw_leak_error
                          +g_uncertainty*abs(conductance_basis[:last+1]))/norm)
        uncertainty += abs(signed_c)*voltage_uncertainty/(delta_v-voltage_uncertainty)
        return signed_c,uncertainty
    def fit_rise(z):
        check()
        return _relative(_lowpass(u*actgain,dt,np.exp(z[0])),v)
    actuator = least_squares(fit_rise,[np.log(dt*5)],bounds=([np.log(dt*.1)],[np.log(t[-1]-t[0])]))
    tau_v = float(np.exp(actuator.x[0]))
    # Include the fast resistive edge in the delay / current-response fit as well.
    current_data = [(t,dt,v,v/settings["resistor"]["ohm"],raw*ig)]
    for req in requests:
        if req["stage"]!="known_rc_bandwidth":
            continue
        key = req["key"]
        time = arrays[key][:,0]
        d = time[1]-time[0]
        if d*req["frequency_hz"]>1/12:
            raise ValueError("RC standard requires >=12 samples per excitation period")
        volt = voltage(key)
        r,c = req["resistance_ohm"],req["capacitance_f"]
        ideal = (volt-_lowpass(volt,d,r*c))/r
        current_data.append((time,d,volt,ideal,(arrays[key][:,1]-offset)*ig))
    def predict_load(time,d,volts,ideal,tau,delay,c):
        conductance_basis,cap_basis = _fixture_basis(time,volts,tau,delay)
        load_current = np.interp(time-delay,time,_lowpass(ideal,d,tau))
        return load_current+g*conductance_basis+c*cap_basis
    def current_residual(z):
        check()
        tau,delay = np.exp(z[0]),z[1]*grid
        residuals = []
        signed_c,c_uncertainty = open_capacitance(tau,delay)
        c = signed_c if signed_c>c_uncertainty else 0.
        for time,d,volts,ideal,observed in current_data:
            check()
            predicted = predict_load(time,d,volts,ideal,tau,delay,c)
            residuals.append(_relative(predicted,observed)/np.sqrt(len(time)))
        return np.concatenate(residuals)
    grid = min(row[1] for row in current_data)
    duration = min(row[0][-1]-row[0][0] for row in current_data)
    response = least_squares(current_residual,[np.log(grid*3),1.],
                             bounds=([np.log(grid*.1),0.],[np.log(duration),duration/grid/4]))
    tau_i, delay = float(np.exp(response.x[0])),float(response.x[1]*grid)
    if min(tau_i/grid,tau_v/dt)<1 or np.max(np.abs(fit_rise(actuator.x)))>tol:
        raise ValueError("Unresolved response or command-path training residual exceeds threshold")
    highest = max(settings["rc_frequencies_hz"])
    bw = 1/(2*np.pi*tau_i)
    if bw < min(settings["rc_frequencies_hz"]):
        raise ValueError("RC excitations start above the fitted current-path bandwidth")
    max_rc = 0.
    signed_c,c_uncertainty = open_capacitance(tau_i,delay)
    if signed_c < -c_uncertainty:
        raise ValueError("Significantly negative open-fixture capacitance; check wiring and offset")
    c = signed_c if signed_c>c_uncertainty else 0.
    for time,d,volts,ideal,observed in current_data:
        check()
        predicted = predict_load(time,d,volts,ideal,tau_i,delay,c)
        max_rc = max(max_rc,float(np.max(abs(_relative(predicted,observed)))))
    key = "holdout_resistor"
    time = arrays[key][:,0]
    d = time[1]-time[0]
    volts = voltage(key)
    predicted = _lowpass(commands[key]*actgain*command_gains["resistor"]/command_gains[key],d,tau_v)
    v_error = float(np.max(abs(_relative(predicted,volts))))
    ideal = volts/settings[key]["ohm"]
    i_error = float(np.max(abs(_relative(predict_load(time,d,volts,ideal,tau_i,delay,c),
                                           (arrays[key][:,1]-offset)*ig))))
    if max(max_rc,v_error,i_error)>tol:
        raise ValueError(f"Standard response qualification failed (RC {max_rc:g}, held-out V {v_error:g}, I {i_error:g})")
    check()
    j = int(np.argmax(np.abs(levels)))
    direct = abs(float(x[j]*gain+intercept))
    loaded = float(np.mean(voltage("buffer_loaded")))*np.sign(levels[j])
    uncertainty = float(settings["max_reference_error_v"])
    buf = settings["buffer"]
    if np.std(voltage("buffer_loaded"),ddof=1) > uncertainty:
        raise ValueError("Buffer loading reference was unstable")
    if loaded <= uncertainty or loaded > direct+2*uncertainty:
        raise ValueError("Invalid loaded buffer voltage for input-impedance verification")
    impedance_lower = buf["series_resistance_ohm"]*(loaded-uncertainty)/max(2*uncertainty,direct-loaded+2*uncertainty)
    if impedance_lower < buf["minimum_input_resistance_ohm"]:
        raise ValueError("Measured lower bound on buffer input resistance is insufficient for OCV")
    # The opposite edge is independent of the rising-edge capacitance estimate.
    conductance_basis,cap_basis = _fixture_basis(open_time,open_v,tau_i,delay)
    fixture_prediction = g*conductance_basis+c*cap_basis
    falling = np.arange(len(open_time))>last
    if falling.sum()<8:
        raise ValueError("Need independent falling-edge fixture data")
    residual = float(np.max(abs(fixture_prediction[falling]-open_i[falling])))
    # Rails/noise, gain qualification and unresolved parameters all contribute.
    allowed = (abs(ig)*raw_leak_error+tol*np.max(abs(open_i[falling]))+
               g_uncertainty*np.max(abs(conductance_basis[falling]))+
               c_uncertainty*np.max(abs(cap_basis[falling])))
    if residual>allowed:
        raise ValueError("Open fixture failed independent falling-edge parallel RC qualification")
    displacement = open_i[:last+1]-g*conductance_basis[:last+1]
    charge = float(np.trapezoid(displacement,open_time[:last+1]))
    fixture = dict(raw_sha256=hashes["open"],capture_folder=folders["open"],delta_voltage_v=delta_v,
                   signed_conductance_s=signed_g,conductance_uncertainty_s=g_uncertainty,
                   signed_capacitance_f=signed_c,capacitance_uncertainty_f=c_uncertainty,
                   falling_edge_max_error_a=residual,falling_edge_error_limit_a=float(allowed),
                   conductance_resolved=bool(g>0),capacitance_resolved=bool(c>0),
                   method="Late-plateau leakage and CH1-filtered rising-edge derivative. Opposite edge held out.")
    evidence = dict(use_current_amp=amp,current_path="current amplifier into DC50" if amp else "direct DC50",
                    bandwidth_ceiling_hz=400e3 if amp else config["hardware"]["scope_bandwidth_hz"],
                    calibrated_voltage_interval_v=[float(min(levels[:-1])),float(max(levels[:-1]))],
                    acquisition_settings={k:config["hardware"][k] for k in
                       ("ch1_scale_v_per_div","ch2_scale_v_per_div","ch1_input","voltage_measurement_location")},
                    open_fixture_measurement=fixture,open_parallel_conductance_s=g,open_parallel_capacitance_f=c,
                    open_plateau_leakage_a=open_leak,open_equivalent_conductance_s=signed_g,
                    open_equivalent_capacitance_f=signed_c,
                    open_transition_charge_c=charge,standards=deepcopy(settings), raw_sha256=hashes, capture_folders=folders,
                    buffer_input_resistance_lower_bound_ohm=float(impedance_lower), fitted_bandwidth_hz=bw, max_reference_error_v=float(error), rc_max_relative_error=max_rc,
                    heldout_voltage_relative_error=v_error, heldout_current_relative_error=i_error,
                    baseline_current_noise_a=float(np.std(zero[:,1],ddof=1)*abs(ig)),
                    baseline_voltage_noise_v=float(np.std(zero[:,2],ddof=1)*gain),
                    shorted_voltage_mean_v=float(np.mean(voltage("short"))),
                    shorted_block_mean_spread_v=float(np.ptp([np.mean(z)*gain for z in np.array_split(zero[:,2],8)])),
                    open_peak_current_a=float(np.max(abs(open_i))),
                    open_peak_voltage_v=float(np.max(abs(voltage("open")))))
    updated = deepcopy(config)
    updated["hardware"].update(current_gain_a_per_v=float(abs(ig)),current_sign=int(np.sign(ig)),
        current_offset_v=float(offset),voltage_offset_v=float(-intercept/gain),voltage_scale_factor=float(gain),
        voltage_gain=float(actgain*command_gains["resistor"]),current_bandwidth_hz=min(bw,highest,evidence["bandwidth_ceiling_hz"]),ocv_buffer_verified=True,
        measured_step_rise_s=tau_v*np.log(9),current_response_rise_s=tau_i*np.log(9),
        current_delay_s=delay, data_hardware_calibrated=True,first_order_independently_qualified=True,
        calibration_evidence=evidence,current_gain_source="measured resistor",voltage_scale_source="external DC references")
    updated["calibration_id"]=sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest()
    check()
    return updated












