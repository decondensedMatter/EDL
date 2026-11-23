"""Two identical Stern–Langmuir interfaces with two kinetic populations and Frumkin interactions.

Adapted from a historical TF implementation. State order is scaled
left metal charge density, then eight coverages (side, population, ion+, ion-).
Terminal current is positive into the cell; the model contains no shunt current.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


class TFModel:
    charge_scale_c_m2 = 0.01
    fit_parameter_names = ("R", "CH", "alpha", "tau", "slow_fraction", "tau_ratio", "g_frumkin")
    physical_parameter_names = fit_parameter_names + (
        "Gamma", "beta", "area_m2", "concentration_mol_m3", "temp_K", "eps_r"
    )

    def __init__(self, parameters):
        missing = set(self.physical_parameter_names).difference(parameters)
        if missing:
            raise ValueError(f"Missing TF parameters: {sorted(missing)}")
        for key in self.physical_parameter_names:
            value = float(parameters[key])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite TF parameter {key}")
            setattr(self, key, value)
        for key in ("R", "CH", "alpha", "tau", "Gamma", "area_m2",
                    "concentration_mol_m3", "temp_K", "eps_r"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not (0 <= self.beta <= 1 and 0 < self.slow_fraction < 1
                and self.tau_ratio > 1 and self.g_frumkin > -4):
            raise ValueError("Invalid compact split, population fraction, time ratio or interaction")
        self.vt = 8.314462618 * self.temp_K / 96485.33212
        epsilon = 8.8541878128e-12 * self.eps_r
        self.qgc = np.sqrt(8 * epsilon * self.vt * 96485.33212 * self.concentration_mol_m3)
        self.site_charge_density = 96485.33212 * self.Gamma
        self.weights = np.array([1-self.slow_fraction, self.slow_fraction])
        self.times = self.tau * np.array([1., self.tau_ratio])

    @property
    def characteristic_time_s(self):
        """Scale for resolving model dynamics; full-state reset needs verification."""
        return max(self.times[1], self.R*self.area_m2*self.CH/2)

    def asdict(self):
        return {key: getattr(self, key) for key in self.physical_parameter_names}

    def _surface(self, state):
        theta = np.asarray(state[1:]).reshape(2, 2, 2)
        coverage = theta.sum(axis=2) @ self.weights
        q_ads = self.site_charge_density * ((theta[:, :, 0]-theta[:, :, 1]) @ self.weights)
        qmetal = self.charge_scale_c_m2 * state[0] * np.array([1., -1.])
        qd = qmetal + q_ads
        phi = 2*self.vt*np.arcsinh(qd/self.qgc) + qd*self.beta/self.CH
        eta = phi + qmetal*(1-self.beta)/self.CH
        return theta, coverage, phi, eta

    def current(self, state, voltage_v, voltage_rate_v_s=0):
        eta = self._surface(state)[3]
        return float((voltage_v-eta[0]+eta[1])/self.R)

    def charge(self, state, voltage_v=0):
        return float(self.area_m2*self.charge_scale_c_m2*state[0])

    def rhs(self, t, state, voltage_v):
        theta, coverage, phi, eta = self._surface(state)
        current = (voltage_v-eta[0]+eta[1])/self.R
        vacancy = 1-theta.sum(axis=2)
        exponent = (np.log(self.alpha) - phi[:, None]*np.array([1., -1.])/self.vt
                    - self.g_frumkin*coverage[:, None])
        # Same finite-iterate guard as the historical implicit solver.
        activity = np.exp(np.clip(exponent, -700., 60.))
        rates = (activity[:, None, :]*vacancy[:, :, None]-theta)/self.times[None, :, None]
        return np.r_[current/(self.area_m2*self.charge_scale_c_m2), rates.ravel()]

    def equilibrium(self, voltage_v):
        """Solve symmetric steady state by bracketing left displacement density."""
        voltage_v = float(voltage_v)
        if not np.isfinite(voltage_v):
            raise ValueError("Nonfinite voltage")
        def at_displacement(qd):
            phi = 2*self.vt*np.arcsinh(qd/self.qgc) + qd*self.beta/self.CH
            logs = np.log(self.alpha) + np.array([-phi, phi])/self.vt
            log_sum = np.logaddexp(*logs)
            occupied = brentq(lambda x: x-expit(log_sum-self.g_frumkin*x),
                              0., 1., xtol=5e-16)
            fractions = occupied*np.exp(logs-log_sum)
            qm = qd-self.site_charge_density*(fractions[0]-fractions[1])
            return phi+qm*(1-self.beta)/self.CH, qm, fractions
        target = abs(voltage_v)/2
        upper = max(self.qgc, self.CH*target)
        while at_displacement(upper)[0] < target:
            upper *= 2
            if not np.isfinite(upper):
                raise ValueError("Could not bracket equilibrium voltage")
        qd = brentq(lambda q: at_displacement(q)[0]-target, 0., upper, xtol=5e-22)
        _, qm, fractions = at_displacement(np.copysign(qd, voltage_v))
        return np.r_[qm/self.charge_scale_c_m2, np.tile(fractions, 2), np.tile(fractions[::-1], 2)]

    def validate_state(self, state):
        """Return True for nine finite variables and physically admissible coverages."""
        state = np.asarray(state)
        if state.shape != (9,) or not np.isfinite(state).all():
            return False
        theta = state[1:].reshape(4, 2)
        return bool(np.min(theta) >= -1e-8 and np.max(theta.sum(axis=1)) <= 1+1e-8)

    def normalized_state_error(self, state, target, initial):
        """Largest componentwise relative miss, with 1e-3 floors in state units."""
        target = np.asarray(target)
        scales = np.maximum(np.abs(target-np.asarray(initial)), 1e-3)
        return float(np.max(np.abs(np.asarray(state)-target)/scales))

    @staticmethod
    def historical_seed():
        """Return archived 0.1 M fit and geometry as initial guesses for calibration."""
        source = Path(__file__).with_name("historical_tf_seed.json")
        return json.loads(source.read_text(encoding="utf-8-sig"))["parameters"]
