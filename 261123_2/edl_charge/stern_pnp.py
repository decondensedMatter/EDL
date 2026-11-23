"""Closed, planar, two-blocking-contact Stern–PNP for one monovalent positive carrier.

State entries are cell-average concentrations [mol/m³]. Positive terminal current
enters the left metal, whose voltage relative to right is the command voltage.
Optional finite site density gives ideal lattice mixing and vacancy exclusion.
The dilute branch uses ideal mixing. Neither branch includes electronic leakage.
"""
import numpy as np
from scipy.constants import epsilon_0, R, physical_constants
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, xlogy
from scipy.optimize import least_squares

F = physical_constants["Faraday constant"][0]


def _bernoulli(x):
    """x/expm1(x), including the removable singularity and both extreme tails."""
    x = np.asarray(x)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    large = x > 50
    reg = ~(large | small)
    out[small] = 1 - x[small] / 2 + x[small] ** 2 / 12
    out[large] = x[large] * np.exp(-x[large]) / (-np.expm1(-x[large]))
    out[reg] = x[reg] / np.expm1(x[reg])
    return out


class SternPNPModel:
    def __init__(self, parameters):
        self.parameters = dict(parameters)

        def positive(key):
            v = float(parameters[key])
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be positive and finite")
            return v

        self.area = positive("area_m2")
        self.length = positive("length_m")
        self.D = positive("D_m2_s")
        self.c0 = positive("concentration_mol_m3")
        self.epsilon = epsilon_0 * positive("eps_r")
        self.cs = positive("stern_capacitance_f_m2")  # one contact
        self.rt = R * positive("temp_K")
        self.vt = self.rt / F
        self.site = (positive("site_concentration_mol_m3")
                     if parameters.get("site_concentration_mol_m3") is not None else np.inf)
        if self.site <= self.c0:
            raise ValueError("site concentration must exceed the neutral concentration")
        n = int(parameters.get("cells", 32))
        if n < 4 or n != parameters.get("cells", 32):
            raise ValueError("cells must be an integer >= 4")
        self.cells = n
        chi = self.c0 * (1 - self.c0 / self.site)
        self.debye_length_m = np.sqrt(self.epsilon * self.vt / (F * chi))
        grade = float(parameters.get("mesh_grading", max(0, np.log(self.length / self.debye_length_m))))
        if not np.isfinite(grade) or not 0 <= grade <= 35:
            raise ValueError("mesh_grading must be in [0,35]")
        s = np.linspace(0, 1, n + 1)
        u = 2 * np.minimum(s, 1 - s)
        half = 0.5 * self.length * (np.expm1(grade * u) / np.expm1(grade) if grade > 1e-8 else u)
        faces = np.where(s <= 0.5, half, self.length - half)
        self.faces = faces
        self.dx = np.diff(faces)
        self.x = 0.5 * (faces[1:] + faces[:-1])
        self.spacing = np.diff(self.x)
        if np.any(self.dx <= 0) or np.any(self.spacing <= 0):
            raise ValueError("mesh resolution exceeds floating point precision")
        # Conductances here are dielectric capacitances per area of adjacent faces.
        g = self.epsilon / self.spacing
        gl, gr = 1 / (1 / self.cs + self.dx[[0, -1]] / (2 * self.epsilon))
        self.edge = np.r_[gl, g, gr]
        self.K = np.diag(self.edge[:-1] + self.edge[1:]) - np.diag(g, 1) - np.diag(g, -1)
        self._chol = cho_factor(self.K)
        drive = np.zeros(n)
        drive[[0, -1]] = [gl / 2, -gr / 2]
        self._field = cho_solve(self._chol, drive)
        self._q_y = -gl * cho_solve(self._chol, np.eye(n)[0]) * F * self.dx
        self._q_v = 1 / (2 / self.cs + self.length / self.epsilon)
        sigma = self.D * F * chi / self.vt
        self.characteristic_time_s = self.length / sigma / (2 / self.cs + 2 * self.debye_length_m / self.epsilon)

    def _potential(self, state, v):
        rho = F * self.dx * (np.asarray(state) - self.c0)
        return cho_solve(self._chol, rho) + self._field * v

    def charge(self, state, voltage_v=0):
        return float(self.area * (self._q_y @ (np.asarray(state) - self.c0) + self._q_v * voltage_v))

    def rhs(self, t, state, voltage_v):
        c = np.asarray(state)
        v = voltage_v(t) if callable(voltage_v) else voltage_v
        p = np.diff(self._potential(c, v)) / self.vt
        left, right = c[:-1], c[1:]
        flux = self.D / self.spacing * (
            _bernoulli(p) * left * (1 - right / self.site)
            - _bernoulli(-p) * right * (1 - left / self.site))
        return -np.diff(np.r_[0, flux, 0]) / self.dx

    def current(self, state, voltage_v, voltage_rate_v_s=0):
        return float(self.area * (self._q_y @ self.rhs(0, state, voltage_v)
                                  + self._q_v * voltage_rate_v_s))

    def equilibrium(self, voltage_v):
        """Solve discrete Poisson–Boltzmann with a mass multiplier and blocking contacts."""
        v = float(voltage_v)
        if not np.isfinite(v):
            raise ValueError("nonfinite voltage")
        if v == 0:
            return np.full(self.cells, self.c0)
        finite_site = np.isfinite(self.site)
        if finite_site:
            log_activity = np.log(self.c0 / (self.site - self.c0))

        def population(w):
            if finite_site:
                c = self.site * expit(log_activity + w)
                return c, c * (1 - c / self.site)
            # C1 extension for distant solver trials. Accepted equilibria must
            # lie below the extension and satisfy the unmodified dilute law.
            hi = 80.
            c = self.c0 * np.exp(np.minimum(w, hi))
            return c * (1 + np.maximum(w - hi, 0)), c

        fdx = F * self.dx
        weight = self.dx / (self.length * self.c0)
        scale = self.K.diagonal() * self.vt + fdx * self.c0
        state = np.zeros(self.cells + 1)

        def residual(a):
            c, _ = population(a[-1] - a[:-1])
            r = self.K @ (a[:-1] * self.vt) - drive - fdx * (c - self.c0)
            return np.r_[r / scale, weight @ (c - self.c0)]

        def jac(a):
            _, dc = population(a[-1] - a[:-1])
            j = np.empty((self.cells + 1, self.cells + 1))
            j[:-1, :-1] = (self.vt * self.K + np.diag(fdx * dc)) / scale[:, None]
            j[:-1, -1] = -fdx * dc / scale
            j[-1, :-1] = -weight * dc
            j[-1, -1] = weight @ dc
            return j

        for voltage in np.linspace(0, v, 1 + max(1, int(np.ceil(abs(v) / (8 * self.vt)))))[1:]:
            drive = np.zeros(self.cells)
            drive[[0, -1]] = [voltage * self.edge[0] / 2, -voltage * self.edge[-1] / 2]
            fit = least_squares(residual, state, jac=jac, xtol=2e-14, ftol=2e-14, gtol=2e-14)
            state = fit.x
            if np.linalg.norm(residual(state), np.inf) > 1e-10:
                raise ValueError("equilibrium solve failed; inspect site parameters and refine mesh")
        w = state[-1] - state[:-1]
        if not finite_site and np.max(w) >= 80:
            raise ValueError("dilute equilibrium exceeds numerical domain")
        c, _ = population(w)
        if not self.validate_state(c):
            raise ValueError("equilibrium violates inventory or concentration bounds")
        return c

    def remap_state(self, state, destination):
        """Conservative piecewise-constant cell-average transfer to another mesh."""
        if self.length != destination.length or not self.validate_state(state):
            raise ValueError("remapping requires equal lengths and a valid source state")
        left = np.maximum(destination.faces[:-1, None], self.faces[None, :-1])
        right = np.minimum(destination.faces[1:, None], self.faces[None, 1:])
        return (np.maximum(0, right - left) @ np.asarray(state)) / destination.dx

    def normalized_state_error(self, state, target, initial):
        """Volume-weighted L2 error divided by the target excursion."""
        norm = lambda y: np.sqrt(self.dx @ np.square(y))
        err = norm(np.asarray(state) - target)
        span = norm(np.asarray(target) - initial)
        return float(err / span if span else (0 if err == 0 else np.inf))

    def validate_state(self, state):
        """True for finite physical concentrations with conserved mobile inventory."""
        c = np.asarray(state)
        return bool(c.shape == self.dx.shape and np.all(np.isfinite(c)) and np.all(c >= 0)
                    and np.all(c <= self.site)
                    and abs(self.dx @ (c - self.c0)) <= 1e-8 * self.length * self.c0)

    def energy(self, state, voltage_v):
        """Discrete stored field (including both Stern layers) and mixing energy [J]."""
        c = np.asarray(state)
        field = self._potential(c, voltage_v)
        drops = np.diff(np.r_[voltage_v / 2, field, -voltage_v / 2])
        mixing = xlogy(c, c / self.c0) - c + self.c0
        if np.isfinite(self.site):
            holes = self.site - c
            mixing += xlogy(holes, holes / (self.site - self.c0)) - holes + self.site - self.c0
        return float(self.area * (0.5 * self.edge @ drops**2 + self.rt * self.dx @ mixing))

