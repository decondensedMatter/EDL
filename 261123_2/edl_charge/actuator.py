"""Optional measured first-order command-to-terminal voltage approximation."""
import numpy as np


class ActuatedModel:
    """State [cell state..., actual terminal voltage]. Qualify the filter on held-out steps."""
    def __init__(self,cell_model,rise_time_s):
        self.cell_model = cell_model
        self.rise_time_s = float(rise_time_s)
        if not np.isfinite(self.rise_time_s) or self.rise_time_s<=0:
            raise ValueError("a positive measured 10–90% rise time is required")
        self.tau_s = self.rise_time_s/np.log(9.)
        self.characteristic_time_s = max(getattr(cell_model,"characteristic_time_s",0.),self.tau_s)

    def equilibrium(self,voltage_v):
        return np.r_[self.cell_model.equilibrium(voltage_v),float(voltage_v)]

    def terminal_voltage(self,state,command):
        return float(state[-1])

    def rhs(self,t,state,voltage_v):
        v = state[-1]
        return np.r_[self.cell_model.rhs(t,state[:-1],v),(voltage_v-v)/self.tau_s]

    def charge(self,state,voltage_v=0):
        return self.cell_model.charge(state[:-1],state[-1])

    def current(self,state,voltage_v,voltage_rate_v_s=0):
        return self.cell_model.current(state[:-1],state[-1],(voltage_v-state[-1])/self.tau_s)

    def normalized_state_error(self,state,target,initial):
        cell_error = self.cell_model.normalized_state_error(state[:-1],target[:-1],initial[:-1])
        delta = abs(target[-1]-initial[-1])
        e = abs(state[-1]-target[-1])
        return float(max(cell_error,e/delta if delta else (0 if e==0 else np.inf)))

    def validate_state(self,state):
        return bool(np.all(np.isfinite(state)) and
                    (not hasattr(self.cell_model,"validate_state") or self.cell_model.validate_state(state[:-1])))

    def refined(self):
        """Double the underlying spatial grid, if present, keeping the measured filter."""
        m = self.cell_model
        if hasattr(m,"refined"):
            finer = m.refined()
        elif hasattr(m,"remap_state") and hasattr(m,"parameters"):
            finer = type(m)(dict(m.parameters,cells=2*m.cells))
        else:
            return None
        return ActuatedModel(finer,self.rise_time_s) if finer is not None else None
