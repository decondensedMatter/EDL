"""Unverified numerical search defaults and their calibration provenance."""
from copy import deepcopy
import json
from pathlib import Path


def model_fit_defaults(config):
    """Populate unverified optimizer starting values when creating a fresh session template."""
    cfg = deepcopy(config)
    qualification=cfg.setdefault("qualification",{})
    for key,value in dict(max_validation_noise_rmse=3.,max_condition_number=1e6,
                          max_sample_voltage_error_fraction=.02).items():
        if qualification.get(key) is None:
            qualification[key]=value
    physical = cfg["model_parameters"]
    if cfg["model"]=="tf":
        history = json.loads(Path(__file__).with_name("historical_tf_seed.json").read_text(encoding="utf-8"))
        seeds = dict(history["parameters"])
        is_source_concentration = physical.get("concentration_mol_m3")==seeds["concentration_mol_m3"]
        # Preserve measured/nominal specimen inputs. Gamma and beta were fixed
        # conventions of the historical reduced model, not fitted observables.
        assumptions = {}
        for name in ("Gamma","beta"):
            if physical.get(name) is None and name not in (cfg.get("parameter_bounds") or {}):
                assumptions[name] = physical[name] = seeds[name]
        bounds = dict(R=[300.,3000.] if is_source_concentration else [3.,300000.],CH=[.01,1.],
                      alpha=[1e-7,10.],tau=[1e-7,.02],slow_fraction=[.02,.98],
                      tau_ratio=[1.001,.2/seeds["tau"]],g_frumkin=[-3.5,20.],eps_r=[1.,120.])
        # The original slow_fraction and g fits approached opposite boundaries.
        # Start just inside both, allowing finite differences in either direction.
        seeds["slow_fraction"] = min(seeds["slow_fraction"],.979)
        seeds["g_frumkin"] = max(seeds["g_frumkin"],-3.49)
        provenance = dict(model="TF",historical_seed=history["source"],seed_sha256=history["source_sha256"],
            historical_bounds_source="Historical TF parameter-search bounds",
            fixed_conventions=assumptions,transferred_across_concentrations=not is_source_concentration,
            scope="Local historical 0.1 M fit initialization. Nominal concentration and area remain specimen-specific.",
            bounds_notes="Historical R, CH, alpha, tau, fraction, and interaction intervals; transfer R range widened 100-fold each way from the source interval. Ratio search uses 0.2 s / historical fast tau as its independent upper limit. Relative permittivity interval [1,120] is an exploratory positive prior.")
        linear = ["slow_fraction","g_frumkin"]
    elif cfg["model"]=="stern_pnp":
        seeds = dict(D_m2_s=1e-11,concentration_mol_m3=1.,eps_r=30.,stern_capacitance_f_m2=.08)
        bounds = dict(D_m2_s=[1e-20,1e-8],concentration_mol_m3=[1e-4,1e5],eps_r=[1.,300.],
                      stern_capacitance_f_m2=[8e-4,8.])
        provenance = dict(model="blocking Stern-PNP",seed_source="edl_charge/test_stern_pnp.py: synthetic P",
            scope="Numerical seeds from a synthetic solver test, used for both solids. No specimen material constants or geometry inferred.",
            bounds_notes="Exploratory positive log-space searches around test seeds. Site concentration left unspecified selects the dilute closure unless an independently supported site density is supplied.")
        linear = []
    else:
        raise ValueError("Unknown specimen model family")
    # Existing measured values and explicit fit selections take precedence.
    defaults_selected = not cfg.get("parameter_bounds")
    if defaults_selected:
        cfg["parameter_bounds"] = {k:v for k,v in bounds.items() if physical.get(k) is None}
        cfg["linear_parameters"] = [k for k in linear if k in cfg["parameter_bounds"]]
    existing = cfg.get("initial_guess") or {}
    guess = dict(existing)
    for name in cfg["parameter_bounds"]:
        if name in seeds and (guess.get(name) is None or
                (defaults_selected and cfg["model"]=="tf" and name in ("g_frumkin","slow_fraction")
                 and guess[name]==history["parameters"][name])):
            guess[name] = seeds[name]
    cfg["initial_guess"] = guess
    cfg["fit_initialization"] = dict(provenance,defaults_selected=defaults_selected,
        status="unverified_initialization",requires_independent_validation=True)
    return cfg

