import numpy as np
import pandas as pd
import json, os

#Parameters (calibrated on real data) or manually adjusted
"""if LOAD_FROM_PROJECT1 = True, the values Project 1 saved in model_params.json"""
LOAD_FROM_PROJECT1 = False
PROJECT1_PARAMS_FILE = "model_params.json"


"""default values"""
PARAMS = {
    #population (always manual)
    "prevalence_mut": 0.10,         #fraction of samples that carry the mutation (the prior)
    "n_samples": 5000,
    "vaf_max": 0.5,                 #heterozygous mutant: VAF max is 0.5
    "eff": 0.65,                    #ln(1 + E), E = 0.92
    "ct_wt_baseline": 25.0,         #arbitrary reference point
    #calibrated on Project 1 (overwritten when LOAD_FROM_PROJECT1 = True)
    "pure_mutant_delta_ct": 1.1415,  #GMM Mut cluster mean
    "background_mean": 15.2465,      #GMM WT cluster mean
    "background_std": 3.0197,         #WT std after removing the concentration-explained part
    "conc_slope": -3.8582,          #delta_ct vs log10 concentration, WT only
    "conc_ref": 67.01,               #concentration at which the background equals background_mean
    "quality_noise_std": 3.6243,      #residual std of ct_wt
    "signal_noise_std": 1.6598,       #real Mut cluster std
    "conc_min": 13, "conc_max": 412,
    "ct_max": 50,
    "real_wt_std": 3.0852, "real_mut_std": 1.6598,   #reference values for the calibration check
}
param_source = {k: "default" for k in PARAMS}

if LOAD_FROM_PROJECT1:
    if not os.path.exists(PROJECT1_PARAMS_FILE):
        raise FileNotFoundError(f"{PROJECT1_PARAMS_FILE} not found")
    with open(PROJECT1_PARAMS_FILE) as f:
        project1 = json.load(f)
    if "simulation_calibration" not in project1:
        raise KeyError("model_params.json has no 'simulation_calibration' section")
    for key, value in project1["simulation_calibration"].items():
        if key in PARAMS:
            PARAMS[key] = value
            param_source[key] = "Project 1"


"""manual adjustments, applied last (override both the defaults and the values loaded from Project 1)"""
MANUAL_OVERRIDES = {
    #"prevalence_mut": 0.07,
    #"background_std": 3.0,
}
for key, value in MANUAL_OVERRIDES.items():
    if key not in PARAMS:
        raise KeyError(f"unknown parameter: {key}")
    PARAMS[key] = value
    param_source[key] = "manual"

#keyword arguments for the generators
SWEEP_KWARGS = {k: PARAMS[k] for k in [
    "vaf_max", "eff", "pure_mutant_delta_ct", "background_mean", "background_std", "conc_slope", "conc_ref",
    "quality_noise_std", "signal_noise_std", "ct_wt_baseline", "conc_min", "conc_max", "ct_max"]}
GEN_KWARGS = {**SWEEP_KWARGS, "p_wt": 1 - PARAMS["prevalence_mut"]}

print(pd.DataFrame({"value": pd.Series(PARAMS, dtype=object), "source": param_source}).to_string())

with open("synthetic_params.json", "w") as f:
    json.dump({"param_source": param_source, "PARAMS": PARAMS,
               "SWEEP_KWARGS": SWEEP_KWARGS, "GEN_KWARGS": GEN_KWARGS}, f, indent=2, default=float)

#synthetic data generator
def _generate_vaf(n_samples, vaf_max, p_wt, rng):
    """
    Generates true Variant Allele Frequency per sample, creating a mixture of WT and mutant samples.
    p_wt is the probability of a sample being a true wild-type (VAF = 0, no mutant
    allele present), p_wt = 1 - prevalence, set in the parameter cell (PARAMS['prevalence_mut'])

    Mutant samples' VAF is drawn from Beta(a=6, b=2), scaled to vaf_max;
    skewed toward full heterozygosity, matching the spread of the mutant fragments from the cohort.
    WT samples are attributed a VAF = 0
    """
    is_wt = rng.random(n_samples) < p_wt
    continuous_vaf = rng.beta(a=6, b=2, size=n_samples) * vaf_max
    vaf = np.where(is_wt, 0.0, continuous_vaf)
    return vaf


def _generate_concentration(n_samples, conc_min, conc_max, rng):
    """
    Generates DNA concentration (ng/uL) per sample, log-uniform, matching
    roughly the observed range from real samples (13-412 ng/uL)
    """
    log_conc = rng.uniform(np.log10(conc_min), np.log10(conc_max), size=n_samples)
    return 10 ** log_conc


def _generate_ct_wt(n_samples, ct_wt_baseline, quality_noise_std, rng):
    """
    Generates Ct_wt per sample as a baseline plus per-sample quality_noise (related to sample quality & handling)
    Real data showed Ct_wt has essentially no dependence on initial DNA concentration 
    because of the subsequent dilution of all samples to a standard concentration before 
    performing the PCR reactions 
    """
    quality_noise = rng.normal(0, quality_noise_std, size=n_samples)
    return ct_wt_baseline + quality_noise


def _logit(x):
    return np.log(x / (1 - x))


def _generate_delta_ct_signal(vaf, pure_mutant_delta_ct, eff, signal_noise_std, n_samples, rng):
    """
    Computes delta_ct from the real VAF-dependent signal:
        delta_ct_signal = pure_mutant_delta_ct - logit(VAF) / eff
        eff = ln(1+E), where E represents PCR reaction efficiency (which we theoretically expect to be between 90-100%;
        here we arbitrarily assume E as being 92%, the resulting eff being 0.65)

    As VAF approaches vaf_max, this converges to pure_mutant_delta_ct (calibrated
    on the real Mut cluster mean from GMM). 
    As VAF approaches 0, delta_ct diverges to +infinity. True WT samples (VAF exactly 0) get +infinity directly,
    so the min() step below always picks the background for them.

    signal_noise_std - technical noise on the signal channel, calibrated
    empirically against the real Mut cluster std
    """
    delta_ct_signal = np.where(
        vaf > 0,
        pure_mutant_delta_ct - _logit(np.clip(vaf, 1e-10, None)) / eff,
        np.inf
    )
    noise = rng.normal(0, signal_noise_std, size=n_samples)
    return delta_ct_signal + noise


def _generate_delta_ct_background(n_samples, dna_conc, background_mean, background_std, conc_slope, conc_ref, rng):
    """
    Background noise (non-specific amplification), dependent on
    concentration, calibrated on a real data regression (delta_ct vs
    log10(concentration), WT-only patients)

    At dna_conc == conc_ref, the background noise is exactly background_mean 
    (the real WT cluster delta_ct mean from GMM)
    """
    log_conc = np.log10(dna_conc)
    log_conc_ref = np.log10(conc_ref)
    mean_shift = conc_slope * (log_conc - log_conc_ref) #the residual effect of initial dna concentration on background amplification
    noise = rng.normal(0, background_std, size=n_samples)
    return background_mean + mean_shift + noise


def _simulate_ct_values(
    vaf, dna_conc,
    ct_wt_baseline, quality_noise_std,
    pure_mutant_delta_ct, eff, signal_noise_std,
    background_mean, background_std, conc_slope, conc_ref,
    ct_max, rng
):
    """
    Generates the Ct values for both the WT and mutant channels, as well as the ΔCt values, 
    based on the true VAFs, DNA concentrations, and the various noise parameters. 
    It also applies censoring to the Ct values above the detection limit
    """
    n_samples = len(vaf)

    ct_wt = _generate_ct_wt(n_samples, ct_wt_baseline, quality_noise_std, rng)

    delta_ct_signal = _generate_delta_ct_signal(vaf, pure_mutant_delta_ct, eff, signal_noise_std, n_samples, rng)
    delta_ct_background = _generate_delta_ct_background(
        n_samples, dna_conc, background_mean, background_std, conc_slope, conc_ref, rng
    )
    delta_ct = np.minimum(delta_ct_signal, delta_ct_background)

    ct_mut = ct_wt + delta_ct

    ct_wt_censored = ct_wt > ct_max
    ct_mut_censored = ct_mut > ct_max

    return pd.DataFrame({
        "true_vaf": vaf,
        "dna_conc": dna_conc,
        "ct_wt": np.where(ct_wt_censored, np.nan, ct_wt),
        "ct_mut": np.where(ct_mut_censored, np.nan, ct_mut),
        "delta_ct": np.where(ct_wt_censored | ct_mut_censored, np.nan, delta_ct),
        "ct_wt_censored": ct_wt_censored,
        "ct_mut_censored": ct_mut_censored,
    })


def generate_synthetic_qpcr_data(
    n_samples, *,
    vaf_max, p_wt, eff, pure_mutant_delta_ct,
    background_mean, background_std, conc_slope, conc_ref,
    quality_noise_std, signal_noise_std, ct_wt_baseline,
    conc_min, conc_max, ct_max,
    random_seed=42,
):
    """Generates a synthetic qPCR dataset"""
    rng = np.random.default_rng(random_seed)

    vaf = _generate_vaf(n_samples, vaf_max, p_wt, rng)
    dna_conc = _generate_concentration(n_samples, conc_min, conc_max, rng)

    df = _simulate_ct_values(
        vaf, dna_conc,
        ct_wt_baseline, quality_noise_std,
        pure_mutant_delta_ct, eff, signal_noise_std,
        background_mean, background_std, conc_slope, conc_ref,
        ct_max, rng,
    )
    df.insert(0, "sample_id", [f"SYN-{i:04d}" for i in range(n_samples)])
    return df


def generate_vaf_sweep_data(
    *,
    vaf_max, eff, pure_mutant_delta_ct,
    background_mean, background_std, conc_slope, conc_ref,
    quality_noise_std, signal_noise_std, ct_wt_baseline,
    conc_min, conc_max, ct_max,
    n_replicates=50, vaf_grid=None, n_grid_points=25,
    random_seed=123,
):
    """
    Generates a VAF dilution-series-style dataset: a fixed grid of known
    VAF values (log-spaced, densely covering the low-VAF range near the
    detection boundary), each replicated n_replicates times.
    """
    rng = np.random.default_rng(random_seed)

    if vaf_grid is None:
        log_grid = np.linspace(np.log10(1e-5), np.log10(vaf_max), n_grid_points)
        vaf_grid = np.concatenate([[0.0], 10 ** log_grid])

    vaf = np.repeat(vaf_grid, n_replicates)
    n_samples = len(vaf)

    dna_conc = _generate_concentration(n_samples, conc_min, conc_max, rng)

    df = _simulate_ct_values(
        vaf, dna_conc,
        ct_wt_baseline, quality_noise_std,
        pure_mutant_delta_ct, eff, signal_noise_std,
        background_mean, background_std, conc_slope, conc_ref,
        ct_max, rng,
    )
    df.insert(0, "sample_id", [f"SWEEP-{i:04d}" for i in range(n_samples)])
    return df


#Generate the synthetic population and the VAF sweep dataset
"""Synthetic population, used for classification, PCA, calibration, 
and as the fixed classifier training set for downstream analyses"""
df = generate_synthetic_qpcr_data(PARAMS["n_samples"], **GEN_KWARGS)
print(df["true_vaf"].describe())
print(df[["ct_wt", "ct_mut", "delta_ct"]].describe())

"""right-censoring of undetermined Ct values: an amplification that never crossed
the threshold within the run (CT_MAX cycles) is set to CT_MAX instead of dropping the fragment"""
CT_MAX = float(PARAMS["ct_max"])
df["ct_mut_undetermined"] = df["ct_mut"].isna()
df["ct_wt_undetermined"] = df["ct_wt"].isna()
df["ct_mut"] = df["ct_mut"].fillna(CT_MAX)
df["ct_wt"] = df["ct_wt"].fillna(CT_MAX)
df["delta_ct"] = df["ct_mut"] - df["ct_wt"]
print(f"\nFragments with right-censored Ct values: {(df['ct_mut_undetermined'] | df['ct_wt_undetermined']).sum()} / {len(df)}")
df.to_csv("synthetic_qpcr_data.csv", index=False)
print("\nSaved as synthetic_qpcr_data.csv")

df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)


"""VAF sweep (dilution-series style, known grid, dense near the detection boundary), 
used for continuous VAF regression and sensitivity analysis"""
df_sweep = generate_vaf_sweep_data(n_replicates=50, n_grid_points=50, **SWEEP_KWARGS)
df_sweep.to_csv("vaf_sweep_data.csv", index=False)
print("\nSaved as vaf_sweep_data.csv")
df_sweep = df_sweep.dropna(subset=["delta_ct"]).reset_index(drop=True)
print(f"Sweep dataset: {df_sweep['true_vaf'].nunique()} VAF grid points, {len(df_sweep)} total samples")
