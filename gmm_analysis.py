import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
import json, os
from scipy.optimize import brentq

with open("synthetic_params.json") as f:
    _p = json.load(f)
param_source, PARAMS, SWEEP_KWARGS, GEN_KWARGS = _p["param_source"], _p["PARAMS"], _p["SWEEP_KWARGS"], _p["GEN_KWARGS"]

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


df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)
df_sweep = pd.read_csv("vaf_sweep_data.csv")
df_sweep = df_sweep.dropna(subset=["delta_ct"]).reset_index(drop=True)


#GMM
X = df["delta_ct"].values.reshape(-1, 1)

#GMM with 2 components
gmm2 = GaussianMixture(n_components=2, random_state=42)
gmm2.fit(X)

print("GMM with 2 components:")
print("Component means:", np.round(gmm2.means_.ravel(), 3))
print("Component variances:", np.round(gmm2.covariances_.ravel(), 3))
print("Component weights:", np.round(gmm2.weights_, 3))

mutant_component = np.argmin(gmm2.means_.ravel())
cluster_labels = gmm2.predict(X)
df["gmm_label"] = np.where(cluster_labels == mutant_component, "Mut", "WT")

#fitted GMM parameters
mut_mean = gmm2.means_.ravel()[mutant_component]
mut_var = gmm2.covariances_.ravel()[mutant_component]
mut_weight = gmm2.weights_[mutant_component]
wt_mean = gmm2.means_.ravel()[1 - mutant_component]
wt_var = gmm2.covariances_.ravel()[1 - mutant_component]
wt_weight = gmm2.weights_[1 - mutant_component]


#BIC sweep for k=1 to 5
print("\nBIC sweep, k=1 to 5:")
bic_scores = []
for k in range(1, 6):
    gmm_k = GaussianMixture(n_components=k, random_state=42)
    gmm_k.fit(X)
    score = gmm_k.bic(X)
    bic_scores.append(score)
    print(f"k={k}: BIC={score:.2f}")
best_k = np.argmin(bic_scores) + 1
print(f"Best k according to BIC: {best_k}")

#plotting BIC scores
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(range(1, 6), bic_scores, marker="o", color="black")
ax.axvline(2, color="steelblue", linestyle="--", alpha=0.6, label="k=2 (biologically motivated)")
ax.set_xlabel("Number of GMM components (k)")
ax.set_ylabel("BIC (lower = better)")
ax.set_title("BIC across candidate numbers of clusters, synthetic data")
ax.legend()
plt.tight_layout()
plt.savefig("bic_sweep_synthetic.png", dpi=150)
print("Saved as bic_sweep_synthetic.png")
plt.show()


#GMM accuracy against true genotype
print("\nGMM vs true genotype:")
print(f"Accuracy: {accuracy_score(df['true_label'], df['gmm_label']):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(df["true_label"], df["gmm_label"], labels=["Mut", "WT"]))
print("\nClassification report:")
print(classification_report(df["true_label"], df["gmm_label"]))
print("\nMisclassified:")
misclassified = df[df["true_label"] != df["gmm_label"]]
print(misclassified[["sample_id", "true_vaf", "delta_ct", "true_label", "gmm_label"]])

#GMM decision boundary: where the weighted component densities cross (P(Mut) = 0.5)
x_search = np.linspace(mut_mean, wt_mean, 10000)
dens_mut = mut_weight * norm.pdf(x_search, mut_mean, np.sqrt(mut_var))
dens_wt = wt_weight * norm.pdf(x_search, wt_mean, np.sqrt(wt_var))
gmm_boundary = x_search[np.argmin(np.abs(dens_mut - dens_wt))]
print(f"\nGMM decision boundary: delta_ct = {gmm_boundary:.3f}")

#Distribution plot with fitted GMM components overlaid
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(df["delta_ct"], bins=40, density=True, alpha=0.4, color="gray", label="observed data")

x_range = np.linspace(df["delta_ct"].min() - 2, df["delta_ct"].max() + 2, 500)
component_info = [
    ("Mut component", "red", mutant_component),
    ("WT component", "steelblue", 1 - mutant_component),
]
for label, color, comp_idx in component_info:
    mean = gmm2.means_.ravel()[comp_idx]
    std = np.sqrt(gmm2.covariances_.ravel()[comp_idx])
    weight = gmm2.weights_[comp_idx]
    curve = weight * norm.pdf(x_range, mean, std)
    ax.plot(x_range, curve, color=color, linewidth=2, label=label)

ax.axvline(gmm_boundary, color="black", linestyle="--", label=f"boundary = {gmm_boundary:.2f}")
ax.set_xlabel("delta_ct")
ax.set_ylabel("density")
ax.set_title("GMM fit on synthetic delta_ct distribution")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_fit_synthetic.png", dpi=150)
print("\nSaved as gmm_fit_synthetic.png")
plt.show()

df.to_csv("gmm_classified_synthetic.csv", index=False)
print("Saved as gmm_classified_synthetic.csv")


#Prevalence scenarios - class-conditional densities fitted on the true labels, prior set explicitly
"""The unsupervised GMM has to find the Mut cluster by itself, and it learns the Mut weight from the sample.
Since in this simulation the true labels are known, here each class gets its own normal density, and the prior
(the prevalence) is stated instead of learned. The boundary is where prior_mut * f_mut(x) = (1 - prior_mut) * f_wt(x).
For every prevalence a new population is generated at that prevalence (training set), and sensitivity, specificity and PPV
are measured on a second, independent population (test set)."""


def make_dataset(prevalence, n, seed):
    d = generate_synthetic_qpcr_data(n, **{**GEN_KWARGS, "p_wt": 1 - prevalence, "random_seed": seed})
    d = d.dropna(subset=["delta_ct"]).reset_index(drop=True)
    d["is_mut"] = d["true_vaf"] > 0
    return d


def bayes_boundary(prior_mut, mut_mean, mut_sd, wt_mean, wt_sd):
    """delta_ct where the prior-weighted Mut and WT densities are equal, searched between the two class means"""
    f = lambda t: prior_mut * norm.pdf(t, mut_mean, mut_sd) - (1 - prior_mut) * norm.pdf(t, wt_mean, wt_sd)
    return brentq(f, mut_mean, wt_mean)


PREVALENCE_GRID = sorted(set([0.50, 0.40, 0.30, 0.25, 0.20, 0.14, 0.10, 0.05, 0.03, 0.01, PARAMS["prevalence_mut"]]), reverse=True)

rows = []
for prev in PREVALENCE_GRID:
    train = make_dataset(prev, 5000, seed=42)
    test = make_dataset(prev, 20000, seed=7)

    #supervised: one normal density per class, fitted on the true labels
    mut_x = train.loc[train["is_mut"], "delta_ct"]
    wt_x = train.loc[~train["is_mut"], "delta_ct"]
    boundary = bayes_boundary(prev, mut_x.mean(), mut_x.std(), wt_x.mean(), wt_x.std())

    #performance on the independent test population
    sens = (test.loc[test["is_mut"], "delta_ct"] < boundary).mean()
    spec = (test.loc[~test["is_mut"], "delta_ct"] >= boundary).mean()
    ppv = prev * sens / (prev * sens + (1 - prev) * (1 - spec))

    #unsupervised GMM on the same training data (no labels), for comparison
    gmm_p = GaussianMixture(n_components=2, random_state=42).fit(train[["delta_ct"]].values)
    k = np.argmin(gmm_p.means_.ravel())

    rows.append({
        "prevalence": prev, "n_mut_train": int(train["is_mut"].sum()),
        "boundary": boundary, "sensitivity": sens, "specificity": spec, "PPV": ppv,
        "GMM_mut_mean": gmm_p.means_.ravel()[k], "GMM_mut_weight": gmm_p.weights_[k],
    })

prevalence_table = pd.DataFrame(rows)
print(f"Supervised class densities + stated prior, vs unsupervised GMM (current PARAMS prevalence = {PARAMS['prevalence_mut']:.0%}):\n")
print(prevalence_table.round(3).to_string(index=False))
print("\nThe unsupervised GMM is correct only while its Mut mean stays near the true Mut cluster and its Mut weight near the prevalence.")

#plot 1: decision boundary as a function of prevalence
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prevalence_table["prevalence"], prevalence_table["boundary"], marker="o", color="black")
ax.set_xscale("log")
ax.set_xlabel("Mut prevalence (log scale)")
ax.set_ylabel("decision boundary (delta_ct)")
ax.set_title("Decision boundary vs prevalence")
plt.tight_layout()
plt.savefig("prevalence_boundary.png", dpi=150)
print("\nSaved as prevalence_boundary.png")
plt.show()

#plot 2: sensitivity, specificity and PPV as a function of prevalence
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prevalence_table["prevalence"], prevalence_table["sensitivity"], marker="o", color="red", label="sensitivity")
ax.plot(prevalence_table["prevalence"], prevalence_table["specificity"], marker="o", color="steelblue", label="specificity")
ax.plot(prevalence_table["prevalence"], prevalence_table["PPV"], marker="o", color="green", label="PPV")
ax.set_xscale("log")
ax.set_xlabel("Mut prevalence (log scale)")
ax.set_ylabel("value on the independent test population")
ax.set_title("Sensitivity, specificity and PPV vs prevalence")
ax.legend()
plt.tight_layout()
plt.savefig("prevalence_performance.png", dpi=150)
print("\nSaved as prevalence_performance.png")
plt.show()


#plain GMM vs initialised GMM (both unsupervised) vs supervised benchmark, across prevalence
rows = []
for prev in PREVALENCE_GRID:
    train = make_dataset(prev, 5000, seed=42)
    test = make_dataset(prev, 20000, seed=7)
    y_test = test["is_mut"].values

    #supervised benchmark: class densities fitted on the true labels + the stated prior
    mut_x = train.loc[train["is_mut"], "delta_ct"]
    wt_x = train.loc[~train["is_mut"], "delta_ct"]
    boundary = bayes_boundary(prev, mut_x.mean(), mut_x.std(), wt_x.mean(), wt_x.std())
    sup_pred = test["delta_ct"].values < boundary
    results = {"supervised": (sup_pred, mut_x.mean(), prev)}

    #two unsupervised GMMs, no labels: plain (default start) and initialised (with the real data means + prevalence as weights)
    gmm_settings = {
        "plain GMM": {},
        "init GMM": {"means_init": [[PARAMS["pure_mutant_delta_ct"]], [PARAMS["background_mean"]]],
                     "weights_init": [prev, 1 - prev]},
    }
    for name, init in gmm_settings.items():
        g = GaussianMixture(n_components=2, random_state=42, **init).fit(train[["delta_ct"]].values)
        k = np.argmin(g.means_.ravel())
        results[name] = (g.predict(test[["delta_ct"]].values) == k, g.means_.ravel()[k], g.weights_[k])

    for name, (pred, mut_mean_m, mut_weight_m) in results.items():
        rows.append({
            "prevalence": prev, "method": name,
            "sensitivity": pred[y_test].mean(),
            "specificity": (~pred[~y_test]).mean(),
            "accuracy": (pred == y_test).mean(),
            "mut_mean": mut_mean_m, "mut_weight": mut_weight_m,
        })

compare_table = pd.DataFrame(rows)
METHODS = ["supervised", "init GMM", "plain GMM"]
wide = compare_table.pivot(index="prevalence", columns="method").sort_index(ascending=False)
print("Supervised benchmark vs initialised GMM vs plain GMM:\n")
print(wide[["accuracy", "mut_mean", "mut_weight"]].round(3)
      .reindex(columns=pd.MultiIndex.from_product([["accuracy", "mut_mean", "mut_weight"], METHODS])).to_string())

for name in ["init GMM", "plain GMM"]:
    gap = wide["accuracy"]["supervised"] - wide["accuracy"][name]
    print(f"Largest accuracy gap to the benchmark, {name}: {gap.max():.4f} (at prevalence {gap.idxmax():.0%})")

STYLES = {"supervised": ("-o", "black"), "init GMM": ("--s", "tab:green"), "plain GMM": (":^", "tab:orange")}

#plot 1: sensitivity, all methods
fig, ax = plt.subplots(figsize=(8, 5))
for name in METHODS:
    sub = compare_table[compare_table["method"] == name].sort_values("prevalence")
    style, color = STYLES[name]
    ax.plot(sub["prevalence"], sub["sensitivity"], style, color=color, label=name)
ax.set_xscale("log")
ax.set_xlabel("prevalence of the mutation")
ax.set_ylabel("sensitivity")
ax.set_title("Sensitivity on independent test set")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_vs_supervised_sensitivity.png", dpi=150)
print("\nSaved as gmm_vs_supervised_sensitivity.png")
plt.show()

#plot 2: specificity, all methods
fig, ax = plt.subplots(figsize=(8, 5))
for name in METHODS:
    sub = compare_table[compare_table["method"] == name].sort_values("prevalence")
    style, color = STYLES[name]
    ax.plot(sub["prevalence"], sub["specificity"], style, color=color, label=name)
ax.set_xscale("log")
ax.set_xlabel("prevalence of the mutation")
ax.set_ylabel("specificity")
ax.set_title("Specificity on independent test set")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_vs_supervised_specificity.png", dpi=150)
print("\nSaved as gmm_vs_supervised_specificity.png")
plt.show()

#plot 3: where each GMM puts the Mut cluster mean, vs the true Mut cluster mean
fig, ax = plt.subplots(figsize=(8, 5))
for name in METHODS:
    sub = compare_table[compare_table["method"] == name].sort_values("prevalence")
    style, color = STYLES[name]
    ax.plot(sub["prevalence"], sub["mut_mean"], style, color=color, label="true Mut mean" if name == "supervised" else f"{name} Mut mean")
ax.set_xscale("log")
ax.set_xlabel("prevalence of the mutation")
ax.set_ylabel("Mut cluster mean (delta_ct)")
ax.set_title("GMM Mut cluster mean comparison")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_vs_supervised_mutmean.png", dpi=150)
print("\nSaved as gmm_vs_supervised_mutmean.png")
plt.show()


#Calibration curve for the GMM classifier
"""Comparing the predicted probabilities of the classifier to the actual observed frequencies"""

def posterior_mut_prob(delta_ct):
    """P(Mut | delta_ct) - posterior probability of a sample being mutant given its delta_ct value, 
    according to the already fitted GMM"""
    density_mut = mut_weight * norm.pdf(delta_ct, mut_mean, np.sqrt(mut_var))
    density_wt = wt_weight * norm.pdf(delta_ct, wt_mean, np.sqrt(wt_var))
    return density_mut / (density_mut + density_wt)

df["p_mut"] = posterior_mut_prob(df["delta_ct"].values)

"""Calibration curve - bin by predicted probability, compare to actual fraction Mut in each bin. 
We divide the probability range [0, 1] into 20 bins.
The bins are chosen so that each one contains approximately the same number of samples (strategy="quantile")
"""
prob_true, prob_pred = calibration_curve(
    df["true_label_binary"], df["p_mut"], n_bins=20, strategy="quantile"
)

print("Calibration curve (predicted probability vs observed frequency):")
for pt, pp in zip(prob_pred, prob_true):
    print(f"predicted={pt:.3f}  observed={pp:.3f}")

brier = brier_score_loss(df["true_label_binary"], df["p_mut"])
print(f"\nBrier score: {brier:.5f}  (0 = perfect, 0.25 = no better than always predicting 0.5)")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(prob_pred, prob_true, marker="o", color="red", linewidth=2, label="GMM posterior")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")
ax.set_xlabel("Predicted P(Mut)")
ax.set_ylabel("Observed fraction actually Mut")
ax.set_title("Calibration curve, GMM posterior probability vs true genotype")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_calibration_curve.png", dpi=150)
print("\nSaved as gmm_calibration_curve.png")
plt.show()


#sensitivity analysis (GMM classifier)
"""Determining the limit of detection (LOD) of the qPCR assay, model-implied sensitivity"""

print("Already fitted on the sample dataset GMM Classifier:")
print(f"Mut component: mean={mut_mean:.3f}, std={np.sqrt(mut_var):.3f}, weight={mut_weight:.3f}")
print(f"WT component: mean={wt_mean:.3f}, std={np.sqrt(wt_var):.3f}, weight={wt_weight:.3f}")


def classify(delta_ct):
    """GMM posterior probabilities-based classification: Mut if P(Mut | delta_ct) > 0.5"""
    density_mut = mut_weight * norm.pdf(delta_ct, mut_mean, np.sqrt(mut_var))
    density_wt = wt_weight * norm.pdf(delta_ct, wt_mean, np.sqrt(wt_var))
    return np.where(density_mut > density_wt, "Mut", "WT")


#Applying the same classifier to the sweep dataset
df_sweep["predicted_label"] = classify(df_sweep["delta_ct"].values)

#Specificity
wt_reference = df_sweep[df_sweep["true_vaf"] == 0]
specificity = (wt_reference["predicted_label"] == "WT").mean()
print(f"\nSpecificity: {specificity:.4f}, n={len(wt_reference)} true WT correctly classified as WT")


#Sensitivity for each VAF grid point
mut_points = df_sweep[df_sweep["true_vaf"] > 0].copy()
sensitivity_by_vaf = mut_points.groupby("true_vaf").apply(
    lambda g: (g["predicted_label"] == "Mut").mean()
).reset_index(name="sensitivity")
sensitivity_by_vaf["n"] = mut_points.groupby("true_vaf").size().values
print("\nSensitivity by VAF level:")
print(sensitivity_by_vaf)

#LOD95: lowest VAF grid point with sensitivity >= 0.95
LOD_THRESHOLD = 0.95
above_threshold = sensitivity_by_vaf[sensitivity_by_vaf["sensitivity"] >= LOD_THRESHOLD]
if len(above_threshold) > 0:
    lod95 = above_threshold["true_vaf"].min()
    print(f"\nLOD95 (lowest VAF with sensitivity >= {LOD_THRESHOLD}): {lod95:.6f}")
else:
    lod95 = None
    print(f"\nNo grid point reached sensitivity >= {LOD_THRESHOLD}")


#Plotting sensitivity as a function of VAF
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sensitivity_by_vaf["true_vaf"], sensitivity_by_vaf["sensitivity"],
        marker="o", color="black", markersize=4)
ax.axhline(LOD_THRESHOLD, color="red", linestyle="--", label=f"{LOD_THRESHOLD*100:.0f}% sensitivity threshold")
if lod95 is not None:
    ax.axvline(lod95, color="green", linestyle=":", label=f"LOD95 = {lod95:.5f}")
ax.set_xscale("log")
ax.set_xlabel("True VAF (log scale)")
ax.set_ylabel("Sensitivity (fraction correctly classified as Mut)")
ax.set_title("GMM Classification sensitivity vs true VAF")
ax.set_ylim(-0.05, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig("gmm_sensitivity_curve.png", dpi=150)
print("\nSaved as gmm_sensitivity_curve.png")
plt.show()


#Physical sampling limit (Poisson)
"""A reaction can only detect a mutant allele if at least one mutant molecule is physically in the tube.
With 20 ng of DNA (6.6 pg per diploid genome) there are ~6061 alleles per reaction, so the number of mutant
molecules in a reaction is Poisson distributed with mean = VAF x total alleles."""

NG_PER_REACTION = 20
PG_PER_GENOME = 6.6
ALLELES_PER_GENOME = 2
total_alleles = (NG_PER_REACTION * 1000 / PG_PER_GENOME) * ALLELES_PER_GENOME
print(f"Total alleles per reaction: {total_alleles:.0f}")

N_REACTIONS = 3000
vaf_grid = np.logspace(-5, np.log10(0.5), 40)
rng = np.random.default_rng(99)

rows = []
for vaf_value in vaf_grid:
    vaf = np.full(N_REACTIONS, vaf_value)
    dna_conc = _generate_concentration(N_REACTIONS, conc_min=PARAMS["conc_min"], conc_max=PARAMS["conc_max"], rng=rng)
    signal = _generate_delta_ct_signal(vaf, PARAMS["pure_mutant_delta_ct"], PARAMS["eff"], PARAMS["signal_noise_std"], N_REACTIONS, rng)
    background = _generate_delta_ct_background(
        N_REACTIONS, dna_conc, PARAMS["background_mean"], PARAMS["background_std"], PARAMS["conc_slope"], PARAMS["conc_ref"], rng
    )
    #draw the number of mutant molecules actually present, signal exists only if there is at least one
    has_molecule = rng.poisson(vaf_value * total_alleles, size=N_REACTIONS) >= 1
    gated = np.minimum(np.where(has_molecule, signal, np.inf), background)
    rows.append({
        "vaf": vaf_value,
        "p_molecule_present": 1 - np.exp(-vaf_value * total_alleles),
        "sens_gated": (classify(gated) == "Mut").mean(),
    })
poisson_df = pd.DataFrame(rows)

#false-positive rate of the classifier on true WT reactions (background only, no mutant signal at all)
N_WT = 20000
wt_conc = _generate_concentration(N_WT, conc_min=PARAMS["conc_min"], conc_max=PARAMS["conc_max"], rng=rng)
wt_background = _generate_delta_ct_background(
    N_WT, wt_conc, PARAMS["background_mean"], PARAMS["background_std"], PARAMS["conc_slope"], PARAMS["conc_ref"], rng
)
fp_rate = (classify(wt_background) == "Mut").mean()
print(f"False-positive rate on true WT reactions (n={N_WT}): {fp_rate:.4f}")

def first_vaf_reaching(column, level=0.95):
    hit = poisson_df[poisson_df[column] >= level]
    return hit["vaf"].min() if len(hit) else None

for label, column in [("molecule present in the tube (physical ceiling)", "p_molecule_present"),
                      ("classifier, Poisson-gated", "sens_gated")]:
    v = first_vaf_reaching(column)
    print(f"Lowest VAF with 95% single-reaction sensitivity, {label}: {v:.5f}" if v else f"{label}: not reached")

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(poisson_df["vaf"], poisson_df["p_molecule_present"], color="gray", linestyle="--", label="P(at least 1 mutant molecule in the tube)")
ax.plot(poisson_df["vaf"], poisson_df["sens_gated"], color="red", marker="o", markersize=3, label="classifier sensitivity (Poisson-gated)")
ax.axhline(fp_rate, color="steelblue", linestyle=":", label=f"false-positive rate on WT ({fp_rate:.3f})")
ax.axhline(0.95, color="red", linestyle=":", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("True VAF (log scale)")
ax.set_ylabel("Single-reaction sensitivity")
ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Physical sampling limit vs classifier limit (single reaction, {NG_PER_REACTION} ng)")
ax.legend()
plt.tight_layout()
plt.savefig("poisson_sampling_limit.png", dpi=150)
print("\nSaved as poisson_sampling_limit.png")
plt.show()


#GMM robustness (bootstrap, cross-validation, sample size)
def fit_gmm_params(x):
    #Fit a 2-component GMM
    g = GaussianMixture(n_components=2, random_state=42).fit(x)
    m = np.argmin(g.means_.ravel())
    mu_m, mu_w = g.means_.ravel()[m], g.means_.ravel()[1 - m]
    xs = np.linspace(mu_m, mu_w, 10000)
    d_m = g.weights_[m] * norm.pdf(xs, mu_m, np.sqrt(g.covariances_.ravel()[m]))
    d_w = g.weights_[1 - m] * norm.pdf(xs, mu_w, np.sqrt(g.covariances_.ravel()[1 - m]))
    return g, m, mu_m, mu_w, xs[np.argmin(np.abs(d_m - d_w))], g.weights_[m]

"""a refit is "stable" if the lower component is still the minority (Mut weight < 0.5);
otherwise the 2-component split has latched onto the WT tail instead of the Mut group.
Here the lower component must also sit on the Mut side of the full-data boundary: at low prevalence the plain GMM can
latch onto the WT tail with a minority weight, and the weight criterion alone would miss that"""
def is_stable(mu_m, w_m):
    return (w_m < 0.5) & (mu_m < gmm_boundary)

#Bootstrap: resample samples, refit the GMM each time, record parameters + boundary
n_bootstrap = 1000
rng = np.random.default_rng(42)
boot = []
for _ in range(n_bootstrap):
    idx = rng.integers(0, len(X), len(X))
    xb = X[idx]
    if len(np.unique(xb)) < 4:
        continue
    g, m, mu_m, mu_w, b, w_m = fit_gmm_params(xb)
    boot.append((mu_m, mu_w, b, w_m))
boot = np.array(boot)

stable = is_stable(boot[:, 0], boot[:, 3])
print(f"Bootstrap GMM refits: {stable.sum()}/{len(boot)} resamples recover a minority Mut component")
print("95% CI over those stable resamples:")
for name, col, orig in [("Mutant mean", 0, mut_mean), ("WT mean", 1, wt_mean), ("Boundary", 2, gmm_boundary)]:
    lo, hi = np.percentile(boot[stable, col], [2.5, 97.5])
    print(f"  {name}: original {orig:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

#plotting bootstrap distribution of the boundary
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(boot[stable, 2], bins=30, color="steelblue", edgecolor="white")
ax.axvline(gmm_boundary, color="black", linestyle="--", label=f"full-data boundary = {gmm_boundary:.2f}")
ax.set_xlabel("GMM boundary (delta_ct)")
ax.set_ylabel("count")
ax.set_title("Bootstrap distribution of the GMM boundary (stable refits)")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_boundary_bootstrap.png", dpi=150)
print("\nSaved as gmm_boundary_bootstrap.png")
plt.show()


#GMM cross-validation (stratified 5-fold)
"""Refitting the GMM without a fold of samples, then classify that fold with the refitted model. 
Samples here are independent, so folds replace patients."""
y_bin = df["true_label_binary"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_labels = np.empty(len(df), dtype=object)
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_bin), start=1):
    g, m, mu_m, mu_w, b, w_m = fit_gmm_params(X[train_idx])
    pred = g.predict(X[test_idx])
    cv_labels[test_idx] = np.where(pred == m, "Mut", "WT")
    print(f"Held out fold {fold}: refit Mut mean = {mu_m:.2f}, WT mean = {mu_w:.2f}, "
          f"boundary = {b:.2f}, Mut weight = {w_m:.2f}, stable = {bool(is_stable(mu_m, w_m))}")
df["gmm_label_cv"] = cv_labels

n_dis = (df["gmm_label"] != df["gmm_label_cv"]).sum()
print(f"\nCross-validation: disagreements with full-data GMM = {n_dis}, accuracy vs true_label = {accuracy_score(df['true_label'], df['gmm_label_cv']):.4f}")
if n_dis > 0:
    print(df.loc[df["gmm_label"] != df["gmm_label_cv"], ["sample_id", "delta_ct", "true_label", "gmm_label", "gmm_label_cv"]]
          .sort_values("delta_ct").to_string(index=False))

print("\nConfusion matrix (rows=test call, cols=true genotype), cross-validated labels:")
cm = confusion_matrix(df["true_label"], df["gmm_label_cv"], labels=["Mut", "WT"]).T
print(pd.DataFrame(cm, index=["call Mut", "call WT"], columns=["true Mut", "true WT"]))


#Sample size: refit the GMM on random subsets of increasing size
SAMPLE_SIZES = [20, 40, 100, 300, 1000, 3000]
N_SUBSETS = 200
rng = np.random.default_rng(7)
rows = []
for n in SAMPLE_SIZES:
    for _ in range(N_SUBSETS):
        xs_sub = X[rng.choice(len(X), size=n, replace=False)]
        if len(np.unique(xs_sub)) < 4:
            continue
        g, m, mu_m, mu_w, b, w_m = fit_gmm_params(xs_sub)
        rows.append({"n": n, "boundary": b, "stable": bool(is_stable(mu_m, w_m))})
size_df = pd.DataFrame(rows)

size_summary = size_df.groupby("n").apply(lambda s: pd.Series({
    "fraction stable": s["stable"].mean(),
    "boundary median (stable)": s.loc[s["stable"], "boundary"].median(),
    "boundary 5%": s.loc[s["stable"], "boundary"].quantile(0.05),
    "boundary 95%": s.loc[s["stable"], "boundary"].quantile(0.95),
})).round(3)
print(f"\nSample-size sweep ({N_SUBSETS} random subsets per size, mutant prevalence {(df['true_label'] == 'Mut').mean():.0%}):")
print(size_summary.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(size_summary.index, size_summary["boundary median (stable)"], "o-", color="black", label="median")
ax.fill_between(size_summary.index, size_summary["boundary 5%"], size_summary["boundary 95%"], color="gray", alpha=0.3, label="5-95% range")
ax.axhline(gmm_boundary, color="red", linestyle="--", label=f"full-data boundary = {gmm_boundary:.2f}")
ax.set_xscale("log")
ax.set_xlabel("Number of samples used to fit the GMM")
ax.set_ylabel("GMM boundary (delta_ct), stable refits only")
ax.set_title("GMM boundary vs sample size")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_sample_size_boundary.png", dpi=150)
print("\nSaved as gmm_sample_size_boundary.png")
plt.show()
