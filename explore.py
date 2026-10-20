import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, os

with open("synthetic_params.json") as f:
    _p = json.load(f)
param_source, PARAMS, SWEEP_KWARGS, GEN_KWARGS = _p["param_source"], _p["PARAMS"], _p["SWEEP_KWARGS"], _p["GEN_KWARGS"]

df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)


#exploration of the generated data & calibration checks
is_wt = df["true_vaf"] == 0
near_vaf_max = 0.9 * PARAMS["vaf_max"]   #samples close to the maximum VAF (0.45 when vaf_max = 0.5)

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(df.loc[is_wt, "delta_ct"], bins=40, alpha=0.5, color="steelblue", label="true WT (VAF=0)")
ax.hist(df.loc[~is_wt, "delta_ct"], bins=40, alpha=0.5, color="red", label="true Mut (VAF>0)")
ax.set_xlabel("Simulated delta_ct")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of simulated delta_ct, by true genotype")
ax.legend()
plt.tight_layout()
plt.savefig("delta_ct_distribution_synthetic.png", dpi=150)
print("Saved as delta_ct_distribution_synthetic.png")
plt.show()

print(f"\nCalibration check - Mut cluster delta_ct (VAF > {near_vaf_max:g}):")
print(df.loc[df["true_vaf"] > near_vaf_max, "delta_ct"].describe())

print("\nCalibration check - WT cluster delta_ct (VAF == 0):")
print(df.loc[is_wt, "delta_ct"].describe())

print("\nCalibration check - compared to GMM parameters from real dataset")
calibration_summary = pd.DataFrame({
    "cluster": ["WT - mean", "WT - std", "Mut - mean (near vaf_max)", "Mut - std (near vaf_max)"],
    "real data GMM parameters": [PARAMS["background_mean"], PARAMS["real_wt_std"], PARAMS["pure_mutant_delta_ct"], PARAMS["real_mut_std"]], #comparing with the real data parameters
    "synthetic": [
        df.loc[df["true_vaf"] == 0, "delta_ct"].mean(),
        df.loc[df["true_vaf"] == 0, "delta_ct"].std(),
        df.loc[df["true_vaf"] > near_vaf_max, "delta_ct"].mean(),
        df.loc[df["true_vaf"] > near_vaf_max, "delta_ct"].std(),
    ],
})
print(calibration_summary)


#Technical failures QC (based on Ct_wt)
CT_WT_FAILURE = 38.0  #Ct_wt >= this, means the wild-type reaction failed, the fragment cannot be genotyped

df["technical_failure"] = df["ct_wt"] >= CT_WT_FAILURE
print(f"Technical failures (Ct_wt >= {CT_WT_FAILURE:g}): {df['technical_failure'].sum()} / {len(df)}")
print(df.loc[df["technical_failure"], ["sample_id", "ct_wt", "ct_mut", "delta_ct", "true_label"]].round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["ct_wt"], bins=60, color="steelblue", edgecolor="white")
ax.axvline(CT_WT_FAILURE, color="black", linestyle="--", linewidth=1.5, label=f"failure threshold (Ct_wt = {CT_WT_FAILURE:g})")
ax.set_xlabel("Ct_wt")
ax.set_ylabel("Number of fragments")
ax.set_title("Ct_wt distribution and technical-failure threshold")
ax.legend()
plt.tight_layout()
plt.savefig("ct_wt_technical_failure.png", dpi=150)
print("\nSaved as ct_wt_technical_failure.png")
plt.show()
