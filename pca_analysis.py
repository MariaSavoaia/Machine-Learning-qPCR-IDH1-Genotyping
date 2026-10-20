import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)


#PCA
df["log10_conc"] = np.log10(df["dna_conc"])

#ct_wt_residual (as a proxy for quality), obtained from the linear regression of ct_wt ~ log10_conc
slope_wt, intercept_wt, r_wt, p_wt, se_wt = stats.linregress(df["log10_conc"], df["ct_wt"])
print(f"ct_wt ~ log10_conc linear regression: slope={slope_wt:.4f}, R^2={r_wt**2:.4f}, p={p_wt:.4f}")
df["ct_wt_predicted"] = slope_wt * df["log10_conc"] + intercept_wt
df["ct_wt_residual"] = df["ct_wt"] - df["ct_wt_predicted"]

#feature matrix
features = df[["ct_mut", "log10_conc", "ct_wt_residual"]].copy()
print("\nSummary statistics for the 3 PCA input variables (before standardization):")
print(features.describe())

#standardize and fit PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA()
pca_result = pca.fit_transform(features_scaled)
print("\nPCA variance breakdown:")
print("Explained variance ratio per principal component (PC1, PC2, PC3):", np.round(pca.explained_variance_ratio_, 4))
print("Cumulative explained variance:", np.round(np.cumsum(pca.explained_variance_ratio_), 4))

#loadings of each original variable on each principal component
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2", "PC3"],
    index=["ct_mut", "log10_conc", "ct_wt_residual"]
)
print("\nLoadings of each original variable on each principal component:")
print(loadings.round(4))


df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]
df["PC3"] = pca_result[:, 2]

df.to_csv("pca_synthetic.csv", index=False)
print("Saved as pca_synthetic.csv")


#PC1 vs PC2 biplot, colored by true genotype
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"Mut": "red", "WT": "steelblue"}
for label, group in df.groupby("true_label"):
    ax.scatter(group["PC1"], group["PC2"], label=label, color=colors[label], alpha=0.3, s=40)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA - PC1 vs PC2")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig("pca_pc1pc2.png", dpi=150)
print("\nSaved as pca_pc1pc2.png")
plt.show()

#PC1 vs PC3 biplot
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"Mut": "red", "WT": "steelblue"}
for label, group in df.groupby("true_label"):
    ax.scatter(group["PC1"], group["PC3"], label=label, color=colors[label], alpha=0.3, s=40)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)")
ax.set_title("PCA - PC1 vs PC3")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig("pca_pc1pc3.png", dpi=150)
print("\nSaved as pca_pc1pc3.png")
plt.show()

#PC2 vs PC3 biplot
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"Mut": "red", "WT": "steelblue"}
for label, group in df.groupby("true_label"):
    ax.scatter(group["PC2"], group["PC3"], label=label, color=colors[label], alpha=0.3, s=40)
ax.set_xlabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% variance)")
ax.set_title("PCA - PC2 vs PC3")
ax.legend()
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig("pca_pc2pc3.png", dpi=150)
print("\nSaved as pca_pc2pc3.png")
plt.show()
