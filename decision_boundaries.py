import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)

X = df["delta_ct"].values.reshape(-1, 1)

gmm2 = GaussianMixture(n_components=2, random_state=42)
gmm2.fit(X)
mutant_component = np.argmin(gmm2.means_.ravel())
cluster_labels = gmm2.predict(X)
df["gmm_label"] = np.where(cluster_labels == mutant_component, "Mut", "WT")
mut_mean = gmm2.means_.ravel()[mutant_component]
mut_var = gmm2.covariances_.ravel()[mutant_component]
mut_weight = gmm2.weights_[mutant_component]
wt_mean = gmm2.means_.ravel()[1 - mutant_component]
wt_var = gmm2.covariances_.ravel()[1 - mutant_component]
wt_weight = gmm2.weights_[1 - mutant_component]
x_search = np.linspace(mut_mean, wt_mean, 10000)
dens_mut = mut_weight * norm.pdf(x_search, mut_mean, np.sqrt(mut_var))
dens_wt = wt_weight * norm.pdf(x_search, wt_mean, np.sqrt(wt_var))
gmm_boundary = x_search[np.argmin(np.abs(dens_mut - dens_wt))]

X_km = df["delta_ct"].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
km_raw = kmeans.fit_predict(X_km)
centers = kmeans.cluster_centers_.ravel()
mut_cluster = np.argmin(centers)
df["kmeans_label"] = np.where(km_raw == mut_cluster, "Mut", "WT")
kmeans_boundary = (centers[mut_cluster] + centers[1 - mut_cluster]) / 2

X = df[["ct_wt", "ct_mut"]].values
y = df["true_label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
logreg = LogisticRegression(C=1.0)
logreg.fit(X_train, y_train)
coef_wt, coef_mut = logreg.coef_.ravel()
intercept = logreg.intercept_[0]

CT_WT_FAILURE = 38.0  #Ct_wt >= this, means the wild-type reaction failed, the fragment cannot be genotyped
df["technical_failure"] = df["ct_wt"] >= CT_WT_FAILURE


#All decision boundaries compared (GMM, K-means, logistic regression)

lr_pred_all = np.where(coef_wt * df["ct_wt"] + coef_mut * df["ct_mut"] + intercept > 0, "WT", "Mut")
#logreg.classes_ is ['Mut', 'WT'], so a positive decision function means WT

#absolute Ct thresholds, CT_WT_FAILURE already defined
CT_MUT_POSITIVE = 34.0   #Ct_mut < this means Mut, otherwise WT

df["absolute_call"] = np.where(df["ct_mut"] < CT_MUT_POSITIVE, "Mut", "WT")
df["absolute_label"] = np.where(df["technical_failure"], "Technical failure", df["absolute_call"])
df["absolute_wrong"] = df["absolute_label"] != df["true_label"]

print(f"Absolute thresholds: technical failure if Ct_wt >= {CT_WT_FAILURE:g}; Mut if Ct_mut < {CT_MUT_POSITIVE:g}, otherwise WT")
print(f"Technical failures: {df['technical_failure'].sum()} / {len(df)}")
print("\nGenotype call vs true label (before removing technical failures):")
print(pd.crosstab(df["absolute_call"], df["true_label"], rownames=["test call"], colnames=["true genotype"]))
print()

boundary_summary = pd.DataFrame({
    "method": ["GMM", "K-means", "Logistic regression"],
    "boundary": [
        f"delta_ct = {gmm_boundary:.3f}",
        f"delta_ct = {kmeans_boundary:.3f}",
        f"Ct_mut = {-intercept/coef_mut:.3f} + {-coef_wt/coef_mut:.3f} x Ct_wt",
    ],
    "misclassified": [
        (df["gmm_label"] != df["true_label"]).sum(),
        (df["kmeans_label"] != df["true_label"]).sum(),
        (lr_pred_all != df["true_label"]).sum(),
    ],
})
boundary_summary["accuracy"] = 1 - boundary_summary["misclassified"] / len(df)
print("Decision boundaries, all methods (evaluated on the full dataset, n={}):".format(len(df)))
print(boundary_summary.to_string(index=False))

#all boundaries in the ct_wt and ct_mut space
fig, ax = plt.subplots(figsize=(9, 8))
colors = {"Mut": "red", "WT": "steelblue"}
for label, group in df.groupby("true_label"):
    ax.scatter(group["ct_wt"], group["ct_mut"], color=colors[label], label=f"true {label}", alpha=0.25, s=12)

ct_wt_line = np.linspace(df["ct_wt"].min() - 2, df["ct_wt"].max() + 2, 100)
ax.plot(ct_wt_line, ct_wt_line + gmm_boundary, color="black", linestyle="-", linewidth=2,
        label=f"GMM (delta_ct = {gmm_boundary:.2f})")
ax.plot(ct_wt_line, ct_wt_line + kmeans_boundary, color="purple", linestyle="--", linewidth=2,
        label=f"K-means (delta_ct = {kmeans_boundary:.2f})")
ax.plot(ct_wt_line, -(intercept + coef_wt * ct_wt_line) / coef_mut, color="orange", linestyle=":", linewidth=2.5,
        label="Logistic regression")
ax.axvline(CT_WT_FAILURE, color="green", linestyle="-.", linewidth=2, label=f"Absolute: Ct_wt failure = {CT_WT_FAILURE:g}")
ax.axhline(CT_MUT_POSITIVE, color="green", linestyle=":", linewidth=2, label=f"Absolute: Ct_mut positive = {CT_MUT_POSITIVE:g}")

ax.set_ylim(df["ct_mut"].min() - 2, df["ct_mut"].max() + 2)
ax.set_xlabel("Ct_wt")
ax.set_ylabel("Ct_mut")
ax.set_title("Decision boundaries of all classifiers, synthetic data")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("all_decision_boundaries_synthetic.png", dpi=150, bbox_inches="tight")
print("\nSaved as all_decision_boundaries_synthetic.png")
plt.show()


#all boundaries on the delta_ct axis
a_lr = -intercept / coef_mut
b_lr = -coef_wt / coef_mut
ct_wt_lo, ct_wt_hi = np.percentile(df["ct_wt"], [5, 95]) #logistic regression boundary shown as a band covering the 5th-95th percentile of ct_wt
lr_thr_lo, lr_thr_hi = sorted([a_lr + (b_lr - 1) * ct_wt_lo, a_lr + (b_lr - 1) * ct_wt_hi])

fig, ax = plt.subplots(figsize=(9, 6))
is_wt = df["true_label"] == "WT"
ax.hist(df.loc[is_wt, "delta_ct"], bins=40, alpha=0.5, color="steelblue", label="true WT")
ax.hist(df.loc[~is_wt, "delta_ct"], bins=40, alpha=0.5, color="red", label="true Mut")
ax.axvspan(lr_thr_lo, lr_thr_hi, color="orange", alpha=0.35,
           label=f"Logistic regression ({lr_thr_lo:.2f} to {lr_thr_hi:.2f}, depends on ct_wt)")
ax.axvline(gmm_boundary, color="black", linestyle="-", linewidth=2, label=f"GMM = {gmm_boundary:.2f}")
ax.axvline(kmeans_boundary, color="purple", linestyle="--", linewidth=2, label=f"K-means = {kmeans_boundary:.2f}")
ax.set_xlabel("Simulated delta_ct")
ax.set_ylabel("Frequency")
ax.set_title("Decision boundaries on the delta_ct axis, synthetic data")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("all_boundaries_delta_ct_synthetic.png", dpi=150, bbox_inches="tight")
print("Saved as all_boundaries_delta_ct_synthetic.png")
plt.show()
