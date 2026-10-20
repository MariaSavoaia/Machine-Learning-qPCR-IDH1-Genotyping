import numpy as np
import pandas as pd
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
lr_pred_all = np.where(coef_wt * df["ct_wt"] + coef_mut * df["ct_mut"] + intercept > 0, "WT", "Mut")
CT_MUT_POSITIVE = 34.0   #Ct_mut < this means Mut, otherwise WT
df["absolute_call"] = np.where(df["ct_mut"] < CT_MUT_POSITIVE, "Mut", "WT")
df["absolute_label"] = np.where(df["technical_failure"], "Technical failure", df["absolute_call"])


#Summary (all methods comparison and disagreements)
df["logreg_label"] = lr_pred_all
method_cols = {"GMM": "gmm_label", "K-means": "kmeans_label", "Logistic regression": "logreg_label"}
if "absolute_label" in df.columns:
    method_cols["Absolute Ct thresholds"] = "absolute_label"

#performance against the true label
rows = []
for name, col in method_cols.items():
    wrong = df[col] != df["true_label"]
    rows.append({
        "method": name,
        "accuracy": round(1 - wrong.mean(), 4),
        "Mut called WT": int(((df["true_label"] == "Mut") & (df[col] == "WT")).sum()),
        "WT called Mut": int(((df["true_label"] == "WT") & (df[col] == "Mut")).sum()),
        "technical failure": int((df[col] == "Technical failure").sum()),
    })
print(f"Performance vs true label (n={len(df)}: {(df['true_label'] == 'Mut').sum()} Mut, {(df['true_label'] == 'WT').sum()} WT):")
print(pd.DataFrame(rows).to_string(index=False))

#pairwise disagreements: number of samples on which two methods give different labels
names = list(method_cols)
disagree = pd.DataFrame(
    [[(df[method_cols[a]] != df[method_cols[b]]).sum() for b in names] for a in names],
    index=names, columns=names,
)
print("\nPairwise disagreements (number of samples with different labels):")
print(disagree.to_string())

#combinations of calls, and who is right when the methods split
label_cols = list(method_cols.values())
patterns = (
    df.groupby(label_cols)
    .agg(n=("delta_ct", "size"),
         true_Mut=("true_label", lambda s: (s == "Mut").sum()),
         delta_ct_min=("delta_ct", "min"), delta_ct_max=("delta_ct", "max"))
    .reset_index()
)
patterns.columns = list(method_cols) + list(patterns.columns[len(label_cols):])
all_agree = patterns[patterns[list(method_cols)].nunique(axis=1) == 1]
split = patterns[patterns[list(method_cols)].nunique(axis=1) > 1].sort_values("n", ascending=False)
print(f"\nAll methods agree on {all_agree['n'].sum()} of {len(df)} samples.")
print("Samples where the methods split (n = samples, true_Mut = how many of them are truly Mut):")
print(split.round(2).to_string(index=False) if len(split) else "none")
