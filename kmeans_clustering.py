import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, silhouette_score

df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)
X = df["delta_ct"].values.reshape(-1, 1)
gmm2 = GaussianMixture(n_components=2, random_state=42)
gmm2.fit(X)
mutant_component = np.argmin(gmm2.means_.ravel())
cluster_labels = gmm2.predict(X)
df["gmm_label"] = np.where(cluster_labels == mutant_component, "Mut", "WT")


#K-means clustering on delta_ct
X_km = df["delta_ct"].values.reshape(-1, 1)

#K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
km_raw = kmeans.fit_predict(X_km)
centers = kmeans.cluster_centers_.ravel()

#the mutant cluster is the one with the lower delta_ct center
mut_cluster = np.argmin(centers)
df["kmeans_label"] = np.where(km_raw == mut_cluster, "Mut", "WT")

print("K-means clustering (k=2):")
print("Centers (Mut, WT):", np.round([centers[mut_cluster], centers[1 - mut_cluster]], 3))
print(f"Silhouette score: {silhouette_score(X_km, km_raw):.4f}")

#K-means vs true genotype
print("\nK-means vs true genotype:")
print(f"Accuracy: {accuracy_score(df['true_label'], df['kmeans_label']):.4f}")
print("\nConfusion matrix (rows=true, cols=predicted), labels=[Mut, WT]:")
print(confusion_matrix(df["true_label"], df["kmeans_label"], labels=["Mut", "WT"]))
print("\nClassification report:")
print(classification_report(df["true_label"], df["kmeans_label"]))

#decision boundary
kmeans_boundary = (centers[mut_cluster] + centers[1 - mut_cluster]) / 2
print(f"K-means decision boundary: delta_ct = {kmeans_boundary:.3f}")


n_disagree = (df["gmm_label"] != df["kmeans_label"]).sum()
print(f"Disagreements GMM vs K-means: {n_disagree} of {len(df)}")


#Silhouette score sweep, k=2 to 5
print("\nSilhouette score sweep:")
sil_scores = []
for k in range(2, 6):
    labels_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_km)
    sil_scores.append(silhouette_score(X_km, labels_k))
    print(f"k={k}: silhouette={sil_scores[-1]:.4f}")
print(f"Best k according to silhouette score: {np.argmax(sil_scores) + 2}")


#Plot: delta_ct distribution colored by K-means label, with centers and boundary
fig, ax = plt.subplots(figsize=(9, 6))
is_km_wt = df["kmeans_label"] == "WT"
ax.hist(df.loc[is_km_wt, "delta_ct"], bins=40, alpha=0.5, color="steelblue", label="K-means WT")
ax.hist(df.loc[~is_km_wt, "delta_ct"], bins=40, alpha=0.5, color="red", label="K-means Mut")
ax.axvline(centers[mut_cluster], color="darkred", linestyle="-", label=f"Mut center = {centers[mut_cluster]:.2f}")
ax.axvline(centers[1 - mut_cluster], color="darkblue", linestyle="-", label=f"WT center = {centers[1 - mut_cluster]:.2f}")
ax.axvline(kmeans_boundary, color="black", linestyle="--", label=f"boundary = {kmeans_boundary:.2f}")
ax.set_xlabel("Simulated delta_ct")
ax.set_ylabel("Frequency")
ax.set_title("K-means clustering of simulated delta_ct")
ax.legend()
plt.tight_layout()
plt.savefig("kmeans_cluster_view_synthetic.png", dpi=150)
print("\nSaved as kmeans_cluster_view_synthetic.png")
plt.show()
