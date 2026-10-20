import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

df = pd.read_csv("synthetic_qpcr_data.csv")
df["true_label"] = np.where(df["true_vaf"] > 0, "Mut", "WT")
df["true_label_binary"] = (df["true_vaf"] > 0).astype(int)


#Logistic Regression
X = df[["ct_wt", "ct_mut"]].values
y = df["true_label"].values

#Stratified split: preserves the class balance set in PARAMS in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

logreg = LogisticRegression(C=1.0)
logreg.fit(X_train, y_train)

coef_wt, coef_mut = logreg.coef_.ravel()
print("Logistic regression, trained on train split:")
print(f"Coefficients (ct_wt, ct_mut): {coef_wt:.4f}, {coef_mut:.4f}")
print(f"Coefficient ratio (ct_mut / ct_wt): {coef_mut / coef_wt:.4f}")
print(f"\nTest set size: {len(y_test)}")

y_pred = logreg.predict(X_test)

#Logistic regression accuracy
acc_test = accuracy_score(y_test, y_pred)
print(f"\nLogistic regression accuracy, held-out test set (n={len(y_test)}): {acc_test:.4f}")

#Accuracy on the whole dataset, directly comparable to the GMM and K-means accuracies (which use all samples)
y_pred_all = logreg.predict(df[["ct_wt", "ct_mut"]].values)
acc_all = accuracy_score(df["true_label"], y_pred_all)
print(f"Logistic regression accuracy, full dataset (n={len(df)}, includes training samples): {acc_all:.4f}")


print("\nConfusion matrix (rows=true, cols=predicted), labels=[Mut, WT]:")
print(confusion_matrix(y_test, y_pred, labels=["Mut", "WT"]))
print("\nClassification report:")
print(classification_report(y_test, y_pred))


#Logistic regression decision boundary in the (ct_wt, ct_mut) plane
"""The boundary is the line where the model's decision function is 0 (P(Mut) = 0.5):
intercept + coef_wt * ct_wt + coef_mut * ct_mut = 0"""
intercept = logreg.intercept_[0]
print(f"\nDecision boundary equation: Ct_mut = {-intercept/coef_mut:.3f} + {-coef_wt/coef_mut:.3f} x Ct_wt")

fig, ax = plt.subplots(figsize=(8, 7))
colors = {"Mut": "red", "WT": "steelblue"}
for label, group in df.groupby("true_label"):
    ax.scatter(group["ct_wt"], group["ct_mut"], color=colors[label], label=f"true {label}", alpha=0.3, s=15)

ct_wt_line = np.linspace(df["ct_wt"].min() - 2, df["ct_wt"].max() + 2, 100)
ct_mut_line = -(intercept + coef_wt * ct_wt_line) / coef_mut
ax.plot(ct_wt_line, ct_mut_line, color="black", linewidth=2, label="logistic regression boundary")

ax.set_ylim(df["ct_mut"].min() - 2, df["ct_mut"].max() + 2)
ax.set_xlabel("Ct_wt")
ax.set_ylabel("Ct_mut")
ax.set_title("Decision boundary from logistic regression, synthetic data")
ax.legend()
plt.tight_layout()
plt.savefig("logreg_decision_boundary_synthetic.png", dpi=150)
print("\nSaved as logreg_decision_boundary_synthetic.png")
plt.show()


#Logistic regression (Precision-Recall, ROC-AUC, repeated stratified k-fold cross-validation)
"""Stratified k-fold to ensure that each fold maintains the class balance set in PARAMS"""

X = df[["ct_wt", "ct_mut"]].values
y = df["true_label_binary"].values

logreg = LogisticRegression(C=1.0)
N_SPLITS = 5
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

#Cross-validated ROC AUC and PR AUC scores
roc_scores = cross_val_score(logreg, X, y, cv=cv, scoring="roc_auc")
pr_scores = cross_val_score(logreg, X, y, cv=cv, scoring="average_precision")

print(f"\nstratified {N_SPLITS}-fold cross-validation:")
print(f"ROC-AUC: mean={roc_scores.mean():.4f}, std={roc_scores.std():.4f}, per-fold={np.round(roc_scores, 4)}")
print(f"PR-AUC (average precision): mean={pr_scores.mean():.4f}, std={pr_scores.std():.4f}, per-fold={np.round(pr_scores, 4)}")


#Out-of-fold predictions for the entire dataset
"""For each sample, the prediction is made by the classifier that 
was trained on the folds that did not include that sample. 
Then we constructed a single precision-recall curve that reflects the classifier's performance on the entire dataset."""

y_proba_oof = cross_val_predict(logreg, X, y, cv=cv, method="predict_proba")[:, 1]
precision, recall, thresholds = precision_recall_curve(y, y_proba_oof)
avg_precision = average_precision_score(y, y_proba_oof)
baseline = y.mean()  #Mutant class prevalence - the "no-skill" baseline for PR (precision if we randomly guessed for each sample)

print(f"\nOverall PR-AUC (out-of-fold): {avg_precision:.4f}")
print(f"Class prevalence (no-skill baseline for PR): {baseline:.4f}")

#Plotting the precision-recall curve
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(recall, precision, color="red", linewidth=2, label=f"logistic regression (PR-AUC={avg_precision:.3f})")
ax.axhline(baseline, color="gray", linestyle="--", label=f"no-skill baseline (prevalence={baseline:.3f})")
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall curve, {N_SPLITS}-fold cross-validated (out-of-fold predictions)")
ax.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
print("\nSaved as precision_recall_curve.png")
plt.show()

#ROC curve from the out-of-fold predictions
fpr, tpr, _ = roc_curve(y, y_proba_oof)
roc_auc = roc_auc_score(y, y_proba_oof)
print(f"\nOverall ROC-AUC (out-of-fold): {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, color="red", linewidth=2, label=f"logistic regression (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random guess (AUC=0.5)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title(f"ROC curve, {N_SPLITS}-fold cross-validated (out-of-fold predictions)")
ax.legend()
plt.tight_layout()
plt.savefig("roc_curve_synthetic.png", dpi=150)
print("\nSaved as roc_curve_synthetic.png")
plt.show()
