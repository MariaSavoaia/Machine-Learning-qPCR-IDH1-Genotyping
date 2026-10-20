import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import json, os

with open("synthetic_params.json") as f:
    _p = json.load(f)
param_source, PARAMS, SWEEP_KWARGS, GEN_KWARGS = _p["param_source"], _p["PARAMS"], _p["SWEEP_KWARGS"], _p["GEN_KWARGS"]

df_sweep = pd.read_csv("vaf_sweep_data.csv")


#Linear regression between VAF and delta_ct using the VAF sweep
df_sweep = df_sweep.dropna(subset=["delta_ct"]).reset_index(drop=True)

df_signal = df_sweep[df_sweep["true_vaf"] > 0].copy()
df_signal["logit_vaf"] = np.log(df_signal["true_vaf"] / (1 - df_signal["true_vaf"]))


#Linear Regression 1: full range, all VAF>0 grid points
slope_full, intercept_full, r_full, p_full, se_full = stats.linregress(
    df_signal["logit_vaf"], df_signal["delta_ct"]
)
print("Linear Regression 1, delta_ct ~ logit(VAF), full range:")
print(f"slope={slope_full:.4f}, intercept={intercept_full:.4f}, R^2={r_full**2:.4f}")
print(f"(theoretically: slope=-1/eff={-1/PARAMS['eff']:.4f}, intercept=pure_mutant_delta_ct={PARAMS['pure_mutant_delta_ct']})")


#Linear Regression 2: restricted to VAF above a threshold of 0.01, where the signal is expected to dominate over background
threshold = 0.01
df_restricted = df_signal[df_signal["true_vaf"] > threshold]
slope_r, intercept_r, r_r, p_r, se_r = stats.linregress(
    df_restricted["logit_vaf"], df_restricted["delta_ct"]
)
print(f"\nLinear Regression 2, delta_ct ~ logit(VAF), restricted to VAF > {threshold}:")
print(f"slope={slope_r:.4f}, intercept={intercept_r:.4f}, R^2={r_r**2:.4f}")


#Summary of mean, std, count for each true_vaf value in the dataset
grid_summary = df_sweep.groupby("true_vaf")["delta_ct"].agg(["mean", "std", "count"]).reset_index()
print("\nSummary for each VAF value (delta_ct mean, std, count across replicates):")
print(grid_summary)


#Plotting delta_ct vs VAF (log x-axis), observed means with error bars, plus both fitted regression lines
fig, ax = plt.subplots(figsize=(10, 7))

plot_data = grid_summary[grid_summary["true_vaf"] > 0]
ax.errorbar(
    plot_data["true_vaf"], plot_data["mean"], yerr=plot_data["std"],
    fmt="o", color="black", ecolor="gray", capsize=3, label="observed (mean +/- std per VAF point)"
)

wt_point = grid_summary[grid_summary["true_vaf"] == 0]
ax.axhline(wt_point["mean"].values[0], color="steelblue", linestyle=":", label="true WT background level")

x_vaf = np.logspace(np.log10(df_signal["true_vaf"].min()), np.log10(df_signal["true_vaf"].max()), 200)
x_logit = np.log(x_vaf / (1 - x_vaf))
ax.plot(x_vaf, slope_full * x_logit + intercept_full, color="red", linestyle="--",
        label=f"fit, full range (R^2={r_full**2:.2f})")
ax.plot(x_vaf, slope_r * x_logit + intercept_r, color="green",
        label=f"fit, VAF>{threshold} (R^2={r_r**2:.2f})")

ax.set_xscale("log")
ax.set_xlabel("True VAF (log scale)")
ax.set_ylabel("delta_ct")
ax.set_title("delta_ct vs true VAF - dilution-series-style curve")
ax.legend()
plt.tight_layout()
plt.savefig("linear_regression_curve.png", dpi=150)
print("\nSaved as linear_regression_curve.png")
plt.show()


#VAF random forest nonlinear regressor
df_signal = df_sweep[df_sweep["true_vaf"] > 0].copy()
df_signal["log_vaf"] = np.log10(df_signal["true_vaf"])

X = df_signal[["log_vaf"]].values
y = df_signal["delta_ct"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("\nRandom Forest regressor, delta_ct ~ log10(VAF):")
print(f"Test R^2: {r2_score(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

#Linear regression (full range), for comparison
df_signal["logit_vaf"] = np.log(df_signal["true_vaf"] / (1 - df_signal["true_vaf"]))
lin_slope, lin_intercept, r, p, se = stats.linregress(df_signal["logit_vaf"], df_signal["delta_ct"])
print(f"\nLinear regression curve (for comparison): slope={lin_slope:.4f}, intercept={lin_intercept:.4f}, R^2={r**2:.4f}")


#Predicting across a smooth VAF grid for plotting
vaf_grid = np.logspace(np.log10(df_signal["true_vaf"].min()), np.log10(df_signal["true_vaf"].max()), 300)
log_vaf_grid = np.log10(vaf_grid).reshape(-1, 1)
rf_curve = rf.predict(log_vaf_grid)

logit_grid = np.log(vaf_grid / (1 - vaf_grid))
linear_regression_curve = lin_slope * logit_grid + lin_intercept

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df_signal["true_vaf"], df_signal["delta_ct"], alpha=0.15, color="gray", s=15, label="observed replicates")
ax.plot(vaf_grid, linear_regression_curve, color="red", linestyle="--", linewidth=2, label="Linear regression fit (logit-linear)")
ax.plot(vaf_grid, rf_curve, color="green", linewidth=2, label="Random Forest fit")
sweep_wt_mean = df_sweep.loc[df_sweep["true_vaf"] == 0, "delta_ct"].mean()
ax.axhline(sweep_wt_mean, color="steelblue", linestyle=":", label="true WT background level")
ax.set_xscale("log")
ax.set_xlabel("True VAF (log scale)")
ax.set_ylabel("delta_ct")
ax.set_title("Linear regression vs Random Forest fit, delta_ct ~ VAF")
ax.legend()
plt.tight_layout()
plt.savefig("rf_linreg_comparison.png", dpi=150)
print("\nSaved as rf_linreg_comparison.png")
plt.show()
