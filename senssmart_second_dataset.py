"""Reproducible SensSmartTech benchmark used by monte_carlo.ipynb."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import GroupKFold
from IPython.display import display

S_ROOT = Path("public_datasets/senssmarttech")
S_CSV = S_ROOT / "CSV"
S_URL = "https://physionet.org/files/senssmarttech/1.0.0"
S_ACTIONS = {
    0: "Raw", 1: "Moving average (5)", 2: "Moving average (11)",
    3: "Savitzky-Golay (21,3)", 4: "Butterworth (0.5-5 Hz, order 4)",
    5: "Butterworth (0.7-4 Hz, order 3)",
}
S_FS, S_SEEDS, S_EPOCHS = 100, [11, 22, 33], 40


def s_download():
    S_ROOT.mkdir(parents=True, exist_ok=True); S_CSV.mkdir(exist_ok=True)
    demographics = S_ROOT / "Demographics.csv"
    if not demographics.exists(): urlretrieve(f"{S_URL}/Demographics.csv", demographics)
    frame = pd.read_csv(demographics, skiprows=1).iloc[:, :14]
    frame = frame[frame["PPG"].notna()].copy()
    names = frame["PPG"].dropna().astype(str).tolist()
    def one(name):
        destination = S_CSV / f"{name}.csv"
        if not destination.exists(): urlretrieve(f"{S_URL}/CSV/{name}.csv", destination)
    with ThreadPoolExecutor(max_workers=12) as pool: list(pool.map(one, names))
    return frame


def s_filter(x, action):
    if action == 0: return x.copy()
    if action == 1: return np.convolve(x, np.ones(5)/5, mode="same")
    if action == 2: return np.convolve(x, np.ones(11)/11, mode="same")
    if action == 3: return savgol_filter(x, 21, 3)
    band, order = ([0.5, 5.0], 4) if action == 4 else ([0.7, 4.0], 3)
    return sosfiltfilt(butter(order, band, btype="bandpass", fs=S_FS, output="sos"), x)


def s_hr(x):
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    peaks, _ = find_peaks(z, distance=int(.25*S_FS), prominence=.35)
    if len(peaks) < 2: return np.nan
    value = 60 / np.median(np.diff(peaks) / S_FS)
    return value if 35 <= value <= 220 else np.nan


def s_features(x):
    z = (x - x.mean()) / (x.std() + 1e-8)
    smooth = savgol_filter(z, 31, 3)
    noise = np.std(z-smooth) / (np.std(z)+1e-8)
    hist, _ = np.histogram(z, bins=32, density=True); p = hist[hist > 0]; p = p/p.sum()
    entropy = -np.sum(p*np.log(p))
    peaks, props = find_peaks(z, distance=int(.25*S_FS), prominence=.35)
    peak_quality = float(np.median(props["prominences"])) if len(peaks) else 0.0
    return [1-noise, entropy, peak_quality]


def s_load(frame):
    rows = []
    for _, row in frame.iterrows():
        name = str(row["PPG"]); path = S_CSV / f"{name}.csv"
        if not path.exists():
            continue
        signal = pd.read_csv(path).iloc[:, 1].to_numpy(float)
        if len(signal) >= 500 and np.isfinite(signal).all():
            rows.append({"record": name, "subject": int(row["Subject number"]),
                         "reference_hr": float(row["Median heart rate (bpm)"]),
                         "activity": row["Before (B)  / after (A) activity"],
                         "signal": signal, "features": s_features(signal)})
    return pd.DataFrame(rows)


def s_train(errors, states, train, algorithm, seed):
    rng = np.random.default_rng(seed); q = np.zeros((27, len(S_ACTIONS))); counts = np.zeros_like(q, int)
    epsilon = 1.0
    for _ in range(S_EPOCHS):
        for index in rng.permutation(train):
            state = states[index]
            action = rng.integers(len(S_ACTIONS)) if rng.random() < epsilon else rng.choice(np.flatnonzero(q[state] == q[state].max()))
            counts[state, action] += 1
            alpha = 1/counts[state, action] if algorithm != "q" else .10/np.sqrt(counts[state, action])
            q[state, action] += alpha * (-errors[index, action] - q[state, action])
        epsilon = max(.05, epsilon*.97)
    return q


def s_metrics(pred, ref):
    finite = np.isfinite(pred); err = np.where(finite, np.minimum(abs(pred-ref), 50), 50)
    return {"MAE (BPM)": err.mean(), "RMSE (BPM)": np.sqrt(np.mean(err**2)), "Failed (%)": 100*(1-finite.mean())}


def run_senssmart_experiment():
    source = s_download(); data = s_load(source)
    signals = data.signal.to_list(); ref = data.reference_hr.to_numpy(float); groups = data.subject.to_numpy()
    estimates = np.array([[s_hr(s_filter(x, a)) for a in S_ACTIONS] for x in signals])
    errors = np.where(np.isfinite(estimates), np.minimum(abs(estimates-ref[:, None]), 50), 50)
    features = np.vstack(data.features.to_numpy())
    rows, predictions = [], []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=4).split(features, groups=groups), 1):
        edges = np.quantile(features[train], [.33, .66], axis=0)
        states = np.clip(np.digitize(features[:, 0], edges[:, 0])*9 + np.digitize(features[:, 1], edges[:, 1])*3 + np.digitize(features[:, 2], edges[:, 2]), 0, 26)
        for action, name in S_ACTIONS.items():
            m = s_metrics(estimates[test, action], ref[test]); rows.append({"Fold":fold,"Method":"Fixed | "+name,"Seed":0, **m})
            predictions.append(pd.DataFrame({"Fold":fold,"Method":"Fixed | "+name,"Seed":0,
                "record":data.record.to_numpy()[test], "subject":groups[test], "activity":data.activity.to_numpy()[test],
                "reference_hr":ref[test],"prediction":estimates[test, action],"action":action}))
        for algorithm, label in [("mc", "Monte-Carlo"), ("q", "Q-learning"), ("bandit", "Contextual bandit")]:
            for seed in S_SEEDS:
                q = s_train(errors, states, train, algorithm, seed); chosen = np.argmax(q[states[test]], axis=1)
                pred = estimates[test, chosen]; m = s_metrics(pred, ref[test])
                rows.append({"Fold":fold,"Method":label,"Seed":seed, **m})
                predictions.append(pd.DataFrame({"Fold":fold,"Method":label,"Seed":seed,
                    "record":data.record.to_numpy()[test], "subject":groups[test], "activity":data.activity.to_numpy()[test],
                    "reference_hr":ref[test],"prediction":pred,"action":chosen}))
    results = pd.DataFrame(rows)
    fold_results = results.groupby(["Fold", "Method"], as_index=False)[["MAE (BPM)", "RMSE (BPM)", "Failed (%)"]].mean()
    summary = fold_results.groupby("Method", as_index=False).agg(**{"Mean MAE (BPM)":("MAE (BPM)","mean"), "SD MAE (BPM)":("MAE (BPM)","std"), "Mean RMSE (BPM)":("RMSE (BPM)","mean"), "Mean Failed (%)":("Failed (%)","mean")}).sort_values("Mean MAE (BPM)")
    return data, results, fold_results, summary, pd.concat(predictions, ignore_index=True)


def build_senssmart_paper_outputs():
    """Create paper-ready figures and analysis tables from held-out predictions only."""
    output = Path("paper_outputs")
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed_methods = S_FOLD_RESULTS[S_FOLD_RESULTS.Method.str.startswith("Fixed |")] 
    best_fixed = fixed_methods.groupby("Method")["MAE (BPM)"].mean().idxmin()
    selected = [best_fixed, "Monte-Carlo", "Contextual bandit", "Q-learning"]
    display_folds = S_FOLD_RESULTS[S_FOLD_RESULTS.Method.isin(selected)].copy()
    q_seed = S_SEEDS[0]
    selected_predictions = S_PREDICTIONS[((S_PREDICTIONS.Method == best_fixed) & (S_PREDICTIONS.Seed == 0)) | ((S_PREDICTIONS.Method == "Q-learning") & (S_PREDICTIONS.Seed == q_seed))].copy()
    selected_predictions["error"] = np.where(np.isfinite(selected_predictions.prediction), np.minimum(abs(selected_predictions.prediction-selected_predictions.reference_hr), 50), 50)
    subject_mae = selected_predictions.groupby(["Method", "subject"], as_index=False)["error"].mean().rename(columns={"error":"Subject MAE (BPM)"})
    activity_mae = selected_predictions.groupby(["Method", "activity"], as_index=False)["error"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean":"MAE (BPM)","std":"SD (BPM)","count":"N recordings"})
    condition_names = {"A": "After activity", "B": "Before activity"}
    activity_mae["Condition"] = activity_mae["activity"].map(condition_names).fillna(activity_mae["activity"])
    q_actions = selected_predictions[selected_predictions.Method == "Q-learning"].groupby(["activity", "action"], as_index=False).size()
    q_actions["Action"] = q_actions.action.map(S_ACTIONS)
    q_actions["Percentage"] = 100*q_actions["size"]/q_actions.groupby("activity")["size"].transform("sum")
    q_actions["Condition"] = q_actions["activity"].map(condition_names).fillna(q_actions["activity"])
    paired = S_FOLD_RESULTS[S_FOLD_RESULTS.Method.isin([best_fixed, "Q-learning"])].pivot(index="Fold", columns="Method", values="MAE (BPM)").dropna()
    delta = paired["Q-learning"] - paired[best_fixed]
    try: t_p = float(ttest_rel(paired["Q-learning"], paired[best_fixed]).pvalue)
    except Exception: t_p = np.nan
    try: w_p = float(wilcoxon(delta).pvalue)
    except Exception: w_p = np.nan
    stats = pd.DataFrame([{"Best fixed method":best_fixed, "Q-learning minus best-fixed MAE (BPM)":delta.mean(), "Paired folds":len(delta), "Paired t-test p":t_p, "Wilcoxon p":w_p}])
    analysis = pd.concat([subject_mae.assign(Table="Subject-level MAE"), activity_mae.assign(Table="Activity-stratified MAE"), q_actions.assign(Table="Q-learning action distribution")], ignore_index=True, sort=False)
    analysis.to_csv(output / "senssmarttech_paper_analysis.csv", index=False)
    stats.to_csv(output / "senssmarttech_qlearning_vs_best_fixed.csv", index=False)
    subject_mae.to_csv(output / "senssmarttech_subject_level_mae.csv", index=False)
    activity_mae.to_csv(output / "senssmarttech_activity_mae.csv", index=False)
    q_actions.to_csv(output / "senssmarttech_qlearning_action_distribution.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    order = display_folds.groupby("Method")["MAE (BPM)"].mean().sort_values().index
    sns.boxplot(data=display_folds, x="MAE (BPM)", y="Method", order=order, color="lightsteelblue", ax=axes[0,0])
    sns.stripplot(data=display_folds, x="MAE (BPM)", y="Method", order=order, color="black", size=5, ax=axes[0,0])
    axes[0,0].set_title("Held-out subject-fold MAE")
    sns.boxplot(data=subject_mae, x="Subject MAE (BPM)", y="Method", order=[best_fixed,"Q-learning"], palette=["#4C78A8", "#F58518"], ax=axes[0,1])
    sns.stripplot(data=subject_mae, x="Subject MAE (BPM)", y="Method", order=[best_fixed,"Q-learning"], color="black", size=4, ax=axes[0,1])
    axes[0,1].set_title("Subject-level generalization")
    sns.barplot(data=activity_mae, x="Condition", y="MAE (BPM)", hue="Method", hue_order=[best_fixed,"Q-learning"], ax=axes[1,0])
    axes[1,0].set_title("Error before vs after activity"); axes[1,0].set_xlabel("Recording condition")
    for method, color in [(best_fixed,"#4C78A8"),("Q-learning","#F58518")]:
        subset = selected_predictions[selected_predictions.Method == method].dropna(subset=["prediction"])
        mean_hr = (subset.prediction + subset.reference_hr)/2; diff = subset.prediction-subset.reference_hr
        axes[1,1].scatter(mean_hr, diff, s=12, alpha=.28, label=method, color=color)
        bias, sd = diff.mean(), diff.std(ddof=1)
        axes[1,1].axhline(bias, color=color, linewidth=1)
        axes[1,1].axhline(bias+1.96*sd, color=color, linestyle="--", linewidth=.8)
        axes[1,1].axhline(bias-1.96*sd, color=color, linestyle="--", linewidth=.8)
    axes[1,1].axhline(0, color="black", linewidth=.7); axes[1,1].legend(fontsize=8)
    axes[1,1].set_title("Bland-Altman agreement"); axes[1,1].set_xlabel("Mean HR (BPM)"); axes[1,1].set_ylabel("Estimated − reference (BPM)")
    fig.suptitle("SensSmartTech external replication", fontsize=16); fig.tight_layout()
    fig.savefig(figure_dir / "senssmarttech_external_replication.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8,4.5))
    sns.barplot(data=q_actions, x="Action", y="Percentage", hue="Condition", palette="Set2")
    plt.xticks(rotation=25, ha="right"); plt.ylabel("Held-out selections (%)"); plt.xlabel("")
    plt.title("Q-learning filter choices by recording condition"); plt.tight_layout()
    plt.savefig(figure_dir / "senssmarttech_qlearning_actions.png", dpi=300, bbox_inches="tight")
    plt.show()
    best_mae = S_SUMMARY.iloc[0]["Mean MAE (BPM)"]
    q_mae = S_SUMMARY.loc[S_SUMMARY.Method == "Q-learning", "Mean MAE (BPM)"].iloc[0]
    interpretation = (f"The best fixed method was {best_fixed} (mean MAE {best_mae:.3f} BPM). "
        f"Q-learning had {q_mae:.3f} BPM mean MAE, a {delta.mean():+.3f} BPM fold-mean difference. "
        "The p-values are exploratory because only four subject-grouped folds are available; report effect sizes and fold distributions, not a definitive significance claim.")
    print(interpretation); display(stats.round(4)); display(activity_mae.round(3)); display(q_actions.round(2))
    return subject_mae, activity_mae, q_actions, stats, interpretation


S_DATA, S_RESULTS, S_FOLD_RESULTS, S_SUMMARY, S_PREDICTIONS = run_senssmart_experiment()
Path("paper_outputs").mkdir(exist_ok=True)
S_RESULTS.to_csv("paper_outputs/senssmarttech_seed_results.csv", index=False)
S_FOLD_RESULTS.to_csv("paper_outputs/senssmarttech_fold_results.csv", index=False)
S_SUMMARY.to_csv("paper_outputs/senssmarttech_summary.csv", index=False)
S_PREDICTIONS.to_csv("paper_outputs/senssmarttech_heldout_predictions.csv", index=False)
print(f"SensSmartTech: {len(S_DATA)} recordings, {S_DATA.subject.nunique()} subjects")
print(S_SUMMARY.round(3).to_string(index=False))
