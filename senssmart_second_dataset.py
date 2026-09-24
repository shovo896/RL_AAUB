"""Reproducible SensSmartTech benchmark used by monte_carlo.ipynb."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks
from sklearn.model_selection import GroupKFold

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
        for algorithm, label in [("mc", "Monte-Carlo"), ("q", "Q-learning"), ("bandit", "Contextual bandit")]:
            for seed in S_SEEDS:
                q = s_train(errors, states, train, algorithm, seed); chosen = np.argmax(q[states[test]], axis=1)
                pred = estimates[test, chosen]; m = s_metrics(pred, ref[test])
                rows.append({"Fold":fold,"Method":label,"Seed":seed, **m})
                predictions.append(pd.DataFrame({"Fold":fold,"Method":label,"Seed":seed,"subject":groups[test],"reference_hr":ref[test],"prediction":pred,"action":chosen}))
    results = pd.DataFrame(rows)
    fold_results = results.groupby(["Fold", "Method"], as_index=False)[["MAE (BPM)", "RMSE (BPM)", "Failed (%)"]].mean()
    summary = fold_results.groupby("Method", as_index=False).agg(**{"Mean MAE (BPM)":("MAE (BPM)","mean"), "SD MAE (BPM)":("MAE (BPM)","std"), "Mean RMSE (BPM)":("RMSE (BPM)","mean"), "Mean Failed (%)":("Failed (%)","mean")}).sort_values("Mean MAE (BPM)")
    return data, results, fold_results, summary, pd.concat(predictions, ignore_index=True)


S_DATA, S_RESULTS, S_FOLD_RESULTS, S_SUMMARY, S_PREDICTIONS = run_senssmart_experiment()
Path("paper_outputs").mkdir(exist_ok=True)
S_RESULTS.to_csv("paper_outputs/senssmarttech_seed_results.csv", index=False)
S_FOLD_RESULTS.to_csv("paper_outputs/senssmarttech_fold_results.csv", index=False)
S_SUMMARY.to_csv("paper_outputs/senssmarttech_summary.csv", index=False)
S_PREDICTIONS.to_csv("paper_outputs/senssmarttech_heldout_predictions.csv", index=False)
print(f"SensSmartTech: {len(S_DATA)} recordings, {S_DATA.subject.nunique()} subjects")
print(S_SUMMARY.round(3).to_string(index=False))
