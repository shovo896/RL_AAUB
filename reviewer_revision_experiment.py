"""Reviewer-driven confirmatory experiments for the ECG5000 study.

This module deliberately keeps the official ECG5000 test split untouched during
model search.  It also exposes one-factor-at-a-time controls for the controller,
optimizer, and checkpoint-reuse policy.  The primary warm-start arm uses a
validation gate: whenever the architecture is shape compatible, an inherited
candidate and a fresh candidate receive the same update budget; inheritance is
accepted only if its validation MSE is no worse.  This is an empirical safeguard,
not a theoretical guarantee against negative transfer.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


URLS = {
    "train": "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/classification/UCR/ECG5000/ECG5000_TRAIN.ts?download=true",
    "test": "https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/main/classification/UCR/ECG5000/ECG5000_TEST.ts?download=true",
}
HIDDEN_SPACE = (32, 64, 128)
LAYER_SPACE = (1, 2)
DROPOUT_SPACE = (0.0, 0.2, 0.4)
STATES = tuple((h, l, d) for h in HIDDEN_SPACE for l in LAYER_SPACE for d in DROPOUT_SPACE)
ACTIONS = (
    "increase_hidden",
    "decrease_hidden",
    "increase_layers",
    "decrease_layers",
    "increase_dropout",
    "decrease_dropout",
    "keep",
)


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str
    optimizer: str
    reuse: str


PRIMARY_ARMS = (
    Arm("Q + Adam + gated reuse", "q", "adam", "gated"),
    Arm("Random + Adam + gated reuse", "random", "adam", "gated"),
    Arm("Q + SGD + gated reuse", "q", "sgd", "gated"),
    Arm("Q + Adam + cold start", "q", "adam", "none"),
    Arm("Q + Adam + direct reuse", "q", "adam", "direct"),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _parse_ts(text: str) -> Tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        values, label = line.rsplit(":", 1)
        rows.append([float(v) for v in values.split(",")])
        labels.append(int(float(label)))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_official_ecg5000() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parsed = {}
    for split, url in URLS.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        parsed[split] = _parse_ts(response.text)
    train_x, train_label = parsed["train"]
    test_x, test_label = parsed["test"]
    if train_x.shape != (500, 140) or test_x.shape != (4500, 140):
        raise RuntimeError(f"Unexpected official split: train={train_x.shape}, test={test_x.shape}")
    return train_x, train_label, test_x, test_label


def windows_by_trace(signals: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_trace, length = signals.shape
    n_per_trace = length - window
    x = np.empty((n_trace * n_per_trace, window, 1), dtype=np.float32)
    y = np.empty((n_trace * n_per_trace, 1), dtype=np.float32)
    trace_id = np.repeat(np.arange(n_trace, dtype=np.int64), n_per_trace)
    cursor = 0
    for trace in signals:
        for t in range(window, length):
            x[cursor, :, 0] = trace[t - window : t]
            y[cursor, 0] = trace[t]
            cursor += 1
    return x, y, trace_id


def capped_subset(
    x: np.ndarray,
    y: np.ndarray,
    trace_id: np.ndarray,
    cap: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cap is None or len(x) <= cap:
        return x, y, trace_id
    idx = np.sort(rng.choice(len(x), size=cap, replace=False))
    return x[idx], y[idx], trace_id[idx]


def prepare_splits(
    seed: int,
    train_cap: Optional[int],
    val_cap: Optional[int],
    window: int = 20,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    official_train, train_labels, official_test, _ = load_official_ecg5000()
    indices = np.arange(len(official_train))
    try:
        tr_idx, va_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=train_labels
        )
    except ValueError:
        tr_idx, va_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    scaler.fit(official_train[tr_idx].reshape(-1, 1))
    train_signal = scaler.transform(official_train[tr_idx].reshape(-1, 1)).reshape(len(tr_idx), -1)
    val_signal = scaler.transform(official_train[va_idx].reshape(-1, 1)).reshape(len(va_idx), -1)
    test_signal = scaler.transform(official_test.reshape(-1, 1)).reshape(len(official_test), -1)

    rng = np.random.default_rng(seed)
    train = capped_subset(*windows_by_trace(train_signal, window), train_cap, rng)
    val = capped_subset(*windows_by_trace(val_signal, window), val_cap, rng)
    test = windows_by_trace(test_signal, window)
    return {"train": train, "val": val, "test": test}


def loader(split, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    x, y, _ = split
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


class ECGForecaster(nn.Module):
    def __init__(self, state: Tuple[int, int, float]):
        super().__init__()
        hidden, layers, dropout = state
        effective_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=effective_dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


def next_state(state: Tuple[int, int, float], action: str) -> Tuple[int, int, float]:
    h, layers, dropout = state
    hi, li, di = HIDDEN_SPACE.index(h), LAYER_SPACE.index(layers), DROPOUT_SPACE.index(dropout)
    if action == "increase_hidden": hi = min(hi + 1, len(HIDDEN_SPACE) - 1)
    elif action == "decrease_hidden": hi = max(hi - 1, 0)
    elif action == "increase_layers": li = min(li + 1, len(LAYER_SPACE) - 1)
    elif action == "decrease_layers": li = max(li - 1, 0)
    elif action == "increase_dropout": di = min(di + 1, len(DROPOUT_SPACE) - 1)
    elif action == "decrease_dropout": di = max(di - 1, 0)
    return HIDDEN_SPACE[hi], LAYER_SPACE[li], DROPOUT_SPACE[di]


def shape_compatible(model: nn.Module, checkpoint: Optional[Dict[str, torch.Tensor]]) -> bool:
    if checkpoint is None:
        return False
    current = model.state_dict()
    return current.keys() == checkpoint.keys() and all(current[k].shape == checkpoint[k].shape for k in current)


def optimizer_for(name: str, model: nn.Module, lr: float):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(name)


def train_candidate(model, train_loader, optimizer_name, epochs, lr, device):
    model.to(device)
    opt = optimizer_for(optimizer_name, model, lr)
    loss_fn = nn.MSELoss()
    updates = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            updates += 1
    return updates


@torch.no_grad()
def evaluate(model, data_loader, device) -> Dict[str, float]:
    model.eval()
    targets, preds = [], []
    for xb, yb in data_loader:
        pred = model(xb.to(device)).cpu().numpy().reshape(-1)
        preds.append(pred)
        targets.append(yb.numpy().reshape(-1))
    y = np.concatenate(targets)
    p = np.concatenate(preds)
    err = y - p
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom else 0.0
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "r2": r2}


class Controller:
    def __init__(self, seed: int, alpha=0.4, gamma=0.8, epsilon=0.5):
        self.rng = random.Random(seed)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = {state: {action: 0.0 for action in ACTIONS} for state in STATES}

    def act(self, state, policy):
        if policy == "random" or self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        values = self.q[state]
        best = max(values.values())
        return self.rng.choice([a for a, value in values.items() if value == best])

    def update(self, state, action, reward, new_state):
        current = self.q[state][action]
        target = reward + self.gamma * max(self.q[new_state].values())
        self.q[state][action] = current + self.alpha * (target - current)


def run_arm(
    arm: Arm,
    seed: int,
    splits,
    episodes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    set_seed(seed)
    train_dl = loader(splits["train"], batch_size, True, seed)
    val_dl = loader(splits["val"], batch_size, False, seed)
    test_dl = loader(splits["test"], batch_size, False, seed)
    controller = Controller(seed)
    state = (64, 1, 0.0)
    previous_checkpoint = None
    best = None
    log = []
    start = time.perf_counter()
    total_updates = 0

    for episode in range(1, episodes + 1):
        action = controller.act(state, arm.policy)
        candidate_state = next_state(state, action)
        inherited = ECGForecaster(candidate_state)
        compatible = shape_compatible(inherited, previous_checkpoint)
        if compatible and arm.reuse != "none":
            inherited.load_state_dict(previous_checkpoint)

        inherited_updates = train_candidate(
            inherited, train_dl, arm.optimizer, epochs, lr, device
        )
        total_updates += inherited_updates
        inherited_val = evaluate(inherited, val_dl, device)
        selected_model = inherited
        selected_val = inherited_val
        reuse_accepted = bool(compatible and arm.reuse != "none")

        if compatible and arm.reuse == "gated":
            # Equal-budget counterfactual used only to reject harmful transfer.
            set_seed(seed * 1000 + episode)
            cold = ECGForecaster(candidate_state)
            total_updates += train_candidate(cold, train_dl, arm.optimizer, epochs, lr, device)
            cold_val = evaluate(cold, val_dl, device)
            if cold_val["mse"] < inherited_val["mse"]:
                selected_model, selected_val, reuse_accepted = cold, cold_val, False

        checkpoint = {k: v.detach().cpu().clone() for k, v in selected_model.state_dict().items()}
        previous_checkpoint = checkpoint
        reward = -selected_val["mse"]
        if arm.policy == "q":
            controller.update(state, action, reward, candidate_state)
        row = {
            "arm": arm.name,
            "seed": seed,
            "episode": episode,
            "state": str(state),
            "action": action,
            "new_state": str(candidate_state),
            "shape_compatible": compatible,
            "reuse_accepted": reuse_accepted,
            **{f"val_{k}": v for k, v in selected_val.items()},
        }
        log.append(row)
        if best is None or selected_val["mse"] < best["val_mse"]:
            best = {**row, "checkpoint": checkpoint, "model_state": candidate_state}
        state = candidate_state

    best_model = ECGForecaster(best["model_state"]).to(device)
    best_model.load_state_dict(best.pop("checkpoint"))
    test_metrics = evaluate(best_model, test_dl, device)
    summary = {
        "arm": arm.name,
        "policy": arm.policy,
        "optimizer": arm.optimizer,
        "reuse": arm.reuse,
        "seed": seed,
        "best_episode": best["episode"],
        "best_state": best["new_state"],
        "val_mse": best["val_mse"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "gradient_updates": total_updates,
        "wall_seconds": time.perf_counter() - start,
    }
    return summary, log


def confidence_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["test_mse", "test_rmse", "test_mae", "test_r2", "wall_seconds"]
    rows = []
    for arm, group in results.groupby("arm", sort=False):
        row = {"arm": arm, "seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            mean = values.mean()
            half = 0.0 if len(values) < 2 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = half
        row["gradient_updates_mean"] = group["gradient_updates"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(
    output_dir: Path,
    seeds: Sequence[int] = (11, 22, 33, 44, 55),
    episodes: int = 12,
    epochs: int = 3,
    train_cap: Optional[int] = 20000,
    val_cap: Optional[int] = 5000,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries, episodes_log = [], []
    for seed in seeds:
        splits = prepare_splits(seed, train_cap, val_cap)
        for arm in PRIMARY_ARMS:
            summary, log = run_arm(
                arm, seed, splits, episodes, epochs, batch_size, lr, device
            )
            summaries.append(summary)
            episodes_log.extend(log)
            print(json.dumps(summary, indent=2))

    raw = pd.DataFrame(summaries)
    aggregate = confidence_summary(raw)
    raw.to_csv(output_dir / "reviewer_ablation_per_seed.csv", index=False)
    pd.DataFrame(episodes_log).to_csv(output_dir / "reviewer_ablation_episodes.csv", index=False)
    aggregate.to_csv(output_dir / "reviewer_ablation_summary.csv", index=False)
    print("\nAggregate mean and 95% normal-approximation CI across seeds:")
    print(aggregate.to_string(index=False))
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reviewer_revision_results")
    parser.add_argument("--quick", action="store_true", help="Smoke test only; not paper evidence")
    args = parser.parse_args()
    if args.quick:
        run_experiment(Path(args.output_dir), seeds=(11,), episodes=2, epochs=1, train_cap=2048, val_cap=512)
    else:
        run_experiment(Path(args.output_dir))


if __name__ == "__main__":
    main()
