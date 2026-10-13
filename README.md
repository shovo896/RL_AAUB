# RL_AAUB

Research notebooks and reproducible experiments for reinforcement learning,
sequential decision-making, and neural time-series forecasting.

## ECG forecasting reviewer-revision experiment

The repository includes a controlled revision of the Bellman-guided LSTM
configuration-search experiment used in Paper 93, *Bellman-Guided Warm-Start
Search for LSTM Configuration in ECG Forecasting*.

Primary entry point:

- `lstm_bellman_reviewer_revision.ipynb` — stepwise notebook for the revised
  five-seed experiment.
- `reviewer_revision_experiment.py` — script version of the same controlled
  protocol.
- `lstm+bellman_eqn.ipynb` — original Bellman-guided LSTM exploration.

### Evaluation controls

- ECG traces are partitioned before window extraction.
- Normalization statistics are fitted only on training traces.
- The official ECG5000 test split is excluded from search and model selection.
- Five declared seeds are used: `11`, `22`, `33`, `44`, and `55`.
- Q-learning, random control, optimizer choice, and checkpoint-reuse policies
  are compared through one-factor-at-a-time ablations.
- Every selected architecture is reinitialized and trained under a common
  Adam and early-stopping protocol before final test evaluation.
- Results report seed-level uncertainty and gradient-update counts alongside
  prediction metrics.

The experiment is a methodological signal-forecasting study. ECG class labels
are not prediction targets, and the reported one-step errors should not be
interpreted as diagnostic or clinical-performance measures.

## Reproducibility

Run the notebook from top to bottom in a Python environment containing
PyTorch, NumPy, pandas, scikit-learn, and Matplotlib. Generated experiment
tables and histories are written to the configured results directory; large
runtime artifacts and local environments are intentionally excluded from Git.
