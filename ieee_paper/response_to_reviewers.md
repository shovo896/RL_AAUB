# Response to Reviewers

We thank the reviewers for identifying weaknesses in causal attribution, data partitioning, uncertainty reporting, scalability analysis, dataset provenance, and the recency of the related work. We substantially revised both the experiment and manuscript. The revision preserves the official ECG5000 test split, partitions complete development traces before preprocessing and window generation, repeats every search over five declared seeds, separates controller/optimizer/reuse effects, and retrains each selected architecture from scratch under a common final protocol before test evaluation.

## Reviewer 4

### Comment 1: What theoretical criterion prevents negative transfer or suboptimal trapping?

**Response.** Tensor-shape compatibility is only a loading-safety condition and does not theoretically guarantee positive transfer, global optimality, or convergence within a 12-episode horizon. We now state this limitation explicitly. We added a validation-gated rollback: when two architectures are shape-compatible, inherited and freshly initialized branches receive the same three-epoch candidate budget, and only the branch with lower validation MSE is propagated. This is an empirical safeguard, not a theorem. The revised discussion also reports seed-to-seed state variability and avoids any convergence claim.

### Comment 2: How are Adam, inheritance, and effective training age decoupled from the RL controller?

**Response.** We replaced the original joint comparison with five matched arms: (i) Q + Adam + gated reuse, (ii) random + Adam + gated reuse, (iii) Q + SGD + gated reuse, (iv) Q + Adam + cold start, and (v) Q + Adam + direct reuse. All arms use identical data partitions, seeds, search horizon, candidate epoch budget, and reward. More importantly, after search, the validation-selected architecture from every arm is reinitialized; inherited parameters and optimizer state are discarded. It is then trained with the same Adam optimizer for at most 50 epochs with validation patience five. The official test split is accessed only after this common final-training stage.

### Comment 3: What are the scalability and computational overhead?

**Response.** We added an explicit analysis. The current Q table contains only 18 x 7 = 126 entries. For input dimension d, width H, depth L, and window length w, recurrent computation scales approximately as O(wL(H^2+dH)); multi-lead input increases the first-layer term linearly. Adding a discrete hyperparameter with m values multiplies the tabular state count by m. We also report mean gradient updates and wall times. The primary validation-gated Q arm uses 4,128 mean updates, compared with 3,320 for cold-start Q, quantifying the cost of the rollback safeguard.

### Comment 4: Cite the authentic ECG5000 source.

**Response.** We now cite the UCR/UEA archive paper, the ECG5000 Zenodo archive record (DOI 10.5281/zenodo.11186692), and PhysioNet. We state that ECG5000 was derived from the BIDMC Congestive Heart Failure record `chf07`, and we explicitly limit claims because the beats do not constitute independent patients.

### Comment 5: Add recent references and comparison with 2020--2026 techniques.

**Response.** The revision discusses Informer (2021), PatchTST (2023), TimesNet (2023), DLinear (2023), and ModernTCN (2024). Their published long-horizon results are not numerically commensurate with our one-step, length-20 ECG endpoint, so we do not copy incomparable numbers. The empirical comparison instead uses a matched random-search controller and optimizer/reuse ablations on the identical LSTM family. We explicitly identify task-adapted evaluation of modern forecasting architectures on the same endpoint as required before any numerical state-of-the-art claim.

## Reviewer 8

### Comment: Add controlled experiments, multiple seeds, an untouched test set, and separate ablations.

**Response.** Completed. All five arms were run with seeds 11, 22, 33, 44, and 55. Development traces are split before scaling and window generation; the scaler is fitted only on training traces. The official 4,500-trace test partition is excluded from search, gating, architecture selection, and early stopping. We report per-seed metrics and mean plus 95% Student-t interval half-widths. Under common final retraining, Q + Adam + gated reuse obtains test MSE 0.02355 +/- 0.00162, whereas Q + Adam + cold start obtains 0.02288 +/- 0.00092. The intervals overlap, so the revision does not claim a statistically resolved advantage for RL or reuse. This controlled result replaces the earlier confounded tenfold-improvement narrative.
