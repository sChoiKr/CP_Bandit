# Bandit CP recovery pipeline (Julia)

This project tests one claim:

> Low-dimensional agent coordinates obtained from a CP approximation of
> observable Q-learning trajectories retain predictive information about the
> ground-truth `(alpha, beta)` parameters of unseen agents.

The main tensor contains only four observable features: two choice indicators
and the obtained reward masked by the chosen arm. Ground-truth parameters,
potential outcomes, and Q-values are kept outside the main tensor.

## Pipeline

1. Draw continuous `alpha ~ U(0.05, 0.95)` and `beta ~ U(0.5, 8)`.
2. Generate one reversal-bandit potential-reward schedule shared by all agents.
3. Simulate Q-learning and construct `X_obs :: N x T x 4`.
4. Split agents before fitting; CPD sees training agents only.
5. Fit ordinary CPD with TensorKitchen from multiple random starts.
6. Absorb CP weights into training coordinates: `Z_train = A * Diagonal(lambda)`.
7. Project each unseen agent through fixed `B, C` factors.
8. Predict `alpha` and `beta` with the same standardized ridge model used for
   PCA and raw-trajectory baselines.
9. Report rank-wise recovery, reconstruction, multistart stability, and a
   numerical Kruskal sufficient-condition check.

Parameter draws, potential rewards, stochastic choices, agent splitting, and
CPD initialization use separate seeds. This allows environment-only repeated
runs while keeping the ground-truth agent population fixed.

The default reward environment reverses the high-probability arm every 30
trials. This makes learning rate behaviorally relevant while preserving the
shared-environment design.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'

# Fast structural check
julia --project=. scripts/run_pilot.jl --smoke

# N=500, T=150, 400/100 split, ranks 2:6, 10 starts
julia --project=. scripts/run_pilot.jl

# Explicit reference condition containing Q1 and Q2
julia --project=. scripts/run_pilot.jl --oracle

# Paired environment replications (five seeds by default)
julia --project=. scripts/run_environment_replicates.jl

# Fast two-environment structural check
julia --project=. scripts/run_environment_replicates.jl --smoke
```

Use `--output PATH` to choose a result directory. Each run writes:

- `rank_metrics.csv`: held-out `R2`, MAE, RMSE and compression ratios;
- `stability.csv`: objective spread and aligned factor congruence;
- `identifiability.csv`: numerical ranks, pairwise feature-factor separation,
  and the Kruskal sufficient-condition flag;
- `projection_conditioning.csv`: singular-value and condition-number diagnostics
  for the fixed out-of-sample basis `C ⊙ B`;
- `projection_sensitivity.csv`: held-out recovery after relative ridge penalties
  are applied only to unseen-agent CP projection;
- `run.jls`: the full reproducibility artifact, including configuration,
  trajectories, split indices, and best factors.

Environment replication additionally writes `replicate_metrics.csv` with every
seed-level result and `metric_summary.csv` with the mean, sample standard
deviation, standard error, and Student-t 95% confidence interval.
Use `--seeds 101,102,103` to supply an explicit seed set.

The Kruskal flag concerns essential uniqueness of the returned CP
representation. It is not a claim that the optimizer found a unique global
minimizer. Recoverability, CP identifiability, and optimization stability are
reported separately.
