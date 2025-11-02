# Optuna Hyperparameter Optimization

This directory contains tools for Bayesian hyperparameter optimization using Optuna.

## Overview

**Goal**: Find optimal hyperparameters for the Evolvion evolutionary algorithm using TPE (Tree-structured Parzen Estimator) Bayesian optimization.

**Evaluation task**: Spiral classification (2→8→8→1 network, 150 generations, 3 seeds averaged)

**Fitness metric**: Negative MSE (mean squared error). Higher is better (closer to 0 = better classification).

## Components

1. **Evolvatron.OptunaEval/** - C# CLI tool that runs evolution with given hyperparameters
2. **optuna_sweep.py** - Python script that orchestrates the optimization
3. **optuna_evolvion.db** - SQLite database storing trial history (created on first run)

## Hyperparameter Space

### Complete List of Tunable Hyperparameters

#### Population Structure (NEAT-style vs Traditional)
- `species_count` (int, 4-30): Number of species in population
- `individuals_per_species` (int, 20-100): Population size per species
- `min_species_count` (int, 2-10): Minimum species count (prevents collapse)

#### Selection Pressure
- `elites` (int, 1-4): Number of elites preserved per species
- `tournament_size` (int, 4-32): Tournament size for parent selection
- `parent_pool_percentage` (float, 0.5-1.0): Fraction of population eligible as parents

#### Culling Thresholds (NEW - Phase 10)
- `grace_generations` (int, 0-3): Generations of protection for new species
- `stagnation_threshold` (int, 3-15): Generations without improvement before culling eligibility
- `species_diversity_threshold` (float, 0.01-0.20): Minimum fitness variance to avoid culling
- `relative_performance_threshold` (float, 0.3-0.9): Performance fraction vs best species

#### Weight Mutations (Phase 7 + Phase 9 insights)
- `weight_jitter` (float, 0.8-1.0): Probability of adding Gaussian noise to weight
- `weight_jitter_stddev` (float, 0.1-0.5): Std dev for weight jitter (as fraction of weight magnitude)
- `weight_reset` (float, 0.0-0.2): Probability of resetting weight to random value
- `weight_l1_shrink` (float, 0.0-0.4): Probability of shrinking weight toward zero
- `l1_shrink_factor` (float, 0.85-0.95): Shrinkage factor (0.9 = reduce by 10%)
- `activation_swap` (float, 0.0-0.20): Probability of swapping activation function

#### Node Parameters (Phase 7: disabled was best, but allows verification)
- `node_param_mutate` (float, 0.0-0.1): Probability of mutating node parameters (alpha, beta)
- `node_param_stddev` (float, 0.05-0.2): Std dev for node parameter mutations

#### Topology Mutations
- `edge_add` (float, 0.0-0.15): Probability of adding new edge
- `edge_delete_random` (float, 0.0-0.05): Probability of deleting random edge
- `edge_split` (float, 0.0-0.05): Probability of splitting edge (adds node)
- `edge_redirect` (float, 0.0-0.10): Probability of redirecting edge source/dest
- `edge_swap` (float, 0.0-0.05): Probability of swapping two edges

#### Weak Edge Pruning
- `weak_edge_pruning_enabled` (bool): Enable automatic pruning of weak edges
- `weak_edge_pruning_threshold` (float, 0.001-0.05): Weight threshold for "weak" edge
- `weak_edge_pruning_base_rate` (float, 0.5-0.9): Base probability of pruning weak edges
- `weak_edge_pruning_on_birth` (bool): Apply pruning when species is born
- `weak_edge_pruning_during_evolution` (bool): Apply pruning during normal evolution

**Total: 31 hyperparameters**

### Hardcoded Values in Codebase (Not Exposed)

The following are hardcoded in `SpeciesDiversification.cs` and could be surfaced if needed:

- **Bias initialization scale**: `0.1f` (line 191, 207, 406)
- **Node parameter defaults**: `alpha=0.01f, beta=1.0f` (lines 203-204, 397-398)
- **Hidden layer size mutation range**: `±2 nodes, clamped [2, 16]` (lines 268-272)
- **Activation toggle count**: `1-3 activations` (line 288)
- **MaxInDegree mutation range**: `±1, clamped [4, 12]` (lines 314-317)

These were not included in the sweep because they relate to structural mutations (topology diversification) which are less critical than weight/selection parameters.

## Installation

### Prerequisites
- .NET 8.0 SDK
- Python 3.8+
- Optuna: `pip install optuna optuna-dashboard`

### Build the CLI Tool
```bash
cd Evolvatron.OptunaEval
dotnet build -c Release
```

## Usage

### Quick Test (1 trial)
```bash
python optuna_sweep.py --n-trials 1
```

### Full Optimization (100 trials)
```bash
python optuna_sweep.py --n-trials 100 --study-name evolvion_sweep_v1
```

### Parallel Execution (use with caution - high CPU usage!)
```bash
python optuna_sweep.py --n-trials 100 --n-jobs 4
```

### Continue Previous Study
```bash
python optuna_sweep.py --n-trials 50 --study-name evolvion_sweep_v1 --storage sqlite:///optuna_evolvion.db
```

## Monitoring

### View Real-Time Dashboard
```bash
optuna-dashboard sqlite:///optuna_evolvion.db
```
Then open http://localhost:8080 in your browser.

### Check Best Parameters So Far
```bash
cat optuna_best_params.txt
```

## Expected Runtime

- **Single trial**: ~2-3 minutes (3 seeds × 150 generations each)
- **100 trials**: ~3-5 hours (sequential)
- **100 trials (4 parallel)**: ~1-2 hours (high CPU usage!)

## Results Interpretation

- **Fitness**: Negative MSE (e.g., -0.86 means MSE=0.86)
- **Higher is better** (closer to 0 = better classification)
- **Baseline**: Current default config achieves ~-0.88 to -0.89

## Phase History Context

This Optuna sweep builds on findings from previous phases:

- **Phase 2**: Tournament size (+0.743 correlation), low elitism (-0.264)
- **Phase 6**: 4×200 beats 8×100 by +3.3% (but lacks topology exploration)
- **Phase 7**: Bias mutations +42.4%, mixed activations +33.3%, disable node params +38.9%
- **Phase 8**: Comprehensive sweep of 16 hyperparameters (20 combos × 5 seeds)
- **Phase 9**: Mutation ablation study (weight jitter, L1 shrink, activation swap all beneficial)
- **Phase 10**: NEAT-style population (20×40 + culling) enables topology exploration

## Troubleshomarks

### "optuna not found"
```bash
pip install optuna
```

### "dotnet not found"
Install .NET 8.0 SDK from https://dot.net

### Trial timeouts
Increase timeout in `optuna_sweep.py` line 118:
```python
timeout=600,  # Increase to 900 for slower machines
```

### Out of memory
Reduce `individuals_per_species` range or use fewer parallel jobs.

## Advanced: Custom Parameter Ranges

Edit `optuna_sweep.py` function `objective()` to adjust ranges:

```python
tournament_size = trial.suggest_int("tournament_size", 8, 64)  # Expand range
```

## Next Steps

After optimization completes:
1. Review `optuna_best_params.txt`
2. Update `EvolutionConfig.cs` defaults with best parameters
3. Run validation sweep with best parameters (10+ seeds)
4. Consider Phase 11: Multi-objective optimization (fitness vs topology diversity)
