#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Evolvion
Runs Bayesian optimization using TPE sampler to find optimal hyperparameters.

Usage:
    python optuna_sweep.py --n-trials 100 --storage sqlite:///optuna_evolvion.db --study-name evolvion_sweep_v1

Dependencies:
    pip install optuna
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def objective(trial):
    """
    Optuna objective function.
    Runs the C# CLI tool with sampled hyperparameters and returns fitness.

    Returns:
        float: Mean fitness across 3 seeds (negative MSE - we want to MAXIMIZE this = minimize loss)
    """

    # Population structure (NEAT-style vs traditional tradeoff)
    species_count = trial.suggest_int("species_count", 4, 30)
    individuals_per_species = trial.suggest_int("individuals_per_species", 20, 100)
    # Keep total population ~800 by suggesting complementary values
    # (Optuna will explore deviations naturally)

    min_species_count = trial.suggest_int("min_species_count", 2, max(2, species_count // 3))

    # Selection pressure
    elites = trial.suggest_int("elites", 1, 4)
    tournament_size = trial.suggest_int("tournament_size", 4, 32)
    parent_pool_percentage = trial.suggest_float("parent_pool_percentage", 0.5, 1.0)

    # Culling thresholds (NEW - Phase 10)
    grace_generations = trial.suggest_int("grace_generations", 0, 3)
    stagnation_threshold = trial.suggest_int("stagnation_threshold", 3, 15)
    species_diversity_threshold = trial.suggest_float("species_diversity_threshold", 0.01, 0.20)
    relative_performance_threshold = trial.suggest_float("relative_performance_threshold", 0.3, 0.9)

    # Weight mutations (Phase 7 + Phase 9 findings)
    weight_jitter = trial.suggest_float("weight_jitter", 0.8, 1.0)
    weight_jitter_stddev = trial.suggest_float("weight_jitter_stddev", 0.1, 0.5)
    weight_reset = trial.suggest_float("weight_reset", 0.0, 0.2)
    weight_l1_shrink = trial.suggest_float("weight_l1_shrink", 0.0, 0.4)
    l1_shrink_factor = trial.suggest_float("l1_shrink_factor", 0.85, 0.95)
    activation_swap = trial.suggest_float("activation_swap", 0.0, 0.20)

    # Node parameter mutation (Phase 7: disabled was best, but allow Optuna to verify)
    node_param_mutate = trial.suggest_float("node_param_mutate", 0.0, 0.1)
    node_param_stddev = trial.suggest_float("node_param_stddev", 0.05, 0.2)

    # Topology mutations
    edge_add = trial.suggest_float("edge_add", 0.0, 0.15)
    edge_delete_random = trial.suggest_float("edge_delete_random", 0.0, 0.05)
    edge_split = trial.suggest_float("edge_split", 0.0, 0.05)
    edge_redirect = trial.suggest_float("edge_redirect", 0.0, 0.10)
    edge_swap = trial.suggest_float("edge_swap", 0.0, 0.05)

    # Weak edge pruning
    weak_edge_pruning_enabled = trial.suggest_categorical("weak_edge_pruning_enabled", [True, False])
    weak_edge_pruning_threshold = trial.suggest_float("weak_edge_pruning_threshold", 0.001, 0.05)
    weak_edge_pruning_base_rate = trial.suggest_float("weak_edge_pruning_base_rate", 0.5, 0.9)
    weak_edge_pruning_on_birth = trial.suggest_categorical("weak_edge_pruning_on_birth", [True, False])
    weak_edge_pruning_during_evolution = trial.suggest_categorical("weak_edge_pruning_during_evolution", [True, False])

    # Build command-line arguments
    args = [
        "dotnet", "run", "--project", "Evolvatron.OptunaEval/Evolvatron.OptunaEval.csproj",
        "-c", "Release", "--",
        f"species_count={species_count}",
        f"individuals_per_species={individuals_per_species}",
        f"min_species_count={min_species_count}",
        f"elites={elites}",
        f"tournament_size={tournament_size}",
        f"parent_pool_percentage={parent_pool_percentage}",
        f"grace_generations={grace_generations}",
        f"stagnation_threshold={stagnation_threshold}",
        f"species_diversity_threshold={species_diversity_threshold}",
        f"relative_performance_threshold={relative_performance_threshold}",
        f"weight_jitter={weight_jitter}",
        f"weight_jitter_stddev={weight_jitter_stddev}",
        f"weight_reset={weight_reset}",
        f"weight_l1_shrink={weight_l1_shrink}",
        f"l1_shrink_factor={l1_shrink_factor}",
        f"activation_swap={activation_swap}",
        f"node_param_mutate={node_param_mutate}",
        f"node_param_stddev={node_param_stddev}",
        f"edge_add={edge_add}",
        f"edge_delete_random={edge_delete_random}",
        f"edge_split={edge_split}",
        f"edge_redirect={edge_redirect}",
        f"edge_swap={edge_swap}",
        f"weak_edge_pruning_enabled={str(weak_edge_pruning_enabled).lower()}",
        f"weak_edge_pruning_threshold={weak_edge_pruning_threshold}",
        f"weak_edge_pruning_base_rate={weak_edge_pruning_base_rate}",
        f"weak_edge_pruning_on_birth={str(weak_edge_pruning_on_birth).lower()}",
        f"weak_edge_pruning_during_evolution={str(weak_edge_pruning_during_evolution).lower()}",
    ]

    # Run evaluation
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per evaluation (3 seeds × 150 gens each)
            check=True
        )

        # Parse output (single line with fitness value)
        fitness_str = result.stdout.strip().split('\n')[-1]
        fitness = float(fitness_str)

        # Report intermediate values for monitoring
        trial.set_user_attr("fitness", fitness)

        return fitness  # Maximize fitness (less negative = better)

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out!")
        return float('-inf')  # Worst possible fitness
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error: {e}")
        print(f"Stderr: {e.stderr}")
        return float('-inf')
    except ValueError as e:
        print(f"Trial {trial.number} failed to parse fitness: {e}")
        print(f"Output: {result.stdout}")
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Evolvion")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_evolvion.db",
                      help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--study-name", type=str, default="evolvion_sweep_v1",
                      help="Name of the Optuna study")
    parser.add_argument("--n-jobs", type=int, default=1,
                      help="Number of parallel jobs (use with caution - high CPU usage)")
    args = parser.parse_args()

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",  # Maximize fitness (less negative MSE = better)
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    )

    print(f"Starting Optuna optimization:")
    print(f"  Study: {args.study_name}")
    print(f"  Storage: {args.storage}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print()

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best fitness: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results to file
    results_file = Path("optuna_best_params.txt")
    with open(results_file, "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best fitness: {study.best_value:.6f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}={value}\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Study database: {args.storage}")
    print("\nView results interactively:")
    print(f"  optuna-dashboard {args.storage}")


if __name__ == "__main__":
    main()
