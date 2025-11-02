#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Rosenbrock Navigation
Runs Bayesian optimization to find hyperparameters that solve Rosenbrock valley navigation.

Usage:
    python optuna_sweep_rosenbrock.py --n-trials 200 --storage sqlite:///optuna_rosenbrock.db --study-name rosenbrock_v1

Dependencies:
    pip install optuna
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def objective(trial):
    """
    Optuna objective function for Rosenbrock optimization.

    Returns:
        float: Solve rate (0-100) + avg fitness (higher = better)
               Example: 60% solve rate + avg fitness -0.05 = 59.95
    """

    # Population structure
    species_count = trial.suggest_int("species_count", 10, 40)
    individuals_per_species = trial.suggest_int("individuals_per_species", 50, 150)
    min_species_count = trial.suggest_int("min_species_count", 3, max(3, species_count // 3))

    # Selection pressure - CRITICAL FOR VALLEY NAVIGATION
    # Hypothesis: Lower tournament size = less selection pressure = better diversity for valley exploration
    elites = trial.suggest_int("elites", 2, 8)
    tournament_size = trial.suggest_int("tournament_size", 4, 30)
    parent_pool_percentage = trial.suggest_float("parent_pool_percentage", 0.5, 1.0)

    # Culling thresholds - may need relaxation for valley navigation
    grace_generations = trial.suggest_int("grace_generations", 0, 5)
    stagnation_threshold = trial.suggest_int("stagnation_threshold", 4, 20)
    species_diversity_threshold = trial.suggest_float("species_diversity_threshold", 0.01, 0.25)
    relative_performance_threshold = trial.suggest_float("relative_performance_threshold", 0.3, 0.95)

    # Weight mutations
    weight_jitter = trial.suggest_float("weight_jitter", 0.7, 1.0)
    weight_jitter_stddev = trial.suggest_float("weight_jitter_stddev", 0.05, 0.6)
    weight_reset = trial.suggest_float("weight_reset", 0.0, 0.25)
    weight_l1_shrink = trial.suggest_float("weight_l1_shrink", 0.0, 0.3)
    l1_shrink_factor = trial.suggest_float("l1_shrink_factor", 0.8, 0.98)
    activation_swap = trial.suggest_float("activation_swap", 0.0, 0.30)

    # Node parameter mutation
    node_param_mutate = trial.suggest_float("node_param_mutate", 0.0, 0.15)
    node_param_stddev = trial.suggest_float("node_param_stddev", 0.02, 0.3)

    # Topology mutations - might help escape plateaus
    edge_add = trial.suggest_float("edge_add", 0.0, 0.20)
    edge_delete_random = trial.suggest_float("edge_delete_random", 0.0, 0.08)
    edge_split = trial.suggest_float("edge_split", 0.0, 0.08)
    edge_redirect = trial.suggest_float("edge_redirect", 0.0, 0.15)
    edge_swap = trial.suggest_float("edge_swap", 0.0, 0.08)

    # Weak edge pruning
    weak_edge_pruning_enabled = trial.suggest_categorical("weak_edge_pruning_enabled", [True, False])
    weak_edge_pruning_threshold = trial.suggest_float("weak_edge_pruning_threshold", 0.001, 0.08)
    weak_edge_pruning_base_rate = trial.suggest_float("weak_edge_pruning_base_rate", 0.4, 0.95)
    weak_edge_pruning_on_birth = trial.suggest_categorical("weak_edge_pruning_on_birth", [True, False])
    weak_edge_pruning_during_evolution = trial.suggest_categorical("weak_edge_pruning_during_evolution", [True, False])

    # Build command-line arguments
    args = [
        "dotnet", "run", "--project", "Evolvatron.OptunaEval.Rosenbrock/Evolvatron.OptunaEval.Rosenbrock.csproj",
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

    # Run evaluation (5 seeds, 100 generations each)
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout (5 seeds × 100 gens can take time)
            check=True
        )

        # Parse output (single line with fitness value)
        fitness_str = result.stdout.strip().split('\n')[-1]
        fitness = float(fitness_str)

        # Report for monitoring
        trial.set_user_attr("fitness", fitness)

        return fitness  # Maximize: solve_rate% + avg_fitness

    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out!")
        return float('-inf')
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error: {e}")
        print(f"Stderr: {e.stderr}")
        return float('-inf')
    except ValueError as e:
        print(f"Trial {trial.number} failed to parse fitness: {e}")
        print(f"Output: {result.stdout}")
        return float('-inf')


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Rosenbrock")
    parser.add_argument("--n-trials", type=int, default=200,
                      help="Number of trials to run")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_rosenbrock.db",
                      help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="rosenbrock_v1",
                      help="Name of the Optuna study")
    parser.add_argument("--n-jobs", type=int, default=1,
                      help="Number of parallel jobs (careful: very CPU intensive)")
    args = parser.parse_args()

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",  # Maximize solve rate + fitness
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30)
    )

    print("="*80)
    print("ROSENBROCK HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nStudy: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"\nEach trial tests 5 seeds × 100 generations")
    print(f"Success threshold: -0.10 (10% of optimum)")
    print(f"Fitness: solve_rate% + avg_fitness (higher = better)")
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
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best fitness: {study.best_value:.6f}")

    # Decode fitness (solve_rate% + avg_fitness)
    solve_rate_approx = int(study.best_value) if study.best_value > 0 else 0
    avg_fitness_approx = study.best_value - solve_rate_approx
    print(f"  (≈ {solve_rate_approx}% solve rate, avg fitness {avg_fitness_approx:.6f})")

    print("\nBest hyperparameters:")
    for key, value in sorted(study.best_params.items()):
        print(f"  {key}: {value}")

    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path(f"optuna_rosenbrock_results_{timestamp}.txt")
    with open(results_file, "w") as f:
        f.write(f"Rosenbrock Hyperparameter Optimization Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best fitness: {study.best_value:.6f}\n")
        f.write(f"  Solve rate: ≈{solve_rate_approx}%\n")
        f.write(f"  Avg fitness: ≈{avg_fitness_approx:.6f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in sorted(study.best_params.items()):
            f.write(f"{key}={value}\n")

        # Write top 10 trials
        f.write(f"\n{'='*80}\n")
        f.write("Top 10 Trials:\n")
        f.write(f"{'='*80}\n")
        trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
        for i, t in enumerate(trials[:10], 1):
            if t.value is not None:
                sr = int(t.value) if t.value > 0 else 0
                af = t.value - sr
                f.write(f"\n#{i} Trial {t.number}: {t.value:.6f} (≈{sr}% solve, {af:.6f} fitness)\n")
                for key, value in sorted(t.params.items()):
                    f.write(f"  {key}={value}\n")

    print(f"\nResults saved to: {results_file}")
    print(f"Study database: {args.storage}")
    print("\nView results interactively:")
    print(f"  pip install optuna-dashboard")
    print(f"  optuna-dashboard {args.storage}")


if __name__ == "__main__":
    main()
