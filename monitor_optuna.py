#!/usr/bin/env python3
"""
Monitor Optuna study progress in real-time.

Usage:
    python monitor_optuna.py --storage sqlite:///optuna_rosenbrock.db --study-name rosenbrock_valley_v1
"""

import argparse
import time
from datetime import datetime

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", default="sqlite:///optuna_rosenbrock.db")
    parser.add_argument("--study-name", default="rosenbrock_valley_v1")
    parser.add_argument("--interval", type=int, default=60, help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"Monitoring Optuna study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Refresh every {args.interval} seconds")
    print(f"Press Ctrl+C to exit\n")

    while True:
        try:
            study = optuna.load_study(study_name=args.study_name, storage=args.storage)

            print(f"\n{'='*80}")
            print(f"Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")

            trials = study.trials
            completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            running_trials = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]

            print(f"Trials: {len(completed_trials)} completed, {len(running_trials)} running, {len(trials)} total")

            if completed_trials:
                best_trial = study.best_trial
                best_value = study.best_value

                # Decode fitness
                solve_rate = int(best_value) if best_value > 0 else 0
                avg_fitness = best_value - solve_rate

                print(f"\nBest trial so far: #{best_trial.number}")
                print(f"Best fitness: {best_value:.6f}")
                print(f"  Solve rate: ≈{solve_rate}% ({solve_rate // 20}/5 seeds)")
                print(f"  Avg fitness: ≈{avg_fitness:.6f}")

                print(f"\nTop 5 trials:")
                top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
                for i, t in enumerate(top_trials, 1):
                    sr = int(t.value) if t.value > 0 else 0
                    af = t.value - sr
                    print(f"  #{i}. Trial {t.number}: {t.value:.4f} (≈{sr}%, {af:.4f})")

                # Calculate progress
                if len(completed_trials) > 1:
                    recent = completed_trials[-10:]
                    if len(recent) > 1:
                        avg_time = sum((t.datetime_complete - t.datetime_start).total_seconds() for t in recent) / len(recent)
                        remaining = 200 - len(completed_trials)
                        eta_seconds = remaining * avg_time
                        eta_hours = eta_seconds / 3600
                        print(f"\nProgress: {len(completed_trials)}/200 ({len(completed_trials)*100//200}%)")
                        print(f"Avg time per trial: {avg_time:.1f}s")
                        print(f"ETA: {eta_hours:.1f} hours")

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
