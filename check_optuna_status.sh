#!/bin/bash
# Quick status check for Optuna sweep

echo "=== Optuna Rosenbrock Sweep Status ==="
echo "Process: $(ps aux | grep 'optuna_sweep_rosenbrock.py' | grep -v grep | awk '{print "PID "$2}')"
echo ""
echo "Latest progress:"
tail -3 optuna_rosenbrock_log.txt | grep -E "Best trial:|%\|"
echo ""
echo "To see full monitor: python monitor_optuna.py"
