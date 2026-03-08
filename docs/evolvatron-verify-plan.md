# Evolvatron-Verify: PPO Baseline Comparison

## Goal

Create `C:\Dev\Evolvatron-Verify\` — a Python project that trains PPO (and variants) on the **exact same** Double Pole No-Velocity (DPNV) benchmark used by Evolvion, using the **exact same** 16 fixed starting positions. This gives us a rigorous apples-to-apples comparison: neuroevolution (Evolvion) vs gradient-based RL (PPO).

If Evolvion can't beat PPO on a GPU, we've picked the wrong algorithm.

## Physics Parity (CRITICAL)

The Python env must match our C#/CUDA physics **exactly**:

| Constant | Value |
|----------|-------|
| Gravity | -9.8 m/s^2 |
| MassCart | 1.0 kg |
| Length1 (half-length pole1) | 0.5 m |
| MassPole1 | 0.1 kg |
| Length2 (half-length pole2) | 0.05 m |
| MassPole2 | 0.01 kg |
| ForceMag | 10.0 N |
| TimeDelta | 0.01 s |
| Mup (friction) | 0.000002 |
| Track half-length | 2.4 m |
| Pole angle threshold | 36 degrees |
| Integration | **2x RK4 per tick** |
| Max steps | 100,000 |

The integration is Colonel-style: each "tick" calls RK4 twice with the same action, advancing 0.02s total per tick.

### Observation space (no-velocity mode)
- 3 floats: `[cartPos/2.4, pole1Angle/0.6283, pole2Angle/0.6283]`
- Normalized to roughly [-1, 1]

### Action space
- 1 float in [-1, 1], scaled to force = action * 10.0 N

### Terminal conditions
- Cart position outside [-2.4, 2.4]
- Either pole angle outside [-36deg, 36deg] ([-0.6283, 0.6283] rad)
- Steps >= 100,000 (solved!)

## 16 Fixed Starting Positions

Stored in `starting_positions.json` — shared between Evolvatron and Evolvatron-Verify.

State format: `[cartPos, cartVel, pole1Angle, pole1AngVel, pole2Angle, pole2AngVel]`

Angles in radians. 4 degrees = 0.0698 rad, 10 deg = 0.1745, 15 deg = 0.2618.

### Tier 1: Gentle (positions 0-3)
Classic-style perturbations. Any decent controller handles these.

| # | cartPos | cartVel | p1Angle | p1AngVel | p2Angle | p2AngVel | Description |
|---|---------|---------|---------|----------|---------|----------|-------------|
| 0 | 0.0 | 0.0 | 0.0698 | 0.0 | 0.0 | 0.0 | Standard (4 deg) |
| 1 | 0.0 | 0.0 | -0.0698 | 0.0 | 0.0 | 0.0 | Mirror of standard |
| 2 | 0.0 | 0.0 | 0.0524 | 0.0 | -0.0524 | 0.0 | Both poles off (3 deg) |
| 3 | 0.5 | 0.0 | 0.0698 | 0.0 | 0.0 | 0.0 | Cart offset right |

### Tier 2: Moderate (positions 4-7)
Larger perturbations requiring real control.

| # | cartPos | cartVel | p1Angle | p1AngVel | p2Angle | p2AngVel | Description |
|---|---------|---------|---------|----------|---------|----------|-------------|
| 4 | 0.0 | 0.0 | 0.1745 | 0.0 | 0.0 | 0.0 | Wide pole1 (10 deg) |
| 5 | 0.0 | 1.0 | 0.0698 | 0.0 | 0.0 | 0.0 | Moving cart |
| 6 | 0.0 | 0.0 | 0.0349 | 0.0 | 0.1396 | 0.0 | Pole2 falling (8 deg) |
| 7 | 0.0 | 0.0 | 0.0698 | 1.0 | 0.0 | 0.0 | Pole1 spinning |

### Tier 3: Hard (positions 8-11)
Multiple state dimensions perturbed simultaneously.

| # | cartPos | cartVel | p1Angle | p1AngVel | p2Angle | p2AngVel | Description |
|---|---------|---------|---------|----------|---------|----------|-------------|
| 8 | 1.0 | 0.0 | 0.1396 | 0.0 | -0.0873 | 0.0 | Cart right, poles diverge |
| 9 | 0.0 | 1.5 | -0.1047 | 0.0 | 0.0698 | 0.0 | Fast cart, opposing poles |
| 10 | 0.0 | 0.0 | 0.0873 | -1.5 | -0.0524 | 1.0 | Angular motion |
| 11 | -0.8 | -0.8 | 0.1745 | 0.5 | -0.0873 | -0.5 | Everything perturbed |

### Tier 4: Extreme (positions 12-15)
Near the edge of solvability. Tests robustness.

| # | cartPos | cartVel | p1Angle | p1AngVel | p2Angle | p2AngVel | Description |
|---|---------|---------|---------|----------|---------|----------|-------------|
| 12 | 1.5 | 0.0 | 0.2094 | 0.0 | -0.1396 | 0.0 | Cart near edge (12+8 deg) |
| 13 | 0.0 | 0.0 | 0.2618 | 2.0 | 0.0 | 0.0 | Big angle + spin (15 deg) |
| 14 | -1.0 | 1.2 | -0.1745 | -1.0 | 0.1396 | 1.5 | Full chaos |
| 15 | 0.3 | -1.5 | 0.2094 | -0.8 | -0.1047 | 0.8 | Mixed perturbation |

## RL Frameworks and Algorithms

### Primary: Stable Baselines3 + sb3-contrib
- **RecurrentPPO** (LSTM policy) — the proper solution for non-Markovian DPNV
- **PPO + frame stacking** (stack last 8-16 frames) — simpler baseline
- **SAC + frame stacking** — off-policy alternative, often better for continuous control

### Why RecurrentPPO is the right competitor
DPNV without velocity is non-Markovian — the agent can't observe velocities. Our Evolvion uses Elman recurrence (ctx=2) to give the network memory. The fair PPO comparison must also have memory:
- RecurrentPPO uses LSTM cells in the policy — direct analog to Elman recurrence
- Frame stacking approximates memory by concatenating recent observations

### Secondary (stretch): CleanRL
- Simpler, single-file implementations
- Good for understanding exactly what's happening
- Less tuned than SB3 but more transparent

## Project Structure

```
Evolvatron-Verify/
  starting_positions.json     # THE 16 positions (shared with Evolvatron)
  requirements.txt            # sb3, sb3-contrib, gymnasium, torch, numpy
  envs/
    double_pole.py            # Gymnasium env with exact physics parity
    test_physics.py           # Verify physics matches C# output
  train/
    train_recurrent_ppo.py    # RecurrentPPO (LSTM) training
    train_ppo_stacked.py      # PPO + frame stacking
    train_sac_stacked.py      # SAC + frame stacking
  eval/
    evaluate.py               # Evaluate saved model on all 16 positions
    gruau_fitness.py           # Gruau anti-jiggle metric (match our impl)
  results/
    (generated training logs and evaluation results)
  README.md
```

## Evaluation Protocol

### Success Criteria
- **Solved**: Survive 100,000 steps on a single starting position
- **Benchmark solved**: Survive 100,000 steps on ALL 16 starting positions
- **Gruau score**: For solvers, `0.75 * 100000 / jiggle_sum` bonus (smoothness)

### Metrics to Compare
| Metric | Evolvion | PPO/SAC |
|--------|----------|---------|
| Wall-clock to solve all 16 | seconds | seconds |
| Total env interactions | count | count |
| Training seeds that solve (of 10) | X/10 | X/10 |
| Gruau fitness (smoothness) | score | score |
| Network size (params) | ~50-200 | ~thousands |

### Hardware
- Same machine: RTX 4090, same CPU
- PPO/SAC use PyTorch with CUDA
- Evolvion uses ILGPU with CUDA

### Wall-clock budget
- 120 seconds per training run (matching our sparse study)
- Also test with 300s and 600s budgets

## Physics Verification

Before any training, verify physics parity:
1. Run the C# `DoublePoleEnvironment` for 1000 steps from position 0 with action=0.5
2. Run the Python env for 1000 steps from position 0 with action=0.5
3. Compare all 6 state values — must match to <1e-5

This catches any integration bugs, constant mismatches, or normalization differences.

## Implementation Order

1. `starting_positions.json` — define the 16 positions
2. `envs/double_pole.py` — Gymnasium env with exact physics
3. `envs/test_physics.py` — physics parity verification
4. `train/train_recurrent_ppo.py` — primary competitor
5. `train/train_ppo_stacked.py` — frame stacking baseline
6. `eval/evaluate.py` + `eval/gruau_fitness.py` — evaluation pipeline
7. Run comparison experiments
8. Generate results table

## What Victory Looks Like

**Evolvion wins if**: It solves all 16 positions faster (wall-clock) than PPO with equivalent hardware, OR achieves higher Gruau fitness in the same time budget.

**PPO wins if**: It solves all 16 positions faster AND more reliably (higher seed success rate) than Evolvion.

**Draw**: Both solve within similar timeframes — then we compare sample efficiency, network simplicity, and Gruau smoothness as tiebreakers.

Note: Evolvion's advantage is massive parallelism (16K+ individuals evaluated simultaneously on GPU). PPO's advantage is gradient-based optimization (much more sample-efficient per interaction). The question is whether PPO's efficiency overcomes Evolvion's throughput.
