# Landscape Navigation Benchmarks

**Date**: 2025-11-02
**Status**: Completed
**Configuration**: Optuna Trial 23 (27 species × 88 individuals = 2,376 population)

---

## Table of Contents

1. [Overview](#overview)
2. [Test Plan](#test-plan)
3. [Results Summary](#results-summary)
4. [Detailed Results](#detailed-results)
5. [Analysis & Conclusions](#analysis--conclusions)
6. [Next Steps](#next-steps)

---

## Overview

### Goal

Verify that Optuna-optimized hyperparameters (Trial 23) generalize beyond spiral classification to continuous optimization landscapes.

### Why This Matters

Spiral classification is **too easy** - solves in 3 generations regardless of seed. We need to test on harder problems to:
1. Validate hyperparameters generalize beyond spiral
2. Establish evaluation budget for harder problems
3. Identify if any hyperparameter tuning is needed for continuous control tasks
4. Benchmark the algorithm against standard optimization problems

### Test Landscapes

From `Evolvatron.Evolvion.Benchmarks/OptimizationLandscapes.cs`:

#### 1. Sphere (Easy Baseline) ✅
- **Function**: f(x) = Σ x_i²
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Smooth convex, single global optimum
- **Expected**: Should solve easily (sanity check)

#### 2. Rosenbrock (Medium) ⚠️
- **Function**: f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
- **Optimum**: x = [1, 1, ..., 1], f(x) = 0
- **Characteristics**: Narrow parabolic valley, hard to navigate
- **Expected**: Moderate difficulty, tests sequential decision-making

#### 3. Rastrigin (Hard - Multimodal) ❌
- **Function**: f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Many local optima (highly multimodal)
- **Expected**: Tests exploration vs exploitation balance

#### 4. Ackley (Hard - Deceptive) [Not Yet Tested]
- **Function**: Complex with exp terms (see OptimizationLandscapes.cs)
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Nearly flat outer regions, sharp center
- **Expected**: Tests gradient-based navigation

#### 5. Schwefel (Very Hard) [Not Yet Tested]
- **Function**: f(x) = 418.9829n - Σ [x_i * sin(√|x_i|)]
- **Optimum**: x = [420.9687, ...], f(x) ≈ 0
- **Characteristics**: Global optimum far from good local optima (deceptive)
- **Expected**: Hardest test, may not solve without specific tuning

---

## Test Plan

### Configuration Matrix

| Landscape  | Dimensions | Timesteps | Bounds        | Observation Type     | Priority |
|------------|-----------|-----------|---------------|----------------------|----------|
| Sphere     | 5D        | 50        | [-5, 5]       | FullPosition         | HIGH     |
| Rosenbrock | 5D        | 150       | [-2, 2]       | FullPosition         | HIGH     |
| Rastrigin  | 8D        | 100       | [-5.12, 5.12] | FullPosition         | HIGH     |
| Ackley     | 8D        | 100       | [-5, 5]       | FullPosition         | MEDIUM   |
| Schwefel   | 12D       | 200       | [-500, 500]   | FullPosition         | LOW      |

### Success Criteria

Define "solved" as reaching within X% of global optimum:

| Landscape  | Threshold (% of optimum) | Notes                           |
|------------|--------------------------|---------------------------------|
| Sphere     | 0.01 (1%)                | Should get very close           |
| Rosenbrock | 0.10 (10%)               | Valley is tricky                |
| Rastrigin  | 0.15 (15%)               | Local optima are challenging    |
| Ackley     | 0.20 (20%)               | Flat regions hard to navigate   |
| Schwefel   | 0.30 (30%)               | Very deceptive, lower bar       |

### Evaluation Budget

From spiral results: 3 gens × 2,376 = 7,128 evals to solve

**Proposed budgets by difficulty**:
- Sphere: 20,000 evals (~8 generations) - should be easy
- Rosenbrock: 150,000 evals (~63 generations)
- Rastrigin: 100,000 evals (~42 generations)
- Ackley: 100,000 evals (~42 generations)
- Schwefel: 200,000 evals (~84 generations)

### Test Protocol

For each landscape:

1. **Network topology**: Match input/output dimensions
   - Inputs = dimensions (for FullPosition)
   - Outputs = dimensions (movement vector)
   - Hidden = [8, 8] (same as spiral)
   - Output activation = Tanh (movement bounded [-1, 1])

2. **Seeds**: Test 10 seeds (0-9) for statistical significance

3. **Metrics to track**:
   - Generations to solve (primary)
   - Total evaluations to solve
   - Final distance from optimum
   - Success rate across seeds (% that solve)
   - Convergence curves (best fitness per generation)

4. **Early stopping**: Stop at solve threshold or max budget

---

## Results Summary

| Landscape     | Difficulty | Solve Rate   | Avg Gens | Status      |
|---------------|-----------|--------------|----------|-------------|
| Sphere 5D     | Easy      | 10/10 (100%) | 14.0     | ✅ PASS     |
| Rosenbrock 5D | Medium    | 2/10 (20%)   | 63.5     | ⚠️ PARTIAL  |
| Rastrigin 8D  | Hard      | 0/10 (0%)    | N/A      | ❌ FAIL     |

---

## Detailed Results

### Sphere 5D (EASY BASELINE) ✅ PASSED

**Task**: Navigate to origin [0,0,0,0,0] in 5D space

**Configuration**:
- Dimensions: 5
- Timesteps: 50
- Bounds: [-5, 5]
- Step size: 0.1
- Observation: FullPosition (5 inputs → 5 outputs)
- Solve threshold: -0.01 (within 1% of optimum)
- Max generations: 20

**Results**:
```
Solved: 10/10 (100.0%)
Avg generations: 14.0
Avg evaluations: 33,264
Generation range: 8 - 20
```

**Per-seed breakdown**:

| Seed | Result  | Gen | Fitness    | Evaluations |
|------|---------|-----|------------|-------------|
| 0    | SOLVED  | 9   | -0.006716  | 21,384      |
| 1    | SOLVED  | 19  | -0.008428  | 45,144      |
| 2    | SOLVED  | 13  | -0.006242  | 30,888      |
| 3    | SOLVED  | 11  | -0.007937  | 26,136      |
| 4    | SOLVED  | 8   | -0.006501  | 19,008      |
| 5    | SOLVED  | 20  | -0.006068  | 47,520      |
| 6    | SOLVED  | 18  | -0.007281  | 42,768      |
| 7    | SOLVED  | 13  | -0.007232  | 30,888      |
| 8    | SOLVED  | 13  | -0.007723  | 30,888      |
| 9    | SOLVED  | 16  | -0.002119  | 38,016      |

**Analysis**:
- ✅ **100% solve rate** - excellent baseline performance
- Fast convergence: fastest solve at gen 8, average at gen 14
- Seed 9 achieved near-perfect solution (-0.002119, very close to 0)
- Validates that hyperparameters work for continuous control tasks
- No failures, all seeds converged reliably

**Conclusion**: Optuna hyperparameters generalize excellently to simple convex continuous optimization landscapes.

---

### Rosenbrock 5D (MEDIUM DIFFICULTY) ⚠️ PARTIAL SUCCESS

**Task**: Navigate to optimum [1,1,1,1,1] through narrow parabolic valley

**Configuration**:
- Dimensions: 5
- Timesteps: 150
- Bounds: [-2, 2]
- Step size: 0.1
- Observation: FullPosition (5 inputs → 5 outputs)
- Solve threshold: -0.10 (within 10% of optimum)

#### Budget Progression Study

| Max Gens | Solve Rate | Seeds Solved | Notes                        |
|----------|------------|--------------|------------------------------|
| 50       | 0/10 (0%)  | -            | Seed 0 at -0.118 (close!)    |
| 100      | 2/10 (20%) | 0, 5         | Seeds solved at gen 52 & 75  |
| 150      | 2/10 (20%) | 0, 5         | **No improvement from 100!** |

**Final Results (150 generations)**:
```
Solved: 2/10 (20.0%)
Best unsolved: -0.112451 (Seed 7, very close!)
Avg generations to solve: 63.5
Avg evaluations to solve: 150,876
```

**Per-seed breakdown (150 gen run)**:

| Seed | Best Fitness | Status           | Notes               |
|------|--------------|------------------|---------------------|
| 5    | -0.081361    | ✅ SOLVED (gen 52) | Excellent!        |
| 0    | -0.099496    | ✅ SOLVED (gen 75) | Just made it      |
| 7    | -0.112451    | ❌ Plateaued      | 0.012 away, stuck |
| 8    | -0.113736    | ❌ Plateaued      | 0.014 away, stuck |
| 3    | -0.162260    | ❌                | Not improving     |
| 2    | -0.201225    | ❌                |                   |
| 1    | -0.248691    | ❌                |                   |
| 9    | -0.289048    | ❌                |                   |
| 6    | -0.313856    | ❌                |                   |
| 4    | -0.499223    | ❌                |                   |

**Analysis**:
- ⚠️ **20% solve rate** - proves Rosenbrock IS solvable with these hyperparameters
- **Plateauing observed**: Seeds 7 & 8 stuck at -0.112/-0.113 from gen 100-150
  - They're only 0.012-0.014 away from solving but can't improve further
  - Classic symptom of premature convergence in narrow valley
- **Two distinct outcomes**:
  - 2 seeds found the valley and navigated successfully
  - 8 seeds either couldn't enter the valley or got stuck partway
- **More generations won't help** - algorithm needs different approach for stuck seeds

**Conclusion**: Hyperparameters CAN solve Rosenbrock (20% success), but 80% of runs get stuck. The narrow valley defeats most runs, and plateauing indicates fundamental limitations rather than insufficient budget.

---

### Rastrigin 8D (HARD - MULTIMODAL) ❌ FAILED

**Task**: Navigate to origin [0,0,...,0] in highly multimodal landscape with many local optima

**Configuration**:
- Dimensions: 8
- Timesteps: 100
- Bounds: [-5.12, 5.12]
- Step size: 0.1
- Observation: FullPosition (8 inputs → 8 outputs)
- Solve threshold: -0.15 (within 15% of optimum)
- Max generations: 50

**Results**:
```
Solved: 0/10 (0.0%)
Best fitness: -9.224364 (Seed 1)
Avg fitness: -14.464403
Worst fitness: -24.521740 (Seed 8)
```

**Per-seed breakdown**:

| Seed | Best Fitness | Note  |
|------|--------------|-------|
| 1    | -9.224364    | Best  |
| 2    | -10.669385   |       |
| 3    | -10.034487   |       |
| 6    | -11.219760   |       |
| 7    | -11.228243   |       |
| 5    | -13.535193   |       |
| 0    | -15.452950   |       |
| 9    | -18.187986   |       |
| 4    | -20.569925   |       |
| 8    | -24.521740   | Worst |

**Analysis**:
- ❌ **Complete failure** - all seeds far from optimum
- Best fitness -9.22 is nowhere near the -0.15 threshold
- Algorithm is **stuck in local optima** - classic Rastrigin behavior
- The highly multimodal landscape with dense local optima defeats the current hyperparameters
- Would require:
  - Much stronger exploration mechanisms
  - Diversity maintenance strategies
  - Potentially different network architecture or observation type
  - Many more generations (100s or 1000s)

**Conclusion**: Optuna hyperparameters do NOT generalize to highly multimodal landscapes like Rastrigin.

---

## Analysis & Conclusions

### Key Insights

1. **Hyperparameters work for simple landscapes**: 100% solve rate on Sphere demonstrates that the algorithm can handle basic continuous optimization

2. **Partial success on narrow valleys**: Rosenbrock achieved 20% solve rate with 150 generations
   - **Plateauing problem**: Increasing budget from 100→150 gens provided NO additional solves
   - Seeds 7 & 8 stuck just 0.012 away from threshold - premature convergence
   - More generations won't help without hyperparameter changes

3. **Fail on multimodal landscapes**: Rastrigin completely defeats the algorithm - stuck in local optima, nowhere near the global optimum

4. **Parallelization works great**: 16-core parallelization enables rapid experimentation
   - 150 gens × 10 seeds in 90 seconds (vs ~9 minutes sequential)

5. **Generalization is limited**: Hyperparameters optimized for spiral classification show:
   - ✅ Excellent transfer to convex landscapes (Sphere)
   - ⚠️ Limited transfer to valley navigation (Rosenbrock 20%)
   - ❌ No transfer to multimodal landscapes (Rastrigin 0%)

### Final Verdict

**Minimum bar**: Sphere + Rosenbrock solve at 80%+ rate
**Result**: ❌ FAILED (Sphere: 100%, Rosenbrock: 20%)

**Strong success**: Sphere + Rosenbrock + Rastrigin all solve at 60%+ rate
**Result**: ❌ FAILED (only Sphere passed at 60%+)

The Optuna Trial 23 hyperparameters:
- ✅ **Generalize excellently to convex landscapes** (Sphere 5D: 100%)
- ⚠️ **Partially work on narrow valleys** (Rosenbrock 5D: 20%, plateaus early)
- ❌ **Completely fail on multimodal landscapes** (Rastrigin 8D: stuck in local optima)

**Overall assessment**:
- Hyperparameters are **excellent for simple landscapes**
- **Limited generalization** to medium difficulty (20% vs desired 80%)
- **No generalization** to hard multimodal problems
- **Budget extension doesn't help** - plateauing indicates hyperparameter limitations, not insufficient generations
- **Landscape-specific tuning required** for Rosenbrock and Rastrigin

**Key discovery**: The plateauing behavior (100→150 gens yielding identical results) proves that throwing more compute at the problem won't solve it. The hyperparameters fundamentally struggle with valley navigation - a qualitatively different challenge than spiral classification.

---

## Next Steps

### Immediate Actions (Completed)

- [x] Re-run Rosenbrock with 100 generations - **Result: 2/10 solved (20%)**
- [x] Re-run Rosenbrock with 150 generations - **Result: 2/10 solved (20%, same seeds!)**
- **Conclusion**: Plateauing at ~gen 75, more budget doesn't help

### Priority 1: Rosenbrock Improvements

- [ ] Try step size 0.05 instead of 0.1 (finer valley navigation)
- [ ] Try more timesteps (200 instead of 150)
- [ ] Try ObservationType.GradientOnly (follow the gradient down the valley)
- [ ] Mini Optuna sweep for valley-specific hyperparameters

### Priority 2: Landscape-Specific Optuna Sweep

Run targeted hyperparameter optimization for:
1. **Rosenbrock navigation**: Optimize for valley-following behavior
2. **Rastrigin exploration**: Optimize for escaping local optima

Parameters to sweep:
- Population size (increase for multimodal?)
- Diversity threshold (increase for Rastrigin?)
- Mutation rates (higher exploration for Rastrigin?)
- Tournament size (lower for Rastrigin to reduce selection pressure?)

### Priority 3: Alternative Approaches

- **Gradient observation**: Try ObservationType.GradientOnly for Rosenbrock
- **Hybrid algorithm**: Combine evolution with gradient descent for valley refinement
- **Adaptive diversity**: Detect stagnation and boost diversity automatically

### Priority 4: Test Remaining Landscapes

- [ ] Ackley 8D (medium priority)
- [ ] Schwefel 12D (low priority - very hard)

---

## Implementation Details

**Test harness**: `Evolvatron.Tests/Evolvion/LandscapeNavigationTest.cs`
**Environment**: `Evolvatron.Evolvion/Environments/LandscapeEnvironment.cs`
**Task**: `Evolvatron.Evolvion/Benchmarks/LandscapeNavigationTask.cs`
**Landscapes**: `Evolvatron.Evolvion/Benchmarks/OptimizationLandscapes.cs`

**Network**: Input→8→8→Output (Tanh output activation)
**Fitness function**: Final landscape value (negated) - maximize negative error to minimize distance from optimum
**Evaluator**: SimpleFitnessEvaluator runs network forward passes for each timestep, accumulating position updates

**Performance**: All tests use deterministic initialization (seed controls both topology and evolution). Parallel execution on 16 cores: ~11-30 seconds per landscape (10 seeds). Results are reproducible.

---

**Document created from**:
- `scratch/LANDSCAPE_NAVIGATION_VERIFICATION_PLAN.md`
- `scratch/LANDSCAPE_NAVIGATION_RESULTS.md`
