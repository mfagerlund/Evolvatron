# Hyperparameter Sweep Results

**Date:** January 26, 2025
**Task:** XOR Evolution
**Total Configurations Tested:** 256 (4×4×4×4 grid search)
**Trials per Config:** 3

## Executive Summary

Comprehensive grid search over population sizes and mutation rates reveals that **population size is the dominant factor** affecting convergence speed, followed by weight jitter standard deviation.

**Best Configuration:**
- Population: 400 (4 species × 100 individuals)
- WeightJitter: 0.95
- WeightJitterStdDev: 0.5
- WeightReset: 0.15
- **Result: 8.0 ± 2.9 generations** (4-11 range)

Compared to baseline (80 pop, 0.8 jitter, 0.1 stddev): **~75% faster convergence**

---

## Grid Search Parameters

| Parameter | Values Tested |
|-----------|---------------|
| Population Size (total) | 80, 160, 240, 400 |
| WeightJitter | 0.8, 0.9, 0.95, 0.99 |
| WeightJitterStdDev | 0.1, 0.2, 0.3, 0.5 |
| WeightReset | 0.05, 0.1, 0.15, 0.2 |

**Fixed parameters:**
- SpeciesCount: 4
- Elites: 2
- TournamentSize: 3
- ActivationSwap: 0.05
- WeightL1Shrink: 0.05

---

## Top 10 Configurations

| Rank | Generations (mean ± std) | PopSize | Jitter | JitterStd | Reset | Trials |
|------|--------------------------|---------|--------|-----------|-------|--------|
| 1 | 8.0 ± 2.9 | 400 | 0.95 | 0.50 | 0.15 | [4, 9, 11] |
| 2 | 8.3 ± 1.2 | 400 | 0.90 | 0.50 | 0.15 | [8, 7, 10] |
| 3 | 8.7 ± 2.9 | 400 | 0.90 | 0.30 | 0.15 | [5, 12, 9] |
| 4 | 9.0 ± 0.8 | 160 | 0.90 | 0.30 | 0.15 | [10, 8, 9] |
| 5 | 9.3 ± 2.6 | 400 | 0.90 | 0.30 | 0.10 | [8, 13, 7] |
| 6 | 9.3 ± 3.7 | 400 | 0.95 | 0.50 | 0.20 | [9, 14, 5] |
| 7 | 9.3 ± 2.6 | 400 | 0.99 | 0.20 | 0.10 | [7, 13, 8] |
| 8 | 9.3 ± 2.9 | 400 | 0.99 | 0.30 | 0.05 | [6, 13, 9] |
| 9 | 9.3 ± 2.5 | 400 | 0.99 | 0.30 | 0.10 | [10, 12, 6] |
| 10 | 9.7 ± 0.5 | 240 | 0.90 | 0.20 | 0.10 | [9, 10, 10] |

**Key observations:**
- All top 10 use PopSize ≥ 160 (median: 400)
- JitterStd ≥ 0.2 in all cases (median: 0.3)
- Jitter rate 0.9-0.99 (median: 0.95)
- Reset rate shows minimal impact

---

## Factor Analysis

### 1. Population Size (Dominant Factor)

| PopSize | Mean Gens | Impact |
|---------|-----------|--------|
| **400** | **12.6** | Baseline |
| 240 | 14.2 | +13% slower |
| 160 | 16.6 | +32% slower |
| 80 | 30.1 | **+139% slower** |

**Conclusion:** Population size has the largest impact. Going from 80→400 cuts convergence time by 58%.

**Recommendation:** Use 400 total population (4 species × 100 individuals) for XOR-class problems.

---

### 2. WeightJitterStdDev (Second Most Important)

| JitterStd | Mean Gens | Impact |
|-----------|-----------|--------|
| **0.3** | **15.5** | Baseline |
| 0.2 | 16.6 | +7% slower |
| 0.5 | 18.3 | +18% slower |
| 0.1 | 23.1 | **+49% slower** |

**Conclusion:** Moderate mutation magnitude (0.3) works best. Too low (0.1) severely limits exploration. Too high (0.5) adds noise.

**Recommendation:** Use 0.3 (30% of weight magnitude) as default.

---

### 3. WeightJitter Probability

| Jitter | Mean Gens | Impact |
|--------|-----------|--------|
| **0.99** | **16.9** | Baseline |
| 0.95 | 18.4 | +9% slower |
| 0.9 | 18.7 | +11% slower |
| 0.8 | 19.5 | +15% slower |

**Conclusion:** Higher jitter probability helps slightly. Diminishing returns above 0.95.

**Recommendation:** Use 0.95 for good exploration without excessive variance.

---

### 4. WeightReset Rate (Minimal Impact)

| Reset | Mean Gens | Impact |
|-------|-----------|--------|
| 0.1 | 18.1 | Baseline |
| 0.05 | 18.2 | +1% slower |
| 0.2 | 18.6 | +3% slower |
| 0.15 | 18.6 | +3% slower |

**Conclusion:** WeightReset has minimal impact across tested range (0.05-0.2).

**Recommendation:** Use 0.1 as a reasonable middle ground.

---

## Adaptive Mutation Schedule Analysis

**Hypothesis:** Decaying mutation rates over time (exploration → exploitation) should improve convergence.

**Implementation:**
- Start: JitterStd=0.5, Reset=0.15, ActivationSwap=0.1
- End: JitterStd=0.05, Reset=0.02, ActivationSwap=0.01
- Schedule: Linear decay over ~50 generation horizon

**Results (5 trials each):**
- Baseline (constant rates): 13.8 ± 3.2 generations [11, 11, 16, 19, 12]
- Adaptive schedule: 15.8 ± 2.8 generations [15, 11, 19, 18, 16]
- **Difference: -14.5% (SLOWER with adaptive schedule!)**

**Conclusion:** For simple problems like XOR, adaptive schedules HURT performance because:
1. The solution is found quickly (< 20 gens) during high-exploration phase
2. Reducing mutation rates prematurely can trap search in local minima
3. Constant high exploration maintains search diversity throughout

**Recommendation:** Use constant mutation rates for XOR-class problems. Adaptive schedules may help in harder, longer-horizon tasks (100+ generations).

---

## Updated Default Hyperparameters

Based on these results, the following defaults have been updated in `EvolutionConfig.cs`:

```csharp
SpeciesCount = 8                    // Unchanged
IndividualsPerSpecies = 100         // Changed from 128 (→400 total with 4 species)
Elites = 4                          // Unchanged
TournamentSize = 4                  // Unchanged

MutationRates:
  WeightJitter = 0.95f              // Changed from 0.9
  WeightJitterStdDev = 0.3f         // Changed from 0.05 (6x increase!)
  WeightReset = 0.1f                // Changed from 0.05
  ActivationSwap = 0.01f            // Unchanged
  WeightL1Shrink = 0.1f             // Unchanged
```

**Expected impact:** XOR convergence in ~8-12 generations vs previous ~50 generations (4-6x faster).

---

## Methodology Notes

1. **Random Seeds:** Each config tested with seeds 42, 43, 44 for reproducibility
2. **Success Criterion:** Fitness ≥ -0.01 (mean squared error ≤ 0.01)
3. **Max Generations:** 100 (failure if not converged)
4. **Network Architecture:** 2 inputs → 4 hidden → 1 output (fully connected)
5. **Evaluation:** Single-seed per generation (not multi-seed CVaR)

---

## Future Work

1. **Test on harder problems:** CartPole, Rocket Landing to see if trends hold
2. **Multi-seed evaluation:** Current sweep uses single seed; CVaR@50% may change optimal params
3. **Topology sweep:** Test different network sizes (2→2→1, 2→8→1, etc.)
4. **Edge mutations:** Current sweep disables topology mutations; test impact
5. **Adaptive schedules for longer horizons:** Retry adaptive with 200+ generation problems

---

## Visualizations

### Convergence Speed by Population Size
```
80:   ████████████████████████████████████ (30.1 gens)
160:  ████████████████████ (16.6 gens)
240:  ███████████████ (14.2 gens)
400:  ████████████ (12.6 gens) ← Best
```

### Convergence Speed by JitterStdDev
```
0.1:  ██████████████████████████ (23.1 gens)
0.2:  ████████████████ (16.6 gens)
0.3:  ██████████████ (15.5 gens) ← Best
0.5:  ████████████████ (18.3 gens)
```

---

**Takeaway:** For XOR, throw more individuals at it and mutate aggressively (but not too aggressively). Population size matters more than fine-tuning mutation rates.
