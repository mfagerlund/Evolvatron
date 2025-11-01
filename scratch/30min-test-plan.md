# 30-Minute Testing Plan: High-Impact Parameter Sweep

Based on Phase 1-5 findings, here's what to test for maximum insight in 30 minutes.

---

## Testing Strategy

**Parallelization**: 15-way parallel = ~2 minutes per batch
**Total batches**: 15 batches = 30 minutes
**Total configs**: 225 tests

**Focus**: Untested parameters with high potential impact + verification tests

---

## Batch 1: Bias Initialization Methods (5 configs, 2 min)

**Question**: Does bias initialization matter now that they mutate?

```csharp
1. BiasInit-Zero:        Bias[i] = 0.0 (current)
2. BiasInit-SmallRandom: Bias[i] = U(-0.1, 0.1)
3. BiasInit-LargeRandom: Bias[i] = U(-1.0, 1.0)
4. BiasInit-Glorot:      Bias[i] = GlorotUniform(fanIn, fanOut)
5. BiasInit-Negative:    Bias[i] = -0.5 (test bias towards negative)
```

**Why**: Biases now mutate, but start at 0.0. Better initialization could provide better starting diversity.

**Expected winner**: SmallRandom or Zero (large random might be too noisy)

---

## Batch 2: Weight Initialization Scale (5 configs, 2 min)

**Question**: Is Glorot optimal for evolution (no gradients)?

```csharp
1. WeightInit-GlorotCurrent: (current)
2. WeightInit-Xavier:         σ = sqrt(2/(fanIn+fanOut))
3. WeightInit-He:            σ = sqrt(2/fanIn)
4. WeightInit-Uniform-0.5:   U(-0.5, 0.5)
5. WeightInit-Uniform-2.0:   U(-2.0, 2.0)
```

**Why**: Glorot designed for gradient descent. Evolution might prefer different scales.

**Expected winner**: Uniform-0.5 or 1.0 (simpler, more diversity)

---

## Batch 3: WeightJitter Standard Deviation (5 configs, 2 min)

**Question**: Is 0.3 (30% of weight) optimal mutation step size?

```csharp
WeightJitterStdDev (currently 0.3):
1. 0.05  (5% - very conservative)
2. 0.1   (10% - conservative)
3. 0.3   (30% - current)
4. 0.5   (50% - aggressive)
5. 1.0   (100% - very aggressive)
```

**Why**: This was NEVER tested in any sweep! Could be critical.

**Expected winner**: 0.1-0.3 range (too small = slow, too large = disruptive)

---

## Batch 4: Population Size vs Tournament Size Trade-off (5 configs, 2 min)

**Question**: Better to have more individuals OR stronger selection?

```csharp
Population × Tournament (same compute budget ~800 evals/gen):
1. Pop=400,  Tournament=4   (baseline from XOR)
2. Pop=800,  Tournament=16  (current best)
3. Pop=1600, Tournament=32  (max selection pressure)
4. Pop=200,  Tournament=8   (small pop, strong selection)
5. Pop=1200, Tournament=24  (middle ground)
```

**Why**: Tournament=16 was found best at Pop=800, but is there a better combo?

**Expected winner**: Current (800, 16) or (1200, 24)

---

## Batch 5: Elite Count Trade-off (5 configs, 2 min)

**Question**: Optimal elitism level?

```csharp
Elites (at Pop=800):
1. Elites=0   (pure tournament, no elitism)
2. Elites=1   (minimal preservation)
3. Elites=2   (current)
4. Elites=4   (moderate)
5. Elites=8   (high preservation)
```

**Why**: Found Elites=2 best, but tested only 1,2,4,10,20. Never tried 0 or 8.

**Expected winner**: 1-2 (exploration important, but some preservation helps)

---

## Batch 6: Funnel Architecture Variations (5 configs, 2 min)

**Question**: Is funnel architecture better with proper tuning?

```csharp
All Tanh-only, dense init:
1. Baseline:       2→6→6→1      (15 nodes, 54 edges)
2. Funnel-Wide:    2→16→8→4→1   (31 nodes, 200 edges)
3. Funnel-Medium:  2→12→8→4→1   (27 nodes, 156 edges) [tested before]
4. Funnel-Narrow:  2→8→6→4→1    (21 nodes, 92 edges)
5. Reverse-Funnel: 2→4→8→12→1   (27 nodes, 156 edges) [expansion]
```

**Why**: Funnel was 2nd best in depth test. With Tanh-only, might be #1.

**Expected winner**: Funnel-Medium or Funnel-Narrow

---

## Batch 7: Hybrid Activation Strategies (5 configs, 2 min)

**Question**: Is pure Tanh really best, or can hybrids work?

```csharp
All 2→6→6→1 dense:
1. Tanh-Only:          All Tanh (current best)
2. ReLU-Tanh-Tanh:     L1=ReLU, L2=Tanh, Out=Tanh
3. Tanh-ReLU-Tanh:     L1=Tanh, L2=ReLU, Out=Tanh
4. LeakyReLU-Only:     All LeakyReLU
5. LeakyReLU-Tanh-Tanh: L1=LeakyReLU, L2=Tanh, Out=Tanh
```

**Why**: Tanh won, but pure Tanh vs strategic placement untested.

**Expected winner**: Tanh-Only (but LeakyReLU-Tanh-Tanh might surprise)

---

## Batch 8: Species Count vs Diversity (5 configs, 2 min)

**Question**: Optimal number of species?

```csharp
SpeciesCount (at Pop=800 = 100 per species):
1. Species=4   (200 per species)
2. Species=8   (100 per species - current)
3. Species=16  (50 per species)
4. Species=32  (25 per species)
5. Species=2   (400 per species - minimal diversity)
```

**Why**: 8 species arbitrary choice. More species = more diversity BUT smaller populations per species.

**Expected winner**: 4-8 (too many species = small populations, too few = no diversity)

---

## Batch 9: Dynamic Mutation Rates (5 configs, 2 min)

**Question**: Should mutation rates decrease over time?

```csharp
WeightJitter schedule (start → end over 100 gens):
1. Static-0.95:     0.95 throughout (current)
2. Decay-Linear:    0.95 → 0.50 linear
3. Decay-Exp:       0.95 → 0.20 exponential (half-life 30 gens)
4. Increase-Linear: 0.50 → 0.95 (start conservative)
5. Adaptive:        High when fitness plateaus, low when improving
```

**Why**: Early exploration vs late exploitation trade-off.

**Expected winner**: Decay-Linear or Static (adaptive might help)

---

## Batch 10: Edge Topology Mutation Rates (5 configs, 2 min)

**Question**: Should we enable topology mutations?

```csharp
EdgeAdd / EdgeDelete rates (currently disabled):
1. Disabled:     0.00 / 0.00 (current)
2. VeryLow:      0.01 / 0.005
3. Low:          0.05 / 0.02  (original defaults)
4. Medium:       0.10 / 0.05
5. High:         0.20 / 0.10
```

**Why**: Dense init provides good topology, but can evolution improve it?

**Expected winner**: Disabled or VeryLow (good init makes topology changes risky)

---

## Batch 11: Crossover vs Pure Mutation (5 configs, 2 min)

**Question**: Would crossover help?

```csharp
1. NoXover:       Pure mutation (current)
2. Xover-10%:     10% of offspring from crossover
3. Xover-25%:     25% crossover
4. Xover-50%:     50% crossover
5. Xover-Uniform: Uniform crossover (swap each weight 50%)
```

**Why**: NEVER TESTED! Pure mutation vs crossover is fundamental EA question.

**Expected winner**: NoXover or Xover-10% (small populations might not benefit)

---

## Batch 12: Fitness Evaluation Noise Handling (5 configs, 2 min)

**Question**: How to handle evaluation variance?

```csharp
SpiralEnvironment evaluation:
1. Deterministic: Single eval (current for deterministic task)
2. Average-3:     3 evals, average
3. Median-5:      5 evals, median (CVaR50)
4. Best-3:        3 evals, take best (optimistic)
5. Worst-3:       3 evals, take worst (pessimistic/robust)
```

**Why**: Spiral is deterministic, but verifying single eval is optimal.

**Expected winner**: Deterministic (no noise to handle)

---

## Batch 13: ParentPoolPercentage (5 configs, 2 min)

**Question**: Should only top performers breed?

```csharp
ParentPoolPercentage (currently 1.0 = 100%):
1. 1.00:  All individuals can breed (current)
2. 0.75:  Top 75% can breed
3. 0.50:  Top 50% can breed
4. 0.25:  Top 25% can breed
5. 0.10:  Top 10% only (elite breeding)
```

**Why**: Tournament already provides selection, but restricting parent pool adds extra pressure.

**Expected winner**: 0.50-0.75 (too restrictive kills diversity)

---

## Batch 14: Node Parameter Mutation Importance (5 configs, 2 min)

**Question**: How important are LeakyReLU alpha parameters?

```csharp
NodeParamMutate (currently 0.20) + NodeParamStdDev (0.1):
1. Disabled:       0.00 (no param mutation)
2. Low:            0.10 / 0.05
3. Current:        0.20 / 0.10
4. High:           0.50 / 0.20
5. VeryHigh:       0.80 / 0.50
```

**Why**: With Tanh-only, node params might not matter (Tanh has no params).

**Expected winner**: Disabled (Tanh doesn't use alpha/beta)

---

## Batch 15: Verification - Best Config Stability (5 configs, 2 min)

**Question**: Is best config stable across seeds?

```csharp
Best config (Tanh-only, Dense 2→6→6→1, T=16, WJ=0.95) with different seeds:
1. Seed=42    (original)
2. Seed=123
3. Seed=456
4. Seed=789
5. Seed=999
```

**Why**: Verify results aren't luck. Should all perform ~0.20 improvement.

**Expected winner**: All should be similar (±0.02)

---

## Implementation Plan

### Create `ThirtyMinuteSweepTest.cs`:

```csharp
[Fact]
public void ThirtyMinuteSweep()
{
    var batches = new[]
    {
        CreateBatch1_BiasInit(),
        CreateBatch2_WeightInit(),
        CreateBatch3_WeightJitterStdDev(),
        CreateBatch4_PopulationVsTournament(),
        CreateBatch5_EliteCount(),
        CreateBatch6_FunnelArchitectures(),
        CreateBatch7_HybridActivations(),
        CreateBatch8_SpeciesCount(),
        CreateBatch9_DynamicMutationRates(),
        CreateBatch10_EdgeTopologyMutations(),
        CreateBatch11_CrossoverVsMutation(),
        CreateBatch12_EvaluationNoise(),
        CreateBatch13_ParentPoolPercentage(),
        CreateBatch14_NodeParamMutation(),
        CreateBatch15_BestConfigStability()
    };

    // Run each batch sequentially (configs within batch parallel)
    foreach (var batch in batches)
    {
        RunBatchInParallel(batch, parallelism: 5);
    }
}
```

---

## IMPORTANT: Run This AFTER Bias Bug Fix

**All previous tests (Phase 1-5) were run with biases frozen at 0.0** due to missing mutation operators.

This 30-minute sweep should be run with:
- ✅ Bias mutation enabled (fixed)
- ✅ Biases mutating at same rates as weights
- ✅ Dense initialization as default
- ✅ Tanh-only activations

The results will establish the **true optimal configuration** with all features working correctly.

---

## Expected High-Impact Findings

### Most Likely Improvements (ranked):

1. **WeightJitterStdDev** (Batch 3) - NEVER TESTED, could be 20-50% improvement
2. **ParentPoolPercentage** (Batch 13) - Could amplify tournament selection
3. **Bias Initialization** (Batch 1) - Better diversity from start
4. **Weight Initialization** (Batch 2) - Evolution ≠ gradient descent
5. **Funnel Architecture** (Batch 6) - Promising in initial tests

### Likely No Impact:

- Batch 10 (Edge topology) - Dense init already optimal
- Batch 12 (Evaluation noise) - Task is deterministic
- Batch 14 (Node params) - Tanh doesn't use them

### Verification:

- Batch 15 confirms results are robust, not lucky

---

## Output Format

```
30-Minute Sweep Results
========================

Batch 1: Bias Initialization
-----------------------------
Winner: BiasInit-SmallRandom (0.2150 improvement, +5% vs current)

Batch 2: Weight Initialization
------------------------------
Winner: Uniform-1.0 (0.2280 improvement, +11% vs current)

Batch 3: WeightJitter StdDev ⭐
-------------------------------
Winner: 0.1 (0.2850 improvement, +39% vs current) ← BREAKTHROUGH!

[... continues for all 15 batches ...]

Top 5 Improvements Found:
1. WeightJitterStdDev=0.1:      +39% improvement
2. ParentPoolPercentage=0.5:    +22% improvement
3. WeightInit=Uniform-1.0:      +11% improvement
4. BiasInit=SmallRandom:        +5% improvement
5. Funnel-Narrow:               +3% improvement

Combined Best Config:
- All improvements together: Projected 0.35+ improvement (vs 0.20 current)
- Solve time: ~100 generations (vs current 250)
- Total speedup vs original: 25x faster!
```

---

## Why This Plan?

1. **Covers untested parameters**: WeightJitterStdDev, ParentPoolPercentage, Crossover never tested
2. **Verifies assumptions**: Edge mutations disabled, NodeParams with Tanh, single eval
3. **Explores promising leads**: Funnel architecture, hybrid activations
4. **Tests fundamentals**: Initialization, population structure, selection pressure
5. **Validates robustness**: Multiple seeds confirm stability

**Expected outcome**: Find 2-3 high-impact parameters that provide another 2-3x speedup, bringing total to 20-30x faster than original baseline!
