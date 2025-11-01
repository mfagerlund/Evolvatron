# One-Hour Comprehensive Hyperparameter Sweep Plan

**Target Duration**: 60 minutes
**Parallelism**: 8 threads
**Generations per config**: 150 (for better signal in untested params)
**Total configs**: ~120
**Goal**: Test ALL untested/under-tested parameters post-bias-fix

---

## Hyperparameter Audit Summary

### ✅ Thoroughly Tested (Post-Bias-Fix, Phase 6):
1. **Architecture depth** (2-6 layers) - Winner: 6 layers
2. **Architecture width** (3-12 nodes/layer) - Winner: 3 nodes (narrow & deep)
3. **Architecture shape** (rectangular, funnel, reverse) - Winner: Rectangular
4. **SpeciesCount** (2, 4, 8, 16, 32) - Winner: 4
5. **IndividualsPerSpecies** (implicit via total pop) - Winner: 200
6. **TournamentSize** (2, 4, 8, 16, 32) - Winner: 16
7. **Elites** (0, 1, 2, 4, 8) - Winner: 2
8. **WeightJitter** rate (0.5-0.99) - Winner: 0.95
9. **WeightJitterStdDev** (0.05-1.0) - Winner: 0.3
10. **WeightReset** rate (0.01-0.5) - Winner: 0.10
11. **ParentPoolPercentage** (0.1-1.0) - Winner: 1.0
12. **EdgeAdd/EdgeDelete** rates (0.0-0.2) - Winner: Any (no impact)

### ⚠️ Tested But Suspect (Pre-Bias-Fix):
All Phase 1-5 absolute values invalid (biases frozen)
Relative rankings likely still valid, but should re-verify

### ❓ NEVER Tested or Minimally Tested:

#### Culling Parameters (NEVER tested):
1. **StagnationThreshold** (15) - Gens without improvement before culling eligibility
2. **GraceGenerations** (3) - New species protection period
3. **SpeciesDiversityThreshold** (0.15) - Minimum fitness variance
4. **RelativePerformanceThreshold** (0.5) - Performance ratio for culling

#### Mutation Parameters (Minimally tested):
5. **WeightL1Shrink** rate (0.1) - Only tested on/off, not rate sweep
6. **L1ShrinkFactor** (0.9) - Only tested default value
7. **ActivationSwap** rate (0.01) - Only tested on/off
8. **NodeParamMutate** rate (0.2) - Limited sweep (with Tanh, params don't matter)
9. **NodeParamStdDev** (0.1) - Limited sweep

#### Evaluation Parameters (NEVER tested):
10. **SeedsPerIndividual** (5) - Audited but never swept
11. **FitnessAggregation** ("CVaR50") - Never tested ("Mean" vs "CVaR50")

#### Initialization (Partially tested):
12. **WeightInitialization** - Only tested "GlorotUniform", not "Xavier", "He", etc.

---

## Testing Strategy

### Batch Size Calculation:
- **Parallelism**: 8 threads
- **Time per config**: ~30 seconds (150 gens × 800 individuals)
- **Configs per minute**: 8 × 2 = 16 configs/min
- **Total configs in 60 min**: 16 × 60 = 960 config-minutes
- **Target**: ~120 configs (leaves buffer for variance)

### Batch Design:
- 15 batches × 8 configs = 120 total
- Each batch runs 8 configs in parallel (8 threads)
- Each batch takes ~4 minutes
- Total: 15 batches × 4 min = 60 minutes

---

## Batch Definitions

### Batch 1: StagnationThreshold (8 configs, 4 min)

**Question**: When should stagnant species become culling-eligible?

```csharp
StagnationThreshold:
1. 5    (very aggressive - cull fast)
2. 10   (aggressive)
3. 15   (current default)
4. 20   (moderate)
5. 30   (conservative)
6. 50   (very conservative)
7. 100  (minimal culling)
8. INF  (disable stagnation-based culling)
```

**Expectation**: 15-30 range optimal (too low kills diversity, too high wastes resources)

---

### Batch 2: GraceGenerations (8 configs, 4 min)

**Question**: How long should new species be protected?

```csharp
GraceGenerations:
1. 0    (no protection)
2. 1    (minimal)
3. 3    (current default)
4. 5    (moderate)
5. 10   (high protection)
6. 20   (very high)
7. 30   (extreme)
8. 50   (nearly permanent)
```

**Expectation**: 3-10 range optimal (too low kills innovation, too high wastes slots)

---

### Batch 3: SpeciesDiversityThreshold (8 configs, 4 min)

**Question**: Minimum fitness variance to avoid culling?

```csharp
SpeciesDiversityThreshold:
1. 0.01  (very low - easy to avoid culling)
2. 0.05  (low)
3. 0.10  (moderate-low)
4. 0.15  (current default)
5. 0.25  (moderate-high)
6. 0.50  (high)
7. 1.00  (very high - hard to avoid culling)
8. INF   (disable diversity-based culling)
```

**Expectation**: 0.10-0.25 range optimal

---

### Batch 4: RelativePerformanceThreshold (8 configs, 4 min)

**Question**: Performance ratio for culling eligibility?

```csharp
RelativePerformanceThreshold (species median / best species):
1. 0.1   (very strict - top 10% only)
2. 0.25  (strict - top 25%)
3. 0.5   (current default - top 50%)
4. 0.75  (lenient - top 75%)
5. 0.9   (very lenient - top 90%)
6. 1.0   (disable - all equal)
7. 0.33  (top third)
8. 0.67  (top two-thirds)
```

**Expectation**: 0.5-0.75 range optimal

---

### Batch 5: WeightL1Shrink Rate (8 configs, 4 min)

**Question**: Optimal L1 regularization mutation rate?

```csharp
WeightL1Shrink:
1. 0.0   (disabled)
2. 0.01  (very rare)
3. 0.05  (rare)
4. 0.1   (current default)
5. 0.2   (frequent)
6. 0.3   (very frequent)
7. 0.5   (extreme)
8. 0.15  (moderate)
```

**Expectation**: 0.05-0.15 range optimal

---

### Batch 6: L1ShrinkFactor (8 configs, 4 min)

**Question**: How much to shrink each time?

```csharp
L1ShrinkFactor (multiply weight by this):
1. 0.5   (shrink by 50% - aggressive)
2. 0.7   (shrink by 30%)
3. 0.8   (shrink by 20%)
4. 0.9   (current default - shrink by 10%)
5. 0.95  (shrink by 5%)
6. 0.99  (shrink by 1% - minimal)
7. 0.85  (shrink by 15%)
8. 0.75  (shrink by 25%)
```

**Expectation**: 0.85-0.95 range optimal

---

### Batch 7: ActivationSwap Rate (8 configs, 4 min)

**Question**: Optimal activation function mutation rate?

```csharp
ActivationSwap:
1. 0.0   (disabled - pure Tanh)
2. 0.001 (very rare)
3. 0.01  (current default)
4. 0.05  (moderate)
5. 0.1   (frequent)
6. 0.2   (very frequent)
7. 0.005 (rare)
8. 0.02  (moderate-rare)
```

**Expectation**: 0.0 optimal (Tanh-only won in Phase 5)

---

### Batch 8: NodeParamMutate + NodeParamStdDev Combined (8 configs, 4 min)

**Question**: Node parameter mutation strategy (for LeakyReLU alpha, etc.)?

```csharp
NodeParamMutate / NodeParamStdDev:
1. 0.0 / -     (disabled - Tanh doesn't use params)
2. 0.1 / 0.05  (low)
3. 0.2 / 0.10  (current default)
4. 0.3 / 0.15  (moderate)
5. 0.5 / 0.20  (high)
6. 0.2 / 0.05  (current rate, lower stddev)
7. 0.2 / 0.20  (current rate, higher stddev)
8. 0.1 / 0.10  (low rate, current stddev)
```

**Expectation**: 0.0 optimal (Tanh-only architecture doesn't need param mutation)

---

### Batch 9: SeedsPerIndividual (8 configs, 4 min)

**Question**: Optimal number of evaluation seeds?

```csharp
SeedsPerIndividual (for stochastic tasks):
1. 1    (single evaluation - deterministic)
2. 3    (minimal averaging)
3. 5    (current default)
4. 7    (moderate averaging)
5. 10   (high averaging)
6. 15   (very high)
7. 20   (extreme)
8. 2    (pair)
```

**Expectation**: 1 optimal for deterministic spiral task (no stochasticity)

---

### Batch 10: FitnessAggregation Method (8 configs, 4 min)

**Question**: Mean vs CVaR for multi-seed fitness?

```csharp
FitnessAggregation (when SeedsPerIndividual > 1):
1. Mean          (average across seeds)
2. CVaR50        (current default - median, pessimistic)
3. CVaR25        (worst 25% - very pessimistic/robust)
4. CVaR75        (worst 75% - moderately pessimistic)
5. Min           (worst seed - extremely pessimistic)
6. Max           (best seed - optimistic)
7. TrimmedMean10 (drop top/bottom 10%, mean rest)
8. Median        (same as CVaR50 for clarity)
```

**Expectation**: Mean optimal for deterministic tasks, CVaR for stochastic

---

### Batch 11: WeightInitialization Methods (8 configs, 4 min)

**Question**: Is GlorotUniform optimal for evolution?

```csharp
WeightInitialization:
1. GlorotUniform (current default)
2. GlorotNormal
3. HeUniform
4. HeNormal
5. XavierUniform
6. XavierNormal
7. Uniform(-0.5, 0.5)
8. Uniform(-1.0, 1.0)
```

**Expectation**: GlorotUniform or simple Uniform optimal

---

### Batch 12: Combined Culling Strategies (8 configs, 4 min)

**Question**: Interactions between culling parameters?

```csharp
StagnationThreshold / GraceGenerations / DiversityThreshold / PerformanceThreshold:
1. 15/3/0.15/0.5   (current defaults)
2. 10/5/0.10/0.5   (fast cull, high protection)
3. 30/3/0.25/0.75  (slow cull, lenient)
4. 20/10/0.15/0.5  (balanced, high protection)
5. 15/3/0.10/0.25  (strict diversity + performance)
6. 50/3/0.50/0.9   (very conservative)
7. 5/1/0.05/0.25   (very aggressive)
8. INF/0/INF/1.0   (culling disabled)
```

**Expectation**: Current defaults or slightly more conservative

---

### Batch 13: Mutation Rate Combinations (8 configs, 4 min)

**Question**: Interactions between mutation operators?

```csharp
WeightL1Shrink / ActivationSwap / NodeParamMutate:
1. 0.1/0.01/0.2    (current defaults)
2. 0.0/0.0/0.0     (disable all three)
3. 0.2/0.0/0.0     (L1 only)
4. 0.0/0.05/0.0    (activation swap only)
5. 0.0/0.0/0.5     (node param only)
6. 0.1/0.0/0.0     (L1 at default, others off)
7. 0.05/0.005/0.1  (all reduced)
8. 0.2/0.02/0.3    (all increased)
```

**Expectation**: L1 helps, others minimal impact (Tanh-only architecture)

---

### Batch 14: Deep Architecture Variations (8 configs, 4 min)

**Question**: Fine-tune the 6-layer architecture?

```csharp
All Tanh-only, Dense:
1. 2→3→3→3→3→3→3→1  (current best from Phase 6)
2. 2→4→4→4→4→4→4→1  (wider)
3. 2→2→2→2→2→2→2→1  (very narrow, very deep)
4. 2→3→3→3→3→3→1    (5 layers - one fewer)
5. 2→3→3→3→3→3→3→3→1 (7 layers - one more)
6. 2→5→5→5→5→5→5→1  (wider still)
7. 2→3→4→4→4→3→1    (bottle shape)
8. 2→4→3→3→3→4→1    (hourglass)
```

**Expectation**: Current 6×3 optimal, but variations may surprise

---

### Batch 15: Extreme Deep Networks (8 configs, 4 min)

**Question**: How deep can we go?

```csharp
All narrow (2-3 nodes), Tanh-only, Dense:
1. 2→3→3→3→3→3→3→1       (6 layers - current)
2. 2→2→2→2→2→2→2→2→2→1   (8 layers, very narrow)
3. 2→3→3→3→3→3→3→3→3→1   (8 layers)
4. 2→2×10→1               (10 layers, 2 nodes each!)
5. 2→3×8→1                (8 layers, 3 nodes each)
6. 2→2×12→1               (12 layers!)
7. 2→3→3→3→3→3→3→3→3→3→1 (9 layers)
8. 2→2×15→1               (15 layers - extreme!)
```

**Expectation**: Diminishing returns past 8 layers, but worth testing

---

## Implementation Structure

```csharp
[Fact]
public void OneHourComprehensiveSweep()
{
    var batches = new[]
    {
        CreateBatch1_StagnationThreshold(),
        CreateBatch2_GraceGenerations(),
        CreateBatch3_SpeciesDiversityThreshold(),
        CreateBatch4_RelativePerformanceThreshold(),
        CreateBatch5_WeightL1ShrinkRate(),
        CreateBatch6_L1ShrinkFactor(),
        CreateBatch7_ActivationSwapRate(),
        CreateBatch8_NodeParamMutation(),
        CreateBatch9_SeedsPerIndividual(),
        CreateBatch10_FitnessAggregation(),
        CreateBatch11_WeightInitialization(),
        CreateBatch12_CombinedCulling(),
        CreateBatch13_MutationCombinations(),
        CreateBatch14_DeepArchitectureVariations(),
        CreateBatch15_ExtremeDeepNetworks()
    };

    foreach (var batch in batches)
    {
        RunBatchInParallel(batch, parallelism: 8);
    }
}
```

---

## Expected High-Impact Findings (Ranked)

### Tier 1: Likely Significant Impact (>10% improvement)
1. **Extreme Deep Networks** (Batch 15) - Might find 8-10 layers beats 6
2. **Deep Architecture Variations** (Batch 14) - Fine-tuning width/shape
3. **StagnationThreshold** (Batch 1) - Could free up species slots faster

### Tier 2: Moderate Impact (5-10% improvement)
4. **Combined Culling Strategies** (Batch 12) - Interactions matter
5. **RelativePerformanceThreshold** (Batch 4) - Balance diversity vs efficiency
6. **GraceGenerations** (Batch 2) - Protects innovation

### Tier 3: Minimal Impact (<5% improvement)
7. **WeightL1Shrink** (Batch 5, 6, 13) - Regularization helps but not critical
8. **Weight Init** (Batch 11) - Evolution adapts quickly
9. **Seeds & Aggregation** (Batch 9, 10) - Task is deterministic
10. **ActivationSwap** (Batch 7) - Tanh-only already optimal
11. **NodeParamMutate** (Batch 8) - Tanh has no params
12. **SpeciesDiversityThreshold** (Batch 3) - Narrow impact range

---

## Output Format

For each batch, report:
```
Batch X/15: [Name]
------------------------------------
Winner: [ConfigName]
  Gen0: [fitness]  Gen150: [fitness]  Improvement: [delta] ([%])

All Results (sorted):
  [Config1]  [improvement]  ([Gen150 fitness])
  [Config2]  [improvement]  ([Gen150 fitness])
  ...
```

Final summary:
```
TOP 10 IMPROVEMENTS FOUND:
1. [Config] → +X% ([Batch])
2. ...

RECOMMENDED CONFIGURATION CHANGES:
- [Parameter]: [Old] → [New] (+X% improvement)
- ...
```

---

## Success Criteria

1. **Coverage**: Test ALL 12 untested/minimally-tested parameters
2. **Depth**: 8 configs per parameter (explore full range)
3. **Signal**: 150 generations per config (vs 100 in Phase 6)
4. **Speed**: Complete in 60 minutes with 8-thread parallelism
5. **Actionable**: Find 2-3 parameters worth changing in defaults

---

## Risk Mitigation

1. **Bias fix verified**: All tests run post-bias-fix (Phase 6 confirmed)
2. **Baseline consistency**: Use same 2→3→3→3→3→3→1 architecture for param tests
3. **Multiple seeds**: Final verification with 5 seeds if unstable
4. **Incremental testing**: Can stop early if time runs over

---

## Next Steps After Sweep

1. **Update defaults** if >10% improvement found
2. **Long-run verification** (500 gens) with new config
3. **Cross-task validation** (XOR, CartPole, corridor)
4. **Document** final recommended configuration
5. **Publish** findings as research note

---

## How to Run

### File Created:
`Evolvatron.Tests/Evolvion/OneHourSweepTest.cs`

### Command to Execute:
```bash
cd /c/Dev/Evolvatron
dotnet test --filter "FullyQualifiedName~OneHourComprehensiveSweep" --logger "console;verbosity=detailed"
```

**IMPORTANT**: The test is marked with `[Skip]` attribute. You must either:
1. Remove the `Skip` attribute from line 26 in `OneHourSweepTest.cs`, OR
2. Comment out the `Skip = "..."` parameter

### Expected Runtime:
- **Duration**: ~60 minutes
- **Parallelism**: 8 threads
- **Total configs**: 120 (15 batches × 8 configs)
- **Generations per config**: 150
- **Total evaluations**: ~14.4 million

### Output Files:
Results will be printed to console. Recommended to capture to file:
```bash
dotnet test --filter "FullyQualifiedName~OneHourComprehensiveSweep" --logger "console;verbosity=detailed" 2>&1 | tee one-hour-sweep-results.log
```

### Test Structure:
- 15 batch methods
- 120 total configs
- 8-thread parallelism
- Comprehensive reporting (top 20 improvements, recommended changes)
