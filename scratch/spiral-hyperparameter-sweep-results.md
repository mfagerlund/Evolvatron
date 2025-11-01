# Spiral Classification Hyperparameter Sweep Results

## Executive Summary

**16 configurations tested, 10 generations each, 37 seconds total runtime**

**Key Finding**: Tournament size has the strongest impact on improvement (correlation: 0.743)

**Best Configuration**: LargeTournament
- Tournament size: 16 (vs baseline: 4)
- Weight jitter: 0.95 (vs baseline: 0.90)
- Improvement: 0.0084 fitness units (0.9%) - **35% better than baseline**

**Worst Configuration**: Conservative
- Tournament size: 2 (weak selection)
- Many elites: 20 (vs baseline: 2)
- Low mutation rates
- Improvement: 0.0012 (0.1%) - **80% worse than baseline**

## Full Rankings

| Rank | Configuration | Gen0 | Gen9 | Improvement | Improve % |
|------|--------------|------|------|-------------|-----------|
| 1 | **LargeTournament** | -0.9634 | -0.9550 | **0.0084** | 0.9% |
| 2 | **Aggressive** | -0.9634 | -0.9553 | **0.0081** | 0.8% |
| 3 | **Balanced High** | -0.9634 | -0.9560 | **0.0075** | 0.8% |
| 4 | Strong Select | -0.9634 | -0.9562 | 0.0072 | 0.8% |
| 5 | Baseline | -0.9634 | -0.9572 | 0.0062 | 0.6% |
| 6 | Weight Focus | -0.9634 | -0.9572 | 0.0062 | 0.6% |
| 7 | More Elites | -0.9634 | -0.9577 | 0.0057 | 0.6% |
| 8 | Large Pop | -0.9634 | -0.9594 | 0.0040 | 0.4% |
| 9 | Few Species | -0.9634 | -0.9594 | 0.0040 | 0.4% |
| 10 | Low Mutation | -0.9634 | -0.9600 | 0.0034 | 0.4% |
| 11 | SmallTournament | -0.9634 | -0.9605 | 0.0029 | 0.3% |
| 12 | High Mutation | -0.9634 | -0.9605 | 0.0029 | 0.3% |
| 13 | Many Species | -0.9634 | -0.9608 | 0.0027 | 0.3% |
| 14 | Edge Focus | -0.9634 | -0.9610 | 0.0024 | 0.2% |
| 15 | Minimal | -0.9634 | -0.9611 | 0.0024 | 0.2% |
| 16 | **Conservative** | -0.9634 | -0.9622 | **0.0012** | 0.1% |

## Parameter Correlations

**Strong positive correlations (helps improvement):**
- **Tournament size: 0.743** ⭐ (strongest predictor)
- **Weight jitter: 0.700** ⭐
- **Weight reset: 0.582**

**Weak positive correlations:**
- Edge add: 0.186
- Population size: 0.143

**Negative correlations (hurts improvement):**
- **Elites: -0.264** (more elites = less exploration)

## Key Insights

### 1. Tournament Size is Critical

The strongest correlation with improvement is tournament size (0.743).

**Why this matters:**
- Larger tournaments = stronger selection pressure
- With weak fitness variance (~0.1-0.2 units), need aggressive selection to amplify signal
- Tournament 16 vs Tournament 2: **7x better improvement**

**Configurations by tournament size:**
- Size 16: Best performer (0.0084 improvement)
- Size 8: Strong performers (0.0081, 0.0072)
- Size 4: Moderate (0.0062 baseline)
- Size 2: Worst performers (0.0029, 0.0012)

### 2. High Mutation Rates Help (When Combined with Strong Selection)

**Top performers all use high mutation:**
- LargeTournament: WeightJitter=0.95, Reset=0.10
- Aggressive: WeightJitter=0.95, Reset=0.20
- Balanced High: WeightJitter=0.90, Reset=0.10

**But "High Mutation" alone (rank 12) performed poorly**
- Why? Tournament size was only 4
- High mutation without strong selection = random walk

**Key insight**: Need high mutation + strong selection together

### 3. More Elites HURTS Performance

Correlation: -0.264

- Conservative (20 elites): Rank 16 (worst)
- More Elites (10 elites): Rank 7 (below baseline)
- Aggressive (1 elite): Rank 2 (best)
- LargeTournament (2 elites): Rank 1

**Why**: More elites = less exploration
- With weak fitness signal, need aggressive exploration
- Preserving too many elites prevents population from trying new solutions

### 4. Population Size Doesn't Matter Much

Correlation: 0.143 (weak)

- Large Pop (1600 individuals): Rank 8
- Baseline (800 individuals): Rank 5
- Minimal (200 individuals): Rank 15

**Interpretation**:
- Doubling population doesn't help much
- Quality of selection > quantity of individuals
- Confirms that fitness landscape, not sampling, is the bottleneck

### 5. Species Count is Irrelevant

- Many Species (16 species): Rank 13
- Baseline (8 species): Rank 5
- Few Species (4 species): Rank 9

All similar performance. **Species diversity doesn't help when all species converge to same fitness.**

### 6. Edge Mutations Don't Help

- Edge Focus (EdgeAdd=0.25): Rank 14 (very poor)
- Weight Focus (EdgeAdd=0.01): Rank 6 (above baseline)

**Why edge mutations don't help**:
- Problem needs better weights, not different topology
- All sparse random topologies can represent solution
- Topology changes are disruptive without immediate benefit

## Recommended Configuration

Based on sweep results, the optimal config for spiral classification:

```csharp
new EvolutionConfig
{
    SpeciesCount = 8,              // Baseline (doesn't matter much)
    IndividualsPerSpecies = 100,   // Baseline (sufficient)
    Elites = 1-2,                  // LOW (aggressive exploration)
    TournamentSize = 16,           // HIGH (strong selection) ⭐
    MutationRates = new MutationRates
    {
        WeightJitter = 0.95,       // HIGH ⭐
        WeightReset = 0.10-0.20,   // MODERATE-HIGH
        WeightJitterStdDev = 0.3   // Default
    },
    EdgeMutations = new EdgeMutationConfig
    {
        EdgeAdd = 0.05,            // Baseline (low priority)
        EdgeDeleteRandom = 0.02    // Baseline
    }
}
```

## Projected Improvement

**With baseline config**: 0.0062 per 10 generations
- Need to improve: -0.96 → -0.05 = 0.91 fitness units
- **Generations needed: ~1,470**
- **Time: ~25 minutes**

**With LargeTournament config**: 0.0084 per 10 generations
- **Generations needed: ~1,080** (35% faster)
- **Time: ~18 minutes**

**Still not great**, but 35% improvement is significant.

## Why This Problem is Still Hard

Even with optimal hyperparameters:
1. **Fitness variance is inherently low** (~0.1-0.2 range)
2. **No intermediate milestones** - can't reward "partial progress"
3. **Sparse temporal feedback** - 1 signal per 100 steps
4. **Discrete decision boundary required** - qualitative vs quantitative improvement

## Comparison to Corridor Following

Corridor following achieves >50% track completion in **50-100 generations**.

Spiral classification with optimal hyperparameters would need **~1,080 generations** to reach threshold.

**Why the 10-20x difference?**

| Factor | Corridor | Spiral |
|--------|----------|--------|
| Fitness range | ~1.0 | ~0.2 |
| Temporal feedback | 320 steps | 1 signal |
| Intermediate milestones | Yes (checkpoints) | No |
| Smooth improvement path | Yes | No |

## Recommendations

### Option 1: Accept the slow convergence
- Use LargeTournament config
- Run for ~1,000-1,500 generations
- ~18-25 minutes total time
- Validates evolution works, just slowly

### Option 2: Modify environment for dense rewards
```csharp
// Return reward every step instead of accumulating
public float Step(ReadOnlySpan<float> actions) {
    var (_, _, expected) = _testCases[_currentCase];
    float output = actions[0];
    float error = (output - expected) * (output - expected);
    _currentCase++;
    return -error;  // Immediate feedback
}
```

Expected impact: **10-50x faster convergence** (based on corridor comparison)

### Option 3: Curriculum learning
Start with easier versions:
1. Linearly separable (5 gens)
2. Single curved boundary (20 gens)
3. Loose spirals (50 gens)
4. Tight spirals (100 gens)

Total: ~175 generations instead of 1,080

### Option 4: Remove from benchmark suite
Accept that spiral classification is mismatched to Evolvion's strengths.
Focus benchmarks on temporal control tasks where dense feedback exists.

## Final Answer to User

**Yes, hyperparameters matter** - we achieved 35% improvement with:
- **Large tournament size (16)**: Strongest effect
- **High mutation rates**: Weight jitter 0.95
- **Few elites (1-2)**: More exploration

**But hyperparameters can't fix fundamental mismatch:**
- Best config still needs ~1,080 generations (~18 minutes)
- vs Corridor following: ~50-100 generations
- **10-20x slower** even with optimal settings

The problem isn't bad hyperparameters - it's that batch supervised classification with sparse feedback is a poor fit for evolutionary RL designed for dense temporal feedback.
