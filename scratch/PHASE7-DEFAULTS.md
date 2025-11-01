# Phase 7 Recommended Defaults (Post-Comprehensive-Sweep)

**Date**: 2025-11-01
**Based on**: Comprehensive hyperparameter sweep (128 configurations, 16.1 minutes)

---

## Executive Summary

Phase 7 comprehensive sweep tested ALL previously untested parameters post-bias-fix and discovered:
1. **Ultra-deep narrow networks** (15×2) provide +63.4% improvement over previous best (6×3)
2. **Mutation dynamics changed** - Mixed activations help (+33%), NodeParamMutate hurts (+39% when disabled)
3. **Dense initialization still required** - Sparse (<0.5) performs 100x worse

---

## Recommended Default Configuration

### Architecture: Ultra-Deep Narrow (15 layers × 2 nodes)

```csharp
// WINNER: 2→(2 nodes × 15 layers)→1
// Provides +63.4% improvement over 6×3 baseline
var topology = new SpeciesBuilder()
    .AddInputRow(2)
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 1
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 2
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 3
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 4
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 5
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 6
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 7
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 8
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 9
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 10
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 11
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 12
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 13
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 14
    .AddHiddenRow(2, ActivationType.Tanh)  // Layer 15
    .AddOutputRow(1, ActivationType.Tanh)
    .WithMaxInDegree(6)
    .InitializeDense(random, density: 1.0f)  // Dense still wins!
    .Build();
```

**Key insight**: DEPTH >> WIDTH for this architecture

**Alternatives considered**:
- 5×6 (wider mid-depth): +46.6% improvement (second place)
- Bottleneck shape: +41.9% improvement (third place)
- 7×3 (deeper current width): +26.1% improvement

---

### Evolution Config: Updated Mutation Rates

```csharp
var config = new EvolutionConfig
{
    // Population structure (unchanged from Phase 6)
    SpeciesCount = 4,
    IndividualsPerSpecies = 200,  // Total: 800
    Elites = 2,
    TournamentSize = 16,
    ParentPoolPercentage = 1.0f,

    // Culling parameters (NO EFFECT OBSERVED - needs investigation)
    GraceGenerations = 3,
    StagnationThreshold = 15,
    SpeciesDiversityThreshold = 0.15f,
    RelativePerformanceThreshold = 0.5f,

    // Mutation rates (UPDATED in Phase 7)
    MutationRates = new MutationRates
    {
        WeightJitter = 0.95f,           // unchanged
        WeightJitterStdDev = 0.3f,      // unchanged
        WeightReset = 0.10f,            // unchanged

        WeightL1Shrink = 0.20f,         // CHANGED: was 0.1f (+15.1% improvement)
        L1ShrinkFactor = 0.9f,          // unchanged

        ActivationSwap = 0.10f,         // CHANGED: was 0.01f (+33.3% improvement)
        NodeParamMutate = 0.0f,         // CHANGED: was 0.2f (+38.9% when disabled)
        NodeParamStdDev = 0.1f          // unchanged
    },

    EdgeMutations = new EdgeMutationConfig
    {
        EdgeAdd = 0.05f,                // unchanged
        EdgeDeleteRandom = 0.02f        // unchanged
    },

    // NOT IMPLEMENTED (dead code - needs implementation)
    WeightInitialization = "GlorotUniform",  // hardcoded, parameter ignored
    SeedsPerIndividual = 1,                  // not used by SimpleFitnessEvaluator
    FitnessAggregation = "Mean"              // not implemented
};
```

---

## Changes from Phase 6

| Parameter | Phase 6 | Phase 7 | Improvement | Notes |
|-----------|---------|---------|-------------|-------|
| **Architecture** | 6×3 | 15×2 | +63.4% | Ultra-deep narrow wins |
| **WeightL1Shrink** | 0.1 | 0.2 | +15.1% | More regularization |
| **ActivationSwap** | 0.01 | 0.10 | +33.3% | Mixed activations help |
| **NodeParamMutate** | 0.2 | 0.0 | +38.9% | Disable completely |
| **InitDensity** | 1.0 | 1.0 | N/A | Dense still required |

**Combined expected improvement**: ~80-100% (effects may be superlinear)

---

## Critical Findings: Dead Code

### ❌ NOT IMPLEMENTED (must be fixed):

1. **SeedsPerIndividual** - `SimpleFitnessEvaluator` ignores this parameter
   - Always evaluates once per generation
   - No multi-seed evaluation logic exists

2. **FitnessAggregation** - No aggregation implemented
   - Parameter is complete dead code
   - No Mean/CVaR/Min/Max logic exists

3. **WeightInitialization** - Hardcoded to GlorotUniform
   - No switch logic for HeUniform, XavierNormal, etc.
   - All 8 tested variants were actually identical

### ⚠️ LIKELY NO EFFECT (needs investigation):

4. **Culling Parameters** - All 4 showed zero effect
   - Code exists and is called correctly
   - BUT: ALL 4 conditions must be met simultaneously (strict AND logic)
   - With only 4 species, max 2 can be culled (MinSpeciesCount=2)
   - Theory: Culling rarely triggers in 150 generations with 4 species

**Action items**:
- Implement SeedsPerIndividual + FitnessAggregation
- Implement WeightInitialization factory
- Add culling diagnostics or remove dead parameters

---

## Sparse vs Dense Results

**Critical question**: Does NEAT-style sparse-to-dense work post-bias-fix?
**Answer**: NO - dense still dominates

| Density | Gen0→Gen150 | Improvement | vs Dense |
|---------|-------------|-------------|----------|
| **1.0 (Dense)** | -0.9658→-0.7148 | 0.2510 | 100% 🥇 |
| 0.95 | -0.9658→-0.7148 | 0.2510 | 100% |
| 0.85 | -0.9658→-0.7148 | 0.2510 | 100% |
| 0.7 | -0.9650→-0.7837 | 0.1814 | 72% |
| 0.5 | -0.9650→-0.7837 | 0.1814 | 72% |
| **0.3** | -0.9661→-0.9632 | **0.0029** | **1%** ⚠️ |
| 0.2 | -0.9661→-0.9632 | 0.0029 | **1%** |
| **0.1 (Sparse)** | -0.9661→-0.9632 | **0.0029** | **1%** |

**Key findings**:
- Dense (0.85-1.0) still required even after bias fix
- Networks with <50% density barely learn (100x worse!)
- Performance cliff at ~0.7 density threshold
- NEAT-style sparse-to-dense does NOT work for this architecture

---

## Next Steps

### High Priority:
1. ✅ Update EvolutionConfig defaults (mutation rates)
2. ⏳ Implement SeedsPerIndividual + FitnessAggregation
3. ⏳ Implement WeightInitialization variants
4. ⏳ Investigate/fix or remove culling parameters
5. ⏳ Run 500-generation validation with new config
6. ⏳ Test on other tasks (XOR, CartPole, etc.)

### Research Questions:
- Why do ultra-deep narrow networks work so well?
- Can we go even deeper? (20 layers? 25?)
- Does this generalize to other tasks?
- Why does sparse initialization fail catastrophically?

---

## Full Results

See: `scratch/COMPREHENSIVE-SWEEP-RESULTS.md`

**Total execution time**: 16.1 minutes for 128 configurations
**Status**: ✅ All planned tests completed successfully
