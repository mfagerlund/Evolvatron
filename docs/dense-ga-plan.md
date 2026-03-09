# Plan: GA on Dense Kernel

## Goal
Run GA evolutionary operators (tournament selection, elitism, weight jitter) on the same `GPUDenseDoublePoleEvaluator` and `DenseNN` kernel used by CEM/ES. This gives a true apples-to-apples comparison: same GPU kernel, same population size, same wall-clock time — only the optimization algorithm differs.

Currently GA uses the sparse `GPUDoublePoleEvaluator` + `InlineNN`, which is ~2.5x slower per generation. The GA's 314/625 median was achieved with ~600 gens in 100s. On the dense kernel it would get ~1500 gens — potentially even higher scores.

## What Changes

### New file: `ES/DenseGAStrategy.cs`

A population-based optimizer that works with flat param vectors (same interface as `IslandOptimizer`). No distribution (no mu/sigma). Stores the actual population.

```
class DenseGAOptimizer
  - float[] population       // [popSize × paramCount] - current generation
  - float[] fitnesses        // [popSize]
  - int popSize, paramCount
  - Config: eliteCount, tournamentSize, jitterStdDev, parentPoolFraction

  Methods:
  - constructor(DenseTopology, int popSize, int seed)
      Initialize all individuals via Glorot

  - float[] GetParamVectors()
      Return population array (already flat, ready for GPU)

  - void Update(float[] evaluatedFitness)
      Store fitnesses for selection

  - void StepGeneration(Random rng)
      1. Sort by fitness, identify elite indices
      2. Copy elites unchanged into next generation
      3. For remaining slots:
         a. Tournament select a parent from top parentPoolFraction
         b. Clone parent weights
         c. Add Gaussian jitter (jitterStdDev) to all weights
      4. Swap current ↔ next population buffer
```

### Key GA parameters (from existing EvolutionConfig)
- `Elites = 10` (preserve top 10 unchanged)
- `TournamentSize = 5`
- `ParentPoolPercentage = 0.5` (select from top 50%)
- `WeightJitterStdDev = 0.15` (the secret sauce for generalization)

### No changes needed to:
- `GPUDenseDoublePoleEvaluator` — already takes `float[]` param vectors
- `DenseTopology` — already describes the network
- `DenseNN.cs` / `DenseDoublePoleStepKernel.cs` — GPU kernel unchanged
- `IslandOptimizer` / `CEMStrategy` / `ESStrategy` — untouched

### New test: `DenseGABenchmark.cs`

```
[Fact] DenseGA_625Grid_5443()
  - DenseTopology.ForDPNV([4,4], ctx=2)
  - GPUDenseDoublePoleEvaluator (MaxSteps=1000, Gruau)
  - DenseGAOptimizer (16K pop, jitter=0.15)
  - 100s budget, 10 seeds
  - Track best grid every 3s (same as CEM/ES sweeps)
  - Report median, range, pass≥200

[Fact] DenseGA_625Grid_583()
  - Same but 5→8→3 topology

[Fact] DenseGA_JitterSweep()
  - Sweep jitterStdDev: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  - Find optimal jitter for dense kernel
  - 2 seeds × 30s per config
```

## Implementation Steps

1. **Create `ES/DenseGAOptimizer.cs`** (~80 lines)
   - Constructor: Glorot init all individuals
   - `GetParamVectors()`: return flat array
   - `Update(float[] fitness)`: store fitnesses
   - `StepGeneration(Random rng)`: elitism + tournament + jitter
   - Double-buffered (current + next population)

2. **Create `Tests/Evolvion/DenseGABenchmark.cs`** (~150 lines)
   - Reuse Build625Grid() from existing sweeps
   - Same tracked-best-grid pattern as CEMRefinedSweep
   - Run jitter sweep, then 10-seed validation

3. **Run and compare**
   - Expected: GA on dense kernel gets MORE gens/s than sparse
   - Prediction: GA generalization should improve (more gens = more selection pressure)
   - Key question: does the speed advantage translate to higher grid scores?

## Expected Outcome

| Algorithm | Kernel | Gens in 100s | Median Grid |
|-----------|--------|:---:|:---:|
| GA (sparse) | InlineNN | ~600 | 314/625 |
| GA (dense) | DenseNN | ~1500 | ? |
| CEM | DenseNN | ~1500 | 223/625 |
| ES | DenseNN | ~2300 | 164/625 |

If GA on dense kernel scores higher than 314, the speed advantage matters.
If it scores ~314, the extra gens don't help (GA already found the generalization ceiling).
Either way, this eliminates the kernel speed confound from the comparison.

## Non-goals
- No topology mutation / edge add / edge delete
- No species / speciation
- No activation type diversity (all Tanh)
- No node parameter mutations
- This is pure weight-space GA on a fixed dense topology
