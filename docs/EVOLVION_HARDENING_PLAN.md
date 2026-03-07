# Evolvion Hardening Plan

Goal: Make Evolvion a robust, competitive neuroevolution engine for RL tasks (CartPole → Rocket landing).

## Phase 1: Fix Known Bugs (Safety First) ✅ DONE

### 1.1 Node-Index-Shift Bug in Diversification ✅
**File**: `SpeciesDiversification.cs` (AdaptIndividualToTopology)
**Problem**: When `MutateHiddenLayerSizes` shrinks a hidden row, nodes in later rows shift down. Edge matching by raw `(source, dest)` indices fails because the same semantic connection now has different indices. Weights are lost and replaced with random Glorot values.
**Fix**: Rewrote AdaptIndividualToTopology with per-row semantic node remapping via `(row, localIndex)`. Added `RemapAndPruneEdges()` to MutateHiddenLayerSizes to remap edge indices through old→new node mapping and remove edges that become same-row or backward.

### 1.2 TopologiesCompatible Is Insufficient ✅
**File**: `SpeciesDiversification.cs`
**Problem**: Checked only RowCounts and Edges.Count, not actual edge connectivity.
**Fix**: Now checks edge-by-edge identity after BuildRowPlans sorts both.

### 1.3 Hardcoded Magic Number for Activation Count ✅
**Fix**: Replaced all `random.Next(11)` / `for (i < 11)` with `Enum.GetValues<ActivationType>().Length`.

### 1.4 Debug Assertions for Array-Topology Consistency ✅
**Fix**: Added `Debug.Assert(individual.Weights.Length == spec.Edges.Count)` in CPUEvaluator.Evaluate().

### 1.5 Hidden-Layer Mutation Off-by-One ✅
**Problem**: MutateHiddenLayerSizes started from row 2 (assumed nonexistent bias row), skipping the first hidden row entirely.
**Fix**: Changed to start from row 1.

### 1.6 Activation Mask Mutation Violating Input/Output Invariants ✅
**Problem**: MutateAllowedActivations could mutate input row (breaking Linear-only contract) and output row (breaking Linear/Tanh contract).
**Fix**: Now only mutates hidden rows (skips row 0 and last row). Added `Validate()` call after diversification mutations.

### 1.7 Inherited Activations Violating New Masks ✅
**Problem**: After topology adaptation, copied activations could be disallowed by newly mutated activation masks.
**Fix**: Added `ReconcileActivations()` step after inheriting individuals.

### 1.8 Gaussian Sampler NaN ✅
**Problem**: `MathF.Log(0)` possible when `random.NextSingle()` returns 0.
**Fix**: Clamp u1 away from 0 in SampleGaussian.

### 1.9 ComputeRankProbabilities Order Mismatch ✅
**Problem**: Probabilities assigned by loop index, not mapped back to original individual order.
**Fix**: Now maps probabilities back via fitness-ranked index tracking.

### 1.10 Test SDK Version Mismatch ✅
**Problem**: Colonel.Tests pulls Microsoft.NET.Test.Sdk 18.0.1, but Evolvatron.Tests pinned 18.0.0 → NU1605 error blocking all tests.
**Fix**: Bumped to 18.0.1.

### 1.11 MaxInDegree Diversification Regression ✅
**File**: `SpeciesDiversification.cs`
**Problem**: Lowering MaxInDegree via MutateMaxInDegree can make an otherwise valid inherited topology fail Validate() immediately if any node already sits at the old cap.
**Fix**: Added `PruneExcessInDegree()` — after MutateMaxInDegree decreases the cap, randomly removes excess incoming edges from any node that exceeds the new limit.
**Tests needed**: Create a topology with nodes at cap, diversify repeatedly, assert no InvalidOperationException.

### 1.12 Reconcile Activation Params When Activation Type Changes ✅
**File**: `SpeciesDiversification.cs`
**Problem**: ReconcileActivations() changed Activations[] but left stale NodeParams[] from the old activation in place. A node rewritten from a parameterless activation to LeakyReLU/ELU would inherit arbitrary evolved values instead of sane defaults.
**Fix**: ReconcileActivations() now resets NodeParams to defaults for the new activation type (via `GetDefaultParameters()`).
**Tests needed**: Explicitly flip Linear → LeakyReLU/ELU and assert expected default params.

## Phase 2: Enable Edge Topology Mutations

### 2.1 Create Weight Synchronization Infrastructure
**New method**: `SpeciesSpec.ApplyTopologyChange(species, changeAction)` or similar pattern that:
1. Captures old edge list
2. Applies the topology change (add/delete/split/etc.)
3. Calls BuildRowPlans()
4. Builds old-index → new-index mapping by matching (source, dest) tuples
5. Remaps all individuals' weight arrays

**Alternative simpler approach**: After any topology mutation, rebuild weight arrays for all individuals using edge identity matching (same approach as AdaptIndividualToTopology but within the same species).

### 2.2 Integrate into Evolver
**File**: `Evolver.cs:160-181`
- Edge mutations should be applied **per-species, not per-individual** (topology is shared)
- Move edge mutations out of `ApplyMutations` (individual-level) into `EvolveSpecies` (species-level)
- Apply edge mutations AFTER offspring generation but BEFORE weight mutations
- Flow: select → reproduce → edge topology mutation (once per species) → weight mutations (per individual)

### 2.3 Test Weight Integrity
- Add test: evolve for 50 generations with all edge mutations enabled, assert `individual.Weights.Length == spec.Edges.Count` every generation for every individual
- Add test: edge split followed by evaluation produces finite outputs
- Add test: edge delete followed by evaluation produces finite outputs

### 2.4 Topology Mutation Invariant Suite
After every topology mutation and diversification step, assert the full invariant set:
- `Weights.Length == spec.Edges.Count`
- `Biases.Length == spec.TotalNodes`
- `Activations.Length == spec.TotalNodes`
- `NodeParams.Length == spec.TotalNodes * 4`
- All activations allowed by their row's mask
- Evaluation returns finite outputs on smoke inputs
- Topology passes `Validate()`

These should be packaged as a reusable `AssertSpeciesInvariant(species)` method for use in tests and optionally as a Debug-only check in production code.

## Phase 2.5: Genome Alignment Policy

Before implementing crossover, define the canonical identity model for nodes and edges.

### Key decisions:
1. **Node identity**: Nodes are identified by `(row, localIndex)`. Two individuals in the same species have semantically identical nodes at every position. This is guaranteed by the shared-topology constraint.

2. **Edge identity**: Edges are identified by their `(source, dest)` tuple after `BuildRowPlans()` sorting. Within a species, edge index `i` means the same connection for all individuals.

3. **Within-species crossover**: Trivially safe. Shared topology guarantees array alignment — weight index `i` maps to the same edge in both parents. No innovation numbers needed.

4. **Cross-species crossover**: **Not supported.** Different species have different topologies, edge counts, and node counts. Cross-species recombination would require NEAT-style innovation tracking or the semantic node remapping already used by `AdaptIndividualToTopology`. The cost/complexity is not justified given that species diversification already provides topology exploration.

5. **Rationale**: Evolvion's architecture is designed around GPU-batched evaluation of identical topologies. Cross-species crossover would undermine this by requiring per-individual topology variation. The topology search happens at the species level through diversification and culling, not through individual-level structural recombination.

## Phase 3: Add Crossover

### 3.1 Within-Species Weight Crossover
Since all individuals in a species share topology, crossover is straightforward:

```csharp
// Uniform crossover
static Individual Crossover(Individual parent1, Individual parent2, Random random)
{
    var child = new Individual(parent1); // deep copy
    for (int i = 0; i < child.Weights.Length; i++)
    {
        if (random.NextSingle() < 0.5f)
            child.Weights[i] = parent2.Weights[i];
    }
    // Same for biases, activations, nodeParams
    return child;
}
```

Variants to implement:
- **Uniform crossover** (50/50 per weight)
- **Arithmetic crossover** (weighted average: `alpha * p1 + (1-alpha) * p2`)
- **Fitness-proportional blend** (bias toward fitter parent)

### 3.2 Config Integration
Add to `EvolutionConfig`:
- `CrossoverRate` (probability of crossover vs clone-and-mutate)
- `CrossoverType` (enum: Uniform, Arithmetic, FitnessBlend)

### 3.3 Ablation
Test crossover on CartPole: crossover-enabled vs mutation-only, 5+ seeds, 200 generations.

## Phase 4: CartPole Benchmark

### 4.1 Validate CartPole Environment
- Write unit tests for CartPole physics (known trajectories, boundary conditions)
- Verify observation normalization (inputs should be roughly [-1, 1])
- Verify action mapping (continuous output → discrete or continuous force)
- Test that random agent gets low fitness, perfect agent gets max fitness

### 4.2 CartPole Evolution Test
- Start with small topology: 4 inputs → 8 hidden → 1 output
- Multi-seed evaluation (5-10 seeds per individual, average fitness)
- Target: solve in <100 generations
- Compare with known baselines (NEAT typically solves CartPole in 20-50 gens)

### 4.3 CartPole GPU Kernels
- Implement GPU CartPole environment kernels (same pattern as Spiral/XOR)
- CartPole physics is simple enough for GPU: just 4 state variables, 1 action
- Verify GPU matches CPU evaluation within tolerance

## Phase 5: Ablation on RL-Tuned Config

### 5.1 CartPole Ablation Battery
Run all ablations on CartPole (not Spiral/Rosenbrock — RL dynamics differ):

**Batch A: Weight mutation ablations**
- No weight jitter
- No weight reset
- No L1 shrink
- No bias mutations (are they helping for RL?)
- No activation swap
- No node param mutate

**Batch B: Selection ablations**
- Tournament size sweep: 3, 5, 10, 15, 20
- Parent pool percentage sweep: 0.3, 0.5, 0.75, 1.0
- Elites sweep: 1, 3, 5, 10

**Batch C: Species management ablations**
- No culling (all species survive)
- Stagnation-only culling
- Diversity-only culling
- Grace period sweep: 1, 3, 5, 10

**Batch D: Structural ablations**
- Activations restricted to {Tanh, ReLU} only vs all 11
- Single hidden layer vs two hidden layers
- Dense init vs sparse init

**Batch E: Crossover ablations** (after Phase 3)
- No crossover (baseline)
- Uniform crossover at 30%, 50%, 70%
- Arithmetic crossover at 30%, 50%, 70%

**Protocol**: 10 seeds × 200 generations per config. Report: median solve generation, solve rate, final fitness distribution.

### 5.2 Optuna Re-Tune for CartPole
After ablation identifies which operators matter, run Optuna on the reduced parameter space for CartPole specifically.

## Phase 6: Rocket Landing

### 6.1 Multi-Episode Evaluation
**Critical for RL**: Evaluate each individual across N different initial conditions:
- Vary starting height, angle, velocity
- Average fitness across episodes
- Prevents overfitting to single scenario

Add `EpisodesPerIndividual` to `EvolutionConfig`. GPU already supports this via `EvaluateRowForEpisodesKernel`.

### 6.2 Rocket Environment Tuning
- Verify reward shaping produces smooth gradient (not sparse binary reward)
- Test observation normalization (all inputs ~[-1, 1])
- Consider adding distance-to-pad as observation (if not already present)
- Verify terminal conditions are correctly detected

### 6.3 GPU Batched Rocket Evaluation
- `GPUBatchedEnvironment` already exists in Rigidon/GPU/Batched/
- Integrate with Evolvion's `GPUBatchedFitnessEvaluator`
- Benchmark: how many rocket episodes/second on target GPU?

### 6.4 Rocket Ablation
Same ablation battery as CartPole (Phase 5), but on Rocket.
Key question: do CartPole-optimal hyperparameters transfer to Rocket, or does each environment need its own tuning?

## Phase 7: Competitive Benchmarking

### 7.1 Compare Against Baselines
- **NEAT** (SharpNeat): standard neuroevolution baseline
- **CMA-ES**: strong black-box optimizer for continuous control
- **PPO/SAC** (if feasible): RL baselines for same environments
- Report: solve rate, sample efficiency (total evaluations to solve), wall-clock time

### 7.2 Scaling Analysis
- How does performance scale with population size? (1K, 5K, 10K, 50K individuals)
- GPU utilization at each scale
- Diminishing returns threshold

## Phase 8: GPU Evaluation Pipeline Optimization

### 8.0 Eliminate CPU Evaluator via ILGPU CPU Backend (Medium)

The pure-C# `CPUEvaluator` duplicates all GPU kernel logic and must be manually kept in sync. This has caused multiple bugs (velocity stabilization, motor dt, MaxVelocity mismatch). ILGPU can target CPU via `CPUAccelerator`, running the exact same kernel code — zero sync burden by construction.

**Benchmark to run**: Compare `CPUEvaluator` vs `GPUEvaluator` with `CPUAccelerator` on a real workload (2K population, 600 steps, rocket environment). If ILGPU CPU is within 30% of pure C#, delete `CPUEvaluator` entirely.

**What we keep**: `CPUStepper` (physics engine) stays — it serves `TrajectoryOptimizer` which needs per-step Jacobian access that kernels can't provide. The physics stepper is a different layer from the Evolvion evaluator.

**What we lose**: Debuggability of stepping through pure C# evaluation. Mitigated by: (a) `CPUStepper` still exists for physics debugging, (b) NN forward pass can be tested in isolation without a full evaluator.

### 8.1 Remove Redundant Synchronization (Easy)
Current code calls `_accelerator.Synchronize()` after every kernel launch (~300 syncs/species/generation). ILGPU streams are in-order, so consecutive launches on the same stream don't need explicit sync. Only sync when reading results back to CPU.

### 8.2 Multi-Species Batch Evaluation (Medium)
Currently species are evaluated sequentially (`foreach species`). Pack all species into a single GPU pass by concatenating SoA buffers and using per-individual species indices for RowPlan/edge lookup.

### 8.3 Keep Weights on GPU Across Generations (High)
Upload weights once, then apply mutations via GPU kernels. Only download fitness values. Eliminates ~10MB/generation transfer for large populations.

## Priority Order

| Phase | Effort | Impact | Dependency | Status |
|-------|--------|--------|------------|--------|
| 1. Fix Bugs | Low | High (correctness) | None | ✅ DONE |
| 2. Edge Mutations | Medium | High (unlock topology search) | Phase 1 | |
| 2.5 Genome Alignment | Low | High (correctness foundation) | None | |
| 3. Crossover | Low | Medium (exploitation) | Phase 2.5 | |
| 4. CartPole | Low | High (first real RL validation) | None | |
| 5. Ablation | Medium | High (know what works) | Phases 3-4 | |
| 6. Rocket | Medium | High (flagship task) | Phase 5 | |
| 7. Benchmarking | Low | Medium (credibility) | Phase 6 | |
| 8. GPU Pipeline | Medium | High (throughput) | Phase 4 | |

Phases 2.5, 3, and 4 can run in parallel. Phase 2 depends on Phase 1. Phase 3 depends on 2.5. Phase 5 depends on 3+4. Phase 6 depends on 5. Phase 8 can start anytime after Phase 4.
