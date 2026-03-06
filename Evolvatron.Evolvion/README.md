# Evolvion — GPU-Batched Speciated Neuroevolution

Evolvion is a speciated neuroevolution framework designed around **GPU-batched evaluation**. All individuals within a species share identical network topology, enabling massively parallel forward passes with coalesced memory access. Species explore different architectures while individuals within each species are evaluated as a single batched operation on the GPU.

This topology-sharing constraint is the central design decision — it drives the population structure, the mutation system, the species management, and the memory layout. Everything serves efficient parallel evaluation.

**Target domain**: Reinforcement learning and optimization tasks with small observation/action spaces (2–25 signals), where thousands of neural controllers must be evaluated per generation.

**Status**: Optuna hyperparameter search complete. Solves spiral classification in 3 generations.

## Core Concepts

### The GPU-Batching Principle

Traditional neuroevolution (e.g., NEAT) gives every individual a unique topology. This makes GPU parallelism difficult — each network has different shapes, edge counts, and evaluation paths.

Evolvion takes a different approach: **topology is shared per species**. All N individuals in a species have the same edges, the same number of nodes, the same evaluation graph. They differ only in weights, biases, and activation choices. This means:

- **One kernel launch evaluates an entire species** — all individuals simultaneously
- **Memory is Structure-of-Arrays (SoA)** — weights for all individuals are contiguous, enabling coalesced GPU reads
- **RowPlans compile once per species** — the evaluation schedule is computed once, not per-individual
- **Edge sorting by destination** — enables sequential memory access patterns within each kernel invocation

Species-level topology search still happens, but through species diversification and culling rather than per-individual structural mutation (see [NEAT comparison](../docs/NEAT_COMPARISON.md)).

### Hierarchy: Population → Species → Individual

- **Population** — The entire evolutionary cohort. Manages all species, tracks global generation count, and orchestrates evolution steps.
- **Species** — A group of individuals sharing identical network topology (`SpeciesSpec`). Species compete for survival based on collective performance. Each species has an age, stats tracker, and grace period protection against premature culling.
- **Individual** — A single neural controller defined by mutable arrays:
  - `float[] Weights` — Edge weights (count = topology edges)
  - `float[] Biases` — Per-node bias terms (count = total nodes)
  - `ActivationType[] Activations` — Per-node activation function choice
  - `float[] NodeParams` — 4 parameters per node (activation-specific, e.g., LeakyReLU alpha)
  - `bool[] ActiveNodes` — Marks nodes reachable from input to output
  - `float Fitness` — Current fitness value
  - `int Age` — Generation count for this individual

The topology-parameter separation is not just conceptual — it's the mechanism that makes GPU batching possible.

### SpeciesSpec — Immutable Topology Definition

`SpeciesSpec` defines the network structure shared by all members of a species:

```
RowCounts: [6, 8, 6, 3]          — nodes per layer (input, hidden..., output)
Edges: [(0,6), (1,7), (3,12)]    — directed connections (source → dest)
AllowedActivationsPerRow: uint[]  — bitmask of permitted activations per layer
MaxInDegree: 6                    — cap on incoming edges per node
RowPlans: RowPlan[]               — compiled evaluation metadata
```

**Constraints:**
- Strictly **acyclic** (feedforward): for every edge, `row(source) < row(dest)`
- **In-degree capped**: no node receives more than `MaxInDegree` edges
- **Input layer**: always Linear activation
- **Output layer**: restricted to Linear or Tanh

### RowPlan — Compiled Evaluation Metadata

After any topology mutation, `BuildRowPlans()` recompiles evaluation metadata:

```csharp
struct RowPlan {
    int NodeStart;   // first node index in this row
    int NodeCount;   // number of nodes in this row
    int EdgeStart;   // first edge index targeting this row
    int EdgeCount;   // number of edges into this row
}
```

Edges are sorted by destination node during compilation, enabling sequential memory access during the forward pass — critical for GPU kernel performance.

### SpeciesBuilder — Fluent API for Topology Construction

```csharp
var topology = new SpeciesBuilder()
    .AddInputRow(2)                                                    // Linear only
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh,       // allowed activations
                     ActivationType.Sigmoid)
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh)
    .AddOutputRow(1, ActivationType.Tanh)                             // Linear or Tanh only
    .WithMaxInDegree(10)
    .InitializeDense(random, density: 0.3f)                           // or InitializeSparse()
    .Build();
```

- `InitializeSparse(random)`: Create minimal connected topology via EdgeAdd
- `InitializeDense(random, density)`: Create dense fully-connected layers with density control

## Neural Network Evaluation

### Forward Pass (CPUEvaluator)

Row-by-row evaluation:

1. Copy observations into input layer nodes
2. For each subsequent row:
   - Zero-initialize node values
   - Accumulate `weight * source_value` for all incoming edges
   - Add per-node bias
   - Apply per-node activation function
3. Read output layer values as actions

### Activation Functions

11 activation types, each with optional learnable parameters:

| ID | Type | Formula | Learnable Params |
|----|------|---------|-----------------|
| 0 | Linear | f(x) = x | — |
| 1 | Tanh | f(x) = tanh(x) | — |
| 2 | Sigmoid | f(x) = 1/(1+exp(-x)) | — |
| 3 | ReLU | f(x) = max(0, x) | — |
| 4 | LeakyReLU | f(x) = x > 0 ? x : ax | alpha |
| 5 | ELU | f(x) = x > 0 ? x : a(exp(x)-1) | alpha |
| 6 | Softsign | f(x) = x/(1+\|x\|) | — |
| 7 | Softplus | f(x) = log(1+exp(x)) | — |
| 8 | Sin | f(x) = sin(x) | — |
| 9 | Gaussian | f(x) = exp(-x^2) | — |
| 10 | GELU | f(x) = x * Phi(x) | — |

Per-row bitmasks (`AllowedActivationsPerRow`) control which activations are permitted, checked via `IsActivationAllowed(row, type)` before any mutation.

### Connectivity Validation

`ConnectivityValidator` ensures network integrity after topology mutations:

- **ComputeReachableForward()**: BFS from inputs — which nodes are reachable?
- **ComputeReachableBackward()**: Reverse BFS from outputs — which nodes feed into outputs?
- **ComputeActiveNodes()**: Intersection of forward and backward reachable sets — nodes on a live input→output path
- **CanDeleteEdge(spec, src, dst)**: Returns true only if deletion preserves at least one input→output path

Used by EdgeDelete and EdgeSplitSmart to maintain valid topology at all times.

## Evolution System

### Evolution Loop (Evolver.StepGeneration)

Each generation:

1. **Update statistics** — Compute median fitness, variance, stagnation counters per species
2. **Cull stagnant species** — Remove underperforming species, replace with diversified offspring
3. **Per-species evolution:**
   - Preserve elite individuals (top N by fitness, cloned unchanged)
   - Generate offspring via tournament selection
   - Apply parameter mutations (weights, biases, activations, node params)
   - Apply topology mutations (edge add/delete/split/redirect/swap)
   - Combine elites + mutated offspring
4. **Increment generation** and species ages

### Selection and Elitism

- **Tournament Selection**: Pick K random individuals, return the fittest. Selection pressure scales with tournament size.
- **Rank-Based Probabilities**: `ComputeRankProbabilities` provides linear rank scaling as an alternative.
- **Parent Pool Filtering**: Only the top X% of the population is eligible for selection (default ~59.3%).
- **Elitism**: Top N individuals preserved unchanged each generation (default 5). `VerifyElitesPreserved` validates exact clones (for testing).

### Mutation Operators

#### Parameter Mutations (per-individual)

| Mutation | Description | Typical Rate |
|----------|-------------|-------------|
| WeightJitter | Gaussian noise `ε ~ N(0, σ|w|)` proportional to weight magnitude | ~81% |
| WeightReset | Replace random weight with `U(-1,1)` | ~21% |
| WeightL1Shrink | Multiply all weights by `(1 - factor)`, pushing toward zero | ~29% |
| BiasJitter | Gaussian noise on biases | varies |
| BiasReset | Replace random bias with `U(-1,1)` | varies |
| BiasL1Shrink | Same L1 shrink for biases | varies |
| ActivationSwap | Change a node's activation to another allowed type | ~15% |
| NodeParamMutate | Jitter activation-specific parameters (e.g., alpha for LeakyReLU) | ~7% |

New weights use **Glorot uniform** initialization: `U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))`.

#### Topology Mutations (per-species, affects all individuals)

| Mutation | Description | Rate |
|----------|-------------|------|
| EdgeAdd | Add random feedforward edge (respects in-degree cap and acyclic constraint) | ~0.7% |
| EdgeDeleteRandom | Remove edge only if `ConnectivityValidator.CanDeleteEdge` returns true | ~4.2% |
| EdgeSplit | Replace edge A→B with A→C→B through intermediate node | ~0.1% |
| EdgeRedirect | Change source or destination of existing edge | ~13% |
| EdgeSwap | Swap destinations of two edges (acyclic constraint enforced) | ~4.7% |

**Smart Edge Split** (`TryEdgeSplitSmart`): Preferentially activates currently-inactive nodes with low stabilization weights, enabling incremental topology growth without disrupting learned behavior. Adds stabilization edges from source to preserve gradient flow.

**Weak Edge Pruning**: Identifies edges with mean absolute weight below threshold across the entire species population, then probabilistically removes them. Configurable: enabled/disabled separately during evolution vs at species birth.

#### Complexity-Based Adaptive Mutation Rates

`ComplexityBasedMutationRates` automatically adjusts topology mutation probabilities:

- **Targets**: ~20 active hidden nodes, ~50 active edges (configurable)
- **Over target**: increase deletion rate, decrease addition rate
- **Under target**: disable deletion, boost addition to minimum threshold

This creates a self-regulating complexity homeostasis — networks neither bloat unboundedly nor collapse to trivial structures.

### Species Management

#### Species Culling

`SpeciesCuller` applies adaptive culling after a grace period (`GraceGenerations`). A species is culled if **ANY** condition is true:

1. **Stagnant**: No fitness improvement for N generations (`StagnationThreshold`)
2. **Low relative performance**: Median fitness < X% of the best species (`RelativePerformanceThreshold`)
3. **Low diversity**: Fitness variance below threshold (`SpeciesDiversityThreshold`)

The species containing the global best individual is **never** culled.

#### Species Diversification

When a species is culled, `SpeciesDiversification` replaces it:

1. Select top-2 species by median fitness as parents
2. Clone parent topology
3. Apply structural mutations:
   - ±1–2 nodes per hidden row (range 2–16)
   - Toggle 1–3 activation function permissions
   - Adjust MaxInDegree by ±1 (range 4–12)
4. Skip weak edge pruning (topology changed, weight arrays mismatched)
5. Inherit population from parent via `AdaptIndividualToTopology`:
   - Preserve weights for matching edges
   - Initialize new edges with Glorot distribution
   - Copy/initialize activations and biases
6. New species starts at Age=0 (grace period protection)

#### Stagnation Tracking

`StagnationTracker` maintains per-species statistics via `SpeciesStats`:

```csharp
float BestFitnessEver;
int GenerationsSinceImprovement;
float[] FitnessHistory;           // Rolling 10-generation window of medians
float MedianFitness;
float FitnessVariance;
```

Methods: `IsStagnant`, `HasLowDiversity`, `IsPastGracePeriod`, `GetFitnessHistory`.

## Fitness Evaluation

### IEnvironment Interface

```csharp
interface IEnvironment {
    int InputCount { get; }
    int OutputCount { get; }
    int MaxSteps { get; }

    void Reset(int seed);
    void GetObservations(Span<float> observations);
    float Step(ReadOnlySpan<float> actions);
    bool IsTerminal();
    float GetFinalFitness() => 0f;  // Optional override
}
```

### SimpleFitnessEvaluator (CPU)

For each individual:
1. Create `CPUEvaluator` from species topology
2. Reset environment with seed
3. Loop: observe → forward pass → check for NaN (penalty: -1000) → step environment → accumulate reward
4. Return cumulative reward (or environment's `GetFinalFitness()`)

### Available Environments

| Environment | Inputs | Outputs | Description |
|------------|--------|---------|-------------|
| XOR | 2 | 1 | Classic XOR classification (4 test cases) |
| Spiral | 2 | 1 | Two interlocking noisy spirals, classify +1/-1 |
| CartPole | 4 | 1 | Balance pole on cart (classic control) |
| Landscape | N | N | Navigate N-dimensional optimization surfaces |
| Rocket | 8 | 2 | Land a rigid-body rocket using Rigidon physics (throttle + gimbal) |
| TargetChase | varies | varies | Chase moving targets |
| Corridor | varies | varies | Navigate through corridors |

#### Landscape Environment

Continuous optimization over multiple landscape functions:

- **Sphere** (easy) — Convex, single global minimum
- **Rosenbrock** (medium) — Narrow curved valley
- **Rastrigin** (hard) — Multimodal, many local minima
- **Ackley** (hard) — Nearly flat with deep global minimum
- **Schwefel** (very hard) — Deceptive, global minimum far from local minima

Observations = current position (ND), actions = position delta, fitness = -landscape_value at final position.

#### Rocket Environment

8-dimensional observation: `[relPosX, relPosY, velX, velY, upX, upY, gimbal, throttle]`
2-dimensional action: `[throttle ∈ [0,1], gimbal ∈ [-1,1]]`

Terminal conditions: successful landing, crash (high impact), out-of-bounds, or max steps. Integrated with `RigidBodyRocketTemplate` from the Rigidon physics engine.

## GPU Backend

### Architecture

The GPU backend provides massively parallel evaluation using ILGPU:

```
GPUEvaluator          — Orchestrates kernel launches, manages device memory
GPUEvolvionState      — Device memory allocation (edges, weights, node values, environments)
GPUEvolvionKernels    — ILGPU kernel implementations
GPUDataStructures     — Blittable GPU-compatible structs (SoA layout)
GPUFitnessEvaluator   — High-level batch fitness evaluation (drop-in for SimpleFitnessEvaluator)
GPUBatchedFitnessEvaluator — Multi-episode batched evaluation
```

### Memory Layout (Structure-of-Arrays)

For N individuals with M weights each:

```
Weights:     [ind0_w0, ind0_w1, ..., ind0_wM, ind1_w0, ..., indN_wM]
Biases:      [ind0_b0, ind0_b1, ..., ind0_bK, ind1_b0, ...]
Activations: [ind0_a0, ind0_a1, ..., ind0_aK, ind1_a0, ...]
NodeValues:  [ind0_n0, ind0_n1, ..., ind0_nK, ind1_n0, ...]
```

SoA layout enables coalesced memory access when all threads in a warp evaluate the same row simultaneously.

### Kernel Variants

**Core evaluation:**
- `EvaluateRowKernel` — One thread per individual, processes one row at a time
- `EvaluateRowForEpisodesKernel` — One thread per episode, maps episode→individual for multi-episode evaluation
- `SetInputsKernel`, `GetOutputsKernel` — I/O handling

**Environment-specific kernels** (GPU-native episode loops):
- Landscape: `InitializeEpisodesKernel`, `ComputeObservationsKernel`, `StepEnvironmentKernel`, `FinalizeFitnessKernel`
- XOR: `InitializeXORKernel`, `GetXORObservationsKernel`, `StepXORKernel`, `FinalizeXORFitnessKernel`
- Spiral: `InitializeSpiralPointsKernel`, `GetSpiralObservationsKernel`, `StepSpiralKernel`, `FinalizeSpiralFitnessKernel`

### GPU-Specific Notes

- **Tanh workaround**: RTX 4090 PTX compiler issue requires exp-based approximation instead of native tanh
- **LCG random**: Lightweight linear congruential generator for GPU episode initialization
- **No topology mutations on GPU**: Only evaluation runs on GPU; evolution logic stays on CPU
- **Speedup**: ~1.93x over CPU on RTX 4090

### Usage

```csharp
// CPU evaluation
var evaluator = new SimpleFitnessEvaluator();
evaluator.EvaluatePopulation(population, environment, seed: gen);

// GPU evaluation (drop-in replacement)
using var gpuEval = new GPUFitnessEvaluator(
    maxIndividuals: 1000, maxNodes: 100, maxEdges: 500);
gpuEval.EvaluatePopulation(population, environment, episodesPerIndividual: 5, seed: gen);
```

## Configuration (EvolutionConfig)

Current defaults are Optuna-optimized (Phase 10, Rosenbrock):

```
Population:     39 species × 132 individuals = ~5,148 total
                MinSpeciesCount = 13

Selection:      TournamentSize = 10
                ParentPoolPercentage = 0.593
                Elites = 5

Speciation:     GraceGenerations = 1
                StagnationThreshold = 6
                SpeciesDiversityThreshold = 0.113
                RelativePerformanceThreshold = 0.627

Weight mutations:
                WeightJitter = 0.812 (sigma = 0.058)
                WeightReset = 0.212
                WeightL1Shrink = 0.288 (factor = 0.857)
                ActivationSwap = 0.150
                NodeParamMutate = 0.072

Edge mutations:
                EdgeAdd = 0.007
                EdgeDeleteRandom = 0.042
                EdgeSplit = 0.001
                EdgeRedirect = 0.132
                EdgeSwap = 0.047
```

## Visualization

`NeuralNetworkVisualizer` renders networks as SVG:

- **Input nodes**: blue gradient
- **Output nodes**: orange gradient
- **Active hidden nodes**: green gradient
- **Inactive hidden nodes**: gray
- **Edges**: thickness and color scaled by weight magnitude; directional arrows indicate sign (positive/negative)
- **RenderMutationProgression**: Grid view showing topology evolution across generations
- Includes network statistics (active nodes, edge count)

## Project Structure

```
Evolvatron.Evolvion/
├── SpeciesSpec.cs                  # Immutable topology definition (RowCounts, Edges, RowPlans)
├── SpeciesBuilder.cs               # Fluent API for topology construction
├── Individual.cs                   # Per-individual mutable parameters
├── Species.cs                      # Species container (topology + individuals + stats)
├── Population.cs                   # Multi-species population management
├── Evolver.cs                      # Main evolution loop orchestrator
├── EvolutionConfig.cs              # Optuna-optimized hyperparameters
├── RowPlan.cs                      # Compiled row evaluation metadata
├── ActivationType.cs               # 11 activation functions with learnable params
├── IEnvironment.cs                 # Environment interface
│
├── MutationOperators.cs            # Weight/bias/activation parameter mutations
├── EdgeTopologyMutations.cs        # Edge add/delete/split/redirect/swap
├── EdgeMutationConfig.cs           # Edge mutation configuration
├── ComplexityBasedMutationRates.cs  # Adaptive mutation rate adjustment
├── ConnectivityValidator.cs        # BFS-based topology integrity validation
│
├── Selection.cs                    # Tournament selection, rank probabilities
├── Elitism.cs                      # Elite preservation
├── SpeciesCuller.cs                # Adaptive species culling
├── SpeciesDiversification.cs       # New species creation from culled slots
├── StagnationTracker.cs            # Per-species performance monitoring
├── SpeciesStats.cs                 # Fitness statistics struct
│
├── CPUEvaluator.cs                 # CPU neural network forward pass
├── SimpleFitnessEvaluator.cs       # CPU fitness evaluation loop
│
├── GPU/
│   ├── GPUEvaluator.cs             # GPU kernel orchestration
│   ├── GPUFitnessEvaluator.cs      # GPU fitness evaluation (1.93x faster)
│   ├── GPUBatchedFitnessEvaluator.cs # Multi-episode batch evaluation
│   ├── GPUEvolvionKernels.cs       # ILGPU compute kernels
│   ├── GPUEvolvionState.cs         # Device memory management
│   └── GPUDataStructures.cs        # Blittable GPU-compatible structs (SoA)
│
├── Environments/
│   ├── XOREnvironment.cs           # Classic XOR
│   ├── SpiralEnvironment.cs        # 2D spiral classification
│   ├── CartPoleEnvironment.cs      # Pole balancing
│   ├── LandscapeEnvironment.cs     # N-dim optimization landscapes
│   ├── RocketEnvironment.cs        # Rigidon physics rocket landing
│   ├── TargetChaseEnvironment.cs   # Moving target pursuit
│   ├── SimpleCorridorEnvironment.cs
│   ├── FollowCorridorEnvironment.cs
│   └── FollowTheCorridorEnvironment.cs
│
├── Benchmarks/
│   ├── OptimizationLandscapes.cs   # Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel
│   ├── LandscapeNavigationTask.cs  # Benchmark runner
│   └── LandscapeEnvironmentAdapter.cs
│
├── Visualization/
│   └── NeuralNetworkVisualizer.cs  # SVG network rendering
│
├── Utilities/
│   └── SvgPathDecoder.cs           # SVG path parsing
│
└── Refactor/                       # Design sketches (not active code)
    ├── PopulationDesign.cs          # Alternative GenomeDef/SpeciesDef hierarchy
    ├── ExecutableFormat.cs          # Alternative evaluation format
    └── Again.cs                     # Architecture exploration
```

## Benchmark Results

### Spiral Classification (2→8→8→1)
- **Solve time**: 3 generations
- **Evaluations**: 7,128 (3 gens × 2,376 individuals)
- **Success rate**: 100% (all 15 seeds tested)
- **Deterministic**: Verified with multiple runs

## Design Philosophy

1. **GPU batching drives architecture**: The shared-topology-per-species constraint exists to enable massively parallel evaluation. Every other design choice follows from this.
2. **Topology-parameter separation**: Immutable topology per species enables batched kernel launches; mutable parameters (weights, biases, activations) enable individual variation within the batch.
3. **Speciation as architecture search**: Multiple coexisting topologies explore the architecture space in parallel, while individuals within each species exploit parameter space. This is fundamentally different from NEAT — see [NEAT comparison](../docs/NEAT_COMPARISON.md).
4. **Adaptive complexity**: Self-regulating mutation rates prevent both bloat and premature simplification.
5. **Graceful diversification**: New species inherit learned weights from top performers via rank-weighted selection, avoiding catastrophic knowledge loss when topology changes.
6. **Environment-agnostic**: The same evolution engine drives XOR classification, continuous optimization, and physics-based rocket landing.

## Testing

```bash
# Run all evolution tests
dotnet test --filter "FullyQualifiedName~Evolvion"

# Verify determinism
dotnet test --filter "FullyQualifiedName~DeterminismVerificationTest"

# Long-run convergence
dotnet test --filter "FullyQualifiedName~LongRunConvergenceTest"
```

## Hyperparameter Optimization

See `OPTUNA_RESULTS.md` for complete optimization results.

- `optuna_sweep.py` — Bayesian optimization orchestrator
- `Evolvatron.OptunaEval/` — C# evaluation CLI
- `optuna_best_params.txt` — Best trial parameters
