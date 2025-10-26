# **Evolvion: Evolutionary Neural Controller Framework**

## **1. Overview**

Evolvion is an **evolutionary machine learning framework** designed for large-scale parallel execution using **ILGPU**.
It evolves **species of fixed-topology neural controllers**, where each individual differs only by its **weights and node parameters**.
These controllers will be tested on benchmark and physics environments, including the ILGPU-based rocket simulator (Reward Engineer).

---

## **2. Core Principles**

1. **Species = identical topology**, differing constants.
2. **Individuals = unique weight & node parameter sets.**
3. **Networks = layered directed acyclic grids (rows)** — inputs → hidden rows → outputs.
4. **Connections** may only originate from earlier rows (acyclic by construction).
5. **Synchronous row updates** — each row evaluated only after all rows above it have committed.
6. **Bias row (Row 0)** supplies constant 1.0 inputs to all nodes, representing per-node bias.
7. **Execution** occurs entirely on GPU using precompiled kernels that execute 1000s of individuals in parallel.
8. **Evolutionary loop** runs on CPU, managing mutation, selection, and speciation.

---

## **3. Network Structure**

### 3.1 Rows and Nodes

* **Row 0:** Bias (constant 1).
* **Row 1:** Input layer (environment signals).
* **Rows 2 → R-1:** Hidden computation layers.
* **Row R:** Output layer (action/control outputs).

### 3.2 Node Attributes

Each node has:

* `activation`: enum (see §3.5)
* `param[4]`: optional per-node constants (e.g., α for leaky ReLU)
* `incomingEdges[]`: list of weighted connections `(sourceNode, weightIndex)`

### 3.3 Connection Rules

* Allowed only from **row A < row B**.
* Duplicate edges forbidden.
* Each node capped by `MaxInDegree` (species-level).
* Nodes or edges without paths from input→output are marked **inactive** but retained.

### 3.4 Example Layout

```
Row 0 (Bias): 1 node (constant)
Row 1 (Inputs): 6 nodes
Row 2: 8 hidden
Row 3: 6 hidden
Row 4 (Outputs): 3 nodes
```

### 3.5 Activation Set

Default enabled:

```
Linear, Tanh, Sigmoid, ReLU, LeakyReLU, ELU,
Softsign, Softplus, Sin, Gaussian, GELU
```

* **Outputs:** Linear or Tanh only (bounded control).
* Each activation defines required param count (e.g., α for LeakyReLU).

---

## **4. Species Definition**

A **Species** is defined by its grid topology and allowed operators.

| Field                        | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| `RowCounts[]`                | Number of nodes per row.                        |
| `AllowedActivationsPerRow[]` | Bitmask of valid activations.                   |
| `MaxInDegree`                | Maximum incoming edges per node.                |
| `EdgeSchema`                 | Static list of potential edges (row→row pairs). |
| `RowPlans`                   | Compiled per-row evaluation plan (see §7).      |

### 4.1 Individual

Each **Individual** stores:

```csharp
struct Individual {
    float[] Weights;          // per-edge weights
    float[] NodeParams;       // per-node activation params
    ActivationType[] Acts;    // per-node activation
    Bitset ActiveNodes;       // for debug/analytics
    float Fitness;
    int Age;
}
```

### 4.3 Species Statistics

Each **Species** tracks stagnation metrics:

```csharp
struct SpeciesStats {
    float BestFitnessEver;         // historical peak fitness
    int GenerationsSinceImprovement; // stagnation counter
    float[] FitnessHistory;        // rolling window (last 10 gens)
    float MedianFitness;           // current generation median
}
```

### 4.2 Connectivity Integrity

After mutation:

* Recompute `reachableFromInput[]` and `reachesOutput[]`.
* Mark edges inactive if neither condition holds.
* Optionally prune if inactive > T generations.

---

## **5. Evolutionary Lifecycle**

### 5.1 Population Structure

| Parameter                       | Default | Notes                                              |
| ------------------------------- | ------- | -------------------------------------------------- |
| Species Count (N)               | 8       | Parallel species with differing topologies.        |
| Individuals per Species (M)     | 128     | Parallel GPU batch size target.                    |
| Grace Generations               | 3       | Newborns protected from culling.                   |
| Stagnation Threshold            | 15      | Generations without improvement before eligible.   |
| Species Diversity Threshold     | 0.15    | Min fitness variance to avoid premature culling.   |
| Tournament Size                 | 4       | Selection pressure.                                |
| Elites per Species              | 4       | Top performers copied unchanged.                   |

### 5.2 Generation Loop

1. **Evaluation (GPU)**
   Each individual is tested over `K=5` randomized rollouts. Fitness = average or CVaR@50%.

2. **Species Scoring & Stagnation Tracking**
   * Compute median fitness per species
   * Update `BestFitnessEver` and `GenerationsSinceImprovement` counters
   * Track fitness variance within each species

3. **Adaptive Species Culling**
   Species are culled **only if** they meet **all** of the following criteria:
   * **Age**: Past grace period (Age > GraceGenerations)
   * **Stagnation**: No improvement for StagnationThreshold generations (e.g., 15)
   * **Relative Performance**: Median fitness < 50% of best species median
   * **Low Diversity**: Fitness variance < SpeciesDiversityThreshold

   **Culling Strategy**:
   * Identify eligible species using above criteria
   * If 2+ species eligible, cull the worst-performing one
   * Replace with diversified offspring from top-2 performing species
   * Never cull below minimum species count (e.g., 4)

4. **Within-Species Selection**

   * Rank individuals by fitness.
   * Keep top E elites unchanged.
   * Remainder created by **tournament selection** among existing individuals.

5. **Mutation / Crossover**

   * Mutate offspring (see §6).
   * Optionally apply within-species crossover.

6. **Upload New Weights**

   * Push new weight arrays to GPU buffers for next generation.

---

## **6. Mutation Operators**

| Operator              | Description                                   | Probability | Notes                              |
| --------------------- | --------------------------------------------- | ----------- | ---------------------------------- |
| **Weight Jitter**     | Gaussian noise σ = 0.05 × w                   | 0.9         | Primary exploration mechanism      |
| **Weight Reset**      | Replace with random from U(-1, 1)             | 0.05        | Escape local minima                |
| **Weight L1 Shrink**  | Reduce \|w\| by 10%                           | 0.1         | Regularization pressure            |
| **Activation Swap**   | Replace with random allowed activation        | 0.01        | Functional diversity               |
| **Edge Add**          | Add random upward edge if below in-degree cap | 0.05        | Topology complexification          |
| **Edge Delete**       | Remove random edge                            | 0.02        | Topology simplification            |
| **Edge Split**        | Insert intermediate node with two edges       | 0.01        | Depth increase (adds hidden layer) |
| **Node Param Mutate** | Gaussian jitter on α/β params                 | 0.2         | Activation function tuning         |

All probabilities are per-individual and tunable per species.

### 6.1 Weight Initialization

New weights (from Edge Add, Edge Split, species creation) are initialized using **Glorot/Xavier uniform**:

```
w ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
```

Where `fan_in` = incoming edges to target node, `fan_out` = outgoing edges from source node.

---

## **7. ILGPU Execution Architecture**

### 7.1 Kernel Design

* One **thread-group per (individual × rollout)**.
* Each group executes all rows synchronously:

  ```csharp
  for (row = 1; row < R; row++) {
      EvaluateRow(row);
      Group.Barrier();
  }
  ```
* `EvaluateRow` performs segmented reductions using `RowPlan` offsets.

### 7.2 Data Layout (Structure-of-Arrays)

| Buffer                    | Scope      | Description               |
| ------------------------- | ---------- | ------------------------- |
| `edgeSrc[]` / `edgeDst[]` | Species    | Connection topology.      |
| `edgeOffsetPerRow[]`      | Species    | Edge segment prefix sums. |
| `nodeAct[]`               | Species    | Activation enums.         |
| `nodeParams[][]`          | Individual | Node parameter sets.      |
| `weights[][]`             | Individual | Per-edge weights.         |
| `rowValues[][]`           | Runtime    | Node activations.         |

### 7.3 Semi-Compiled RowPlans

CPU-side preprocessing builds compact metadata:

```csharp
struct RowPlan {
    int nodeStart;      // global index of first node in this row
    int nodeCount;      // number of nodes in this row
    int edgeStart;      // global index of first edge targeting this row
    int edgeCount;      // number of edges targeting this row
}
```

**RowPlan Construction Algorithm:**

1. Iterate through all edges in species topology
2. Sort edges by `(destinationRow, destinationNode)` for coalesced access
3. Build prefix sums:
   * `edgeOffsetPerRow[r]` = index of first edge where destination is in row `r`
   * `nodeOffsetPerRow[r]` = cumulative node count up to row `r`
4. Store sorted `edgeSrc[]` and `edgeDst[]` arrays on device
5. Upload RowPlans once per species; remain constant until topology changes

These are uploaded once per species. Kernel uses them for deterministic row iteration.

### 7.4 Determinism

All RNG (mutation, rollouts) uses a **counter-based Philox RNG** keyed by `(speciesId, individualId, generation, rollout)` ensuring reproducibility.

### 7.5 Debugging

* CPU evaluator using same `RowPlan` for parity tests.
* Periodic validation: GPU vs CPU outputs on fixed inputs.
* Optional probe sampling of row activations for visualization.

---

## **8. Fitness Evaluation**

### 8.1 Multi-Seed Protocol

Each individual runs **K = 5** episodes with different RNG seeds.
Fitness = average or **CVaR@β=0.5** (average of worst 50%) to reward robustness.

### 8.2 Early Termination

Simulations abort immediately on:

* NaN in network output.
* Divergent physics (explosion, invalid state).
* Timeout or crash detection.
  Penalty fitness applied.

### 8.3 Shaping & Normalization

* **Fitness normalization**: Relative to species-specific rolling median (last 3 generations)
* **Selection**: Uses **rank-based probabilities** to avoid scale issues
* **Cross-species comparison**: Normalized by global median for culling decisions

---

## **9. Speciation and Replacement**

* Each species retains its topology until culled for stagnation.
* When a species is eliminated:

  1. Select top-2 performing species as parents.
  2. Clone topology from one parent with **diversification mutations**:
     * ±1-2 nodes per hidden row (respecting min/max constraints)
     * Randomly toggle 1-3 allowed activations in bitmask
     * Adjust MaxInDegree by ±1 (range: 4-12)
  3. Initialize new individuals with Glorot/Xavier weights.
  4. Assign Grace Generation protection.

### 9.1 Topology Distance Metric (for future cross-species crossover)

To enable cross-species mating, define structural similarity:

```
d(S1, S2) = w_topo * TopoDistance + w_param * ParamDistance

where:
  TopoDistance = HammingDistance(activeEdges1, activeEdges2) / max(|E1|, |E2|)
  ParamDistance = MeanL2(aligned_weights) over structurally identical edges
```

Species pairs with `d < threshold` are eligible for crossover (not implemented in v1).

---

## **10. Benchmarks and Demos**

1. **Regression/XOR** – validate math correctness.
2. **Spiral classification** – validate non-linearity.
3. **CartPole Continuous** – control baseline.
4. **MountainCar Continuous** – delayed rewards.
5. **Rocket Landing (ILGPU Physics)** – flagship test.
   Each benchmark provides:

* Deterministic rollout harness.
* Standardized input/output scaling.
* Replay recorder for elite individuals.

---

## **11. Debugging and Telemetry**

* **Parity mode:** Every N generations, one individual evaluated on CPU for comparison.
* **Activation probe:** Optional sampling of row activations (M individuals).
* **Stat logs (per generation):**

  * Species mean, median, max fitness
  * `GenerationsSinceImprovement` per species
  * Fitness variance (diversity metric)
  * Active edge ratio
  * Edge count distribution
  * Avg in-degree / out-degree
  * Mutation rate stats
  * Species lineage tree (parent/child relationships)
* **Replay dump:** Store elite network + rollout trace for visualization.

### 11.1 Convergence Detection

Monitor for end-of-run conditions:

* **Fitness plateau**: Best fitness unchanged for 50+ generations
* **Species collapse**: All species within 5% fitness of each other
* **Diversity loss**: Mean fitness variance < 0.01 across all species

Trigger alerts or auto-tuning (increase mutation rates, inject random species).

---

## **12. Code Architecture (C#)**

```
Evolvatron.Evolvion/
│
├─ Species/
│  ├─ SpeciesSpec.cs          // Topology & configuration
│  ├─ SpeciesPlan.cs          // Device row/edge layout
│  └─ SpeciesManager.cs       // Creation, culling, replacement
│
├─ Individuals/
│  ├─ Individual.cs           // Weights, acts, params, fitness
│  ├─ Mutation.cs             // All mutation operators
│  └─ Crossover.cs            // Optional crossover
│
├─ Evaluation/
│  ├─ EvaluatorGPU.cs         // ILGPU kernel launcher
│  ├─ EvaluatorCPU.cs         // Debug parity evaluator
│  └─ FitnessAggregator.cs    // Multi-seed & CVaR logic
│
├─ Evolution/
│  ├─ Population.cs           // N species × M individuals
│  ├─ Evolver.cs              // Tournament, selection, elitism
│  └─ RNG.cs                  // Counter-based PRNG
│
├─ Environments/
│  ├─ RegressionEnv.cs
│  ├─ CartPoleEnv.cs
│  ├─ RocketEnv.cs
│  └─ ...
│
└─ Utils/
   ├─ Logger.cs
   ├─ ProbeSampler.cs
   └─ MathExt.cs
```

### Key Interfaces

```csharp
interface IEnvironment {
    int InputCount { get; }
    int OutputCount { get; }
    float Step(ReadOnlySpan<float> outputs);
    void Reset(int seed);
}

interface IEvaluator {
    void Evaluate(Population pop, IEnvironment env, int seedsPerInd);
}

interface IEvolver {
    Population Step(Population pop);
}
```

---

## **13. Configuration Defaults**

```json
{
  "SpeciesCount": 8,
  "MinSpeciesCount": 4,
  "IndividualsPerSpecies": 128,
  "Rows": [1, 8, 8, 3],
  "MaxInDegree": 6,
  "Elites": 4,
  "TournamentSize": 4,
  "GraceGenerations": 3,
  "StagnationThreshold": 15,
  "SpeciesDiversityThreshold": 0.15,
  "WeightInitialization": "GlorotUniform",
  "MutationRates": {
    "WeightJitter": 0.9,
    "WeightReset": 0.05,
    "WeightL1Shrink": 0.1,
    "ActivationSwap": 0.01,
    "EdgeAdd": 0.05,
    "EdgeDelete": 0.02,
    "EdgeSplit": 0.01,
    "NodeParamMutate": 0.2
  },
  "SeedsPerIndividual": 5,
  "FitnessAggregation": "CVaR50"
}
```

---

## **14. Future Extensions**

* **CMA-ES local search** for elite fine-tuning.
* **Cross-species crossover** using topology distance metric (§9.1).
* **Adaptive mutation rates** based on species stagnation.
* **Novelty search** and behavioral diversity metrics.
* **FP16 mixed precision** for larger batch throughput.
* **Distributed evaluation** across multi-GPU setups.
* **Web dashboard** with species lineage visualization and interactive rollout playback.
* **Hierarchical evaluation** (species-level screening before full rollouts).

---

## **15. Deliverables**

1. **Core library** (`Evolvatron.Evolvion.dll`)
2. **Benchmark suite** with scripts & plots.
3. **Documentation** (API + design).
4. **Visualization tool** for topologies & rollouts.
5. **Integration demo** with Evolvatron Rocket landing.

---

## **16. Implementation Priorities and Critical Details**

### 16.1 Key Design Decisions

1. **Stagnation-Based Culling**: Prevents premature elimination of promising species that are still improving, even if temporarily underperforming.

2. **Diversity Preservation**: Multi-metric approach (fitness variance, relative performance, stagnation time) ensures genetic diversity is maintained.

3. **Glorot Initialization**: Critical for gradient-free methods; ensures new topologies start in viable fitness landscapes.

4. **RowPlan Sorting**: Edge sorting by destination enables coalesced memory access patterns on GPU, dramatically improving throughput.

5. **Deterministic RNG**: Counter-based approach allows reproducible experiments and parallel evaluation without synchronization.

### 16.2 Memory Layout Specifics

**Individual-specific buffers** organized as flat arrays:

```csharp
// For species with N_nodes nodes and E_edges edges, M individuals:
float[] allWeights = new float[M * E_edges];
float[] allNodeParams = new float[M * N_nodes * 4]; // 4 params per node

// Individual i accesses:
//   weights: allWeights[i * E_edges .. (i+1) * E_edges]
//   params:  allNodeParams[i * N_nodes * 4 .. (i+1) * N_nodes * 4]
```

**Edge Schema**: Stores *all active edges* (not complete graph). Sparse topologies represented efficiently. Inactive edges excluded from device buffers but tracked in Individual metadata for mutation.

### 16.3 GPU Execution Details

**Thread-group sizing**: One thread per node in largest row. Example:
* Species with rows [1, 6, 8, 6, 3] → group size = 8 threads
* Thread divergence when evaluating smaller rows (idle threads)
* Alternative: dynamic thread allocation (more complex, potentially faster)

**Synchronization**: `Group.Barrier()` ensures all nodes in row R are computed before any node in row R+1 begins.

### 16.4 Crossover Alignment Strategy (if implemented)

When crossing individuals within same species:
1. Randomly split edges into 3 sets: {parent1, parent2, average}
2. Copy topologically identical edges
3. For differing activations, randomly select parent's activation + params

---

## **17. Recommended Implementation Roadmap**

### Milestone 1: Core Data Structures ✅ COMPLETE
- [x] `Individual`, `SpeciesSpec`, `SpeciesStats` structs
- [x] `RowPlan` builder with edge sorting
- [x] Activation function enum + param requirements (11 activations)
- [x] Weight initialization (Glorot/Xavier)
- [x] Unit tests: topology validation, connectivity checks (20 tests)
- [x] **BONUS:** Support for parallel edges (up to 2)

### Milestone 2: CPU Evaluator ✅ COMPLETE
- [x] CPU-based forward pass using RowPlans
- [x] All activation functions implemented (11 types)
- [x] Complex network test (XOR-equivalent multi-layer)
- [x] Multi-layer computation validation
- [x] Parity validation harness (29 tests)
- [x] Determinism verification

### Milestone 2.5: Edge Topology Mutations ✅ COMPLETE (BONUS!)
- [x] **EdgeAdd** - add connections with MaxInDegree validation
- [x] **EdgeDelete** - remove connections preserving connectivity
- [x] **EdgeSplit** - insert intermediate nodes
- [x] **EdgeRedirect** - change connection endpoints (BONUS)
- [x] **EdgeDuplicate** - create parallel pathways (BONUS)
- [x] **EdgeMerge** - combine parallel edges (BONUS)
- [x] **EdgeSwap** - rewire connections (BONUS)
- [x] **Weak Edge Pruning** - emergent structural learning (MAJOR INNOVATION)
- [x] **ConnectivityValidator** - BFS-based graph validation
- [x] Unit tests: 26 edge mutation tests
- [x] **Result:** 7 edge operators + weak pruning, 116/117 tests passing

### Milestone 3: Evolutionary Core (Week 5-6) ⬅️ **NEXT PHASE**
- [ ] Tournament selection
- [ ] Weight-level mutation operators (§6) - ✅ DONE
- [ ] Edge topology mutations (§6) - ✅ DONE
- [ ] Stagnation tracking + adaptive culling (§5.2)
- [ ] Elitism + generation loop
- [ ] Species replacement with diversification (§9)
- [ ] Population management (multiple species)

### Milestone 4: Multi-Seed Evaluation (Week 7)
- [ ] Philox counter-based RNG
- [ ] CVaR@50% fitness aggregation
- [ ] Early termination on NaN/divergence
- [ ] Fitness normalization (§8.3)

### Milestone 5: ILGPU Kernel (Week 8-10)
- [ ] Device buffer management (`GPUWorldState` equivalent)
- [ ] Forward pass kernel with row-synchronous execution
- [ ] Kernel validation against CPU evaluator
- [ ] Benchmark: CartPole-Continuous

### Milestone 6: Integration & Benchmarks (Week 11-12)
- [ ] Rocket landing environment integration
- [ ] All 5 benchmarks implemented (§10)
- [ ] Telemetry + logging system (§11)
- [ ] Convergence detection (§11.1)

### Milestone 7: Visualization & Tooling (Week 13-14)
- [ ] Elite replay exporter
- [ ] Topology graph renderer
- [ ] Species lineage tracker
- [ ] Live dashboard (optional)

---
