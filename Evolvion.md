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

### 4.2 Connectivity Integrity

After mutation:

* Recompute `reachableFromInput[]` and `reachesOutput[]`.
* Mark edges inactive if neither condition holds.
* Optionally prune if inactive > T generations.

---

## **5. Evolutionary Lifecycle**

### 5.1 Population Structure

| Parameter                   | Default | Notes                                       |
| --------------------------- | ------- | ------------------------------------------- |
| Species Count (N)           | 8       | Parallel species with differing topologies. |
| Individuals per Species (M) | 128     | Parallel GPU batch size target.             |
| Grace Generations           | 3       | Newborns protected from culling.            |
| Tournament Size             | 4       | Selection pressure.                         |
| Elites per Species          | 4       | Top performers copied unchanged.            |

### 5.2 Generation Loop

1. **Evaluation (GPU)**
   Each individual is tested over `K=5` randomized rollouts. Fitness = average or CVaR@50%.
2. **Species Scoring**
   Median species fitness computed.
3. **Culling**
   Lowest 25% species removed, replaced by mutated children of best species.
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

| Operator              | Description                                   | Probability |        |     |
| --------------------- | --------------------------------------------- | ----------- | ------ | --- |
| **Weight Jitter**     | Gaussian noise σ = 0.05 × w                   | 0.9         |        |     |
| **Weight Reset**      | Replace with random                           | 0.05        |        |     |
| **Weight L1 Shrink**  | Reduce                                        | w           | by 10% | 0.1 |
| **Activation Swap**   | Replace with random allowed activation        | 0.01        |        |     |
| **Edge Add**          | Add random upward edge if below in-degree cap | 0.05        |        |     |
| **Edge Delete**       | Remove random edge                            | 0.02        |        |     |
| **Edge Split**        | Insert intermediate node with two edges       | 0.01        |        |     |
| **Node Param Mutate** | Gaussian jitter on α/β params                 | 0.2         |        |     |

All probabilities are per-individual and tunable per species.

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
    int nodeStart;
    int nodeCount;
    int edgeStart;
    int edgeCount;
}
```

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

Fitness normalized relative to previous-generation median;
selection uses **rank-based probabilities** to avoid scale issues.

---

## **9. Speciation and Replacement**

* Each species retains its topology forever.
* When a species is eliminated:

  1. Select top-2 source species.
  2. Clone one topology (with optional slight mutation of `RowCounts`).
  3. Initialize new individuals with random weights.

Optional future: **distance metric** (Hamming + L2 on aligned edges) for cross-species crossover.

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
* **Stat logs:**

  * Species mean, median, max fitness
  * Active edge ratio
  * Edge count distribution
  * Avg in-degree / out-degree
  * Mutation rate stats
* **Replay dump:** Store elite network + rollout trace for visualization.

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
  "IndividualsPerSpecies": 128,
  "Rows": [1, 8, 8, 3],
  "MaxInDegree": 6,
  "Elites": 4,
  "TournamentSize": 4,
  "GraceGenerations": 3,
  "MutationRates": {
    "WeightJitter": 0.9,
    "ActivationSwap": 0.01,
    "EdgeAdd": 0.05,
    "EdgeDelete": 0.02,
    "EdgeSplit": 0.01
  },
  "SeedsPerIndividual": 5
}
```

---

## **14. Future Extensions**

* **CMA-ES local search** for elite fine-tuning.
* **Dynamic topology mutation** between species (NEAT-like macro-evolution).
* **FP16 mixed precision** for larger batch throughput.
* **Distributed evaluation** across multi-GPU setups.
* **Web dashboard** with species lineage visualization.

---

## **15. Deliverables**

1. **Core library** (`Evolvatron.Evolvion.dll`)
2. **Benchmark suite** with scripts & plots.
3. **Documentation** (API + design).
4. **Visualization tool** for topologies & rollouts.
5. **Integration demo** with Evolvatron Rocket landing.

---

Would you like me to extend this spec with **a project roadmap + task breakdown (sprint-wise)** next — e.g. Milestone 1: core structs + CPU evaluator, Milestone 2: ILGPU kernel, etc.?
