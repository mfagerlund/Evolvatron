# Evolvion vs NEAT: Architectural Comparison

Evolvion draws inspiration from NEAT (NeuroEvolution of Augmenting Topologies) but is a fundamentally different system. The core divergence is a single design choice: **all individuals within a species share identical topology**, enabling GPU-batched evaluation. This constraint propagates through the entire architecture.

## The Central Trade-off

**NEAT** gives every genome a unique topology. This maximizes structural diversity but makes parallel evaluation difficult — each network has different shapes, edge counts, and evaluation paths.

**Evolvion** constrains topology to be shared per species. This sacrifices per-individual structural variation in exchange for massively parallel evaluation — an entire species can be evaluated in a single GPU kernel launch with coalesced memory access.

Neither approach is strictly better. NEAT excels at discovering minimal topologies from scratch. Evolvion excels at scaling to large populations on GPU hardware.

## Feature-by-Feature Comparison

### Speciation

| Aspect | NEAT | Evolvion |
|--------|------|----------|
| What defines a species | Compatibility distance computed from gene-level differences (excess, disjoint, weight delta) | Explicit topology bucket — a `SpeciesSpec` struct defining exact network architecture |
| Speciation timing | Genomes are re-clustered into species every generation based on distance to representative | Species are persistent entities; individuals never migrate between species |
| Within-species variation | Both topology and weights differ | Only weights, biases, activations, and node parameters differ |
| Species creation | Automatic clustering when genome distance exceeds threshold | Diversification from culled species: clone parent topology, apply structural mutations |

NEAT's compatibility distance is a continuous similarity metric. Evolvion's species are discrete topology classes — you're either in a species (matching topology exactly) or you're not.

### Topology Evolution

| Aspect | NEAT | Evolvion |
|--------|------|----------|
| Unit of topology change | Per-genome: each individual can mutate its own structure | Per-species: topology mutations affect the entire species |
| Starting point | Minimal network (inputs directly connected to outputs), complexifies over time | Configurable: dense, sparse, or any hand-designed initial topology |
| Structural mutations | Add node (split edge), add connection | Edge add/delete/split/redirect/swap, weak edge pruning, plus species-level diversification (row size changes, activation mask changes, MaxInDegree changes) |
| Complexity control | Historical markings prevent information loss during crossover; structural innovations protected by speciation | Adaptive mutation rates target complexity homeostasis (~20 active hidden nodes, ~50 active edges); weak edge pruning removes unused connections |
| When topology changes | Every generation, per individual (probabilistically) | Species diversification at culling time; edge mutations per-species during evolution |

NEAT's complexification-from-minimal is elegant but depends on innovation numbers to align genes during crossover. Evolvion doesn't need gene alignment because topology is shared — crossover is trivial element-wise blending.

### Crossover

| Aspect | NEAT | Evolvion |
|--------|------|----------|
| Mechanism | Align genes by innovation number; inherit matching genes from either parent, excess/disjoint from fitter parent | Element-wise blending (uniform or arithmetic) — trivial because shared topology guarantees array alignment |
| Cross-species | Yes, via gene alignment (though typically rare/penalized) | No — individuals only exist within a single species topology |
| Importance | Core mechanism; crossover + speciation is NEAT's key contribution | Currently being added (Phase 3 of hardening plan); historically mutation-only |

Innovation numbers are NEAT's most distinctive feature. They provide a historical record that makes cross-topology crossover possible. Evolvion doesn't need them — the shared topology guarantee means weight arrays are semantically aligned by construction.

### Population Structure

| Aspect | NEAT | Evolvion |
|--------|------|----------|
| Population | Single pool of genomes, clustered into species each generation | Fixed species count, each with fixed individual count (e.g., 39 species x 132 individuals) |
| Species lifetime | Species form and dissolve naturally as genomes drift | Species persist until culled for stagnation or underperformance; replaced by diversified offspring |
| Fitness sharing | Explicit fitness sharing within species (divide by species size) to prevent large species from dominating | Implicit via fixed per-species population size — species don't compete for slots |
| Protection | New species get implicit protection through speciation threshold | Explicit grace period (configurable generations of immunity from culling) |

### Evaluation Model

| Aspect | NEAT | Evolvion |
|--------|------|----------|
| Parallelism | Each genome has unique topology — hard to batch | All individuals in a species share topology — trivially batchable |
| Memory layout | Typically per-genome evaluation | Structure-of-Arrays (SoA): contiguous weight/bias/activation arrays across all individuals |
| GPU suitability | Requires heterogeneous graph execution or padding/masking | Native: one kernel per species row, all individuals evaluated simultaneously |
| Scaling | Thousands of evaluations feasible on CPU | Tens of thousands feasible on GPU |

This is the fundamental architectural difference. NEAT's per-genome topology makes each evaluation unique. Evolvion's shared topology makes evaluation a SIMD operation.

### Features Unique to Evolvion

- **Per-row activation bitmasks**: Each layer has a configurable set of allowed activations, enforced during mutation and diversification
- **Learnable activation parameters**: Node-level parameters (e.g., LeakyReLU alpha) are evolved alongside weights
- **Complexity homeostasis**: Adaptive mutation rates self-regulate network size toward configurable targets
- **Weak edge pruning**: Population-level statistical pruning removes edges with consistently near-zero weights
- **Rank-weighted inheritance**: New species inherit weights from parent species' top performers, biased by fitness rank
- **Multi-episode GPU evaluation**: Built-in support for evaluating each individual across multiple environment seeds in parallel
- **Row-compiled evaluation**: RowPlans compile the forward pass schedule once per topology, not per evaluation

### Features Unique to NEAT

- **Innovation numbers**: Global historical tracking of structural mutations, enabling meaningful cross-topology comparison and crossover
- **Minimal initialization**: Start from simplest possible network and discover necessary structure
- **Compatibility distance**: Continuous genome similarity metric based on structural and parametric differences
- **Dynamic speciation threshold**: Automatically adjust to maintain target species count
- **Cross-topology crossover**: Gene alignment via innovation numbers enables recombination between structurally different genomes

## When to Use What

**Evolvion is better when:**
- You need to evaluate large populations (>5K individuals) quickly
- GPU hardware is available
- The problem domain suggests a known approximate architecture (e.g., "2 hidden layers of ~8 nodes for this control task")
- You want to experiment with specific topological features (activation constraints, connectivity patterns)

**NEAT is better when:**
- You want the algorithm to discover minimal sufficient topology from scratch
- Cross-topology recombination is important
- Population sizes are modest (hundreds, not thousands)
- You want a well-studied algorithm with decades of research and benchmarks

## Summary

Evolvion is not NEAT. It's a speciated neuroevolution framework that shares NEAT's high-level insight — protect structural innovation through speciation — but makes a fundamentally different trade-off: shared topology per species in exchange for GPU-batched evaluation. This single constraint changes everything downstream: how species are defined, how crossover works, how topology evolves, and how evaluation scales.

The right mental model is: **Evolvion is to NEAT what data-parallel deep learning is to heterogeneous graph neural networks.** Same broad problem domain, different parallelism strategy, different architectural constraints.
