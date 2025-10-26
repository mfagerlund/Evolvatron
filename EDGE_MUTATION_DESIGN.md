# Enhanced Edge Mutation System

## Overview

This document extends the Evolvion.md specification with additional edge mutation operators and automatic pruning mechanisms for complexity control.

---

## 1. Core Edge Mutations

### 1.1 EdgeAdd (Complexification)
**Purpose:** Add new connections to increase representational capacity

**Algorithm:**
```csharp
1. Select random destination node (not in bias or input rows)
2. Check in-degree < MaxInDegree
3. Select random source from earlier rows (acyclic constraint)
4. Verify edge doesn't already exist
5. Initialize weight with Glorot/Xavier
6. Add to species edge list
7. Mark all individuals' weights array as needing resize
```

**Probability:** 0.05 (5% chance per individual mutation)

---

### 1.2 EdgeDelete (Simplification)
**Purpose:** Remove unnecessary connections, prevent bloat

**Variants:**

#### A. Random EdgeDelete
```csharp
1. Select random edge from species topology
2. Verify removal doesn't disconnect graph (input → output paths exist)
3. Remove edge from species
4. Compact weight arrays for all individuals
```

#### B. Weak EdgeDelete (Your Suggestion ⭐)
```csharp
1. For each edge in species:
   - Compute mean |weight| across all individuals
   - If mean_weight < epsilon_threshold:
     - Delete with probability P(delete|weak)
2. Epsilon threshold: 0.01 (1% of typical Glorot init range)
3. P(delete|weak): 0.5 (50% chance for weak edges)
```

**Benefits:**
- Automatically prunes connections that evolution has down-weighted
- Creates "soft" topology mutations guided by learning
- Emergent structural learning (important connections survive)

**Probability:** 0.02 (random) + automatic weak edge pruning

---

### 1.3 EdgeSplit (Depth Increase)
**Purpose:** Add representational capacity via intermediate computation

**Algorithm:**
```csharp
1. Select random edge (src → dst)
2. Determine insertion row (between src_row and dst_row)
3. If no intermediate row exists:
   - Insert new hidden row
   - Update all RowPlans
4. Create new node in intermediate row
5. Replace edge: src → new_node → dst
6. Initialize both new weights with Glorot
7. Copy activation from destination node (or random allowed)
```

**Probability:** 0.01 (rare, increases depth)

---

## 2. Advanced Edge Mutations

### 2.1 EdgeRedirect
**Purpose:** Change connection topology without changing edge count

**Algorithm:**
```csharp
1. Select random edge (src → dst)
2. Choose new source OR new destination (not both)
3. If changing source:
   - Pick random node from earlier rows than dst
4. If changing destination:
   - Pick random node from later rows than src
5. Verify new edge doesn't exist
6. Replace edge, keep weight value (transfer learning)
```

**Probability:** 0.03
**Benefit:** Explores connectivity patterns without changing network size

---

### 2.2 EdgeDuplicate
**Purpose:** Parallel pathways for robustness

**Algorithm:**
```csharp
1. Select random edge (src → dst)
2. If dst in-degree < MaxInDegree:
   - Add duplicate edge (src → dst)
   - Initialize new weight with original_weight + Gaussian noise
3. Creates two parallel connections with similar weights
```

**Probability:** 0.01
**Benefit:** Allows multiple weighted paths between same nodes

---

### 2.3 EdgeMerge (Complexity Reduction)
**Purpose:** Combine parallel edges into single weighted connection

**Algorithm:**
```csharp
1. Find nodes with multiple edges from same source
   - Example: src → dst (w1), src → dst (w2)
2. Merge into single edge with weight = w1 + w2
3. Reduces edge count while preserving signal strength
```

**Probability:** 0.02
**Benefit:** Simplifies redundant parallel pathways

---

### 2.4 EdgeSwap
**Purpose:** Rewire connections for topology exploration

**Algorithm:**
```csharp
1. Select two edges: (src1 → dst1), (src2 → dst2)
2. Verify rows allow swap:
   - row(src1) < row(dst2) AND row(src2) < row(dst1)
3. Swap destinations: (src1 → dst2), (src2 → dst1)
4. Keep original weights (transfer learning)
```

**Probability:** 0.02
**Benefit:** Explores alternative wirings without changing edge count

---

## 3. Species Birth Pruning (Your Key Insight)

### When Individual Gives Rise to New Species

**Context:** Section 9 of Evolvion.md - when species is eliminated and replaced

**Enhanced Birth Protocol:**

```csharp
// Step 1: Clone parent topology
var newSpecies = CloneTopology(bestParent);

// Step 2: Weak Edge Pruning (YOUR SUGGESTION)
var weakEdges = new List<(int src, int dst, float meanWeight)>();

foreach (var edge in newSpecies.Edges)
{
    // Compute mean absolute weight across parent population
    float meanWeight = ComputeMeanAbsWeight(bestParent.Individuals, edge);

    if (meanWeight < WEAK_EDGE_THRESHOLD)
    {
        weakEdges.Add((edge.Source, edge.Dest, meanWeight));
    }
}

// Step 3: Probabilistic deletion of weak edges
foreach (var (src, dst, weight) in weakEdges.OrderBy(e => e.meanWeight))
{
    // Probability increases as weight approaches zero
    float deleteProb = 1.0f - (weight / WEAK_EDGE_THRESHOLD);
    deleteProb = Math.Clamp(deleteProb * BASE_PRUNE_RATE, 0.0f, 0.9f);

    if (random.NextSingle() < deleteProb)
    {
        if (CanDeleteEdge(newSpecies, src, dst)) // Preserves connectivity
        {
            newSpecies.Edges.Remove((src, dst));
        }
    }
}

// Step 4: Diversification mutations (from original spec)
ApplyDiversificationMutations(newSpecies);

// Step 5: Initialize new individuals with Glorot weights
InitializePopulation(newSpecies);
```

**Parameters:**
- `WEAK_EDGE_THRESHOLD`: 0.01 (configurable)
- `BASE_PRUNE_RATE`: 0.7 (70% max deletion chance for near-zero weights)

**Benefits:**
1. **Inheritance of learned structure:** New species inherit simplified topology
2. **Automatic complexity control:** Evolution doesn't fight against accumulated bloat
3. **Transfer learning:** Strong connections preserved, weak ones pruned
4. **Diversity maintenance:** Each new species starts from different pruned variant

---

## 4. Connectivity Validation

Critical for all edge deletions:

```csharp
bool CanDeleteEdge(SpeciesSpec spec, int src, int dst)
{
    // Create temporary graph without this edge
    var tempEdges = spec.Edges.Where(e => e != (src, dst));

    // BFS from all input nodes
    var reachableFromInput = ComputeReachable(tempEdges, inputNodes, forward: true);

    // BFS backward from all output nodes
    var reachesOutput = ComputeReachable(tempEdges, outputNodes, forward: false);

    // All output nodes must be reachable from input
    foreach (var outputNode in outputNodes)
    {
        if (!reachableFromInput.Contains(outputNode))
            return false; // Deletion would disconnect graph
    }

    // At least one path must exist through each hidden node
    // (Optional: could allow orphaned hidden nodes, marking them inactive)

    return true;
}
```

---

## 5. Updated Mutation Probabilities

| Operator | Probability | Category | Complexity Impact |
|----------|-------------|----------|-------------------|
| **Weight Jitter** | 0.9 | Weight | None |
| **Weight Reset** | 0.05 | Weight | None |
| **Weight L1 Shrink** | 0.1 | Weight | None (soft regularization) |
| **Activation Swap** | 0.01 | Functional | None |
| **Node Param Mutate** | 0.2 | Functional | None |
| **EdgeAdd** | 0.05 | Topology | ↑ Increase |
| **EdgeDelete (Random)** | 0.01 | Topology | ↓ Decrease |
| **EdgeDelete (Weak)** | Auto | Topology | ↓ Decrease |
| **EdgeSplit** | 0.01 | Topology | ↑↑ Increase (adds node) |
| **EdgeRedirect** | 0.03 | Topology | = Neutral |
| **EdgeDuplicate** | 0.01 | Topology | ↑ Increase |
| **EdgeMerge** | 0.02 | Topology | ↓ Decrease |
| **EdgeSwap** | 0.02 | Topology | = Neutral |

**Complexity Balance:**
- Increase: 0.05 + 0.01 + 0.01 = **0.07** (7%)
- Decrease: 0.01 + auto + 0.02 = **0.03 + auto**
- Neutral: 0.03 + 0.02 = **0.05** (5%)

With automatic weak edge pruning, the system has inherent bias toward simplicity.

---

## 6. Implementation Phases

### Phase 1: Core Edge Mutations (Essential)
- [ ] EdgeAdd
- [ ] EdgeDelete (random)
- [ ] EdgeSplit
- [ ] Connectivity validation
- [ ] Tests for all three

### Phase 2: Weak Edge Pruning (Your Suggestion - High Value!)
- [ ] Mean weight computation per edge
- [ ] Probabilistic deletion based on weight magnitude
- [ ] Species birth pruning integration
- [ ] Tests for pruning behavior

### Phase 3: Advanced Mutations (Optional Enhancements)
- [ ] EdgeRedirect
- [ ] EdgeDuplicate
- [ ] EdgeMerge
- [ ] EdgeSwap
- [ ] Comparative benchmarks (with vs without)

---

## 7. Configuration

```json
{
  "EdgeMutationConfig": {
    "EdgeAdd": 0.05,
    "EdgeDeleteRandom": 0.01,
    "EdgeSplit": 0.01,
    "EdgeRedirect": 0.03,
    "EdgeDuplicate": 0.01,
    "EdgeMerge": 0.02,
    "EdgeSwap": 0.02,

    "WeakEdgePruning": {
      "Enabled": true,
      "Threshold": 0.01,
      "BasePruneRate": 0.7,
      "ApplyOnSpeciesBirth": true,
      "ApplyDuringEvolution": false
    }
  }
}
```

---

## 8. Expected Benefits

### Weak Edge Pruning (Your Idea)
1. **Emergent Structural Learning:** Network architecture becomes part of what's evolved
2. **Prevents Bloat:** Networks don't accumulate useless connections over generations
3. **Transfer Learning:** New species inherit learned structural decisions
4. **Automatic Regularization:** Similar to L1 regularization but discrete
5. **Species Diversity:** Different species can have radically different architectures

### Advanced Edge Mutations
1. **Richer Topology Space:** More ways to explore connectivity patterns
2. **Neutral Mutations:** EdgeRedirect/EdgeSwap allow exploration without size changes
3. **Complexity Control:** EdgeMerge provides another simplification mechanism
4. **Parallel Pathways:** EdgeDuplicate enables ensemble-like behavior within single network

---

## 9. Research Questions

1. **Pruning Timing:** Should weak edges be pruned continuously or only at species birth?
2. **Threshold Adaptation:** Should `WEAK_EDGE_THRESHOLD` adapt based on network performance?
3. **Structural Credit Assignment:** How to attribute fitness to specific edges?
4. **Minimum Topology:** Should there be a minimum edge count to prevent over-pruning?

---

## 10. Next Steps

**Immediate:** Implement Phase 1 (core mutations) + Phase 2 (weak edge pruning)

This gives you the full mutation suite with automatic complexity control driven by evolved weight magnitudes - exactly what you suggested!

Would you like me to implement this system?
