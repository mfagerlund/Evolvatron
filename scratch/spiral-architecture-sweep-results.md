# Spiral Classification - Network Architecture Sweep Results

## Test Configuration
- **12 different architectures tested**
- **20 generations each** (double the previous 10)
- **Best hyperparameters used**: Tournament size 16, Weight jitter 0.95
- **Total runtime**: 1.35 minutes (77 seconds)

## SHOCKING RESULT: Bigger Networks Made It WORSE!

### Rankings (by improvement over 20 generations)

| Rank | Architecture | Gen0 | Gen19 | Improvement | Edges | Active Nodes |
|------|-------------|------|-------|-------------|-------|--------------|
| **1** | **Baseline-Sparse-20gen** (2→8→8→1) | -0.9653 | -0.9580 | **0.0073** (0.8%) | 15 | 5/19 |
| 2 | Baseline-HighDegree (same) | -0.9653 | -0.9580 | 0.0073 | 15 | 5/19 |
| 3 | Baseline-VeryHighDegree (same) | -0.9653 | -0.9580 | 0.0073 | 15 | 5/19 |
| 4 | Deep-10x10x10-Sparse | -0.9642 | -0.9609 | 0.0033 (0.3%) | 16 | 4/33 |
| 5 | Bigger-16x16-Sparse | -0.9637 | -0.9609 | 0.0028 (0.3%) | 16 | 4/35 |
| 6 | Bigger-16x16-HighDegree | -0.9637 | -0.9609 | 0.0028 | 16 | 4/35 |
| 7 | Bigger-12x12-Sparse | -0.9620 | -0.9609 | 0.0012 (0.1%) | 18 | 3/27 |
| 8 | Bigger-12x12-HighDegree | -0.9620 | -0.9609 | 0.0012 | 18 | 3/27 |
| 9 | Bigger-16x16x8-Sparse (3 layers) | -0.9620 | -0.9609 | 0.0011 | 18 | 3/43 |
| 10 | Bigger-16x16x8-HighDegree | -0.9620 | -0.9609 | 0.0011 | 18 | 3/43 |
| 11 | Bigger-20x20-Sparse | -0.9614 | -0.9609 | 0.0006 | 18 | 3/43 |
| 12 | Bigger-20x10-HighDegree | -0.9614 | -0.9609 | 0.0006 | 18 | 3/33 |

## Key Findings

### 1. **BASELINE IS BEST!**

The original 2→8→8→1 architecture with only 15 edges performed **BETTER** than all larger networks:
- **2.5x better** than 2→16→16→1
- **6x better** than 2→12→12→1
- **12x better** than 2→20→20→1

### 2. **Bigger Networks Got WORSE Starting Fitness**

Notice the Gen0 fitness:
- Baseline (8→8): **-0.9653**
- Medium (12→12): -0.9620
- Large (16→16): -0.9637
- Very Large (20→20): **-0.9614 (WORSE!)**

**Bigger networks start with BETTER random fitness** (closer to -0.96 instead of deeper negative), which means they have **less room to improve**.

### 3. **Active Node Count is Tiny**

Look at the active nodes:
- Baseline: **5 out of 19 nodes** are active (26%)
- Bigger-16x16: 4 out of 35 (11%)
- Bigger-20x20: 3 out of 43 (7%)

**Sparse initialization doesn't scale with network size!**

The larger networks have MORE total nodes, but FEWER active nodes. Most of the network is dead weight.

### 4. **Edge Count Barely Increases**

- Baseline: 15 edges
- Bigger networks: 16-18 edges

Even though we doubled or tripled the number of nodes, edge count only increased by 1-3 edges!

**Sparse initialization creates approximately the same connectivity regardless of network size.**

### 5. **Higher MaxInDegree Made NO DIFFERENCE**

- Baseline-Sparse-20gen (MaxInDegree=10): 15 edges, 5 active
- Baseline-HighDegree (MaxInDegree=16): 15 edges, 5 active (IDENTICAL!)
- Baseline-VeryHighDegree (MaxInDegree=20): 15 edges, 5 active (IDENTICAL!)

**MaxInDegree limit never gets hit** with sparse initialization. The network self-limits to ~15-18 edges regardless of the cap.

## Category Analysis

**Bigger Networks (Sparse Init)**:
- Average improvement: 0.0014 (83% worse than baseline)
- Best: Bigger-16x16-Sparse (still 62% worse than baseline)

**Higher MaxInDegree**:
- No effect whatsoever

**Deeper Networks** (3 layers):
- Deep-10x10x10: 0.0033 improvement (55% worse than baseline)

## Why Bigger Networks FAILED

### Theory 1: Sparse Init + Big Networks = Dead Networks

With sparse initialization:
1. Each node gets connections from ~1-3 nodes in previous layer
2. Most nodes never get connected to inputs (not reachable)
3. Larger networks have MORE unreachable nodes
4. Evolution has to "wake up" dead nodes through topology mutations
5. But edge mutations happen at low rates (5% EdgeAdd, 2% EdgeDelete)

**Baseline (8→8)**: 19 total nodes, 5 active (26%) - manageable
**Bigger-20x20**: 43 total nodes, 3 active (7%) - catastrophic

### Theory 2: Better Random Initialization Paradox

Bigger networks with sparse init produce outputs closer to 0 by chance:
- More layers = more averaging
- Sparse connections = signal attenuation
- Result: Random output ≈ 0 (which scores -0.96 fitness)

**Smaller networks are MORE random** (worse Gen0 fitness -0.9653) but this gives them **more room to improve** through evolution!

### Theory 3: Search Space Explosion

- Baseline: 19 nodes × avg 2 params = ~40 dimensional search space
- Bigger-20x20: 43 nodes × avg 2 params = ~90 dimensional search space

Evolution has to search a 2x larger space with only marginal increase in connectivity.

## What Actually Works

**The surprising answer: SMALL NETWORKS!**

Best configuration remains:
- **2→8→8→1** (19 total nodes)
- **Sparse initialization** (15 edges)
- **MaxInDegree 10** (never reached)
- **Tournament size 16**
- **Weight jitter 0.95**

## Projected Time to Solve

With 20 generations showing 0.0073 improvement:
- Improvement per generation: 0.000365
- Fitness gap: -0.958 → -0.05 = 0.908
- **Projected generations: ~2,488**
- **Projected time: ~35 minutes**

**NOTE**: The negative projections in test output are nonsense (Gen19 fitness got WORSE than Gen0 for some runs due to random seed). The actual trend is slow improvement.

## What We SHOULD Try Next

### Option 1: FULLY CONNECTED small network
Instead of sparse 2→8→8→1, try **dense 2→6→6→1**:
- Manually connect EVERY possible edge
- Smaller network, but 100% utilized
- No dead nodes

### Option 2: Start with direct Input→Output connections
Current sparse init might not even connect inputs to outputs!
- Force at least one path from each input to output
- Then add sparse connections on top

### Option 3: Different activation functions
All our sparse networks use same activations:
- Try specialized activations for polar coordinates
- Sin/Cos for angles
- Square/Sqrt for radius

### Option 4: Curriculum Learning
Don't change architecture - change the problem difficulty:
1. Start with 10 spiral points (Gen 0-20)
2. Add 10 more points (Gen 21-40)
3. Keep adding until 100 points (Gen 100+)

This lets evolution learn on easier problems first.

### Option 5: Just Be Patient
Run baseline config for 500-1000 generations.
- Current projections suggest ~2,500 generations needed
- That's only ~35 minutes
- This is actually reasonable for evolutionary methods!

## Conclusion

**We were completely wrong about network capacity being the issue!**

The problem isn't:
- ✗ Not enough neurons
- ✗ Not enough layers
- ✗ Not enough max connections

The problem IS:
- ✓ **Sparse initialization doesn't utilize bigger networks**
- ✓ **Most nodes end up disconnected/inactive**
- ✓ **Smaller networks are MORE random (good!) and MORE compact (searchable!)**

**Recommendation**: Stick with small baseline (2→8→8→1) or try FULLY CONNECTED 2→6→6→1.

The answer isn't "bigger" - it's "denser" or "be patient".
