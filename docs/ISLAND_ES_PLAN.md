# Island-Model Population Optimization — Implementation Plan

## Goal

Replace GA-based weight evolution with **distribution-based optimization** (CEM or ES) using **fixed-topology fully-connected networks** and **island-model species** — all within a single GPU kernel launch. Target: solve locomotion-scale problems in minutes of wall-clock time.

## Why Distribution-Based Over GA

GA evaluates 16K individuals but only the top ~100 influence the next generation. The other 15,900 contribute nothing except proving they're bad. Distribution-based methods (CEM, ES) maintain a compact representation of "where good solutions live" (μ, σ) and update it using evaluation results. This is fundamentally more information-efficient than selection+mutation.

**Expected speedup: 5–10× fewer generations to converge**, at near-identical cost per generation.

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   GPU Kernel Launch                    │
│                                                        │
│  Island 0: samples from N(μ₀, σ₀²)  (threads 0..n-1) │
│  Island 1: samples from N(μ₁, σ₁²)  (threads n..2n-1)│
│  Island 2: samples from N(μ₂, σ₂²)  (threads 2n..3n-1)│
│  ...                                                   │
│                                                        │
│  All same topology → one kernel, one launch, full GPU  │
└──────────────────────────────────────────────────────┘
         │ fitness[16K]
         ▼
┌──────────────────────────────────────────────────────┐
│                    CPU (per generation)                │
│                                                        │
│  For each island:                                      │
│    1. Slice fitness values for this island              │
│    2. Update μ and σ using strategy (CEM or ES)        │
│    3. Check stagnation → replace if stuck               │
│                                                        │
│  Sample next generation's weights from N(μᵢ, σᵢ²)     │
│  Upload to GPU                                         │
└──────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Fixed Topology, Fully Connected

No topology search. No edge mutations. No variable-length weight arrays.

**Network structure:** `input → H₁ → H₂ → output`, fully connected, all tanh (or configurable per-layer).

**Why:**
- Same topology across all islands → single kernel launch (islands are free)
- Dense matmul is maximally GPU-efficient (no sparse edge traversal)
- Eliminates entire subsystems: edge mutations, weight remapping, topology validation
- The sparse complexification study showed topology search matters little for known-structure problems — networks need ≥30% density anyway, and fully connected is 100%

**Parameter count examples:**
- Double pole (6→8→8→3): 6×8 + 8×8 + 8×3 = 136 weights + 19 biases = 155 params
- Rocket landing (12→16→16→3): 12×16 + 16×16 + 16×3 = 496 weights + 35 biases = 531 params
- Humanoid (50→64→64→20): 50×64 + 64×64 + 64×20 = 8576 weights + 148 biases = 8724 params

### 2. GPU-Adaptive Population Sizing

The GPU determines total population, not the user. Different GPUs have different optimal population sizes.

**Population sizing logic:**

```
gpuCapacity = NumMultiprocessors × 4 × WarpSize  (e.g., 16384 on RTX 4090)

if gpuCapacity >= IslandCount × MinIslandPop:
    individualsPerIsland = (gpuCapacity / IslandCount) rounded down to even
    totalPop = IslandCount × individualsPerIsland
else:
    // Small GPU: reduce island count
    maxIslands = gpuCapacity / MinIslandPop
    if maxIslands >= 1:
        IslandCount = maxIslands
        individualsPerIsland = (gpuCapacity / IslandCount) rounded down to even
        totalPop = IslandCount × individualsPerIsland
    else:
        // Very small GPU: single island, multiple kernel launches
        IslandCount = 1
        individualsPerIsland = MinIslandPop
        totalPop = individualsPerIsland
        // Will require ceil(totalPop / gpuCapacity) launches per generation
```

**GPU capacity examples:**

| GPU | SMs | Optimal Pop | 5 Islands | Per Island | Launches/Gen |
|-----|-----|-------------|-----------|------------|--------------|
| RTX 4090 | 128 | 16384 | 5 | 3276 | 1 |
| RTX 3080 | 68 | 8704 | 5 | 1740 | 1 |
| RTX 3060 | 28 | 3584 | 5 | 716 | 1 |
| RTX 2060 | 30 | 3840 | 5 | 768 | 1 |
| GTX 1060 | 10 | 1280 | 2 | 640 | 1 |
| Laptop (low) | 4 | 512 | 1 | 512 | 1 |

Even budget GPUs can run at least 1 island of 512. The algorithm adapts — fewer islands means less exploration diversity but still converges.

### 3. Multi-Launch for Overflow

When `totalPop > gpuCapacity` (e.g., user forces larger population on a small GPU):

```
launchesPerGen = ceil(totalPop / gpuCapacity)
for each launch:
    evaluate batch of min(gpuCapacity, remaining) individuals
    collect fitness
// Then do update with all fitness values combined
```

Rare in practice — auto-scaling avoids it.

### 4. Pluggable Update Strategy: CEM vs ES

The core loop is identical for both — sample from distribution, evaluate on GPU, update distribution on CPU. Only the update rule differs.

```csharp
/// <summary>
/// Shared interface for distribution-based update strategies.
/// Both CEM and ES maintain per-island (μ, σ) and update them from fitness results.
/// </summary>
public interface IUpdateStrategy
{
    /// <summary>
    /// Generate parameter vectors for all individuals in an island.
    /// Writes to the provided weight+bias arrays for GPU upload.
    /// Must store any state needed for the update step (e.g., noise vectors for ES).
    /// </summary>
    void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng);

    /// <summary>
    /// Update island's μ and σ given evaluated fitness values.
    /// paramVectors contains the same values generated by GenerateSamples.
    /// </summary>
    void Update(Island island, ReadOnlySpan<float> fitnesses,
                ReadOnlySpan<float> paramVectors, int popSize);
}
```

#### CEM (Cross-Entropy Method) — Primary

CEM is the recommended starting point: simpler, fewer hyperparameters, natural per-parameter σ adaptation.

```csharp
public class CEMStrategy : IUpdateStrategy
{
    public float EliteFraction { get; set; } = 0.1f;   // top 10%
    public float MuSmoothing { get; set; } = 0.2f;     // 0 = full replace, 1 = no change
    public float SigmaSmoothing { get; set; } = 0.2f;  // prevents σ collapse
    public float MinSigma { get; set; } = 0.001f;      // σ floor

    public void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng)
    {
        int paramCount = island.Mu.Length;
        for (int i = 0; i < popSize; i++)
        {
            int offset = i * paramCount;
            for (int p = 0; p < paramCount; p++)
                paramVectors[offset + p] = island.Mu[p] + island.Sigma[p] * SampleGaussian(rng);
        }
    }

    public void Update(Island island, ReadOnlySpan<float> fitnesses,
                       ReadOnlySpan<float> paramVectors, int popSize)
    {
        int paramCount = island.Mu.Length;
        int eliteCount = Math.Max(1, (int)(popSize * EliteFraction));

        // Find elite indices (top-K by fitness)
        var eliteIndices = TopKIndices(fitnesses, popSize, eliteCount);

        // Refit μ and σ to elites
        for (int p = 0; p < paramCount; p++)
        {
            float sum = 0, sumSq = 0;
            for (int i = 0; i < eliteCount; i++)
            {
                float v = paramVectors[eliteIndices[i] * paramCount + p];
                sum += v;
                sumSq += v * v;
            }
            float eliteMean = sum / eliteCount;
            float eliteVar = sumSq / eliteCount - eliteMean * eliteMean;

            // Smoothed update
            island.Mu[p] = (1 - MuSmoothing) * eliteMean + MuSmoothing * island.Mu[p];
            island.Sigma[p] = MathF.Max(MinSigma,
                (1 - SigmaSmoothing) * MathF.Sqrt(eliteVar) + SigmaSmoothing * island.Sigma[p]);
        }
    }
}
```

**CEM advantages:**
- **No gradient computation** — just mean() and std() of elites
- **No Adam optimizer** — no learning rate, β₁, β₂, ε to tune
- **Per-parameter σ for free** — each weight's σ = spread of that weight among elites. If weight #47 is well-determined, its σ shrinks. If weight #12 varies, σ stays large.
- **2 hyperparameters** — elite fraction + smoothing factor
- **Robust at large population** — 10% of 3200 per island = 320 elites = very stable distribution estimate

#### ES (OpenAI Evolution Strategies) — Alternative

ES uses all individuals for gradient estimation, not just elites. Available as a drop-in alternative.

```csharp
public class ESStrategy : IUpdateStrategy
{
    public float Sigma { get; set; } = 0.05f;          // isotropic noise (or per-param)
    public float LearningRate { get; set; } = 0.01f;
    public float AdamBeta1 { get; set; } = 0.9f;
    public float AdamBeta2 { get; set; } = 0.999f;

    // Stores noise vectors between Generate and Update
    private float[] _noiseVectors;

    public void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng)
    {
        // Antithetic sampling: pairs of θ+σε and θ-σε
        int paramCount = island.Mu.Length;
        int numPairs = popSize / 2;
        _noiseVectors = new float[numPairs * paramCount];

        for (int i = 0; i < numPairs; i++)
        {
            int noiseOffset = i * paramCount;
            int plusOffset = (2 * i) * paramCount;
            int minusOffset = (2 * i + 1) * paramCount;

            for (int p = 0; p < paramCount; p++)
            {
                float eps = SampleGaussian(rng);
                _noiseVectors[noiseOffset + p] = eps;
                paramVectors[plusOffset + p] = island.Mu[p] + Sigma * eps;
                paramVectors[minusOffset + p] = island.Mu[p] - Sigma * eps;
            }
        }
    }

    public void Update(Island island, ReadOnlySpan<float> fitnesses,
                       ReadOnlySpan<float> paramVectors, int popSize)
    {
        int paramCount = island.Mu.Length;
        int numPairs = popSize / 2;

        // ES gradient estimate
        Span<float> gradient = stackalloc float[paramCount];
        for (int i = 0; i < numPairs; i++)
        {
            float fPlus = fitnesses[2 * i];
            float fMinus = fitnesses[2 * i + 1];
            float diff = fPlus - fMinus;
            int noiseOffset = i * paramCount;

            for (int p = 0; p < paramCount; p++)
                gradient[p] += diff * _noiseVectors[noiseOffset + p];
        }

        // Normalize and Adam update
        float scale = 1.0f / (numPairs * Sigma);
        for (int p = 0; p < paramCount; p++)
            gradient[p] *= scale;

        island.AdamUpdate(gradient, LearningRate, AdamBeta1, AdamBeta2);
    }
}
```

**ES advantages over CEM:**
- Uses all individuals (not just top 10%) — better in low-population regimes
- Antithetic sampling halves gradient variance
- Smoother updates (gradient step vs distribution jump)

**ES disadvantages vs CEM:**
- More hyperparameters (σ, lr, Adam β₁/β₂/ε)
- No natural per-parameter σ (isotropic unless you add SNES)
- Stores noise vectors between Generate and Update (memory overhead)

**When to prefer ES:** If CEM converges too slowly or collapses σ too aggressively. For most problems with 16K+ population, CEM is expected to be sufficient.

### 5. Island Lifecycle

```csharp
public class Island
{
    public float[] Mu;              // Mean parameter vector (the "solution")
    public float[] Sigma;           // Per-parameter std dev (CEM) or scalar (ES)
    public float BestFitness;       // Best fitness seen by this island
    public int StagnationCounter;   // Generations without improvement

    // ES-only (unused by CEM):
    public AdamState Adam;          // Per-parameter momentum + variance
}
```

**Stagnation detection and replacement:**

```
After each generation:
    For each island:
        currentBest = max fitness in this island's individuals
        if currentBest > island.BestFitness + epsilon:
            island.BestFitness = currentBest
            island.StagnationCounter = 0
        else:
            island.StagnationCounter++

        if island.StagnationCounter >= StagnationThreshold (e.g., 30):
            bestIsland = island with highest BestFitness
            island.Mu = copy of bestIsland.Mu
            // Add perturbation to break symmetry
            for each param p:
                island.Mu[p] += SampleGaussian(rng) * ReinitSigma
                island.Sigma[p] = InitialSigma
            island.Adam = fresh (if ES)
            island.StagnationCounter = 0
            island.BestFitness = -inf
```

**Why this beats GA species culling:** GA species culling loses the entire population when a species is replaced. Island replacement is just copying one float array (μ) and perturbing it — the "population" is implicitly defined by the distribution, so nothing is lost.

### 6. Warm-Starting

When the player changes the environment (adds obstacle, tweaks body), μ vectors carry over:

```
Player changes environment:
    Keep all island μ values
    Reset StagnationCounters
    Multiply all σ by WarmStartSigmaBump (e.g., 2.0×)
    Re-evaluate immediately — first generation with new environment
```

Minor environment change → converges in seconds (μ is already close).
Major environment change → converges in ~1 minute (μ provides starting point).

GA can't do this — knowledge is spread across 16K individuals with no compact representation.

## Dense NN Kernel (Performance-Critical)

### The Problem with the Current Kernel

The existing `InlineNN.ForwardPass` does **sparse edge traversal** — every multiply requires an indirect lookup:

```csharp
// Current: 3 memory accesses per multiply (edge struct + source node + weight)
for (int edgeIdx = rowPlan.EdgeStart; edgeIdx < rowPlan.EdgeStart + rowPlan.EdgeCount; edgeIdx++)
{
    var edge = nn.Edges[edgeIdx];                              // random access: GPUEdge struct
    float weight = nn.Weights[weightOffset + edgeIdx];         // strided access
    float sourceValue = nn.NodeValues[nodeOffset + edge.Source]; // random access via Source
    nn.NodeValues[nodeOffset + edge.Dest] += weight * sourceValue; // scatter to Dest
}
```

This pays three costs per connection:
1. **Edge indirection**: Read `GPUEdge` struct to discover Source/Dest (8 bytes per edge, pollutes cache)
2. **Scatter pattern**: Accumulate into `NodeValues[edge.Dest]` — different edges target different destinations (poor write locality)
3. **`RowPlan` dispatch**: Jump through `RowPlans` array to find edge ranges per row

For a fully connected network, **all of this is unnecessary**. The Source and Dest are known at compile time from the layer sizes.

### Dense Forward Pass

Replace `InlineNN` with `DenseNN` — no edge array, no row plans, no scatter:

```csharp
public static class DenseNN
{
    /// <summary>
    /// Dense forward pass for fixed fully-connected network.
    /// No edge array, no RowPlan, no indirection — just contiguous matmul.
    ///
    /// Weight layout: layers stored contiguously.
    ///   Layer 0→1: layerSizes[0] × layerSizes[1] weights
    ///   Layer 1→2: layerSizes[1] × layerSizes[2] weights
    ///   ...
    /// Within each layer: row-major (dst-major) order:
    ///   w[dst=0,src=0], w[dst=0,src=1], ..., w[dst=0,src=prevSize-1],
    ///   w[dst=1,src=0], w[dst=1,src=1], ..., etc.
    /// This means computing one output node = one contiguous read of prevSize weights.
    ///
    /// Bias layout: one per non-input node, in layer order.
    /// </summary>
    public static void ForwardPass(
        ArrayView<float> weights,      // [totalWeights per individual], contiguous
        ArrayView<float> biases,       // [totalNonInputNodes per individual]
        ArrayView<float> observations, // [inputSize per world]
        ArrayView<float> actions,      // [outputSize per world]
        int worldIdx,
        int totalWeightsPerNet,
        int totalBiasesPerNet,
        int inputSize,
        int outputSize,
        int numHiddenLayers,           // compile-time known (1 or 2 typically)
        int hiddenSize)                // same width for all hidden layers (simplification)
    {
        int wBase = worldIdx * totalWeightsPerNet;
        int bBase = worldIdx * totalBiasesPerNet;
        int obsBase = worldIdx * inputSize;
        int actBase = worldIdx * outputSize;

        // Ping-pong buffers in registers/local memory
        // For small networks (≤64 nodes per layer), these fit in registers
        // ILGPU local arrays compile to register-backed storage for small sizes
        var bufA = LocalMemory.Allocate1D<float>(64);  // max hidden width
        var bufB = LocalMemory.Allocate1D<float>(64);

        // --- Layer 0 → 1 (input → first hidden) ---
        int wOff = wBase;
        int bOff = bBase;
        int prevSize = inputSize;
        int currSize = hiddenSize;

        for (int dst = 0; dst < currSize; dst++)
        {
            float sum = biases[bOff + dst];
            int wRow = wOff + dst * prevSize;
            for (int src = 0; src < prevSize; src++)
                sum += observations[obsBase + src] * weights[wRow + src];
            bufA[dst] = Tanh(sum);
        }
        wOff += prevSize * currSize;
        bOff += currSize;

        // --- Hidden → Hidden layers (if numHiddenLayers > 1) ---
        // Ping-pong between bufA and bufB
        var readBuf = bufA;
        var writeBuf = bufB;
        prevSize = hiddenSize;

        for (int layer = 1; layer < numHiddenLayers; layer++)
        {
            for (int dst = 0; dst < hiddenSize; dst++)
            {
                float sum = biases[bOff + dst];
                int wRow = wOff + dst * prevSize;
                for (int src = 0; src < prevSize; src++)
                    sum += readBuf[src] * weights[wRow + src];
                writeBuf[dst] = Tanh(sum);
            }
            wOff += prevSize * hiddenSize;
            bOff += hiddenSize;

            // Swap buffers
            var tmp = readBuf;
            readBuf = writeBuf;
            writeBuf = tmp;
        }

        // --- Last hidden → output ---
        prevSize = hiddenSize;
        for (int dst = 0; dst < outputSize; dst++)
        {
            float sum = biases[bOff + dst];
            int wRow = wOff + dst * prevSize;
            for (int src = 0; src < prevSize; src++)
                sum += readBuf[src] * weights[wRow + src];
            actions[actBase + dst] = Tanh(sum);
        }
    }

    private static float Tanh(float x)
    {
        float exp2x = XMath.Exp(2.0f * x);
        return (exp2x - 1.0f) / (exp2x + 1.0f);
    }
}
```

### What This Eliminates

| Current (sparse) | Dense | Savings |
|-------------------|-------|---------|
| `GPUEdge[]` array (8 bytes/edge × edges × individuals) | Gone | Memory bandwidth + cache |
| `GPURowPlan[]` per-row dispatch | Gone | Branch + indirection |
| `byte[] Activations` per-node lookup | Hardcoded `Tanh()` | Branch elimination |
| `float[] NodeParams` (4 floats/node) | Gone | Memory + 4 unused reads/node |
| `float[] NodeValues` (all nodes × individuals) | Local registers (ping-pong) | Global memory → registers |
| Scatter writes to `NodeValues[edge.Dest]` | Sequential accumulation into local `sum` | Write coherence |

### Memory Layout Comparison

**Current sparse layout** (for 16K individuals, 6→8→8→3 network, 136 edges, 25 nodes):

| Buffer | Size | Purpose |
|--------|------|---------|
| Edges | 136 × 8B = 1 KB | Shared, read every forward pass |
| RowPlans | 4 × 16B = 64 B | Shared, read every forward pass |
| Weights | 16K × 136 × 4B = 8.7 MB | Per-individual |
| Biases | 16K × 25 × 4B = 1.6 MB | Per-individual |
| NodeParams | 16K × 25 × 4 × 4B = 6.4 MB | Per-individual (mostly unused with fixed activations) |
| Activations | 16K × 25 × 1B = 400 KB | Per-individual (all same value) |
| NodeValues | 16K × 25 × 4B = 1.6 MB | Per-episode, read/written every step |
| **Total** | **~18.7 MB** | |

**Dense layout** (same network):

| Buffer | Size | Purpose |
|--------|------|---------|
| Weights | 16K × 136 × 4B = 8.7 MB | Per-individual (same count, different layout) |
| Biases | 16K × 19 × 4B = 1.2 MB | Per-individual (non-input nodes only) |
| **Total** | **~9.9 MB** | |

**~47% memory reduction.** More importantly, the hot path (forward pass inner loop) reads only `weights` + one local buffer — everything else is in registers.

### GPU Memory Access Pattern

The critical inner loop reads weights sequentially:

```
weights[wBase + dst * prevSize + 0]
weights[wBase + dst * prevSize + 1]
weights[wBase + dst * prevSize + 2]
...
weights[wBase + dst * prevSize + prevSize-1]
```

This is a **contiguous sequential read** — exactly what GPU memory controllers are optimized for. Adjacent threads (adjacent worldIdx) read adjacent weight blocks, enabling **coalesced memory access** across warps.

Compare to the sparse path where `nn.NodeValues[nodeOffset + edge.Source]` jumps to a Source discovered at runtime — effectively random access.

### Activation Function Simplification

With fixed topology, all hidden nodes use the same activation. This eliminates the 11-way `if/else` chain in `EvaluateActivation` (which the GPU can't branch-predict) and replaces it with a single inlined `Tanh()`.

If we later want per-layer activations (e.g., tanh hidden, linear output), that's a compile-time decision — still no runtime branching.

### Variable Hidden Width (Future)

The pseudocode above assumes uniform hidden width. For variable layer sizes (e.g., 6→16→8→3), replace the hidden loop with explicit per-layer code or pass layer sizes as kernel parameters. The matmul pattern is identical — only the loop bounds change.

For maximum performance with ILGPU, we could generate **specialized kernels per topology** at initialization time. ILGPU supports runtime kernel compilation, so a 6→8→8→3 kernel would have all loop bounds as constants, enabling full unrolling.

## Implementation Steps

### Phase 0: Core Data Structures

Create new files in `Evolvatron.Evolvion/`:

```
ES/
├── IslandConfig.cs         // Configuration (island count, sigma, strategy choice, etc.)
├── Island.cs               // Island state (mu, sigma, stagnation)
├── IUpdateStrategy.cs      // Interface for CEM/ES
├── CEMStrategy.cs          // CEM update rule
├── ESStrategy.cs           // ES update rule + AdamOptimizer
├── IslandPopulationManager.cs  // Generates weights, dispatches updates
└── DenseTopology.cs        // Dense fully-connected topology descriptor
```

**`DenseTopology`** — lightweight descriptor, no edge lists:
```csharp
public class DenseTopology
{
    public int[] LayerSizes { get; }          // e.g., [6, 8, 8, 3]
    public int TotalWeights { get; }          // Σ layerSizes[i] × layerSizes[i+1]
    public int TotalBiases { get; }           // Σ layerSizes[i] for i > 0 (non-input)
    public int TotalParams => TotalWeights + TotalBiases;
    public int InputSize => LayerSizes[0];
    public int OutputSize => LayerSizes[^1];
    public int MaxHiddenWidth { get; }        // for local memory allocation
    public int NumHiddenLayers { get; }       // LayerSizes.Length - 2

    // Maps flat parameter vector ↔ per-layer weight/bias segments
    public (int offset, int count) WeightSegment(int layerIdx);
    public (int offset, int count) BiasSegment(int layerIdx);
}
```

**`IslandPopulationManager`** generates flat weight+bias arrays from distributions:
```csharp
public class IslandPopulationManager
{
    private IUpdateStrategy _strategy;

    // Generate weight arrays for all individuals across all islands
    // Layout: [island0_ind0, island0_ind1, ..., island1_ind0, ...]
    public float[] GeneratePopulation(
        List<Island> islands, int individualsPerIsland, Random rng);

    // After evaluation, update all islands via the active strategy
    public void UpdateIslands(
        List<Island> islands, float[] fitnesses, float[] paramVectors, int individualsPerIsland);

    // Stagnation check and island replacement
    public void ManageIslands(List<Island> islands, IslandConfig config, Random rng);
}
```

### Phase 1: Dense NN Kernel

New `DenseNN.cs` in `GPU/MegaKernel/` — the forward pass shown above. This replaces `InlineNN` for the new evaluators but does not break existing sparse evaluators (they keep using `InlineNN`).

New `DenseNNViews` struct — minimal:
```csharp
public struct DenseNNViews
{
    public ArrayView<float> Weights;  // [totalPop × totalWeightsPerNet]
    public ArrayView<float> Biases;   // [totalPop × totalBiasesPerNet]
}
```

No Edges. No RowPlans. No Activations. No NodeParams. No NodeValues.

### Phase 2: Dense Evaluator

New evaluator class (e.g., `GPUDenseDoublePoleEvaluator`) that:
1. Takes `DenseTopology` + flat weight/bias arrays (not `SpeciesSpec` + `Individual`)
2. Allocates only weights + biases buffers (no edge/rowplan/activation/nodeparam buffers)
3. Uses `DenseNN.ForwardPass` in its step kernel
4. Returns `float[] fitnesses`

This is a **new code path**, not a modification of existing evaluators. The old sparse evaluators remain untouched for Evolvion GA compatibility.

### Phase 3: CPU-Only Strategy Validation

Test both CEM and ES strategies against simple fitness functions (sphere, Rastrigin) without GPU:
- CEM: elite selection + distribution refit converges on sphere
- ES: gradient computation + Adam converges on sphere
- Compare convergence speed CEM vs ES at various population sizes
- Island replacement triggers and helps escape local minima on Rastrigin (multi-modal)
- Verify per-parameter σ adaptation in CEM (some params should converge faster)

### Phase 4: Benchmark DPNV

Run head-to-head: CEM vs ES vs current Evolvion GA on double-pole no-velocity. Same GPU, same wall-clock budget. Measure:
- Generations to solve
- Wall-clock time to solve
- Reliability (X/10 seeds)

**Expected:** Both CEM and ES solve in 5–10× fewer generations than GA.

### Phase 5: Rocket Landing + Warm-Start

Test on rocket landing with obstacles. Then test warm-starting:
1. Evolve without obstacles → converge
2. Add funnel obstacles → continue from μ (warm-start)
3. Measure re-convergence time vs cold start

### Phase 6: Elman Recurrence (Fixed Context Size)

For non-Markovian tasks (DPNV, partial observability), add fixed context outputs:

```
Network: input+context → H₁ → H₂ → output+context
Context size: fixed (e.g., 2)
Context outputs fed back as extra inputs next timestep
```

This fits cleanly into the fixed topology — just adds `contextSize` extra inputs and outputs. All islands share the same context size.

## Configuration

```csharp
public class IslandConfig
{
    // --- Population structure ---
    public int IslandCount { get; set; } = 5;
    public int MinIslandPop { get; set; } = 512;
    public bool AutoScale { get; set; } = true;

    // --- Strategy selection ---
    public UpdateStrategyType Strategy { get; set; } = UpdateStrategyType.CEM;

    // --- CEM parameters ---
    public float CEMEliteFraction { get; set; } = 0.1f;    // top 10%
    public float CEMMuSmoothing { get; set; } = 0.2f;      // blending with previous μ
    public float CEMSigmaSmoothing { get; set; } = 0.2f;   // prevents σ collapse

    // --- ES parameters ---
    public float ESSigma { get; set; } = 0.05f;            // isotropic noise scale
    public float ESLearningRate { get; set; } = 0.01f;     // Adam step size
    public float ESAdamBeta1 { get; set; } = 0.9f;
    public float ESAdamBeta2 { get; set; } = 0.999f;

    // --- Shared ---
    public float InitialSigma { get; set; } = 0.1f;        // per-param initial σ (CEM), or scalar (ES)
    public float MinSigma { get; set; } = 0.001f;          // σ floor
    public float MaxSigma { get; set; } = 0.5f;            // σ ceiling

    // --- Island lifecycle ---
    public int StagnationThreshold { get; set; } = 30;     // generations before replacement
    public float ReinitSigma { get; set; } = 0.1f;         // perturbation on replacement

    // --- Network topology ---
    public int[] LayerSizes { get; set; }                   // e.g., [6, 8, 8, 3]
    public int ContextSize { get; set; } = 0;               // Elman recurrence (0 = feedforward)

    // --- Warm-start ---
    public float WarmStartSigmaBump { get; set; } = 2.0f;  // multiply σ on env change
}

public enum UpdateStrategyType
{
    CEM,    // Cross-Entropy Method (recommended default)
    ES      // OpenAI Evolution Strategies
}
```

## Evolution Loop

```csharp
var config = new IslandConfig { LayerSizes = new[] { 6, 8, 8, 3 } };
var topology = new DenseTopology(config.LayerSizes);
var strategy = config.Strategy == UpdateStrategyType.CEM
    ? new CEMStrategy(config) : new ESStrategy(config);
var manager = new IslandPopulationManager(strategy);
var evaluator = new GPUDenseDoublePoleEvaluator();

// Auto-scale islands to GPU
int gpuCapacity = evaluator.OptimalPopulationSize;
var islands = manager.InitializeIslands(config, topology.TotalParams, gpuCapacity);
int perIsland = gpuCapacity / config.IslandCount;

for (int gen = 0; gen < maxGens; gen++)
{
    // 1. Sample from distributions
    float[] paramVectors = manager.GeneratePopulation(islands, perIsland, rng);

    // 2. Upload weights+biases to GPU, evaluate
    float[] fitnesses = evaluator.Evaluate(topology, paramVectors, perIsland * islands.Count);

    // 3. Update distributions (CEM or ES)
    manager.UpdateIslands(islands, fitnesses, paramVectors, perIsland);

    // 4. Replace stagnant islands
    manager.ManageIslands(islands, config, rng);
}

// Best solution: island with highest BestFitness → its μ vector
var best = islands.OrderByDescending(i => i.BestFitness).First();
```

## What Changes vs Current Architecture

| Component | Current (Evolvion GA) | New (Island CEM/ES) |
|-----------|----------------------|---------------------|
| Weight generation | Tournament select → mutate parent | Sample from N(μ, σ²) |
| Selection | Top-K tournament | CEM: elite refit / ES: all contribute to gradient |
| Update rule | Replace losers with mutated winners | CEM: μ=mean(elites) / ES: Adam ascent |
| Topology | Variable per species, edge mutations | Fixed, fully connected |
| Species/Islands | Different topologies, separate launches | Same topology, one launch, free |
| Warm-start | Not possible (knowledge in 16K individuals) | Copy μ, bump σ |
| GPU kernel | Sparse edge traversal + activation dispatch | Dense matmul + inline tanh |
| GPU memory | Edges + RowPlans + Activations + NodeParams + NodeValues (~19MB) | Weights + Biases only (~10MB) |
| CPU overhead | Minimal (mutation is fast) | Minimal (mean/std or gradient is fast) |
| Strategy swap | Requires code changes | Config flag: `Strategy = CEM` or `ES` |

## Risks and Mitigations

**Risk: CEM σ collapses prematurely (all elites converge, σ→0)**
- Mitigation: σ smoothing (blend with previous σ) + MinSigma floor.
- Fallback: Switch to ES which uses fixed σ with Adam-controlled step size.

**Risk: ES gradient too noisy for large networks (>5000 params)**
- Mitigation: Antithetic sampling halves variance. 1500+ pairs per island is sufficient for ~5000 params.
- Fallback: Switch to CEM which doesn't compute gradients.

**Risk: All islands converge to same local minimum**
- Mitigation: Initialize μ vectors with diverse random seeds. Replacement adds large perturbation.
- Mitigation: Optional: periodically inject a fully random island.

**Risk: Fixed topology is wrong for some problems**
- Mitigation: Make layer sizes configurable. The player (or the game) can offer presets:
  - "Small" (in→8→8→out): fast, for simple tasks
  - "Medium" (in→32→32→out): balanced
  - "Large" (in→64→64→out): locomotion-scale
- Future: could add topology search back as a meta-level (try different layer sizes across runs)

## Success Criteria

- [ ] DPNV solves in <0.5s (currently ~1.7s with GA)
- [ ] Rocket landing solves in <10s (currently ~30-60s with GA)
- [ ] Warm-start re-convergence <20% of cold-start time
- [ ] Works on GPUs from GTX 1060 to RTX 4090
- [ ] Island replacement demonstrably rescues stuck runs
- [ ] CEM and ES are interchangeable via config flag

## Future Extensions

- **Fitness shaping**: Rank-based fitness normalization (standard in ES/CEM, reduces sensitivity to fitness scale)
- **Natural gradient (SNES)**: Better curvature adaptation for ES path
- **Hybrid CEM+ES**: Use CEM's elite selection for μ update but add ES-style gradient from non-elites
- **PPO transition**: If CEM/ES proves the architecture works, PPO is the next step for 10-50× further speedup on locomotion-scale problems
