# GPU Scaling Strategy: Maximizing Throughput

## The Core Insight

A modern GPU has thousands of CUDA cores (e.g., RTX 3080 = 8,704, RTX 4090 = 16,384). With the current default of 132 individuals per species, each kernel launch uses ~132 threads — leaving 99% of the GPU idle. Scaling from 132 to 2,048+ individuals per species costs almost zero additional wall-clock time because the GPU parallelizes all threads across its cores in the same clock cycles.

**Individuals per species is the primary scaling lever.** Everything else is secondary.

## Target Architecture

### GPU Side (Runs Continuously)

The GPU owns the entire evaluate-loop for one generation:

```
For each species (sequentially or multi-stream):
    Upload mutated weights (one bulk transfer)
    For each episode seed:
        Reset physics state
        For each simulation step:
            Observe → NN forward pass → Act → Physics step → Terminal check
        Accumulate fitness
    Download fitness array (one bulk transfer)
```

Between weight upload and fitness download, the GPU runs autonomously — no CPU interaction, no intermediate syncs. With the mega-kernel, this is already 1 kernel dispatch per step.

### CPU Side (Runs in Parallel with GPU)

While the GPU evaluates generation N, the CPU prepares generation N+1:

```
Receive fitness from generation N-1
├── Rank individuals
├── Tournament selection
├── Generate offspring (deep copy)
├── Apply weight mutations (jitter, reset, L1 shrink)
├── Apply activation mutations
├── Species culling and diversification (topology mutations)
└── Pack mutated weights into upload buffer
Wait for GPU to finish generation N
Upload generation N+1 weights
Launch GPU generation N+1
```

### The Pipeline

```
Time ──────────────────────────────────────────────────────►

GPU:  [====== Eval Gen 0 ======][====== Eval Gen 1 ======][====== Eval Gen 2 ======]
CPU:                       [= Mutate 0→1 =]          [= Mutate 1→2 =]
                                ↑                          ↑
                           fitness download            fitness download
                           + weight upload             + weight upload
```

The sync points are:
1. **GPU → CPU**: Download fitness array after evaluation completes
2. **CPU → GPU**: Upload mutated weight arrays before next evaluation

Everything else is overlapped. CPU mutation and GPU evaluation happen concurrently.

## Scaling Analysis

### GPU Utilization vs Population Size

| Individuals/Species | Threads per kernel | GPU Utilization (RTX 3080) | Marginal cost |
|---------------------|-------------------|---------------------------|---------------|
| 132 | 132 | ~1.5% | Baseline |
| 256 | 256 | ~3% | ~Free |
| 512 | 512 | ~6% | ~Free |
| 1,024 | 1,024 | ~12% | ~Free |
| 2,048 | 2,048 | ~24% | ~Free |
| 4,096 | 4,096 | ~47% | Slight slowdown |
| 8,192 | 8,192 | ~94% | ~2x baseline |

The "free" range depends on the specific GPU and kernel complexity. The point where marginal cost rises above zero is when thread count exceeds the number of cores × occupancy multiplier (typically 2-4x due to warp switching hiding memory latency).

**For an RTX 3080 (8,704 cores), expect roughly 2,000-4,000 individuals per species to be "free" compared to 132.**

### Memory Scaling

Per individual, the GPU needs:
- Weights: `edges × 4 bytes` (e.g., 50 edges = 200B)
- Biases: `nodes × 4 bytes` (e.g., 26 nodes = 104B)
- Activations: `nodes × 1 byte` (26B)
- Node params: `nodes × 16 bytes` (416B)
- Node values: `nodes × 4 bytes` (104B)
- **Total NN: ~850 bytes per individual**

Per individual for physics (rocket landing):
- 3 rigid bodies × 80 bytes = 240B
- 19 geoms × 32 bytes = 608B
- 2 joints × 64 bytes = 128B
- Contact constraints × ~48 bytes each
- **Total physics: ~1.5 KB per individual**

**Total: ~2.4 KB per individual.** At 4,096 individuals per species: ~10 MB. At 39 species: ~380 MB. Well within GPU memory limits (8-24 GB).

### CPU Mutation Cost

CPU-side mutation is O(individuals × parameters). For 4,096 individuals with 50 weights each:
- Weight jitter: 4,096 × 50 = 204,800 float mutations
- Other mutations: similar scale
- Deep copy for offspring: 4,096 × ~2.4 KB = ~10 MB memcpy

This completes in <10ms on a modern CPU — negligible compared to GPU evaluation time (which runs hundreds of physics steps).

## Multi-Species Evaluation

With 39 species evaluated sequentially, even at 2,048 individuals per species, only one species uses the GPU at a time. Options to improve:

### Option 1: Fewer, Larger Species (Simplest)

Instead of 39 × 132 = 5,148 individuals, use 10 × 2,048 = 20,480 individuals. Fewer species means fewer sequential launches, and each launch saturates the GPU better.

Trade-off: less topological diversity (fewer species exploring different architectures).

### Option 2: Multi-Stream Overlap (No Code Change to Kernels)

Run 4 species concurrently on 4 CUDA streams. The GPU scheduler interleaves warps from different streams, filling idle cores.

```csharp
var streams = new AcceleratorStream[4];
for (int i = 0; i < species.Count; i++)
{
    var stream = streams[i % 4];
    LaunchEvaluation(species[i], stream);  // non-blocking
}
accelerator.Synchronize();  // wait for all
```

No warp divergence (each stream runs the same-topology kernel). ILGPU supports this but kernel loading needs explicit stream parameters — currently uses `LoadAutoGroupedStreamKernel` which binds to default stream.

### Option 3: Sort Species by Topology Similarity

If species are sorted so that adjacent species have similar edge/node counts, multi-stream overlap gets better SM utilization (similar-sized kernels finish around the same time, avoiding tail effects).

## Priority of Changes

| Change | Effort | Impact | Dependencies |
|--------|--------|--------|-------------|
| Increase individuals per species (config change) | None | Very High | Verify memory fits |
| Decrease species count proportionally | None | High | Config tuning |
| CPU/GPU pipeline overlap | Medium | High | Async upload/download |
| Multi-stream species evaluation | Medium | Medium-High | ILGPU stream API |
| Cache topology uploads between generations | Low | Low-Medium | Hash or dirty flag |

### Step 1: Scale Up (Config Only)

Change population config from 39×132 to something like 12×1024. Measure wall-clock time per generation. The per-generation time should barely change despite 2.4x more individuals.

### Step 2: Overlap CPU/GPU

Structure the main evolution loop so CPU mutation happens during GPU evaluation. This requires:
- Async kernel launch (already how ILGPU works — launches are non-blocking)
- Explicit sync only when downloading fitness
- Pre-packing weight upload buffers while GPU runs

### Step 3: Multi-Stream (If Needed)

Only pursue if Step 1 shows that sequential species evaluation is still the bottleneck after scaling up individuals per species.

## Benchmark Tests

### Test 1: Population Size Sweep (No Code Changes)

Run the existing rocket landing evaluator with different population configurations, holding total individuals roughly constant:

| Config | Species | Individuals/Species | Total | Expected time |
|--------|---------|-------------------|-------|---------------|
| A | 39 | 132 | 5,148 | Baseline |
| B | 20 | 256 | 5,120 | ~Same |
| C | 10 | 512 | 5,120 | ~Same |
| D | 5 | 1,024 | 5,120 | ~Same or faster |
| E | 39 | 512 | 19,968 | ~Same as A (free scaling) |
| F | 39 | 1,024 | 39,936 | Slightly slower than A |
| G | 39 | 2,048 | 79,872 | ~2x A |

Config E is the key test: 4x more individuals for (ideally) similar wall-clock time.

### Test 2: GPU Saturation Point

Hold species count at 1. Sweep individuals from 128 to 16,384. Find the knee where wall-clock time starts increasing linearly — that's the GPU saturation point for the rocket evaluation workload.

### Test 3: CPU/GPU Overlap Measurement

Measure:
- Time for GPU evaluation alone (current)
- Time for CPU mutation alone
- Time with overlap (launch GPU async, mutate on CPU, sync)

Expected: overlap is nearly free if CPU mutation << GPU evaluation.

### Test 4: Sequential vs Multi-Stream Species Evaluation

At 39×512, compare:
- Sequential species evaluation (current)
- 4-stream overlapped species evaluation

Measure wall-clock speedup.
