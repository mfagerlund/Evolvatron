# Evolvion GPU Execution Investigation

## Current State

Evolvion is a NEAT-style evolutionary neural network framework designed for large-scale parallel evolution. Currently, it has:

### Existing Components
- **CPUEvaluator** (`CPUEvaluator.cs`): Single-threaded neural network forward pass evaluator
- **Row-based execution plan** (`RowPlan.cs`): Optimized layer-by-layer execution strategy
- **Species-based topology** (`SpeciesSpec.cs`): Fixed topology per species, individuals differ only in weights/biases
- **Fitness evaluation** (`SimpleFitnessEvaluator.cs`): CPU-based fitness computation using environments
- **Population management** (`Evolver.cs`, `Population.cs`): Evolution loop, selection, mutation

### Key Architecture Features
- **Structure of Arrays (SoA)** storage in `Individual` struct:
  - `float[] Weights` - per-edge weights
  - `float[] Biases` - per-node biases
  - `float[] NodeParams` - per-node activation parameters (4 floats each)
  - `ActivationType[] Activations` - per-node activation functions
- **Acyclic feed-forward topology**: Edges sorted by (destRow, destNode) for coalesced access
- **11 activation types**: Linear, Tanh, Sigmoid, ReLU, LeakyReLU, ELU, Softsign, Softplus, Sin, Gaussian, GELU

### Missing GPU Support
Currently, ALL evaluation happens on CPU in a single-threaded loop:
1. For each species
2. For each individual in species
3. For each episode
4. For each timestep: CPUEvaluator.Evaluate() → environment.Step()

This is the bottleneck preventing large-scale evolution.

## Why GPU Execution is Fundamental

The original design intent was **massively parallel fitness evaluation**:
- Evaluate 1000s of individuals simultaneously across species
- Each individual runs multiple environment episodes in parallel
- GPU can handle 10,000+ neural network forward passes per frame

Without GPU:
- 10 species × 100 individuals × 100 timesteps × 10 episodes = 1M forward passes per generation
- At ~0.1ms per CPU forward pass = **100 seconds per generation**
- With GPU: Could achieve **1-2 seconds per generation** (50-100x speedup)

## Implementation Strategy

### Option 1: ILGPU-Based Implementation (RECOMMENDED)

Follow the pattern established by Evolvatron.Rigidon's GPUStepper, which uses **ILGPU 1.5.3**.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ CPU: Evolution Loop (Evolver.cs)                        │
│  - Selection                                            │
│  - Mutation                                             │
│  - Species management                                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ GPU: Batch Fitness Evaluation (GPUEvaluator)            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Phase 1: Upload to GPU                            │  │
│  │  - Species topologies (edges, row plans)          │  │
│  │  - Individual parameters (weights, biases, etc.)  │  │
│  │  - Environment states (observations, positions)   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Phase 2: Parallel Evaluation Kernels              │  │
│  │  Kernel 1: Initialize all episodes                │  │
│  │  Kernel 2: Per-timestep loop:                     │  │
│  │    - Run NN forward passes (all individuals)      │  │
│  │    - Step environments (all episodes)             │  │
│  │    - Accumulate rewards                           │  │
│  │  Kernel 3: Finalize fitness values                │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Phase 3: Download from GPU                        │  │
│  │  - Fitness values for all individuals             │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Required New Components

##### 1. GPU Data Structures (`GPU/GPUDataStructures.cs`)

```csharp
// GPU-compatible edge representation
public struct GPUEdge
{
    public int Source;
    public int Dest;
}

// GPU-compatible row plan (already mostly compatible)
public struct GPURowPlan
{
    public int NodeStart;
    public int NodeCount;
    public int EdgeStart;
    public int EdgeCount;
}

// GPU individual representation (SoA layout)
public struct GPUIndividualBatch
{
    // All individuals' data in contiguous arrays
    public MemoryBuffer1D<float, Stride1D.Dense> Weights;
    public MemoryBuffer1D<float, Stride1D.Dense> Biases;
    public MemoryBuffer1D<float, Stride1D.Dense> NodeParams;
    public MemoryBuffer1D<byte, Stride1D.Dense> Activations; // ActivationType as byte

    // Metadata
    public int IndividualCount;
    public int WeightsPerIndividual;
    public int NodesPerIndividual;
}

// GPU environment state (SoA for all episodes)
public struct GPUEnvironmentBatch
{
    public MemoryBuffer1D<float, Stride1D.Dense> Observations;
    public MemoryBuffer1D<float, Stride1D.Dense> Positions;
    public MemoryBuffer1D<float, Stride1D.Dense> Rewards;
    public MemoryBuffer1D<int, Stride1D.Dense> Steps;
    public MemoryBuffer1D<byte, Stride1D.Dense> IsTerminal;

    public int EpisodeCount;
    public int ObservationSize;
}
```

##### 2. GPU World State Manager (`GPU/GPUEvolvionState.cs`)

Manages GPU memory allocation and transfers, similar to `GPUWorldState.cs` in Rigidon:

```csharp
public class GPUEvolvionState : IDisposable
{
    private readonly Accelerator _accelerator;

    // Topology buffers (read-only during evaluation)
    public MemoryBuffer1D<GPUEdge, Stride1D.Dense> Edges;
    public MemoryBuffer1D<GPURowPlan, Stride1D.Dense> RowPlans;

    // Individual parameter buffers
    public GPUIndividualBatch Individuals;

    // Working buffers for NN evaluation
    public MemoryBuffer1D<float, Stride1D.Dense> NodeValues; // [IndividualCount × NodesPerIndividual]

    // Environment state buffers
    public GPUEnvironmentBatch Environments;

    // Results
    public MemoryBuffer1D<float, Stride1D.Dense> FitnessValues;

    public void UploadTopology(SpeciesSpec spec);
    public void UploadIndividuals(List<Individual> individuals);
    public void UploadEnvironmentConfig(IEnvironment envTemplate);
    public void DownloadFitness(List<Individual> individuals);
}
```

##### 3. GPU Kernels (`GPU/GPUEvolvionKernels.cs`)

Core computation kernels:

```csharp
public static class GPUEvolvionKernels
{
    // Kernel 1: Evaluate a single row for all individuals in parallel
    public static void EvaluateRowKernel(
        Index1D index,  // Thread index = individual_index
        ArrayView<float> nodeValues,  // [IndividualCount × NodesPerIndividual]
        ArrayView<GPUEdge> edges,
        ArrayView<float> weights,  // [IndividualCount × EdgeCount]
        ArrayView<float> biases,
        ArrayView<byte> activations,
        ArrayView<float> nodeParams,
        GPURowPlan rowPlan,
        int individualCount,
        int nodesPerIndividual,
        int weightsPerIndividual)
    {
        int individualIdx = index;
        if (individualIdx >= individualCount) return;

        int nodeOffset = individualIdx * nodesPerIndividual;
        int weightOffset = individualIdx * weightsPerIndividual;

        // Initialize row nodes to zero
        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            nodeValues[nodeOffset + rowPlan.NodeStart + i] = 0.0f;
        }

        // Accumulate weighted inputs
        for (int edgeIdx = rowPlan.EdgeStart; edgeIdx < rowPlan.EdgeStart + rowPlan.EdgeCount; edgeIdx++)
        {
            var edge = edges[edgeIdx];
            float weight = weights[weightOffset + edgeIdx];
            float sourceValue = nodeValues[nodeOffset + edge.Source];

            nodeValues[nodeOffset + edge.Dest] += weight * sourceValue;
        }

        // Add biases
        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            nodeValues[nodeOffset + nodeIdx] += biases[nodeOffset + nodeIdx];
        }

        // Apply activations
        for (int i = 0; i < rowPlan.NodeCount; i++)
        {
            int nodeIdx = rowPlan.NodeStart + i;
            int globalNodeIdx = nodeOffset + nodeIdx;
            float preActivation = nodeValues[globalNodeIdx];
            byte activationType = activations[nodeOffset + nodeIdx];

            // Get node params (4 floats per node)
            int paramOffset = (nodeOffset + nodeIdx) * 4;
            float param0 = nodeParams[paramOffset];
            float param1 = nodeParams[paramOffset + 1];
            float param2 = nodeParams[paramOffset + 2];
            float param3 = nodeParams[paramOffset + 3];

            nodeValues[globalNodeIdx] = EvaluateActivation(
                activationType, preActivation, param0, param1, param2, param3);
        }
    }

    // Kernel 2: Batch environment step
    public static void StepEnvironmentKernel(
        Index1D index,  // Thread index = episode_index
        ArrayView<float> observations,
        ArrayView<float> actions,  // Output from NN
        ArrayView<float> positions,
        ArrayView<float> rewards,
        ArrayView<int> steps,
        ArrayView<byte> isTerminal,
        int observationSize,
        int actionSize,
        float stepSize,
        float minBound,
        float maxBound)
    {
        // Example: Landscape navigation environment
        // Each thread handles one episode
        int episodeIdx = index;
        int obsOffset = episodeIdx * observationSize;
        int actionOffset = episodeIdx * actionSize;
        int posOffset = episodeIdx * actionSize;  // Position dimensions = action dimensions

        if (isTerminal[episodeIdx] != 0) return;  // Skip if terminal

        // Apply actions to position
        for (int i = 0; i < actionSize; i++)
        {
            positions[posOffset + i] += actions[actionOffset + i] * stepSize;
            // Clamp to bounds
            positions[posOffset + i] = XMath.Clamp(positions[posOffset + i], minBound, maxBound);
        }

        // Update step counter
        steps[episodeIdx]++;

        // Check terminal condition
        // (Terminal reward computed in finalization kernel)
    }

    // Kernel 3: Finalize fitness
    public static void FinalizeFitnessKernel(
        Index1D index,  // Thread index = individual_index
        ArrayView<float> fitnessValues,
        ArrayView<float> positions,
        ArrayView<float> rewards,
        ArrayView<byte> isTerminal,
        int episodesPerIndividual,
        int dimensionsPerEpisode)
    {
        // Average fitness across multiple episodes per individual
        int individualIdx = index;
        int episodeStart = individualIdx * episodesPerIndividual;

        float totalFitness = 0.0f;
        for (int ep = 0; ep < episodesPerIndividual; ep++)
        {
            int episodeIdx = episodeStart + ep;

            // Compute final fitness from terminal state
            // Example: negative landscape value
            int posOffset = episodeIdx * dimensionsPerEpisode;
            float landscapeValue = ComputeLandscapeValue(positions, posOffset, dimensionsPerEpisode);
            totalFitness += -landscapeValue;
        }

        fitnessValues[individualIdx] = totalFitness / episodesPerIndividual;
    }

    // Helper: Activation function evaluation on GPU
    private static float EvaluateActivation(
        byte activationType,
        float x,
        float param0,
        float param1,
        float param2,
        float param3)
    {
        switch (activationType)
        {
            case 0: return x;  // Linear
            case 1: return XMath.Tanh(x);  // Tanh
            case 2: return 1.0f / (1.0f + XMath.Exp(-x));  // Sigmoid
            case 3: return XMath.Max(0.0f, x);  // ReLU
            case 4: return x > 0 ? x : param0 * x;  // LeakyReLU
            case 5: return x > 0 ? x : param0 * (XMath.Exp(x) - 1.0f);  // ELU
            case 6: return x / (1.0f + XMath.Abs(x));  // Softsign
            case 7: return XMath.Log(1.0f + XMath.Exp(x));  // Softplus
            case 8: return XMath.Sin(x);  // Sin
            case 9: return XMath.Exp(-x * x);  // Gaussian
            case 10:  // GELU
                return x * 0.5f * (1.0f + XMath.Tanh(
                    XMath.Sqrt(2.0f / XMath.PI) * (x + 0.044715f * x * x * x)));
            default: return x;
        }
    }

    private static float ComputeLandscapeValue(
        ArrayView<float> positions,
        int offset,
        int dimensions)
    {
        // Example: Sphere function
        float sum = 0.0f;
        for (int i = 0; i < dimensions; i++)
        {
            float val = positions[offset + i];
            sum += val * val;
        }
        return sum;
    }
}
```

##### 4. GPU Evaluator (`GPU/GPUEvaluator.cs`)

High-level interface, similar to CPUEvaluator:

```csharp
public class GPUEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private GPUEvolvionState _gpuState;

    // Compiled kernels
    private Action<Index1D, ...> _evaluateRowKernel;
    private Action<Index1D, ...> _stepEnvironmentKernel;
    private Action<Index1D, ...> _finalizeFitnessKernel;

    public GPUEvaluator(int maxIndividuals, int maxNodes, int maxEdges)
    {
        _context = Context.CreateDefault();
        _accelerator = _context.GetPreferredDevice(preferCPU: false)
            .CreateAccelerator(_context);

        // Load kernels
        _evaluateRowKernel = _accelerator.LoadAutoGroupedStreamKernel<...>(
            GPUEvolvionKernels.EvaluateRowKernel);
        // ... load other kernels

        _gpuState = new GPUEvolvionState(_accelerator, maxIndividuals, maxNodes, maxEdges);
    }

    // Batch evaluate all individuals in a species
    public void EvaluateSpecies(
        Species species,
        IEnvironment environmentTemplate,
        int episodesPerIndividual,
        int seed)
    {
        // 1. Upload topology (once per species)
        _gpuState.UploadTopology(species.Topology);

        // 2. Upload all individuals
        _gpuState.UploadIndividuals(species.Individuals);

        // 3. Initialize environments
        _gpuState.InitializeEnvironments(
            species.Individuals.Count,
            episodesPerIndividual,
            environmentTemplate,
            seed);

        // 4. Run episodes
        int maxSteps = environmentTemplate.MaxSteps;
        for (int step = 0; step < maxSteps; step++)
        {
            // 4a. Evaluate all neural networks in parallel
            for (int rowIdx = 1; rowIdx < species.Topology.RowPlans.Length; rowIdx++)
            {
                _evaluateRowKernel(
                    species.Individuals.Count,
                    /* pass GPU buffers */);
                _accelerator.Synchronize();
            }

            // 4b. Step all environments in parallel
            _stepEnvironmentKernel(
                species.Individuals.Count * episodesPerIndividual,
                /* pass GPU buffers */);
            _accelerator.Synchronize();
        }

        // 5. Finalize fitness values
        _finalizeFitnessKernel(species.Individuals.Count, /* ... */);
        _accelerator.Synchronize();

        // 6. Download results
        _gpuState.DownloadFitness(species.Individuals);
    }

    public void Dispose()
    {
        _gpuState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
```

##### 5. Fitness Evaluator Integration (`GPUFitnessEvaluator.cs`)

Replaces SimpleFitnessEvaluator:

```csharp
public class GPUFitnessEvaluator : IDisposable
{
    private readonly GPUEvaluator _gpuEvaluator;

    public GPUFitnessEvaluator(int maxIndividuals, int maxNodes, int maxEdges)
    {
        _gpuEvaluator = new GPUEvaluator(maxIndividuals, maxNodes, maxEdges);
    }

    public void EvaluatePopulation(
        Population population,
        IEnvironment environment,
        int episodesPerIndividual = 5,
        int seed = 0)
    {
        foreach (var species in population.AllSpecies)
        {
            _gpuEvaluator.EvaluateSpecies(
                species,
                environment,
                episodesPerIndividual,
                seed);
        }
    }

    public void Dispose()
    {
        _gpuEvaluator?.Dispose();
    }
}
```

### Implementation Phases

#### Phase 1: Infrastructure (1-2 days)
- [ ] Add ILGPU package reference to Evolvatron.Evolvion.csproj
- [ ] Create `GPU/` folder structure
- [ ] Implement `GPUDataStructures.cs` with ILGPU-compatible structs
- [ ] Implement basic `GPUEvolvionState.cs` with memory allocation

#### Phase 2: Core NN Evaluation Kernels (2-3 days)
- [ ] Implement `EvaluateRowKernel` for parallel row evaluation
- [ ] Implement activation function evaluation on GPU
- [ ] Test single-row evaluation against CPUEvaluator
- [ ] Add unit tests for GPU vs CPU parity

#### Phase 3: Environment Integration (2-3 days)
- [ ] Implement simple environment kernel (e.g., LandscapeEnvironment)
- [ ] Implement `StepEnvironmentKernel`
- [ ] Implement `FinalizeFitnessKernel`
- [ ] Test end-to-end single-species batch evaluation

#### Phase 4: Full Integration (1-2 days)
- [ ] Implement `GPUEvaluator` with full pipeline
- [ ] Implement `GPUFitnessEvaluator`
- [ ] Update `Evolver.cs` to use GPU evaluator
- [ ] Add CPU/GPU toggle flag in EvolutionConfig

#### Phase 5: Optimization & Testing (2-3 days)
- [ ] Profile GPU memory usage and kernel performance
- [ ] Optimize memory transfers (minimize CPU ↔ GPU copies)
- [ ] Implement multi-stream execution for overlapping compute/transfer
- [ ] Benchmark GPU vs CPU speedup
- [ ] Add determinism tests

#### Phase 6: Additional Environments (ongoing)
- [ ] Port CartPoleEnvironment to GPU
- [ ] Port SpiralEnvironment to GPU
- [ ] Port other existing environments
- [ ] Create generic environment kernel template

**Total estimated time: 8-13 days**

## Key Challenges & Solutions

### Challenge 1: Environment State Complexity
**Problem**: IEnvironment interface is designed for CPU objects with methods
**Solution**:
- Create GPU-compatible environment state structs
- Implement environment logic directly in kernels
- For complex environments, may need hybrid CPU/GPU approach

### Challenge 2: Dynamic Memory Allocation
**Problem**: GPU kernels can't allocate dynamic memory
**Solution**:
- Pre-allocate max-sized buffers for all episodes
- Use fixed-size arrays within kernels
- Pool memory across generations

### Challenge 3: Debugging GPU Code
**Problem**: Debugging GPU kernels is difficult
**Solution**:
- Start with small test cases (1-2 individuals, 5-10 nodes)
- Use ILGPU CPU accelerator for debugging
- Implement extensive CPU/GPU parity tests
- Add GPU→CPU result dumps for inspection

### Challenge 4: Memory Bandwidth
**Problem**: Uploading/downloading data every generation is expensive
**Solution**:
- Keep topology data on GPU permanently (only upload once)
- Only upload changed individuals (after mutation)
- Use pinned memory for faster transfers
- Implement double-buffering

### Challenge 5: Kernel Launch Overhead
**Problem**: Launching many small kernels has overhead
**Solution**:
- Batch operations where possible
- Use persistent kernels for timestep loops
- Minimize synchronization points

## Expected Performance Gains

### Current CPU Performance (estimated)
- **Per forward pass**: ~0.1 ms (10-node network, single-threaded)
- **Per episode**: ~10 ms (100 timesteps)
- **Per generation**:
  - 10 species × 100 individuals × 10 episodes = 10,000 episodes
  - 10,000 episodes × 10 ms = **100 seconds**

### GPU Performance (projected)
- **Per forward pass batch**: ~0.01 ms (1000 individuals in parallel)
- **Per episode batch**: ~1 ms (all episodes in parallel)
- **Per generation**:
  - All 10,000 episodes in parallel batches
  - ~100 batches × 1 ms = **0.1-1 seconds**

**Projected speedup: 100-1000x**

This would enable:
- Real-time evolution visualization
- Hyperparameter sweeps (run 100+ experiments overnight)
- Larger populations (10,000+ individuals)
- Longer episodes (1000+ timesteps)

## Alternative Approaches

### Option 2: ComputeSharp (Not Recommended)
- Modern C# GPU library with better syntax
- Pros: More C#-like, easier debugging
- Cons: Newer, less mature than ILGPU, DirectX 12 only (no CUDA)

### Option 3: Hybrid CPU/GPU (Interim Solution)
- Keep NN evaluation on CPU
- Move only environment stepping to GPU
- Pros: Easier to implement
- Cons: Limited speedup (~5-10x), still bandwidth-bound

### Option 4: External Python/PyTorch (Not Recommended)
- Use Python subprocess for GPU evaluation
- Pros: Access to mature ML ecosystem
- Cons: Serialization overhead, harder to maintain, slower

## Recommendation

**Proceed with Option 1: ILGPU-based implementation**

Rationale:
1. Consistent with existing Evolvatron.Rigidon architecture
2. ILGPU 1.5.3 is mature and well-documented
3. Supports CUDA, OpenCL, and CPU fallback
4. Native C# integration, no external dependencies
5. Full control over memory layout and kernel optimization
6. Existing team knowledge from GPUStepper implementation

The architecture is already GPU-friendly:
- SoA data layout ✓
- Fixed topology per species ✓
- No dynamic memory allocation ✓
- Embarrassingly parallel fitness evaluation ✓

This is a natural evolution of the existing design.
