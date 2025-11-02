# Evolvion: Evolutionary Neural Network Framework

NEAT-style evolutionary algorithm for evolving fixed-topology neural networks.

## Status

**✅ Complete and optimized** - Optuna hyperparameter search complete, solves spiral classification in 3 generations.

## Architecture

### Species-Based Evolution
- **Species** = fixed network topology shared by all individuals
- **Individuals** = unique weight and node parameter sets
- **Population** = multiple species evolving in parallel with adaptive culling

### Network Structure
- **Layered feedforward**: Input → Hidden (multiple rows) → Output
- **Row-synchronous evaluation**: Each row computed after all previous rows complete
- **Acyclic by construction**: Connections only from earlier rows to later rows
- **Fixed topology per species**: Topology mutations create new species

### Key Features
1. **Aggressive selection**: Large populations (2,000+) with strong tournament selection
2. **High mutation rates**: 97% weight jitter for broad exploration
3. **Adaptive species culling**: Remove stagnant/underperforming species
4. **Weak edge pruning**: Automatic simplification of networks during evolution
5. **Deterministic**: Seed-controlled RNG for reproducibility

## Core Components

```
Evolvatron.Evolvion/
├── SpeciesSpec.cs           # Topology definition
├── Individual.cs            # Weight/parameter container
├── Population.cs            # Multi-species management
├── Evolver.cs               # Main evolution loop
├── EvolutionConfig.cs       # Optuna-optimized hyperparameters
├── Mutation/
│   ├── MutationOperators.cs    # Weight mutations
│   ├── EdgeMutations.cs        # Topology mutations
│   └── WeakEdgePruning.cs      # Automatic simplification
├── Selection/
│   ├── Selection.cs            # Tournament selection
│   └── Elitism.cs              # Elite preservation
├── Culling/
│   ├── SpeciesCuller.cs        # Adaptive species removal
│   └── StagnationTracker.cs    # Performance tracking
├── Evaluation/
│   ├── SimpleFitnessEvaluator.cs  # Fitness evaluation
│   └── CPUEvaluator.cs            # Network forward pass
├── Environments/
│   ├── SpiralEnvironment.cs       # ✅ Solved
│   ├── CartPoleEnvironment.cs     # 🔄 In progress
│   └── FollowTheCorridorEnvironment.cs
├── Benchmarks/
│   ├── LandscapeNavigationTask.cs
│   └── OptimizationLandscapes.cs  # Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel
└── SpeciesBuilder.cs         # Fluent API for topology construction

```

## Configuration

Default config in `EvolutionConfig.cs` contains Optuna-optimized values (Trial 23):

```csharp
SpeciesCount = 27
IndividualsPerSpecies = 88
MinSpeciesCount = 8
Elites = 4
TournamentSize = 22
ParentPoolPercentage = 0.75f
WeightJitter = 0.97f
WeightJitterStdDev = 0.40f
// ... see file for complete config
```

## Mutation Operators

### Weight Mutations
- **Weight Jitter**: Gaussian noise (97% application rate)
- **Weight Reset**: Random reinitialization (13.7%)
- **L1 Shrink**: Regularization toward zero (9%)
- **Activation Swap**: Change activation functions (18.6%)

### Topology Mutations
- **Edge Add**: Add connections (1.6%)
- **Edge Delete**: Remove connections (0.4%)
- **Edge Split**: Insert intermediate nodes (4.3%)
- **Edge Redirect**: Rewire connections (9.3%)
- **Edge Swap**: Swap connection endpoints (2.9%)
- **Weak Edge Pruning**: Auto-prune weak edges during evolution (79.9% of |weight| < 0.016)

## Usage

### Create and Evolve Population

```csharp
// Build topology
var topology = new SpeciesBuilder()
    .AddInputRow(2)
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid)
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh)
    .AddOutputRow(1, ActivationType.Tanh)
    .WithMaxInDegree(10)
    .InitializeDense(random, density: 0.3f)
    .Build();

// Initialize
var config = new EvolutionConfig(); // Uses optimized defaults
var evolver = new Evolver(seed: 42);
var population = evolver.InitializePopulation(config, topology);
var environment = new SpiralEnvironment();
var evaluator = new SimpleFitnessEvaluator();

// Evolution loop
for (int gen = 0; gen < maxGenerations; gen++)
{
    evaluator.EvaluatePopulation(population, environment, seed: gen);
    evolver.StepGeneration(population);

    var stats = population.GetStatistics();
    Console.WriteLine($"Gen {gen}: Best={stats.BestFitness:F4}");
}
```

## Benchmark Results

### Spiral Classification (2→8→8→1)
**Status**: ✅ **SOLVED**
- **Solve time**: 3 generations
- **Evaluations**: 7,128 (3 gens × 2,376 individuals)
- **Success rate**: 100% (all 15 seeds tested)
- **Deterministic**: Verified with multiple runs

### Landscape Navigation
**Status**: 🔄 In progress

Available landscapes:
- Sphere (easy)
- Rosenbrock (medium)
- Rastrigin (hard - multimodal)
- Ackley (hard - nearly flat)
- Schwefel (very hard - deceptive)

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

Optimization tooling:
- `optuna_sweep.py` - Bayesian optimization orchestrator
- `Evolvatron.OptunaEval/` - C# evaluation CLI
- `optuna_best_params.txt` - Best trial parameters

## Future Plans

- [ ] Multi-objective optimization (fitness vs complexity)
- [ ] Crossover within species
- [ ] Adaptive mutation rates based on stagnation
- [ ] Novelty search / behavioral diversity
- [ ] GPU acceleration (ILGPU) for massive parallelization
