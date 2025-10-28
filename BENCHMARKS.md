# Evolvion Benchmarks

This document describes the benchmark suite for evaluating and tuning Evolvion hyperparameters.

## Overview

The Evolvion benchmark suite provides fast, deterministic optimization tasks for evaluating:
- Mutation operators and rates
- Initialization strategies
- Speciation parameters
- Selection pressure
- Complexity-based mutation schedules

These benchmarks are designed to be **orders of magnitude faster** than RL tasks while still capturing key challenges:
- Temporal dynamics (navigation over multiple timesteps)
- Recurrent network capabilities
- Exploration vs exploitation tradeoffs
- Credit assignment over time
- Multimodal optimization landscapes

## Why Landscape Navigation?

Traditional function optimization (Rastrigin, Rosenbrock, etc.) is fast but doesn't test temporal dynamics or recurrent network behavior. RL tasks test these but are too slow for hyperparameter sweeps.

**Landscape Navigation** bridges this gap:
- Network receives current position (or gradient)
- Network outputs movement direction
- Episode runs for T timesteps
- Fitness = final distance to optimum
- Evaluation takes microseconds, not milliseconds

This tests:
- Whether networks can integrate information over time
- Whether mutations preserve useful behaviors
- Whether initialization strategies create viable starting topologies

## Benchmark Landscapes

### Sphere (Convex, Unimodal)
```
f(x) = Σ x_i²
```
- **Minimum**: 0 at origin
- **Purpose**: Tests basic gradient descent capability
- **Difficulty**: Easy

### Rosenbrock (Narrow valley)
```
f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
```
- **Minimum**: 0 at (1, 1, ..., 1)
- **Purpose**: Tests fine-grained control and patience
- **Difficulty**: Medium (easy to approach, hard to converge)

### Rastrigin (Highly multimodal)
```
f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
```
- **Minimum**: 0 at origin
- **Purpose**: Tests exploration capability (many local optima)
- **Difficulty**: Hard (exponentially many local minima)

### Ackley (Multimodal with flat regions)
```
f(x) = -20exp(-0.2√(Σx_i²/n)) - exp(Σcos(2πx_i)/n) + 20 + e
```
- **Minimum**: 0 at origin
- **Purpose**: Tests exploration in nearly-flat regions
- **Difficulty**: Medium-Hard

### Schwefel (Deceptive)
```
f(x) = 418.9829n - Σ x_i·sin(√|x_i|)
```
- **Minimum**: 0 at (420.9687, ..., 420.9687)
- **Purpose**: Tests resistance to deception (optimum far from origin)
- **Difficulty**: Hard

## Observation Types

### FullPosition
Network sees: `[x_1, x_2, ..., x_n]`

Tests whether network can learn gradient-free optimization.

### GradientOnly
Network sees: `[∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]`

Tests whether network can learn to follow gradients and build momentum.

### PartialObservability
Network sees: `[x_1, ..., x_n, ∂f/∂x_1, ..., ∂f/∂x_n]`

Tests whether network can integrate multiple information sources.

## Running Benchmarks

### Important: Benchmarks Don't Run By Default

Benchmarks are marked with `[Category("Benchmark")]` and `[Explicit]` to prevent them from running during normal test execution. This ensures they don't slow down regression testing.

### Run All Benchmarks
```bash
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj --filter "Category=Benchmark"
```

### Run By Difficulty
```bash
# Easy benchmarks (5-10 seconds)
dotnet test --filter "FullyQualifiedName~RunEasyBenchmark"

# Medium benchmarks (30-60 seconds)
dotnet test --filter "FullyQualifiedName~RunMediumBenchmark"

# Hard benchmarks (2-5 minutes)
dotnet test --filter "FullyQualifiedName~RunHardBenchmark"
```

### Run Specific Benchmark
```bash
dotnet test --filter "FullyQualifiedName~Sphere-5D-Easy"
```

### Verify Landscape Functions
```bash
dotnet test --filter "FullyQualifiedName~VerifyLandscapeFunctions"
```

## Interpreting Results

### Baseline Performance
A **random policy** (outputs random actions) provides the baseline. Any evolved network should significantly outperform this.

Expected random policy scores (lower = better):
- **Sphere-5D-Easy**: ~10 to ~100 (highly variable)
- **Rastrigin-8D-Medium**: ~50 to ~150
- **Rosenbrock-10D**: ~1000 to ~10000

### Success Criteria
A well-tuned evolutionary run should:
1. Beat random policy by 10x-100x
2. Show consistent improvement over generations
3. Discover networks that reach near-optimal values (<1.0 for most landscapes)

### Comparing Configurations
When comparing hyperparameter configurations:
1. Run each config 5-10 times with different seeds
2. Compare **median final fitness** (robust to outliers)
3. Track **generations to threshold** (e.g., how long to reach fitness < 10)
4. Monitor **solution diversity** (are multiple strategies discovered?)

## Benchmark Suite Structure

### Easy (Quick validation - 5-10 seconds each)
- `Sphere-5D-Easy`: 5D, 50 timesteps, full position
- `Rosenbrock-5D-Easy`: 5D, 100 timesteps, full position

Use for quick sanity checks and rapid iteration.

### Medium (Hyperparameter tuning - 30-60 seconds each)
- `Rastrigin-8D-Medium`: 8D, 100 timesteps, full position (multimodal)
- `Ackley-8D-Medium`: 8D, 100 timesteps, full position (flat regions)
- `Rosenbrock-10D-GradientOnly`: 10D, 150 timesteps, gradient observations

Use for comparing mutation operators, initialization strategies, and selection methods.

### Hard (Final validation - 2-5 minutes each)
- `Rastrigin-15D-Hard`: 15D, 200 timesteps, full position (high-dimensional)
- `Schwefel-12D-Hard`: 12D, 200 timesteps, full position (deceptive)
- `Ackley-15D-PartialObs`: 15D, 250 timesteps, partial observability (integration)

Use for final validation before committing hyperparameter changes.

## Creating Custom Benchmarks

### Example: Add New Landscape
```csharp
// In OptimizationLandscapes.cs
public static float MyCustomLandscape(float[] x)
{
    float sum = 0f;
    for (int i = 0; i < x.Length; i++)
    {
        sum += /* your function here */;
    }
    return sum;
}

// In LandscapeNavigationBenchmarks.cs
yield return new BenchmarkConfig(
    "MyLandscape-10D-Custom",
    OptimizationLandscapes.MyCustomLandscape,
    dimensions: 10,
    timesteps: 100,
    minBound: -5f,
    maxBound: 5f,
    observationType: ObservationType.FullPosition);
```

### Example: Using in Evolution Code
```csharp
var task = new LandscapeNavigationTask(
    OptimizationLandscapes.Rastrigin,
    dimensions: 8,
    timesteps: 100,
    seed: 42);

// Your policy must be Func<float[], float[]>
// where input size = dimensions (or 2*dimensions for PartialObservability)
// and output size = dimensions
var fitness = task.Evaluate(network.Activate);
```

## Performance Characteristics

Measured on typical development machine:

| Benchmark | Dimensions | Timesteps | Time per Eval |
|-----------|------------|-----------|---------------|
| Sphere-5D | 5 | 50 | ~5 µs |
| Rastrigin-8D | 8 | 100 | ~15 µs |
| Rosenbrock-10D | 10 | 150 | ~25 µs |
| Rastrigin-15D | 15 | 200 | ~50 µs |

For comparison:
- RL CartPole episode: ~5 ms (1000x slower)
- RL Corridor episode: ~20 ms (4000x slower)

This speed enables:
- **Hyperparameter sweeps** with 100+ configurations
- **Ablation studies** with statistical significance
- **Continuous integration** benchmarks without infrastructure burden

## Recommended Workflow

### 1. Quick Validation
Run easy benchmarks after any code changes:
```bash
dotnet test --filter "FullyQualifiedName~RunEasyBenchmark"
```

### 2. Hyperparameter Tuning
Run medium benchmarks when comparing configurations:
```bash
for i in {1..10}; do
  dotnet test --filter "FullyQualifiedName~RunMediumBenchmark" --logger "console;verbosity=minimal"
done
```

### 3. Final Validation
Run hard benchmarks before committing major changes:
```bash
dotnet test --filter "FullyQualifiedName~RunHardBenchmark"
```

### 4. Integration with CI
Add to CI pipeline (runs only on explicit trigger, not every commit):
```yaml
benchmark-job:
  when: manual
  script:
    - dotnet test --filter "Category=Benchmark"
```

## Future Extensions

Potential additions to the benchmark suite:

1. **Time-series prediction** (test recurrent memory directly)
2. **Sequence classification** (test pattern recognition)
3. **Dynamic landscapes** (optimum moves over time)
4. **Noisy observations** (test robustness)
5. **Constrained optimization** (test handling of invalid regions)
6. **Multi-objective** (test Pareto front discovery)

## Questions?

See `LandscapeNavigationTask.cs` for implementation details.
See `LandscapeNavigationBenchmarks.cs` for example usage.
See `OptimizationLandscapes.cs` for landscape function definitions.
