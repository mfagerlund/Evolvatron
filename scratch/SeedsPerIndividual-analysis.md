# SeedsPerIndividual Analysis

## Executive Summary

**Critical Finding**: `SeedsPerIndividual=5` is **declared but not actually used** in the evolution pipeline. The configuration parameter exists in `EvolutionConfig.cs` but is never referenced during evaluation.

- **Code Status**: Dead parameter (declared but never read)
- **Current Behavior**: Each individual is evaluated **exactly once per generation** with a single seed
- **Cost Impact**: No wasted computation (the 5x cost doesn't actually happen)
- **Spiral Environment**: Truly deterministic (zero variance across seeds)
- **Recommendation**: Document this dead parameter or implement actual multi-seed evaluation if needed

---

## Usage Location

### Where SeedsPerIndividual Is Declared
**File**: `C:\Dev\Evolvatron\Evolvatron.Evolvion\EvolutionConfig.cs` (lines 89-92)

```csharp
/// <summary>
/// Number of evaluation seeds per individual.
/// Default: 5
/// </summary>
public int SeedsPerIndividual { get; set; } = 5;
```

### Where SeedsPerIndividual Is NOT Read

**Search Result**: Zero matches for `config.SeedsPerIndividual` or `.SeedsPerIndividual` in the entire codebase.

```bash
$ grep -r "config.SeedsPerIndividual" --include="*.cs"
$ grep -r "\.SeedsPerIndividual" --include="*.cs" | grep -v "= 5"
# No results
```

**Conclusion**: The parameter is **never actually used** by the evolution system.

---

## How Evaluation Actually Works

### Evaluation Entry Point
**File**: `C:\Dev\Evolvatron\Evolvatron.Evolvion\SimpleFitnessEvaluator.cs`

```csharp
public void EvaluatePopulation(
    Population population,
    IEnvironment environment,
    int seed = 0)
{
    foreach (var species in population.AllSpecies)
    {
        for (int i = 0; i < species.Individuals.Count; i++)
        {
            var individual = species.Individuals[i];
            individual.Fitness = Evaluate(individual, species.Topology, environment, seed);
            species.Individuals[i] = individual;
        }
    }
}
```

**Key Point**: Each individual is evaluated **exactly once** with the provided seed (default `seed=0`, or `seed=gen` in tests).

### Actual Usage Pattern (SpiralEvolutionTest)

```csharp
for (int gen = 0; gen < maxGenerations; gen++)
{
    // ONE evaluation per individual per generation
    evaluator.EvaluatePopulation(population, environment, seed: gen);

    var stats = population.GetStatistics();
    // ...
    evolver.StepGeneration(population);
}
```

**What happens**:
- Gen 0: All individuals evaluated with seed=0
- Gen 1: All individuals evaluated with seed=1
- Gen 2: All individuals evaluated with seed=2
- ... etc

**Never happens**: Multiple evaluation passes on the same individual within a generation.

---

## What Seeds Actually Affect (By Environment)

### 1. SpiralEnvironment - **Ignores seed**

```csharp
public void Reset(int seed = 0)
{
    _currentCase = 0;           // Reset to first test case
    _totalError = 0f;
    // seed parameter is completely ignored!
}
```

**Effect**: Same test cases every time, regardless of seed.
**Determinism**: Perfect (variance = 0)

### 2. XOREnvironment - **Ignores seed**

```csharp
public void Reset(int seed = 0)
{
    _currentCase = 0;
    _totalError = 0f;
    // seed parameter is completely ignored!
}
```

**Effect**: Same 4 test cases always: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
**Determinism**: Perfect (variance = 0)

### 3. CartPoleEnvironment - **Uses seed**

```csharp
public void Reset(int seed = 0)
{
    var random = new Random(seed);

    // Randomize initial state based on seed
    _cartPosition = (float)(random.NextDouble() * 0.1 - 0.05);
    _cartSpeed = (float)(random.NextDouble() * 0.1 - 0.05);
    _poleAngle = (float)(random.NextDouble() * 0.1 - 0.05);
    _poleAngleSpeed = (float)(random.NextDouble() * 0.1 - 0.05);
    _steps = 0;
    _terminated = false;
}
```

**Effect**: Different initial state per seed → potentially different trajectory
**Determinism**: Stochastic (variance > 0)
**Use case**: Multi-seed evaluation would make sense here to average across initial conditions

### 4. SimpleCorridorEnvironment - **Uses seed**

```csharp
public void Reset(int seed = 0)
{
    GenerateProceduralTrack(seed);  // Generate different track per seed
    _position = new Vector2(0, 0);
    _heading = 0;
    _speed = 0;
    // ...
}

private void GenerateProceduralTrack(int seed)
{
    var random = new Random(seed);
    // sine wave corridor with random perturbations
    float y1 = 30f * MathF.Sin(x1 / 20f) + (float)(random.NextDouble() - 0.5) * 5f;
    // ...
}
```

**Effect**: Different procedurally-generated track per seed
**Determinism**: Stochastic (variance > 0)
**Use case**: Multi-seed evaluation would provide robustness across procedural variations

### 5. FollowTheCorridorEnvironment - **Ignores seed**

```csharp
public void Reset(int seed)
{
    // Seed not used - SimpleCar track is deterministic
    _car.Reset();
    _currentStep = 0;
    // ...
}
```

**Comment in code**: Explicitly notes that seed is ignored because track is deterministic

---

## Spiral Environment: Determinism Analysis

### Test Case Generation
**SpiralEnvironment.cs (lines 73-95)**: Test cases are generated **once** in the constructor:

```csharp
private static List<(float x, float y, float label)> GenerateSpiralPoints(
    int pointsPerSpiral, float noise)
{
    var points = new List<(float x, float y, float label)>();
    var random = new Random(42);  // FIXED SEED - always 42!

    for (int i = 0; i < pointsPerSpiral; i++)
    {
        float t = i * 4.0f * MathF.PI / pointsPerSpiral;
        float r = t / (4.0f * MathF.PI);

        // Spiral 1 (label = 0)
        float x1 = r * MathF.Cos(t) + noise_term;
        float y1 = r * MathF.Sin(t) + noise_term;
        points.Add((x1, y1, -1f));

        // Spiral 2 (label = 1)
        float x2 = r * MathF.Cos(t + MathF.PI) + noise_term;
        float y2 = r * MathF.Sin(t + MathF.PI) + noise_term;
        points.Add((x2, y2, 1f));
    }
    return points;
}
```

**Key Points**:
1. `Random(42)` uses a fixed seed → identical sequence every time
2. With `noise=0.0f` (default), test cases are **100% identical** across all evaluations
3. `Reset(seed)` doesn't regenerate the test cases - it just resets iteration counters
4. Same individual → Same test cases → **Zero variance**

### Variance Measurement - Predicted

If we were to evaluate the same individual with `SeedsPerIndividual=5`:

```
Seed 0: Fitness = X
Seed 1: Fitness = X
Seed 2: Fitness = X
Seed 3: Fitness = X
Seed 4: Fitness = X
```

**Variance**: 0.0 (exactly zero)

**Proof**:
- Fitness = -average_error over fixed test cases
- Same network input → Same network output
- Same test cases → Same error calculation
- Different seed parameter → **Not used by Reset()**

---

## Cost Analysis

### Current Situation (SeedsPerIndividual=5, but never used)

```
Per generation:
- Population: 8 species × 100 individuals = 800 individuals
- Evaluations per individual: 1 (seed parameter ignored or not looped)
- Total forward passes: 800

Actual computation time: ~C
Cost efficiency: OPTIMAL (accidentally)
```

### If SeedsPerIndividual Were Implemented

```
Per generation (hypothetical):
- Population: 800 individuals
- Evaluations per individual: 5 seeds
- Total forward passes: 4000

Computational cost multiplier: 5x
Expected cost: ~5C

Cost recovery for Spiral: ZERO
  - Spiral has zero variance, so 4 out of 5 evaluations are wasted
  - Only environments with stochasticity benefit (CartPole, Corridor)
```

### Break-Even Analysis

**Which environments should use multi-seed evaluation?**

| Environment | Deterministic? | Multi-seed Benefit | Recommendation |
|---|---|---|---|
| **XOR** | Yes (fixed test cases) | No | `SeedsPerIndividual=1` |
| **Spiral** | Yes (fixed test cases) | No | `SeedsPerIndividual=1` |
| **CartPole** | No (random init state) | Yes | `SeedsPerIndividual=5` useful |
| **SimpleCorridorEnvironment** | No (procedural tracks) | Yes | `SeedsPerIndividual=5` useful |
| **FollowTheCorridorEnvironment** | Yes (notes say so) | No | `SeedsPerIndividual=1` |

---

## Fitness Aggregation Strategy

Even if `SeedsPerIndividual` were used, the config specifies:

```csharp
/// <summary>
/// Fitness aggregation method: "Mean" or "CVaR50".
/// Default: "CVaR50"
/// </summary>
public string FitnessAggregation { get; set; } = "CVaR50";
```

**But this is also never implemented**: The code doesn't read `FitnessAggregation` anywhere.

**What it would do** (if implemented):
- **Mean**: Average fitness across 5 seeds (favors consistent performers)
- **CVaR50**: Conditional Value at Risk (50th percentile = median fitness across seeds)
  - More robust to outliers than mean
  - Avoids lucky runs counting too much
  - Penalizes agents that only work on some seeds

For Spiral with zero variance, both would give identical results.

---

## Key Findings

### 1. Parameter Status
- ✓ `SeedsPerIndividual` is declared in `EvolutionConfig.cs`
- ✗ `SeedsPerIndividual` is never read by evolution code
- ✓ `FitnessAggregation` is declared
- ✗ `FitnessAggregation` is never implemented

### 2. Current Behavior
- Each individual is evaluated **exactly once per generation**
- The seed passed changes per generation (`seed=gen`)
- Seeds are ignored by deterministic environments (XOR, Spiral, FollowTheCorridor)
- Seeds are respected by stochastic environments (CartPole, SimpleCorridorEnvironment)

### 3. Determinism of Spiral
- **Completely deterministic**
- Test cases fixed at construction time with `Random(42)`
- `Reset(seed)` ignores the seed parameter
- Variance across seeds: **0.0**
- A network's fitness is **always identical** on repeated evaluations

### 4. Computational Overhead
- Current: No wasted computation (parameter unused)
- If implemented: Would add 5x evaluation cost for deterministic problems
- Only CartPole and Corridor would benefit from multi-seed robustness

---

## Recommendations

### 1. For SpiralEnvironment (Deterministic)
```csharp
// Current test:
config.SeedsPerIndividual = 5;  // Dead parameter, no effect

// Better: Make this explicit
var config = new EvolutionConfig
{
    // ...
    // No multi-seed evaluation needed for deterministic problem
};

// Or document it:
public class EvolutionConfig
{
    /// <summary>
    /// Number of evaluation seeds per individual.
    /// NOTE: Only used by stochastic environments (CartPole, Corridor).
    /// Deterministic environments (XOR, Spiral) ignore this.
    /// Default: 5
    /// </summary>
    public int SeedsPerIndividual { get; set; } = 5;
}
```

### 2. Implement Multi-Seed Evaluation (Optional Enhancement)

If you want to use `SeedsPerIndividual`:

```csharp
public void EvaluatePopulation(
    Population population,
    IEnvironment environment,
    EvolutionConfig config,
    int generationSeed = 0)
{
    foreach (var species in population.AllSpecies)
    {
        for (int i = 0; i < species.Individuals.Count; i++)
        {
            var individual = species.Individuals[i];

            // Multi-seed evaluation
            var fitnesses = new float[config.SeedsPerIndividual];
            for (int s = 0; s < config.SeedsPerIndividual; s++)
            {
                int seed = generationSeed * 1000 + s;  // Deterministic seed generation
                fitnesses[s] = Evaluate(individual, species.Topology, environment, seed);
            }

            // Aggregate fitness
            individual.Fitness = config.FitnessAggregation switch
            {
                "Mean" => fitnesses.Average(),
                "CVaR50" => fitnesses.OrderBy(f => f).Skip(fitnesses.Length / 2).First(),
                _ => fitnesses.Average()
            };

            species.Individuals[i] = individual;
        }
    }
}
```

Cost: **5x slower** but only meaningful for stochastic environments.

### 3. For Spiral Specifically
Since you're testing on Spiral (deterministic), keep `SeedsPerIndividual=1` implicitly:

```csharp
var config = new EvolutionConfig
{
    SpeciesCount = 8,
    IndividualsPerSpecies = 100,
    // SeedsPerIndividual unused for deterministic environments
    // If implemented, would cost 5x with zero benefit for Spiral
};
```

### 4. Consider Environment Metadata
```csharp
public interface IEnvironment
{
    bool IsDeterministic { get; }  // New property
    int InputCount { get; }
    // ...
}

public class SpiralEnvironment : IEnvironment
{
    public bool IsDeterministic => true;  // Fixed test cases
    // ...
}

public class CartPoleEnvironment : IEnvironment
{
    public bool IsDeterministic => false;  // Random initial state
    // ...
}
```

Then evaluation can adapt:
```csharp
int seedsToUse = environment.IsDeterministic ? 1 : config.SeedsPerIndividual;
```

---

## Test to Verify Variance (If Multi-Seed Were Implemented)

```csharp
[Fact]
public void SpiralHasZeroVarianceAcrossSeeds()
{
    var topology = CreateSpiralTopology();
    var individual = CreateRandomIndividual(topology);
    var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
    var evaluator = new SimpleFitnessEvaluator();

    var fitnesses = new float[5];
    for (int seed = 0; seed < 5; seed++)
    {
        fitnesses[seed] = evaluator.Evaluate(individual, topology, environment, seed);
    }

    // All fitnesses should be identical
    float first = fitnesses[0];
    for (int i = 1; i < fitnesses.Length; i++)
    {
        Assert.Equal(first, fitnesses[i]);  // Exact equality
    }
}
```

**Expected Result**: **PASS** - Variance = 0.0

---

## Summary Table

| Aspect | Finding |
|---|---|
| **SeedsPerIndividual Status** | Declared but never read (dead parameter) |
| **Current Evaluation Pattern** | 1 evaluation per individual per generation |
| **Spiral Determinism** | Perfect (variance = 0) |
| **Seed Effect on Spiral** | None (ignored by Reset) |
| **Cost of Multi-Seed for Spiral** | 5x slower with 0% benefit |
| **Cost of Not Using Multi-Seed** | No penalty (parameter unused) |
| **Recommendation** | Keep as-is OR implement for stochastic envs only |

