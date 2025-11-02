# Landscape Navigation Verification Plan

## Goal

Verify that Optuna-optimized hyperparameters (Trial 23) can solve landscape navigation benchmarks.

## Why This Matters

Spiral classification is **too easy** - solves in 3 generations regardless of seed. We need to test on harder problems to:
1. Validate hyperparameters generalize beyond spiral
2. Establish evaluation budget for harder problems
3. Identify if any hyperparameter tuning is needed for continuous control tasks
4. Benchmark the algorithm against standard optimization problems

## Test Landscapes

From `Evolvatron.Evolvion.Benchmarks/OptimizationLandscapes.cs`:

### 1. Sphere (Easy Baseline)
- **Function**: f(x) = Σ x_i²
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Smooth convex, single global optimum
- **Expected**: Should solve easily (sanity check)

### 2. Rosenbrock (Medium)
- **Function**: f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
- **Optimum**: x = [1, 1, ..., 1], f(x) = 0
- **Characteristics**: Narrow parabolic valley, hard to navigate
- **Expected**: Moderate difficulty, tests sequential decision-making

### 3. Rastrigin (Hard - Multimodal)
- **Function**: f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Many local optima (highly multimodal)
- **Expected**: Tests exploration vs exploitation balance

### 4. Ackley (Hard - Deceptive)
- **Function**: Complex with exp terms (see OptimizationLandscapes.cs)
- **Optimum**: x = [0, 0, ..., 0], f(x) = 0
- **Characteristics**: Nearly flat outer regions, sharp center
- **Expected**: Tests gradient-based navigation (if using GradientOnly obs)

### 5. Schwefel (Very Hard)
- **Function**: f(x) = 418.9829n - Σ [x_i * sin(√|x_i|)]
- **Optimum**: x = [420.9687, ...], f(x) ≈ 0
- **Characteristics**: Global optimum far from good local optima (deceptive)
- **Expected**: Hardest test, may not solve without specific tuning

## Test Protocol

### Configuration Matrix

| Landscape  | Dimensions | Timesteps | Bounds        | Observation Type     | Priority |
|------------|-----------|-----------|---------------|----------------------|----------|
| Sphere     | 5D        | 50        | [-5, 5]       | FullPosition         | HIGH     |
| Rosenbrock | 5D        | 100       | [-2, 2]       | FullPosition         | HIGH     |
| Rosenbrock | 10D       | 150       | [-2, 2]       | GradientOnly         | MEDIUM   |
| Rastrigin  | 8D        | 100       | [-5.12, 5.12] | FullPosition         | HIGH     |
| Ackley     | 8D        | 100       | [-5, 5]       | FullPosition         | MEDIUM   |
| Ackley     | 15D       | 250       | [-5, 5]       | PartialObservability | LOW      |
| Schwefel   | 12D       | 200       | [-500, 500]   | FullPosition         | LOW      |

**Priority**:
- HIGH: Run first, must solve for hyperparameters to be considered good
- MEDIUM: Important but more challenging
- LOW: Stretch goals, may require tuning

### Success Criteria

Define "solved" as reaching within X% of global optimum:

| Landscape  | Threshold (% of optimum) | Notes                           |
|------------|--------------------------|---------------------------------|
| Sphere     | 0.01 (1%)                | Should get very close           |
| Rosenbrock | 0.10 (10%)               | Valley is tricky                |
| Rastrigin  | 0.15 (15%)               | Local optima are challenging    |
| Ackley     | 0.20 (20%)               | Flat regions hard to navigate   |
| Schwefel   | 0.30 (30%)               | Very deceptive, lower bar       |

**Alternative metric**: Distance to optimum < ε in state space

### Evaluation Budget

From spiral results: 3 gens × 2,376 = 7,128 evals to solve

**Proposed budgets by difficulty**:
- Sphere: 20,000 evals (~8 generations) - should be easy
- Rosenbrock: 50,000 evals (~21 generations)
- Rastrigin: 100,000 evals (~42 generations)
- Ackley: 100,000 evals (~42 generations)
- Schwefel: 200,000 evals (~84 generations)

### Test Execution

For each landscape:

1. **Network topology**: Match input/output dimensions
   ```
   Inputs = dimensions (or 2×dimensions for PartialObs)
   Outputs = dimensions (movement vector)
   Hidden = [8, 8] (same as spiral)
   ```

2. **Seeds**: Test 10 seeds (0-9) for statistical significance

3. **Metrics to track**:
   - Generations to solve (primary)
   - Total evaluations to solve
   - Final distance from optimum
   - Success rate across seeds (% that solve)
   - Convergence curves (best fitness per generation)

4. **Early stopping**: Stop at solve threshold or max budget

## Implementation Steps

### Step 1: Create Test Harness (HIGH PRIORITY)

Create `Evolvatron.Tests/Evolvion/LandscapeNavigationTest.cs`:

```csharp
[Fact]
public void Sphere5D_SolvesWithOptunaHyperparameters()
{
    const int maxGenerations = 10;
    const int numSeeds = 10;
    const float solveThreshold = -0.01f; // Within 1% of optimum

    var task = new LandscapeNavigationTask(
        OptimizationLandscapes.Sphere,
        dimensions: 5,
        timesteps: 50,
        stepSize: 0.1f,
        minBound: -5f,
        maxBound: 5f,
        observationType: ObservationType.FullPosition,
        seed: 42);

    var results = RunEvolutionOnLandscape(task, maxGenerations, numSeeds, solveThreshold);

    // Report and assert
    int solvedCount = results.Count(r => r.Solved);
    float avgGenerations = results.Where(r => r.Solved).Average(r => r.GenerationsToSolve);

    _output.WriteLine($"Solved: {solvedCount}/{numSeeds}");
    _output.WriteLine($"Avg generations: {avgGenerations:F1}");

    Assert.True(solvedCount >= 8, "Should solve at least 80% of seeds");
}
```

### Step 2: Adapt Network Builder

Network needs variable input/output counts:

```csharp
var topology = new SpeciesBuilder()
    .AddInputRow(task.GetObservationSize())
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid)
    .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh)
    .AddOutputRow(task.GetDimensions(), ActivationType.Tanh) // Movement is bounded [-1, 1]
    .WithMaxInDegree(10)
    .InitializeDense(random, density: 0.3f)
    .Build();
```

### Step 3: Create LandscapeEnvironment Wrapper

Need `IEnvironment` implementation for `LandscapeNavigationTask`:

```csharp
public class LandscapeEnvironment : IEnvironment
{
    private readonly LandscapeNavigationTask _task;
    private float[] _currentPosition;
    private int _step;

    public int InputCount => _task.GetObservationSize();
    public int OutputCount => _task.GetDimensions();
    public int MaxSteps => _task.GetTimesteps();

    // Implement Reset(), GetObservations(), Step(), IsTerminal()
}
```

### Step 4: Run Tests Incrementally

**Phase 1: Sphere** (must pass)
- Run 10 seeds
- Should solve 100% within 10 generations
- If fails, hyperparameters don't work for continuous control

**Phase 2: Rosenbrock** (important)
- Run 10 seeds
- Target 70%+ solve rate within 25 generations
- If fails, may need to adjust mutation rates or population size

**Phase 3: Rastrigin** (challenging)
- Run 10 seeds
- Target 50%+ solve rate within 50 generations
- Tests exploration capabilities

**Phase 4: Ackley** (if time permits)
- Test GradientOnly observation type
- Validates gradient-based navigation

**Phase 5: Schwefel** (stretch goal)
- Very hard, may not solve
- Success would be impressive

### Step 5: Analysis

For each landscape, generate:

1. **Success rate table**:
   ```
   | Landscape  | Seeds Solved | Avg Gens | Avg Evals | Min Gens | Max Gens |
   |------------|--------------|----------|-----------|----------|----------|
   | Sphere     | 10/10        | 3.2      | 7,603     | 2        | 5        |
   | Rosenbrock | 8/10         | 18.4     | 43,734    | 12       | 28       |
   ```

2. **Convergence curves**: Plot best fitness vs generation for all seeds

3. **Failure analysis**: For seeds that don't solve, diagnose why
   - Stuck in local optimum?
   - Premature convergence?
   - Species collapse?

## Expected Outcomes

### Best Case
- Sphere: 100% solve, ~5 generations
- Rosenbrock: 80%+ solve, ~20 generations
- Rastrigin: 60%+ solve, ~40 generations

### Likely Case
- Sphere: 100% solve (easy sanity check)
- Rosenbrock: 70% solve (main test)
- Rastrigin: 40-50% solve (harder)

### Worst Case
- Sphere doesn't solve consistently
- **Action**: Hyperparameters don't generalize, need landscape-specific tuning

## If Tests Fail

### Debugging Steps

1. **Check network architecture**: Is output scaled correctly? (Tanh gives [-1, 1])
2. **Check step size**: Is 0.1 appropriate for all landscapes?
3. **Check timesteps**: Are we giving enough time to reach optimum?
4. **Check fitness scaling**: Is fitness normalized properly?

### Potential Adjustments

If Sphere fails:
- Problem is fundamental, revisit hyperparameters entirely

If Rosenbrock/Rastrigin fail:
- **Increase population**: Try 40 species × 100 individuals = 4,000
- **Increase timesteps**: Give agent more time to navigate
- **Adjust step size**: Tune per landscape (Rosenbrock may need smaller steps)
- **Add diversity pressure**: Increase species diversity threshold
- **Landscape-specific tuning**: Run mini Optuna sweep per landscape

If only Schwefel fails:
- Expected, this is very hard
- Consider as success if we pass Sphere/Rosenbrock/Rastrigin

## Timeline Estimate

- **Step 1-3 (Implementation)**: 2-3 hours
- **Phase 1 (Sphere)**: 30 minutes (10 seeds × ~2 mins each)
- **Phase 2 (Rosenbrock)**: 1 hour
- **Phase 3 (Rastrigin)**: 2 hours
- **Analysis**: 1 hour

**Total**: ~6-7 hours for comprehensive validation

## Success Definition

**Minimum bar**: Sphere + Rosenbrock solve at 80%+ rate
**Strong success**: Sphere + Rosenbrock + Rastrigin all solve at 60%+ rate
**Exceptional**: All 5 landscapes show progress toward optimum

## Deliverables

1. `LandscapeNavigationTest.cs` with all tests
2. `LANDSCAPE_NAVIGATION_RESULTS.md` with tables and analysis
3. Updated `OPTUNA_RESULTS.md` to include landscape results
4. If tests pass: Confidence that hyperparameters generalize
5. If tests fail: Action plan for landscape-specific tuning

## Next Steps After Validation

If hyperparameters work well:
- **Apply to CartPole**: Test on classic RL benchmark
- **Apply to Follow The Corridor**: Test on sequential control
- **Scale up dimensions**: Test 20D, 50D landscapes
- **GPU implementation**: Move to ILGPU for massive parallelization

If hyperparameters need tuning:
- **Mini Optuna sweep per landscape**: Quick 20-50 trial optimization
- **Landscape-specific configs**: Allow different hyperparams per problem type
- **Meta-learning**: Can we learn hyperparameter selection policy?

---

## Quick Start Commands

```bash
# Run Sphere test (HIGH PRIORITY)
dotnet test --filter "FullyQualifiedName~LandscapeNavigationTest.Sphere5D"

# Run all HIGH priority tests
dotnet test --filter "FullyQualifiedName~LandscapeNavigationTest" --filter "Priority=HIGH"

# Generate report
dotnet test --logger "console;verbosity=detailed" > landscape_results.txt
```

## Notes

- Keep same network architecture (2→8→8→N) across all tests for fair comparison
- Use same Optuna Trial 23 hyperparameters unchanged
- This is a verification test, not optimization - resist urge to tweak until after first full run
- Document everything - if it fails, we need detailed diagnostics
