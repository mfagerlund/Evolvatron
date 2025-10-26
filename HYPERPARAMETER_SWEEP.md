# Evolvatron Corridor Hyperparameter Sweep

## Overview
Automated hyperparameter optimization for the FollowTheCorridor evolution demo. The goal is to find the configuration that **minimizes wall-time to solve** the corridor-following task (90% track completion).

## Running the Sweep

```bash
cd C:\Dev\Evolvatron
dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj sweep
```

## What It Does

The sweep tests **12 different configurations** of evolution hyperparameters, evaluating each configuration across **5 random seeds** (run in parallel).

### Evaluation Criteria
- **Success**: Agent reaches 90% of track in a single trial
- **Timeout**: 120 seconds per trial (exits early if solved)
- **Metric**: Average time-to-solve across successful seeds
- **Ranking**: By success rate (primary), then by average solve time (secondary)

### Configurations Tested

1. **Baseline** (40x40, E=1, P=0.2, T=4)
   - 40 species, 40 individuals each (1600 total)
   - 1 elite per species
   - Top 20% eligible as parents
   - Tournament size 4

2. **Fewer species** (20x80, E=1, P=0.2, T=4)
   - Tests if fewer, larger species perform better

3. **More species** (80x20, E=1, P=0.2, T=4)
   - Tests if more, smaller species perform better

4. **Smaller population** (40x20, E=1, P=0.2, T=4)
   - 800 total agents (half the baseline)
   - Faster per-generation but less diversity

5. **Tighter selection** (40x40, E=1, P=0.1, T=4)
   - Only top 10% eligible as parents
   - Stronger selection pressure

6. **Looser selection** (40x40, E=1, P=0.4, T=4)
   - Top 40% eligible as parents
   - Weaker selection pressure

7. **No parent pool** (40x40, E=1, P=1.0, T=4)
   - All individuals eligible as parents
   - Tournament selection only

8. **Weaker tournament** (40x40, E=1, P=0.2, T=2)
   - 2-way tournament
   - Less selection pressure

9. **Stronger tournament** (40x40, E=1, P=0.2, T=8)
   - 8-way tournament
   - More selection pressure

10. **More elites** (40x40, E=2, P=0.2, T=4)
    - 2 elites per species preserved

11. **Many elites** (40x40, E=4, P=0.2, T=4)
    - 4 elites per species preserved

12. **Fast + tight** (40x20, E=1, P=0.1, T=8)
    - Combined optimization: smaller pop, tight selection, strong tournament

## Performance Characteristics

### Parallelization
- Each configuration runs 5 seeds **in parallel** (using Task.Run)
- Max time per config: ~120s (if all seeds timeout)
- Best case: ~10-30s if seeds solve quickly
- Total sweep time estimate: **10-40 minutes** depending on solve rates

### CPU Usage
- Expect high CPU usage during sweep (5 parallel evolution runs)
- Each trial runs 1600 agents through deterministic physics simulation
- No GPU acceleration in current implementation

### Memory Usage
- Each parallel seed: ~50-100 MB
- Total during sweep: ~250-500 MB for 5 parallel trials
- Manageable on most modern machines

## Output Format

### Progress (Real-time)
```
[1/12] Testing: Baseline (40x40, E=1, P=0.2, T=4)
  Result: 4/5 solved, avg time: 45.2s, avg gens: 123.5

[2/12] Testing: Fewer species (20x80, E=1, P=0.2, T=4)
  Result: 3/5 solved, avg time: 67.8s, avg gens: 89.3
...
```

### Final Summary
```
=== SWEEP SUMMARY ===

Ranked by success rate, then average time to solve:

#1: Baseline (40x40, E=1, P=0.2, T=4)
    Solved: 4/5 (80.0%)
    Avg time: 45.2s (± 12.3s)
    Avg gens: 123.5 (± 34.2)

#2: Smaller pop (40x20, E=1, P=0.2, T=4)
    Solved: 4/5 (80.0%)
    Avg time: 52.1s (± 15.7s)
    Avg gens: 156.8 (± 41.3)

...

=== BEST CONFIGURATION ===
Baseline (40x40, E=1, P=0.2, T=4)
Species: 40
Individuals per species: 40
Elites: 1
Parent pool: 20%
Tournament size: 4
Success rate: 4/5
Average time to solve: 45.2s
Average generations: 123.5
```

## Technical Details

### Implementation
- File: `Evolvatron.Demo/CorridorHyperparameterSweep.cs`
- Entry point: `CorridorHyperparameterSweep.Run()`
- Invoked via: `Program.cs` with "sweep" argument

### Key Code Sections

**Parallel execution** (CorridorHyperparameterSweep.cs:196-217):
```csharp
// Run all seeds in parallel
var tasks = new Task<(bool solved, long timeMs, int generations)>[SeedsPerConfig];
for (int seed = 0; seed < SeedsPerConfig; seed++)
{
    int seedCopy = seed;
    tasks[seed] = Task.Run(() => RunSingleTrial(config, seedCopy));
}
Task.WaitAll(tasks);
```

**Trial timeout** (CorridorHyperparameterSweep.cs:259-314):
```csharp
var stopwatch = Stopwatch.StartNew();
while (stopwatch.ElapsedMilliseconds < MaxTimeoutSeconds * 1000 && !solved)
{
    // Evaluate all 1600 individuals
    // Check if solved (fitness >= 0.9)
    // Evolve to next generation
}
```

## Expected Results

Based on preliminary testing, we expect:
- **Success rates**: 60-90% across configs (problem is solvable)
- **Best configs**: Likely moderate population (20-40 individuals), tight selection (P=0.1-0.2)
- **Worst configs**: Likely very large/small species counts, or no selection pressure (P=1.0)
- **Solve times**: 20-100s for successful runs

## Interpreting Results

### Key Metrics
1. **Success Rate**: Most important - config must reliably solve the task
2. **Avg Time**: Among successful runs, lower is better (faster evolution)
3. **Avg Generations**: Informational - shows exploration vs exploitation balance
4. **Std Dev**: Lower variance indicates more consistent performance

### Trade-offs
- **Population size**: Larger = more diversity but slower per-generation
- **Parent pool**: Tighter = faster convergence but risk premature convergence
- **Tournament size**: Larger = stronger selection but less exploration
- **Elites**: More = better preservation but less mutation

## Next Steps

After the sweep completes:

1. **Update FollowTheCorridorDemo.cs** with best config:
   ```csharp
   var config = new EvolutionConfig
   {
       SpeciesCount = <best_value>,
       IndividualsPerSpecies = <best_value>,
       Elites = <best_value>,
       TournamentSize = <best_value>,
       ParentPoolPercentage = <best_value>f
   };
   ```

2. **Verify visually**:
   ```bash
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj corridor
   ```

3. **Consider further optimization**:
   - Test intermediate values around best config
   - Test mutation rate variations
   - Test different network topologies

## Troubleshooting

### Sweep hangs
- Check CPU usage - should be high during parallel execution
- Try running with fewer cores if system is overloaded
- Reduce SeedsPerConfig from 5 to 3 in CorridorHyperparameterSweep.cs:17

### All configs fail to solve
- Increase MaxTimeoutSeconds from 120 to 240 in CorridorHyperparameterSweep.cs:16
- Verify baseline demo works: `dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj corridor`
- Check for code changes that broke evolution

### Inconsistent results
- Evolution is stochastic - some variance is expected
- Increase SeedsPerConfig for more reliable statistics
- Check that environments are deterministic (they should be)
