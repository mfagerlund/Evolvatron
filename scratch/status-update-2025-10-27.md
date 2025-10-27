# Hyperparameter Sweep Status Update (2025-10-27)

## Background

**Goal**: Get the FollowTheCorridor hyperparameter sweep (`CorridorHyperparameterSweep.cs`) to successfully solve the task (reach 90% fitness threshold), matching the working demo (`FollowTheCorridorDemo.cs`).

**Context**: See `scratch/hyperparameter-sweep-situation.md` for full problem analysis. The sweep currently gets stuck at 35-40% fitness OR crashes when species diversification is enabled.

## What I Tried This Session

Implemented "Option A" from the situation document: Added array resizing logic to handle species diversification.

### Changes Made to `CorridorHyperparameterSweep.cs`:

1. **Enabled species diversification**: Set `MinSpeciesCount = 4` (line 128) to match demo
2. **Added environment array resizing** after `evolver.StepGeneration()` (lines 227-232):
   ```csharp
   int newPopSize = population.AllSpecies.Sum(s => s.Individuals.Count);
   while (environments.Count < newPopSize)
       environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
   while (environments.Count > newPopSize)
       environments.RemoveAt(environments.Count - 1);
   ```
3. **Added totalRewards array resizing** (lines 234-236)

## Problem Encountered

**CRASH**: Still getting `IndexOutOfRangeException` at `CPUEvaluator.EvaluateRow` line 66

**Critical Detail**: The crash happens on **GENERATION 0**, BEFORE any species diversification occurs. This means the problem is NOT species diversification itself!

Error trace:
```
System.IndexOutOfRangeException: Index was outside the bounds of the array.
   at Evolvatron.Evolvion.CPUEvaluator.EvaluateRow(Int32 rowIdx, Individual individual) in C:\Dev\Evolvatron\Evolvatron.Evolvion\CPUEvaluator.cs:line 66
```

The failing line is: `_nodeValues[plan.NodeStart + i] = 0.0f;`

## Hypothesis

There's a fundamental mismatch in how the sweep initializes evaluators vs how the demo does it. The topology used to create `CPUEvaluator` might not match the topology of the individuals, or the `_nodeValues` array is incorrectly sized.

## Git Status

Changes made:
- Modified sweep parameters (90% threshold instead of 100%, 15min timeout instead of 2min, 1 seed instead of 3)
- Reduced to single baseline config instead of 30 configs
- Added `MinSpeciesCount = 4`
- Added array resizing logic

## Actionable Next Steps

### CRITICAL: Kill Background Processes First

Many background bash processes from previous session attempts are still running, blocking file access and preventing clean builds. Kill them before continuing:

```bash
# Find processes using port (if running demos)
netstat -ano | grep 5000

# Kill specific PID
taskkill //F //PID <pid>

# Or restart IDE/terminal to kill all
```

### Option 1: Debug the Current Crash (Quick Investigation)

**Goal**: Understand why the crash happens on Generation 0 (before any species diversification).

1. **Revert to clean state**:
   ```bash
   cd C:/Dev/Evolvatron
   git diff Evolvatron.Demo/CorridorHyperparameterSweep.cs  # Review changes
   git checkout Evolvatron.Demo/CorridorHyperparameterSweep.cs  # Revert
   ```

2. **Compare how demo vs sweep create evaluators**:
   - Demo: `FollowTheCorridorDemo.cs` lines 78-92 (creates evaluators with `species.Topology`)
   - Sweep: `CorridorHyperparameterSweep.cs` lines 150-164 (also uses `species.Topology`)
   - Look for ANY differences in initialization order or topology usage

3. **Add diagnostic logging** to CPUEvaluator.cs constructor:
   ```csharp
   public CPUEvaluator(SpeciesSpec spec)
   {
       _spec = spec;
       Console.WriteLine($"CPUEvaluator: TotalNodes={spec.TotalNodes}, RowPlans.Length={spec.RowPlans.Length}");
       _nodeValues = new float[spec.TotalNodes];
       // ...
   }
   ```

4. **Test sweep with diagnostics**:
   ```bash
   cd C:/Dev/Evolvatron
   dotnet build Evolvatron.Demo/Evolvatron.Demo.csproj --nologo -v:q
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj sweep
   ```

5. **Compare with working demo**:
   ```bash
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj corridor
   ```

### Option 2: Proper Fix - Refactor to Share Code (RECOMMENDED)

**User's feedback**: "I can't believe you *copied* the code instead of making it serve both purposes. TOTALLY FUCKING BONKERS."

**Why this is better**: Eliminates subtle incompatibilities from code duplication. Single source of truth for evaluation logic.

**Implementation Plan**:

1. **Create shared evaluation class** (`Evolvatron.Demo/CorridorEvaluationRunner.cs`):
   ```csharp
   namespace Evolvatron.Demo;

   public class CorridorEvaluationRunner
   {
       public static (float bestFitness, int generation, bool solved) RunEvolution(
           Population population,
           Evolver evolver,
           EvolutionConfig config,
           int maxGenerations,
           float solvedThreshold,
           int maxTimeoutMs,
           Action<int, float, long>? progressCallback = null)
       {
           var environments = new List<FollowTheCorridorEnvironment>();
           var evaluators = new List<CPUEvaluator>();
           var individuals = new List<Individual>();
           var totalRewards = new float[0];

           // Initialize environments/evaluators
           foreach (var species in population.AllSpecies)
           {
               foreach (var individual in species.Individuals)
               {
                   environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
                   evaluators.Add(new CPUEvaluator(species.Topology));
                   individuals.Add(individual);
               }
           }
           totalRewards = new float[environments.Count];

           var stopwatch = Stopwatch.StartNew();
           int generation = 0;
           bool solved = false;

           while (stopwatch.ElapsedMilliseconds < maxTimeoutMs && generation < maxGenerations && !solved)
           {
               // Reset environments
               for (int i = 0; i < environments.Count; i++)
               {
                   environments[i].Reset(seed: generation);
                   totalRewards[i] = 0f;
               }

               // Run lockstep evaluation (all agents take step 0, then all take step 1, etc.)
               var observations = new float[environments[0].InputCount];
               for (int step = 0; step < 320; step++)
               {
                   for (int i = 0; i < environments.Count; i++)
                   {
                       if (!environments[i].IsTerminal())
                       {
                           environments[i].GetObservations(observations);
                           var actions = evaluators[i].Evaluate(individuals[i], observations);
                           float reward = environments[i].Step(actions);
                           totalRewards[i] += reward;
                       }
                   }
               }

               // Collect fitnesses
               int envIdx = 0;
               foreach (var species in population.AllSpecies)
               {
                   for (int indIdx = 0; indIdx < species.Individuals.Count; indIdx++)
                   {
                       var ind = species.Individuals[indIdx];
                       ind.Fitness = totalRewards[envIdx];
                       species.Individuals[indIdx] = ind;
                       individuals[envIdx] = ind;
                       envIdx++;
                   }
               }

               var best = population.GetBestIndividual();
               float bestFitness = best.HasValue ? best.Value.individual.Fitness : 0f;

               progressCallback?.Invoke(generation, bestFitness, stopwatch.ElapsedMilliseconds);

               if (bestFitness >= solvedThreshold)
               {
                   solved = true;
                   break;
               }

               evolver.StepGeneration(population);
               generation++;

               // Resize arrays to match new population size (species diversification may change it)
               int newPopSize = population.AllSpecies.Sum(s => s.Individuals.Count);
               while (environments.Count < newPopSize)
                   environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
               while (environments.Count > newPopSize)
                   environments.RemoveAt(environments.Count - 1);

               if (totalRewards.Length != newPopSize)
                   totalRewards = new float[newPopSize];

               // Rebuild evaluators after mutation
               evaluators.Clear();
               individuals.Clear();
               foreach (var species in population.AllSpecies)
               {
                   foreach (var individual in species.Individuals)
                   {
                       evaluators.Add(new CPUEvaluator(species.Topology));
                       individuals.Add(individual);
                   }
               }
           }

           stopwatch.Stop();
           var finalBest = population.GetBestIndividual();
           float finalFitness = finalBest.HasValue ? finalBest.Value.individual.Fitness : 0f;

           return (finalFitness, generation, solved);
       }
   }
   ```

2. **Modify CorridorHyperparameterSweep.cs** to use shared runner:
   ```csharp
   private static (bool solved, long timeMs, int generations) RunSingleTrial(
       ConfigToTest config,
       int seed,
       TrialProgress progress)
   {
       var topology = CreateCorridorTopology();
       var evolutionConfig = new EvolutionConfig
       {
           SpeciesCount = config.SpeciesCount,
           IndividualsPerSpecies = config.IndividualsPerSpecies,
           Elites = config.Elites,
           TournamentSize = config.TournamentSize,
           ParentPoolPercentage = config.ParentPoolPercentage,
           MinSpeciesCount = 4  // Match demo
       };

       var evolver = new Evolver(seed: 42);
       var population = evolver.InitializePopulation(evolutionConfig, topology);

       var stopwatch = Stopwatch.StartNew();

       var (bestFitness, generation, solved) = CorridorEvaluationRunner.RunEvolution(
           population,
           evolver,
           evolutionConfig,
           maxGenerations: MaxStepsForSuccess,
           solvedThreshold: SolvedThreshold,
           maxTimeoutMs: MaxTimeoutSeconds * 1000,
           progressCallback: (gen, fitness, elapsedMs) =>
           {
               progress.Generation = gen;
               progress.BestFitness = fitness;
               progress.ElapsedMs = elapsedMs;

               if (gen % 100 == 0)
               {
                   Console.WriteLine($"  Gen {gen}: Best={fitness:F3} ({fitness * 100:F1}%), Time={elapsedMs / 1000.0:F1}s");
               }
           });

       stopwatch.Stop();
       progress.Completed = true;
       progress.Solved = solved;

       return (solved, stopwatch.ElapsedMilliseconds, generation);
   }
   ```

3. **Modify FollowTheCorridorDemo.cs** to use shared runner in background task:
   - Replace lines 115-176 with call to `CorridorEvaluationRunner.RunEvolution`
   - Keep visualization code separate

4. **Test both paths**:
   ```bash
   cd C:/Dev/Evolvatron
   dotnet build Evolvatron.Demo/Evolvatron.Demo.csproj --nologo -v:q

   # Test sweep
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj sweep

   # Test visual demo
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj corridor
   ```

## How to Test After Fix

1. **Quick smoke test** (single baseline config, 1 seed):
   ```bash
   cd C:/Dev/Evolvatron
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj sweep
   ```
   Expected: Should reach 90%+ fitness within 15 minutes

2. **Full sweep** (after baseline works):
   - Modify `GenerateConfigurations()` in CorridorHyperparameterSweep.cs to add more configs
   - Increase `SeedsPerConfig` to 3-5 for statistical significance
   - Run full sweep

## Recommendation

**Strongly recommend Option 2** (refactor to share code) because:
1. User explicitly complained about code duplication
2. Eliminates entire class of bugs from subtle differences
3. Single source of truth for evaluation logic
4. Easier to maintain and debug going forward
5. The crash happening on Generation 0 suggests deeper initialization mismatch that sharing code would eliminate
