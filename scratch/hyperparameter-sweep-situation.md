# FollowTheCorridor Hyperparameter Sweep - Current Situation

## Goal
Establish a working baseline for the FollowTheCorridor hyperparameter sweep that can solve the task (reach 90% fitness threshold).

## What Works
**FollowTheCorridorDemo.cs** - Visual demo that SUCCESSFULLY solves the task:
- Reaches 90%+ fitness reliably in ~100-200 generations
- Uses 40 species × 40 individuals = 1600 total agents
- Configuration: seed=42, Elites=1, ParentPoolPercentage=0.2f, TournamentSize=4
- Uses DEFAULT EvolutionConfig settings (critically: MinSpeciesCount=4, enabling species culling/diversification)
- Creates 1600 separate environment instances (one per agent)
- Runs all agents in lockstep (all take step 0, then all take step 1, etc.)
- Rebuilds evaluators and individuals lists after each mutation

## What Doesn't Work
**CorridorHyperparameterSweep.cs** - Gets stuck at 35-40% fitness after 500+ generations

## Root Problems Discovered

### Problem 1: Code Duplication Instead of Sharing
The sweep COPIED the demo's evaluation logic instead of sharing it. This created subtle differences that broke evolution. The user is rightfully frustrated about this architectural mistake.

### Problem 2: Species Diversification Incompatibility
- **Demo behavior**: Uses MinSpeciesCount=4 (default), which enables species culling and diversification
  - Species diversification is CRITICAL for solving the task (prevents stagnation)
  - When a species is culled, a new one is created with MUTATED TOPOLOGY (different RowCounts)
  - Population size can change (different number of nodes/edges)

- **Sweep behavior**:
  - Currently sets MinSpeciesCount=SpeciesCount (disables species diversification) to avoid crashes
  - Pre-allocates environments/evaluators/individuals arrays at the start
  - Arrays become WRONG SIZE when species diversification changes topology
  - Results in IndexOutOfRangeException when population size changes

### Problem 3: Parallel Execution Issues
- Sweep originally used `Parallel.For` to run multiple seeds concurrently
- This is incompatible with species diversification (shared population state)
- Changed to sequential execution (for loop) but still crashes with diversification enabled

## Current State

**CorridorHyperparameterSweep.cs (as of last edit)**:
```csharp
// Line 121-129: Configuration
var evolutionConfig = new EvolutionConfig
{
    SpeciesCount = config.SpeciesCount,
    IndividualsPerSpecies = config.IndividualsPerSpecies,
    Elites = config.Elites,
    TournamentSize = config.TournamentSize,
    ParentPoolPercentage = config.ParentPoolPercentage,
    MinSpeciesCount = config.SpeciesCount // DISABLED - causes crashes with fixed-size arrays
};

// Line 88-95: Sequential execution (was Parallel.For)
for (int seed = 0; seed < SeedsPerConfig; seed++)
{
    var progress = new TrialProgress();
    results[seed] = RunSingleTrial(config, seed, progress);
}

// Line 141-163: Pre-allocated arrays (PROBLEM!)
var environments = new List<FollowTheCorridorEnvironment>();
var evaluators = new List<CPUEvaluator>();
var individuals = new List<Individual>();
// ... creates 1600 environments/evaluators/individuals

// Line 227-237: Rebuilds evaluators/individuals after mutation
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
// BUT: environments list is NEVER resized!
```

## Why It Fails

1. **Without species diversification** (current state): Evolution gets stuck at 35-40% fitness
   - Lacks the genetic diversity injection that demo relies on
   - Cannot escape local optima

2. **With species diversification enabled**: Crashes with IndexOutOfRangeException
   - Species diversification mutates RowCounts (changes number of nodes)
   - Population size changes (different total number of individuals)
   - Pre-allocated `environments` array is wrong size
   - `evaluators[i].Evaluate()` accesses out-of-bounds indices

## Solution Approaches

### Option A: Fix Array Resizing in Sweep
After `evolver.StepGeneration()`, rebuild ALL arrays to match new population:
```csharp
// After line 224: evolver.StepGeneration(population);

// Resize environments array to match new population size
int newPopSize = population.AllSpecies.Sum(s => s.Individuals.Count);
while (environments.Count < newPopSize)
    environments.Add(new FollowTheCorridorEnvironment(maxSteps: 320));
while (environments.Count > newPopSize)
    environments.RemoveAt(environments.Count - 1);

// Then rebuild evaluators/individuals (already done at lines 227-237)
```

### Option B: Refactor to Share Code (RECOMMENDED by user)
Extract the evaluation logic into a shared class/method that both demo and sweep use:
```csharp
public static class CorridorEvaluator
{
    public static (float bestFitness, int generation) EvaluatePopulation(
        Population population,
        EvolutionConfig config,
        int maxGenerations,
        float solvedThreshold,
        int maxTimeoutMs,
        Action<int, float>? progressCallback = null)
    {
        // Shared logic for evaluation loop
        // Both demo and sweep call this
    }
}
```

## Immediate Next Steps

1. **Quick fix**: Implement Option A (resize environments array) to get baseline working
2. **Proper fix**: Implement Option B (shared evaluation code)
3. **Then**: Run full hyperparameter sweep with multiple configurations

## Performance Issue
User noted: "1.7s per generation is WRONG". Demo appears faster than sweep even without species diversification. Need to investigate why sweep is slower (possibly console output overhead from debug prints every 100 generations).

## Files
- `Evolvatron.Demo/FollowTheCorridorDemo.cs` - Working visual demo (reference implementation)
- `Evolvatron.Demo/CorridorHyperparameterSweep.cs` - Broken sweep (needs fix)
- `Evolvatron.Evolvion/SpeciesDiversification.cs` - Species diversification logic (lines 248-256 mutate RowCounts)
- `Evolvatron.Evolvion/EvolutionConfig.cs` - Configuration class (MinSpeciesCount default = 4)
