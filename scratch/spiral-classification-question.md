# Spiral Classification Performance Question

## Background

Evolvion has demonstrated excellent performance on some benchmarks but poor performance on others. Recent testing reveals:

**Strong Performance:**
- Follow The Corridor: Achieves >50% checkpoint completion within 100 generations
- Landscape optimization tasks: Solves Sphere-20D, Rosenbrock-15D, Rastrigin-20D in Generation 0

**Poor Performance:**
- Spiral Classification: No observable progress after extended training

## The Spiral Classification Task

### Task Definition
- **Input**: 2D coordinates (x, y)
- **Output**: Binary classification (-1 or +1 using tanh activation)
- **Dataset**: 100 points total (50 points per spiral)
  - Two interleaved spirals generated using:
    - `t = i * 4π / 50` (angle from 0 to 4π = 2 full rotations)
    - `r = t / (4π)` (radius grows linearly)
    - Spiral 1: `(r*cos(t), r*sin(t))` → label = -1
    - Spiral 2: `(r*cos(t+π), r*sin(t+π))` → label = +1
- **Difficulty**: Classic non-linearly separable problem requiring complex decision boundaries

### Network Architecture
```
2 inputs (x, y)
  ↓
8 hidden nodes (ReLU, Tanh, Sigmoid, LeakyReLU)
  ↓
8 hidden nodes (ReLU, Tanh, LeakyReLU)
  ↓
1 output (Tanh)
```

- MaxInDegree: 10
- Initialization: Sparse random connectivity
- Glorot/Xavier weight initialization

### Evolution Configuration
```csharp
SpeciesCount: 10
IndividualsPerSpecies: 80
Total population: 800 individuals
Elites: 2
TournamentSize: 4
ParentPoolPercentage: 0.3
```

### Fitness Function
```csharp
// Evaluate all 100 test points
for each (x, y, expectedLabel) in testCases:
    output = network.Evaluate(x, y)
    error = (output - expectedLabel)²
    totalError += error

fitness = -(totalError / 100)  // Average squared error, negated (higher is better)
```

Success threshold: fitness >= -0.05 (average squared error < 0.05)

### Evaluation Protocol
1. Environment resets
2. Network receives observation (x, y) for point 1
3. Network produces output
4. Error accumulated internally
5. Steps 2-4 repeat for all 100 points
6. Final fitness returned at end: `-(totalError / 100)`

## Follow The Corridor Task (For Comparison)

### Task Definition
- **Input**: 9 distance sensors measuring wall proximity
- **Output**: 2 continuous actions (steering, throttle)
- **Environment**: SVG-based track with walls and checkpoints
- **Duration**: 320 timesteps per episode

### Network Architecture
```
9 inputs (distance sensors)
  ↓
12 hidden nodes (11 activation types)
  ↓
2 outputs (Tanh)
```

### Evolution Configuration
```csharp
SpeciesCount: 4
IndividualsPerSpecies: 100
Total population: 400 individuals
Elites: 4
TournamentSize: 4
```

### Fitness Function
- Dense rewards at every timestep:
  - Progress toward next checkpoint: positive reward
  - Wall collision: negative reward + episode termination
  - Speed bonuses/penalties
  - Timeout penalties if too slow
- Final fitness: cumulative reward over entire episode

### Evaluation Protocol
1. Car resets to starting position
2. For each of 320 timesteps:
   - Get 9 sensor readings
   - Network produces (steering, throttle)
   - Physics updates car position
   - Immediate reward calculated
   - Cumulative fitness updated
3. Episode ends on collision, timeout, or completion

## Observed Behavior

### Follow The Corridor
- Generation 0: Some individuals show basic progress
- Generations 1-50: Steady fitness improvement
- Generation 50-100: Achieves >50% track completion
- **Result**: SUCCESS

### Spiral Classification
- Demo process runs but produces no console output
- Test marked as `[Fact(Skip = "Slow: requires topology mutations and many generations")]`
- Test expects 500 generations to reach -0.05 fitness threshold
- Expected behavior: "Nothing!" (per user report)

## Questions to Investigate

1. **Fitness landscape characteristics**: What is the distribution of fitness values in Generation 0?
   - Are all individuals clustering around the same fitness?
   - Is there sufficient variance for selection pressure to operate?

2. **Random baseline performance**: What accuracy does random guessing achieve?
   - Expected: ~50% accuracy (random binary classifier)
   - Expected fitness: ~-0.25 (average squared error for random guess)

3. **Selection pressure**: Can tournament selection differentiate between individuals?
   - If all fitness values are similar, tournament selection becomes random parent selection
   - Without fitness diversity, evolution has no signal to optimize

4. **Task structure differences**: How do the two tasks differ?
   - Temporal structure (sequential vs batch)
   - Reward density (per-step vs end-of-episode)
   - Fitness landscape smoothness
   - Exploration affordances

5. **Network capacity**: Is the 2→8→8→1 architecture sufficient?
   - The spiral problem is known to require hidden layer capacity
   - How does this compare to proven solutions (e.g., 2→10→10→1 with gradient descent)?

6. **Initialization quality**: How well does sparse random initialization perform on this task?
   - Landscape benchmarks showed exceptional Gen 0 performance
   - Does this extend to classification tasks?

## Test Artifacts

- `Evolvatron.Tests/Evolvion/SpiralEvolutionTest.cs`: Unit test (currently skipped)
- `Evolvatron.Demo/SpiralClassificationDemo.cs`: Demo application
- `Evolvatron.Evolvion/Environments/SpiralEnvironment.cs`: Environment implementation
- `LANDSCAPE_BENCHMARK_RESULTS.md`: Shows Gen 0 success on optimization tasks

## Data Needed

To properly diagnose this issue, we need:

1. **Generation 0 statistics**:
   - Best fitness
   - Mean fitness
   - Median fitness
   - Standard deviation
   - Fitness histogram

2. **Evolution trajectory** (first 100 generations):
   - Best fitness per generation
   - Mean fitness per generation
   - Classification accuracy of best individual

3. **Individual analysis**:
   - Sample network outputs on test cases
   - Weight distribution statistics
   - Active edge counts

4. **Comparative baseline**:
   - Random network performance (averaged over 1000 samples)
   - Simple linear classifier performance
   - Known neural network solution (e.g., backprop-trained network)

## Expected vs Actual

### Expected (if working correctly)
- Generation 0: ~50% accuracy, fitness ≈ -0.25
- Generation 50: Noticeable improvement, fitness > -0.20
- Generation 200: Significant progress, fitness > -0.10
- Generation 500: Success, fitness > -0.05 (>80% accuracy)

### Actual (observed)
- Process runs indefinitely with no output
- No convergence within reasonable timeframe
- Reported as "nothing!" happening

## Open Questions

1. Is the fitness function implemented correctly?
2. Is the spiral generation producing separable classes?
3. Are mutations being applied?
4. Is selection pressure operating, or is fitness too uniform?
5. Does the task structure fundamentally mismatch Evolvion's strengths?
