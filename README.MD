# Evolvatron

Physics-based simulation engine with evolutionary neural network training.

## Components

### Evolvatron.Core - XPBD Physics Engine
2D physics simulation with particle-based and rigid body dynamics.
- **CPUStepper**: Pure C# reference implementation
- **GPUStepper**: ILGPU-accelerated for batch simulations (use only with real GPUs)
- XPBD constraints: rods, angles, motors
- Rigid bodies with revolute joints
- Static colliders: circles, capsules, OBBs

See `CLAUDE.md` for detailed architecture documentation.

### Evolvatron.Evolvion - Evolutionary Neural Networks
NEAT-style evolutionary algorithm with species-based topology exploration.

**Status**: ✅ Hyperparameter optimization complete

Optuna-optimized configuration solves spiral classification in **3 generations** with 100% success rate:
- 27 species × 88 individuals = 2,376 population
- Aggressive selection + high mutation rates
- Adaptive species culling
- Weak edge pruning

See `OPTUNA_RESULTS.md` for full results.

**Key Features:**
- Fixed-topology species with weight evolution
- Edge mutations: add, delete, split, redirect, swap
- Weak edge pruning for simplification
- Deterministic evolution with seed control
- Multi-seed validation framework

**Benchmark Tasks:**
- ✅ **Spiral Classification** (solved)
- 🔄 **Landscape Navigation** (Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel)
- 🔄 **Cart Pole**
- 🔄 **Follow The Corridor**

See `Evolvatron.Evolvion.Benchmarks/` for optimization landscape tasks.

### Evolvatron.Rigidon (Deprecated)
Original rigid body implementation. Superseded by Evolvatron.Core.

## Quick Start

### Run Physics Demo
```bash
dotnet run --project Evolvatron.Demo
```

### Run Evolution Tests
```bash
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj --filter "FullyQualifiedName~Evolvion"
```

### Run Hyperparameter Optimization
```bash
pip install optuna
python optuna_sweep.py --n-trials 10
```

## Project Structure

```
Evolvatron.Core/           # XPBD physics engine
Evolvatron.Evolvion/       # Evolutionary neural networks
Evolvatron.Demo/           # Visualization demos (Raylib)
Evolvatron.Tests/          # Unit and integration tests
Evolvatron.OptunaEval/     # Hyperparameter optimization CLI
optuna_sweep.py            # Bayesian optimization orchestrator
```

## Configuration

Default evolution config in `Evolvatron.Evolvion/EvolutionConfig.cs` contains Optuna-optimized hyperparameters.

For physics config see `Evolvatron.Core/SimulationConfig.cs`.

## Documentation

- **CLAUDE.md** - Architecture and development guide
- **OPTUNA_RESULTS.md** - Hyperparameter optimization results
- **LANDSCAPE_BENCHMARK_RESULTS.md** - Benchmark task performance
- **Evolvatron.Evolvion/README.md** - Evolvion module details

## Next Steps

- [ ] Test on landscape navigation benchmarks
- [ ] Scale to higher-dimensional problems (10D+)
- [ ] Integrate GPU-accelerated physics for RL training
- [ ] Multi-objective optimization (fitness vs complexity)
