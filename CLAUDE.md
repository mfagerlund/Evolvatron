# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Evolvatron** is a game centered around **rocket systems design and reward engineering**. Think moon lander in continuous space where the rocket starts far away with obstacles in the way. The player's job is to design reward landscapes to train RL agents to land rockets successfully.

The game has two major workflows:
1. **Rocket Design**: 2D editor for piecing together rocket components (sensors, gimbals, motors, differential steering)
2. **Reward Engineering**: Live editing of reward landscapes while maintaining valid playback history (unless the agent's effectors/sensors change)

**Key technical characteristics:**
- Physics based on particles + XPBD constraints (rods, angles, motors)
- Particle-vs-static-collider collisions only (no particle-particle)
- Deterministic fixed-timestep simulation
- Both CPU and GPU (ILGPU) implementations with identical API
- Rigid body support with impulse-based contact solver
- Reward shaping for RL landing tasks
- Agents train on **2-15 input signals and 2-25 outputs** (not pixels)

**Evolvion Integration**: The evolutionary neural controller framework (`Evolvatron.Evolvion/`) evolves fixed-topology neural controllers using large-scale parallel GPU execution. Two optimization systems exist:
1. **CEM (flagship)**: Cross-Entropy Method with multi-position training. Distribution-based (mu/sigma per parameter), fits to top 1% elites. Uses `IslandOptimizer` + `GPUDenseDoublePoleEvaluator` with dense NN kernel. Multi-pos(10) training produces generalist controllers (median 303-324/625 on DPNV benchmark).
2. **ES (experimental fallback)**: OpenAI-style Evolution Strategies with Adam optimizer. Same infrastructure as CEM. Needs more training positions (25 vs 10) and is more variable.
3. **Evolvion GA (legacy)**: Species-based GA with sparse edge-based kernel. Retained for topology exploration but NOT used for fixed-topology problems — multi-position training hurts GA because weight jitter already provides implicit regularization.

**Elman Recurrence**: The system supports **Elman networks** — dedicated extra output neurons whose values are fed back as additional inputs on the next timestep, giving the network memory. This is critical for non-Markovian environments where the agent cannot observe velocity or other hidden state directly. Controlled by `ContextSize` (number of feedback outputs) and `IsJordan` (false=Elman, true=Jordan/action-feedback) on `GPUDoublePoleEvaluator`. Elman is preferred over Jordan because it allows arbitrary memory capacity independent of action dimensionality.

## Build and Test Commands

### Building
```bash
# Build entire solution
dotnet build Evolvatron.sln

# Build specific project
dotnet build Evolvatron.Rigidon/Evolvatron.Rigidon.csproj
dotnet build Evolvatron.Demo/Evolvatron.Demo.csproj
```

### Running Tests

**Test Framework**: This project uses **xUnit** (not NUnit).

**Why xUnit over NUnit?**
- **Simpler, more modern design**: xUnit was created by the original inventor of NUnit as a ground-up redesign
- **Better isolation**: Each test class gets a new instance per test method by default (prevents shared state bugs)
- **No [SetUp]/[TearDown] attributes**: Uses constructor/Dispose pattern instead (more explicit, better for async)
- **Better parallel execution**: Designed for parallelization from the start (faster test runs)
- **Theory/InlineData**: More elegant parameterized tests than NUnit's TestCase
- **Community adoption**: Preferred by .NET Core team, ASP.NET Core, and modern .NET projects

```bash
# Run all tests
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj

# Run specific test class
dotnet test --filter "FullyQualifiedName~DeterminismTests"
dotnet test --filter "FullyQualifiedName~RigidBodyStabilityTests"

# Run single test
dotnet test --filter "FullyQualifiedName~DeterminismTests.TwoIdenticalSimulations_ProduceIdenticalResults"
```

**xUnit test attributes:**
- `[Fact]` - Single test method (like NUnit's `[Test]`)
- `[Theory]` + `[InlineData(...)]` - Parameterized tests (like NUnit's `[TestCase]`)
- Constructor/Dispose - Setup/teardown (instead of `[SetUp]`/`[TearDown]`)
- `Assert.Equal(expected, actual)` - Note the order (xUnit convention)

### Running Demos
```bash
# Run graphical demo (default)
dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj

# The demo shows real-time XPBD physics with Raylib rendering
```

## Architecture

### Project Structure

```
Evolvatron.Rigidon/        # Physics engine (CPU + GPU) - uses namespace Evolvatron.Core
├── IStepper.cs           # Stepper interface
├── CPUStepper.cs         # CPU reference implementation
├── WorldState.cs         # SoA particle arrays + constraints + colliders
├── SimulationConfig.cs   # All tunable parameters
├── Constraints.cs        # Rod, Angle, MotorAngle constraint structs
├── Colliders.cs          # CircleCollider, CapsuleCollider, OBBCollider
├── RigidBody.cs          # Rigid body struct with multi-geom support
├── RigidBodyJoint.cs     # Revolute joint for rigid bodies
├── RewardModel.cs        # RL reward shaping for landing task
├── Math2D.cs             # 2D geometry utilities
├── Physics/
│   ├── Integrator.cs         # Symplectic Euler integration
│   ├── XPBDSolver.cs         # XPBD constraint solver (particles)
│   ├── ContactConstraint.cs  # Contact detection logic
│   ├── Friction.cs           # Velocity-level friction (particles)
│   ├── ImpulseContactSolver.cs  # Sequential impulse solver (rigid bodies)
│   ├── RigidBodyJointSolver.cs  # Joint constraint solver (rigid bodies)
│   └── CircleCollision.cs    # Circle-circle collision detection
├── GPU/
│   ├── GPUStepper.cs              # ILGPU drop-in replacement
│   ├── GPUKernels.cs              # ILGPU particle kernel implementations
│   ├── GPUWorldState.cs           # Device memory management
│   ├── GPUDataStructures.cs       # GPU-compatible structs (particles, constraints)
│   ├── GPURigidBodyStructs.cs     # GPU structs for rigid bodies, joints, contacts
│   ├── GPURigidBodyContactKernels.cs  # Rigid body contact detection & solving
│   └── GPURigidBodyJointKernels.cs    # Revolute joint solver kernels
├── Templates/
│   ├── RocketTemplate.cs     # 5-particle rocket factory (XPBD particles)
│   ├── RigidBodyRocketTemplate.cs  # Rigid body rocket factory
│   └── RigidBodyFactory.cs   # Helper for rigid body construction
└── Scenes/
    ├── FunnelSceneBuilder.cs  # Creates funnel scene with static colliders
    └── ContraptionSpawner.cs  # Random contraption generation

Evolvatron.Demo/          # Visualization and demos
├── Program.cs            # Entry point
├── GraphicalDemo.cs      # Raylib-based interactive demo
├── FunnelDemo.cs         # Funnel scene demo
└── GPUBenchmark.cs       # CPU vs GPU performance comparison

Evolvatron.Tests/         # Unit and integration tests
├── DeterminismTests.cs          # Verifies deterministic behavior
├── RigidBodyStabilityTests.cs   # Rigid body physics validation
└── UnitTest1.cs                 # Basic unit tests
```

### Dual Physics Systems

This codebase contains **two separate physics systems** that coexist:

#### 1. XPBD Particle System (Original)
- **Components**: Particles (SoA arrays), Rod/Angle/MotorAngle constraints
- **Solver**: XPBDSolver (positional corrections over multiple iterations)
- **Contacts**: Particle vs static colliders only
- **Friction**: Velocity-level Coulomb friction (Friction.cs)
- **Use case**: Soft contraptions, articulated structures
- **Stepper path**: `CPUStepper` → `XPBDSolver` → `Friction.ApplyFriction`

#### 2. Rigid Body System (Added later)
- **Components**: RigidBody structs with multiple circle geoms, RevoluteJoints
- **Solver**: ImpulseContactSolver (sequential impulse, velocity-based)
- **Contacts**: Rigid body circle geoms vs static colliders
- **Friction**: Built into impulse solver with warm-starting
- **Use case**: Stable multi-body systems, joints, restitution
- **Stepper path**: `CPUStepper` → `ImpulseContactSolver` → `RigidBodyJointSolver`

**Important**: The two systems are solved **separately in the same simulation step**. They do not interact with each other. You can have both particles and rigid bodies in the same WorldState, but they won't collide with each other.

### Simulation Loop (CPUStepper.cs:SubStep)

```
1. Apply gravity to particles and rigid bodies
2. Save previous positions (for velocity stabilization)
3. Integrate velocities and positions (symplectic Euler)

PARTICLE CONSTRAINTS (XPBD):
4. Reset XPBD lambdas
5. For XpbdIterations (e.g., 12 times):
   - Solve Rod constraints
   - Solve Angle constraints
   - Solve MotorAngle constraints
   - Solve particle Contacts (vs static colliders)

RIGID BODY CONSTRAINTS (Sequential Impulse):
6. Initialize contact constraints (compute effective masses)
7. Warm-start with cached impulses
8. Initialize joint constraints
9. For XpbdIterations:
   - Solve contact velocity constraints
   - Solve joint velocity constraints
10. Solve joint position constraints

POST-PROCESSING:
11. Velocity stabilization (particles: v = (p_new - p_prev)/dt * beta)
12. Velocity-level friction pass (particles only; rigid bodies handle in solver)
13. Global damping (both systems)
```

### Key Numerical Methods

**XPBD (for particles)**: Positional constraint solver. Each constraint computes Lagrange multiplier update:
```
α = compliance / dt²
w = Σ invMass[i] * |∂C/∂x_i|²  (effective inverse mass)
Δλ = -(C + α*λ) / (w + α)
Δx_i = invMass[i] * Δλ * ∂C/∂x_i
```

**Sequential Impulse (for rigid bodies)**: Velocity-based solver. Computes impulses to correct constraint violations at velocity level, with position stabilization pass for joints.

**Determinism**: Fixed dt, deterministic math ops, CPU reference ensures reproducibility. GPU backend should match CPU within tolerance.

### Static Collider Types

All colliders are **static** (immovable):

1. **CircleCollider**: (cx, cy, radius)
2. **CapsuleCollider**: (cx, cy, ux, uy, halfLength, radius) — pill shape
3. **OBBCollider**: (cx, cy, ux, uy, hx, hy) — oriented rectangle

SDF (signed distance field) methods in Physics/ContactConstraint.cs compute distance and normals for collision response.

### Configuration (SimulationConfig.cs)

Key parameters:
- `Dt`: Fixed timestep (default 1/240s)
- `XpbdIterations`: Constraint solver iterations (default 12)
- `Substeps`: Subdivide each Step call (default 1)
- `GravityX, GravityY`: Gravity vector (default 0, -9.81 m/s²)
- `ContactCompliance`: XPBD contact softness (default 1e-8, nearly rigid)
- `RodCompliance, AngleCompliance`: Constraint softness (default 0 = rigid)
- `MotorCompliance`: Motor servo stiffness (default 1e-6)
- `FrictionMu`: Coefficient of friction (default 0.6)
- `Restitution`: Bounciness for rigid bodies (default 0.0)
- `VelocityStabilizationBeta`: Velocity correction factor (default 1.0)
- `GlobalDamping`: Per-second velocity decay (default 0.01)

Units: **SI (meters, kilograms, seconds)** throughout.

### GPU Backend (ILGPU)

The GPU backend (`GPUStepper.cs`) is a **complete** implementation of `IStepper`:
- Implements `IStepper` interface (drop-in replacement for CPUStepper)
- Uses ILGPU kernels in `GPU/GPUKernels.cs` and `GPU/GPURigidBody*.cs`
- Memory managed by `GPUWorldState.cs`

**Implemented Features:**
- ✅ **Particle XPBD physics** - Rods, angles, motors, contacts
- ✅ **Velocity stabilization kernel** - Corrects XPBD velocity drift
- ✅ **Friction kernel** - Coulomb friction for particles
- ✅ **Global damping kernel** - Velocity decay for both systems
- ✅ **Rigid body physics** - Gravity, integration, damping
- ✅ **Rigid body contact solver** - Circle, capsule, OBB static colliders
- ✅ **Revolute joint solver** - Position/velocity constraints, motors, limits

**Known Limitations:**
- ⚠️ **No warm-starting** - Contact impulses reset each frame (less stable than CPU)
- ⚠️ **No rigid body vs rigid body** - Only rigid body vs static colliders
- ⚠️ **Determinism** - GPU may have minor floating-point differences from CPU

**Use cases:**
- ✅ Particle-based physics with XPBD constraints
- ✅ Rigid body simulations with joints
- ✅ Batch simulations for evolution/RL
- ✅ Systems requiring friction and damping

**CPU stepper remains the reference implementation for correctness validation.**

To use GPU:
```csharp
var stepper = new GPUStepper();  // Drop-in replacement for CPUStepper
stepper.Step(world, config);     // Full particle + rigid body physics
```

## Templates and Factories

### RocketTemplate (Particle-based)
Creates a 5-particle XPBD rocket:
- Particles: 0=top, 1=bottom, 2=leftFoot, 3=rightFoot, 4=engine
- Rods connect all structural elements
- Angle constraints fix leg geometry
- MotorAngle on engine particle enables gimbal control
- Methods: `ApplyThrust`, `SetGimbal`, `GetCenterOfMass`, `GetVelocity`, `GetUpVector`

### RigidBodyRocketTemplate
Creates a rigid body rocket with revolute joint for engine gimbal:
- Core body: multi-geom rigid body (main fuselage + legs)
- Engine: separate rigid body connected via RevoluteJoint
- More stable and suitable for RL compared to particle rocket
- Methods: `ApplyThrust`, `SetGimbalTorque`, `GetCenterOfMass`, etc.

### Scenes
- **FunnelSceneBuilder**: Creates V-shaped funnel with landing pad
- **ContraptionSpawner**: Generates random connected particle structures

## Reward Model (RL Integration)

`RewardModel.cs` provides:
- `RocketObservation`: 8D observation vector (rel pos, vel, up vector, gimbal, throttle)
- `RewardParams`: Tunable weights for position, velocity, angle, control effort
- `StepReward()`: Returns shaped reward + terminal condition checks
- Terminal conditions: success (landed), crash (high impact), out-of-bounds

## Testing Strategy

Tests in `Evolvatron.Tests/`:

1. **DeterminismTests**: Verify identical initial conditions → identical trajectories (particles and rockets)
2. **RigidBodyStabilityTests**: Verify rigid body physics correctness (stacking, resting, joints)
3. Unit tests for individual constraint solvers, colliders, etc.

All tests use `CPUStepper` as ground truth. Tolerance: 1e-6f for determinism checks.

## Common Patterns

### Creating a Simple Simulation
```csharp
var world = new WorldState();
var config = new SimulationConfig();

// Add particles
int p0 = world.AddParticle(x: 0f, y: 5f, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);
int p1 = world.AddParticle(x: 1f, y: 5f, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);

// Add constraint
world.Rods.Add(new Rod(p0, p1, restLength: 1f, compliance: 0f));

// Add ground
world.Obbs.Add(OBBCollider.AxisAligned(cx: 0f, cy: -1f, hx: 10f, hy: 0.5f));

// Simulate
var stepper = new CPUStepper();
for (int i = 0; i < 1000; i++)
{
    stepper.Step(world, config);
}
```

### Working with Rocket Templates
```csharp
var world = new WorldState();
world.Obbs.Add(OBBCollider.AxisAligned(0f, -5f, 20f, 0.5f)); // ground

// Option 1: Particle rocket
var particleRocket = RocketTemplate.CreateRocket(world, centerX: 0f, centerY: 5f);
RocketTemplate.ApplyThrust(world, particleRocket, throttle: 0.7f, maxThrust: 100f);

// Option 2: Rigid body rocket (more stable)
var rbRocket = RigidBodyRocketTemplate.CreateRocket(world, centerX: 0f, centerY: 5f);
RigidBodyRocketTemplate.ApplyThrust(world, rbRocket, throttle: 0.7f, maxThrust: 100f);
```

## CRITICAL: GPU/CPU Physics Parity

**Any physics change MUST be made in BOTH the CPU and GPU code paths simultaneously.**

The codebase has two independent physics implementations:
- **CPU**: `CPUStepper.cs` + supporting solvers in `Physics/`
- **GPU**: `InlinePhysics.cs` (mega-kernel), `GPUKernels.cs`, `GPUBatchedStepper.cs`, etc.

These are completely separate code — they don't share a single line of physics logic. If you change gravity, friction, contact solving, integration, damping, or any other physics behavior in one, you **must** mirror it in the other.

**Why this matters:**
- Rocket evolution/training runs exclusively on GPU (via `GPURocketLandingMegaEvaluator` + `InlinePhysics.cs`)
- Trajectory optimization (`TrajectoryOptimizer`) runs on CPU (needs per-step Jacobian access)
- `RocketEnvironment` was deleted because maintaining two divergent physics paths caused hours of bugs — controllers trained on GPU failed on CPU due to float math differences
- GPU is the **single source of truth** for rocket evolution. There is no CPU rocket environment.

**What uses what:**
| Component | Physics Path | Purpose |
|-----------|-------------|---------|
| `GPURocketLandingMegaEvaluator` | GPU `InlinePhysics.cs` | Rocket evolution/training |
| `ObstacleLanderDemo` | GPU mega-kernel | GPU replay visualization |
| `TrajectoryOptimizer` | CPU `CPUStepper` | Levenberg-Marquardt optimization |
| `LunarLanderDemo` | CPU `CPUStepper` | Trajectory optimization visualization |
| XPBD particle tests | CPU `CPUStepper` | Physics engine unit tests |

## No ID-Based Lookups

**NEVER resolve relationships by scanning a list for a matching ID at runtime.**

IDs exist for exactly two things: serialization (JSON/disk) and GPU array indices (hardware constraint). In all C# and TypeScript runtime code, use direct object references. When deserializing, resolve IDs to references **once** during load, then discard the IDs.

```csharp
// WRONG — O(n) scan every time you need a relationship
var zone = world.Zones.First(z => z.Id == someId);

// RIGHT — direct reference, resolved at deserialization
checkpoint.NextCheckpoint  // already the object
```

If you're writing `.Find(x => x.Id ==` or `.Where(x => x.Id ==` in runtime code, the data model is wrong. Fix the model so the reference is direct.

## Notes and Gotchas

1. **Lambda Reset**: XPBD lambdas must be reset at the start of each step (done in `XPBDSolver.ResetLambdas`)
2. **Velocity Stabilization**: Critical for XPBD to prevent drift. Always enabled (beta=1.0) unless tuning
3. **Compliance Tuning**: Keep contact compliance small (~1e-8) to avoid sinking through ground
4. **Motor Compliance**: Keep motor compliance very small (~1e-6) for stiff servo behavior
5. **Particle Order**: Constraint indices (i, j, k) must be valid particle indices in WorldState
6. **SoA Layout**: Particles use Structure-of-Arrays for cache efficiency; avoid per-particle structs in hot paths
7. **No Particle-Particle Collisions**: Only particle-vs-static and rigid-body-vs-static supported
8. **Rigid Body Limitations**: Rigid bodies don't collide with particles or each other (only with static colliders)
9. **GPU/CPU Parity**: Any physics change must be mirrored in both implementations (see section above)

## GPU Safety — Avoiding Hard Crashes

GPU kernel bugs don't throw exceptions — they silently corrupt memory or poison the CUDA context,
which can hang the GPU driver so hard that Windows can't recover (no TDR event, forced reboot).

### Rules for GPU kernel code

1. **No unguarded math that can produce NaN/Inf.** `XMath.Exp(x)` overflows to Inf for x>88.
   `Inf/Inf = NaN`. NaN propagates silently through all downstream math and corrupts physics state.
   All `tanh` implementations in `DenseNN.cs` and `InlineNN.cs` clamp inputs to [-10,10] for this reason.
   **Any new activation function MUST be overflow-safe.**

2. **DenseNN local buffer limit: MaxLayerWidth=64.** `DenseNN.ForwardPass` allocates
   `LocalMemory.Allocate1D<float>(64 * 2)` for ping-pong hidden activations. A topology with any
   hidden or output layer wider than 64 writes past this buffer → GPU memory corruption.
   Evaluator constructors validate this, but if you add a new code path that creates DenseTopology,
   ensure it also validates. If 64 is ever increased, update ALL evaluators' validation AND the
   `DenseNN.MaxLayerWidth` constant together.

3. **InlinePhysics.cs hardcodes 3 bodies per world.** The cos/sin cache (lines 73-78) and the
   ternary body-index lookups (`geom.BodyIndex == 0 ? cos0 : (... == 1 ? cos1 : cos2)`) assume
   exactly 3 rigid bodies. If `BodiesPerWorld` ever changes, this code silently uses wrong angles
   for bodies with index >= 3, corrupting physics → NaN → crash. To support more bodies, replace
   the hardcoded cache with a loop or local array.

4. **Attractor proximity buffer: hardcoded `* 8`.** `DenseRocketLandingStepKernel.EvaluateZones`
   computes `proxIdx = worldIdx * 8 + ai`. The buffer is allocated as `worldCount * MaxAttractorsPerWorld`
   where `MaxAttractorsPerWorld = 8`. The host validates `Attractors.Count <= 8`, but the kernel
   uses a magic number. If MaxAttractorsPerWorld ever changes, update the kernel constant too.

5. **Always wrap `_accelerator.Synchronize()` in try-catch.** If a kernel crashed (OOB access, etc.),
   the CUDA context is permanently poisoned. Without try-catch, subsequent kernel launches operate on
   a dead context → undefined behavior → driver hang. The evaluators do this; any new GPU code must too.

6. **NaN guard after every NN forward pass.** All step kernels check `float.IsNaN(action)` after
   the NN produces outputs. If NaN is detected, the world is terminated immediately with poison
   fitness. This is defense-in-depth — the tanh clamp should prevent NaN, but if it sneaks through
   a new code path, this guard prevents it from entering the physics pipeline.

### Patterns that can trigger hard crashes (from ILGPU GitHub issues)

- **Multiple ILGPU contexts on the same GPU without a lock** → driver race condition
- **CopyToCPU without pinning memory** → GC relocates target during async copy → VRAM corruption
- **ILGPU OptimizationLevel.O1+** has known loop-unrolling bugs → silent wrong results → cascading corruption
- **Loading the same kernel on multiple GPUs simultaneously** → NVIDIA driver race ~10% of the time
- **Kernel exceeding TDR timeout (2s on Windows)** → if 5+ timeouts in 60s → BSOD `0x117`

## GPU Sync Policy

**Never add `_accelerator.Synchronize()` between GPU kernel launches unless you are about to read results back to CPU.**

ILGPU's default stream guarantees in-order execution — kernel B launched after kernel A will not start until kernel A completes. Explicit sync is only needed when:
1. **Before CPU reads**: `GetAsArray1D()`, `CopyToCPU()`, or any method that downloads GPU memory to host
2. **Before CPU-side branching on GPU state**: e.g., early-exit checks like `AllTerminal()` that read GPU flags
3. **After CPU-side buffer operations**: e.g., `ClearContactCounts()` if it's a CPU method modifying GPU-visible state

Unnecessary syncs force a full pipeline flush and CPU↔GPU round-trip. In a loop with many kernel launches per step (observe → forward pass → act → step), redundant syncs dominate wall-clock time. The rocket evaluators demonstrate the correct pattern: enqueue all kernels freely, sync only every N steps for early-exit checks, and one final sync before downloading results.

## Directory Conventions

- **`docs/`** — Permanent documentation and plans (tracked by git)
- **`scratch/`** — Temporary working files, logs, experiment output (gitignored)

## Development History

The project started with a pure XPBD particle system for soft contraptions. Rigid bodies with impulse-based solver were added later to improve stability for multi-body systems and provide proper joint constraints with revolute joints. This dual-system architecture allows both soft particle structures and stable rigid body mechanisms to coexist in the same simulation.
