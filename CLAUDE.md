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

**Evolvion Integration** (planned): An evolutionary neural controller framework (see Evolvion.md) will eventually be integrated to evolve fixed-topology neural controllers using large-scale parallel GPU execution. Species evolve differing network topologies while individuals within species differ only by weights and node parameters.

## Build and Test Commands

### Building
```bash
# Build entire solution
dotnet build Evolvatron.sln

# Build specific project
dotnet build Evolvatron.Core/Evolvatron.Core.csproj
dotnet build Evolvatron.Demo/Evolvatron.Demo.csproj
```

### Running Tests
```bash
# Run all tests
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj

# Run specific test class
dotnet test --filter "FullyQualifiedName~DeterminismTests"
dotnet test --filter "FullyQualifiedName~RigidBodyStabilityTests"

# Run single test
dotnet test --filter "FullyQualifiedName~DeterminismTests.TwoIdenticalSimulations_ProduceIdenticalResults"
```

### Running Demos
```bash
# Run graphical demo (default)
dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj

# The demo shows real-time XPBD physics with Raylib rendering
```

## Architecture

### Project Structure

```
Evolvatron.Core/           # Core physics engine (CPU + GPU)
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
│   ├── GPUStepper.cs         # ILGPU drop-in replacement
│   ├── GPUKernels.cs         # ILGPU kernel implementations
│   ├── GPUWorldState.cs      # Device memory management
│   └── GPUDataStructures.cs  # GPU-compatible structs
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
├── RocketDemo.cs         # Rocket landing demo
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

The GPU backend (`GPUStepper.cs`) is a drop-in replacement for `CPUStepper`:
- Implements `IStepper` interface identically
- Uses ILGPU kernels in `GPU/GPUKernels.cs`
- Memory managed by `GPUWorldState.cs`
- **Note**: Currently incomplete — missing velocity stabilization, friction, and damping kernels on GPU path. These steps are skipped or done on CPU.
- CPU stepper is the **reference implementation** for correctness.

To use GPU:
```csharp
var stepper = new GPUStepper();  // instead of new CPUStepper()
stepper.Step(world, config);
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

## Notes and Gotchas

1. **Lambda Reset**: XPBD lambdas must be reset at the start of each step (done in `XPBDSolver.ResetLambdas`)
2. **Velocity Stabilization**: Critical for XPBD to prevent drift. Always enabled (beta=1.0) unless tuning
3. **Compliance Tuning**: Keep contact compliance small (~1e-8) to avoid sinking through ground
4. **Motor Compliance**: Keep motor compliance very small (~1e-6) for stiff servo behavior
5. **Particle Order**: Constraint indices (i, j, k) must be valid particle indices in WorldState
6. **SoA Layout**: Particles use Structure-of-Arrays for cache efficiency; avoid per-particle structs in hot paths
7. **No Particle-Particle Collisions**: Only particle-vs-static and rigid-body-vs-static supported
8. **Rigid Body Limitations**: Rigid bodies don't collide with particles or each other (only with static colliders)
9. **GPU Incompleteness**: GPU path skips some post-processing steps; use CPU for correctness validation

## Development History

The project started with a pure XPBD particle system for soft contraptions. Rigid bodies with impulse-based solver were added later to improve stability for multi-body systems and provide proper joint constraints with revolute joints. This dual-system architecture allows both soft particle structures and stable rigid body mechanisms to coexist in the same simulation.
