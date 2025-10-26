# Rigidon

**Rigidon** is the high-performance 2D physics engine powering **Evolvatron**.

## Overview

Rigidon is a deterministic, fixed-timestep physics simulator designed for large-scale parallel execution on both CPU and GPU. It supports two complementary physics systems:

1. **XPBD Particle System**: Position-based constraint solver for soft, articulated structures
2. **Rigid Body System**: Impulse-based velocity solver for stable multi-body systems with joints

Both systems share the same simulation loop and can coexist in the same world, though they don't interact with each other directly.

## Key Features

- **Deterministic simulation**: Fixed timestep with reproducible results across runs
- **Dual backend**: CPU reference implementation + ILGPU GPU acceleration
- **Particle constraints**: Rods (distance), angles (bend), motor angles (servo control)
- **Rigid bodies**: Multi-geometry bodies with revolute joints, restitution, and friction
- **Static colliders**: Circles, capsules, and oriented bounding boxes (OBB)
- **Contact solver**: XPBD for particles, sequential impulse for rigid bodies
- **Numerical stability**: Velocity stabilization, warm-starting, and configurable compliance

## Architecture

### Core Components

- `IStepper`: Common interface for physics steppers
- `CPUStepper`: Reference CPU implementation (always correct)
- `GPUStepper`: ILGPU-accelerated GPU implementation
- `WorldState`: Structure-of-Arrays (SoA) layout for particles, constraints, colliders, and rigid bodies
- `SimulationConfig`: All tunable parameters (timestep, iterations, compliance, friction, etc.)

### Physics Modules

- **Integrator**: Symplectic Euler integration
- **XPBDSolver**: Position-based constraint solver for particles
- **ImpulseContactSolver**: Velocity-based sequential impulse solver for rigid bodies
- **RigidBodyJointSolver**: Revolute joint constraint solver
- **ContactConstraint**: Collision detection using signed distance fields (SDF)
- **Friction**: Coulomb friction model (particles use velocity-level, rigid bodies use impulse-based)

### Templates and Scenes

- **RocketTemplate**: 5-particle XPBD rocket with gimbal control
- **RigidBodyRocketTemplate**: Rigid body rocket with revolute joint for stable flight
- **FunnelSceneBuilder**: Creates V-shaped landing scenarios
- **ContraptionSpawner**: Random connected structure generator

## Physics Systems

### XPBD Particle System

- Particles stored as SoA arrays (position, velocity, mass, radius)
- Constraints: Rod (distance), Angle (bend), MotorAngle (servo)
- Position-based solver with configurable compliance (stiffness)
- Velocity stabilization to prevent drift
- Particle-vs-static collider contacts only

### Rigid Body System

- Multi-geometry rigid bodies (circles only currently)
- Revolute joints for articulation
- Sequential impulse solver with warm-starting
- Restitution (bounciness) and friction
- Position-level joint stabilization

## Usage

```csharp
// Create world and configuration
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

## Configuration

Key tunable parameters in `SimulationConfig`:

- **Dt**: Fixed timestep (default 1/240s)
- **XpbdIterations**: Constraint solver iterations (default 12)
- **GravityX, GravityY**: Gravity vector (default 0, -9.81 m/s²)
- **ContactCompliance**: Contact softness (default 1e-8, nearly rigid)
- **RodCompliance, AngleCompliance**: Constraint softness (default 0 = rigid)
- **FrictionMu**: Coefficient of friction (default 0.6)
- **Restitution**: Bounciness for rigid bodies (default 0.0)
- **GlobalDamping**: Per-second velocity decay (default 0.01)

All units are **SI (meters, kilograms, seconds)**.

## GPU Acceleration

Switch between CPU and GPU by changing the stepper:

```csharp
// CPU (reference)
var stepper = new CPUStepper();

// GPU (ILGPU)
var stepper = new GPUStepper();
```

**Note**: The GPU backend is currently incomplete. Some post-processing steps (velocity stabilization, friction, damping) may be missing or fall back to CPU. Always use `CPUStepper` for validation.

## Testing

Run tests to verify determinism and stability:

```bash
# All tests
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj

# Determinism verification
dotnet test --filter "FullyQualifiedName~DeterminismTests"

# Rigid body stability
dotnet test --filter "FullyQualifiedName~RigidBodyStabilityTests"
```

## Integration with Evolvatron

Rigidon provides the physics substrate for Evolvatron's rocket landing challenge. The reward model (`RewardModel.cs`) shapes RL training by providing:

- **Observations**: Position, velocity, orientation, gimbal angle, throttle
- **Shaped rewards**: Distance to target, velocity alignment, orientation, control effort
- **Terminal conditions**: Success (landed), crash (high impact), out-of-bounds

## License

Part of the Evolvatron project.
