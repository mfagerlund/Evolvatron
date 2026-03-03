# Trajectory Optimizer: Optimal Rocket Landing via Least Squares

## Context

Evolvatron trains neural controllers to land rockets via evolutionary optimization (Evolvion).
The core challenge is **reward engineering** — designing reward landscapes that lead to good
landing behavior. But what if we could compute the *optimal* control sequence directly?

By using Colonel's Levenberg-Marquardt least squares solver, we can find the best possible
throttle+gimbal sequence for any starting position. This gives us:
- **Gold-standard reference trajectories** for validating reward designs
- **Training data** for imitation learning (bypass reward engineering entirely)
- **Insight** into what optimal landing actually looks like

## Approach

**Single-shooting trajectory optimization** with finite-difference Jacobian.

- **Parameters**: `double[2*N]` — throttle and gimbal at each of N control steps
- **Residuals**: Per-step penalties for position error, velocity, angle, control smoothness, fuel
- **Jacobian**: Computed via finite differences through physics simulation, exploiting causal
  sparsity (perturbing step `s` only affects steps `s..N-1`)
- **Solver**: Colonel's dense `NonlinearLeastSquaresSolver.Solve()` — the problem is small enough
  (60 params, 240 residuals) that sparse isn't needed

### Control Frequency

- Physics: 120 Hz (matching RocketEnvironment)
- Control: ~8 Hz (every 15 physics steps)
- Episode: ~3.75 seconds (30 control steps)
- Parameters: 60 (30 × 2), Residuals: ~240 (30 × 8)

### Residual Formulation (8 per control step)

At each control step `s`, after simulating with controls `0..s`:

```
w_t = (s + 1.0) / N                          // Time weight: increases toward landing

r[8s+0] = w_t * wPos * (x - padX)            // Horizontal position error
r[8s+1] = w_t * wPos * (y - padY)            // Vertical position error
r[8s+2] = w_t * wVel * velX                  // Horizontal velocity
r[8s+3] = w_t * wVel * velY                  // Vertical velocity
r[8s+4] = w_t * wAngle * angleError          // Angle from upright
r[8s+5] = wSmooth * (throttle[s] - prev)     // Throttle smoothness
r[8s+6] = wSmooth * (gimbal[s] - prev)       // Gimbal smoothness
r[8s+7] = wFuel * throttle[s]                // Fuel usage
```

The solver minimizes `sum(r_i^2)`, so these are sqrt-weighted.

### Finite-Difference Jacobian with Causal Optimization

1. **Baseline rollout**: Simulate full trajectory, save `WorldStateSnapshot` at each control
   step boundary
2. **Per column**: Perturb `params[col]`, restore snapshot at affected step `s = col/2`,
   re-simulate from step `s` to end, compute `(perturbed - baseline) / epsilon`
3. Only rows `>= 8*s` can change — upper triangle is structurally zero

Total physics cost per Jacobian: `sum_{s=0}^{29} 2*(30-s)*15 ≈ 14,000` physics steps (~70ms).
With ~20 LM iterations: **~2-3 seconds total optimization time**.

## Files to Create

### 1. `Evolvatron.Evolvion/TrajectoryOptimization/WorldStateSnapshot.cs` (~60 lines)

Captures and restores rigid body dynamic state (position, velocity, angle for each body).

```csharp
public sealed class WorldStateSnapshot
{
    private readonly RigidBody[] _bodies;
    private readonly RevoluteJoint[] _joints;

    public static WorldStateSnapshot Capture(WorldState world);
    public void Restore(WorldState world);
}
```

Only snapshots `RigidBodies` and `RevoluteJoints` lists (value-type structs, cheap to copy).
Static colliders and geoms don't change.

### 2. `Evolvatron.Evolvion/TrajectoryOptimization/TrajectoryResult.cs` (~40 lines)

```csharp
public sealed class TrajectoryResult
{
    public double[] Controls;              // [throttle_0, gimbal_0, ...]
    public TrajectoryState[] States;       // State at each control step
    public bool Success;
    public double FinalCost;
    public int Iterations;
    public double ComputationTimeMs;
}

public struct TrajectoryState
{
    public float X, Y, VelX, VelY, Angle, AngularVel;
}
```

### 3. `Evolvatron.Evolvion/TrajectoryOptimization/TrajectoryOptimizer.cs` (~250 lines)

Main class. Key design:

```csharp
public sealed class TrajectoryOptimizer
{
    public TrajectoryOptimizer(
        int controlSteps = 30,
        int physicsStepsPerControl = 15,
        float maxThrust = 200f,
        float maxGimbalTorque = 50f,
        float targetX = 0f,
        float targetY = -4.5f);

    public TrajectoryResult Optimize(
        WorldState world,
        int[] rocketIndices,
        SimulationConfig config,
        double[]? initialControls = null,
        LeastSquaresOptions? solverOptions = null);
}
```

Implementation details:
- Creates a fresh `CPUStepper` for rollouts (avoids warm-start cache interference)
- `initialControls = null` → hover estimate: `throttle = g / (maxThrust * invMass)`, `gimbal = 0`
- Clamps controls to valid range during rollout: `throttle ∈ [0,1]`, `gimbal ∈ [-1,1]`
- Epsilon for finite differences: `1e-4`
- Default solver options: `MaxIterations=30`, `InitialDamping=1e-2`, `AdaptiveDamping=true`

### 4. `Evolvatron.Tests/TrajectoryOptimizerTests.cs` (~80 lines)

```csharp
[Fact] CostDecreasesAfterOptimization()     // FinalCost < initial cost
[Fact] RocketEndsNearPad()                  // Final position within 3m of pad
[Fact] FinalVelocityIsLow()                 // Final speed < 5 m/s
```

## File to Modify

### `Evolvatron.Evolvion/Evolvatron.Evolvion.csproj`

Add reference to Colonel.Core (the optimization solver lives there):
```xml
<ProjectReference Include="..\..\Colonel\Colonel.Core\Colonel.Core.csproj" />
```

## Key Dependencies

| What | Where | Used For |
|------|-------|----------|
| `NonlinearLeastSquaresSolver.Solve()` | Colonel.Core.Optimization | LM solver (dense) |
| `ResidualEvaluation` | Colonel.Core.Optimization | Residuals + Jacobian return type |
| `LeastSquaresOptions` | Colonel.Core.Optimization | Solver configuration |
| `RigidBodyRocketTemplate` | Evolvatron.Core.Templates | Rocket creation + control |
| `WorldState`, `RigidBody` | Evolvatron.Core | Physics state |
| `CPUStepper` | Evolvatron.Core | Physics simulation |
| `SimulationConfig` | Evolvatron.Core | Physics parameters |

## Edge Cases & Mitigations

- **Contact discontinuities**: LM damping naturally handles noisy Jacobians by increasing λ
- **Solver noise from XPBD iterations**: Use epsilon=1e-4 (above float solver noise)
- **Poor initial guess**: Hover estimate provides a reasonable starting point
- **Control bounds**: Clamped in rollout; LM can explore outside bounds but physics sees clamped values

## Verification

```bash
dotnet test Evolvatron.Tests --filter "FullyQualifiedName~TrajectoryOptimizerTests"
```

Then inspect the output: `FinalCost`, rocket final position, convergence reason. A successful
optimization should show cost decreasing over 10-30 iterations, with the rocket ending near
the pad at low velocity.
