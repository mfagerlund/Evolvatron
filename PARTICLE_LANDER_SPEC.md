# Project: Particle Lander (XPBD)

A lightweight 2D physics sandbox for building “contraptions” (e.g., a rocket with legs and a gimbaled engine) from particles and constraints. Collisions are **particle vs. static colliders only**. The system includes reward signals suitable for RL experiments (e.g., landing on a pad). The primary target implementation is **C#** with GPU acceleration optional (ILGPU recommended) and a minimal visualization.

---

## 1) Goals & Non‑Goals

### Goals
- Simple, fast 2D physics based on **particles + XPBD constraints**.
- Static colliders: **OBB (rotatable rectangles), capsules/pills, circles**.
- Contraptions composed from particles with: distance (rod), angle, and motorized angle constraints.
- Contact handling via **XPBD positional solve** + a separate **velocity-level friction** pass.
- Deterministic(ish) single-thread or reproducible GPU execution (fixed dt).
- Reward shaping and termination logic for a “lander” task.
- Tiny demo: spawn random contraptions at the top and **funnel** them through a static scene into a landing area.

### Non‑Goals
- No particle–particle collisions.
- No deformables/fluid/cloth.
- No dynamic colliders.
- No broadphase; scene has few colliders and few particles (up to ~5–200 per scene).

---

## 2) High-Level Architecture

```
ParticleLander
├── Core
│   ├── SimulationConfig
│   ├── WorldState         // particles + constraints + colliders
│   ├── Stepper            // runs one fixed-timestep step
│   └── RewardModel        // computes rewards & terminal states
├── Physics
│   ├── Integrator         // symplectic Euler
│   ├── XPBD               // distance / angle / motor constraints
│   ├── Contacts           // particle vs static collider + friction
│   └── Math2D             // geometry & SDF utilities
├── Rendering (demo)
│   └── SimpleRenderer     // lines, circles, OBB/capsule outlines, debug
└── Demo
    ├── FunnelSceneBuilder // builds static colliders
    └── Spawner            // random contraptions (rockets & junk)
```

- **GPU path (optional):** mirror the kernels with ILGPU; same APIs, different backend. CPU path must exist for simplicity and determinism.

---

## 3) Data Model (SoA for performance, simple POCOs for API)

### Particles (N)
- `float[] posX, posY`
- `float[] velX, velY`
- `float[] invMass`  // 0 = pinned
- `float[] radius`

### Constraints
- **Rod**: `(int i, int j, float rest, float compliance, float lambda)`
- **Angle**: `(int i, int j, int k, float theta0, float compliance, float lambda)`
- **MotorAngle**: `(int i, int j, int k, float target, float compliance, float lambda)`
  - `i–j–k` define the angle at `j`.
  - `target` can be updated per-step (for controllers).
  - `lambda` values persist within a time step (reset each step) and accumulate over XPBD sub-iterations.

### Static Colliders (small counts: 1–64 each)
- **Circle**: `(cx, cy, r)`
- **Capsule**: `(cx, cy, ux, uy, halfLen, r)` // `u` unit axis
- **OBB**: `(cx, cy, ux, uy, hx, hy)` // half-extents `h`, unit axis `u`

### Scene
- Metadata: `gravity`, `dt`, `substeps`, `xpbdIterations`, `contactCompliance`, `frictionMu`, `velocityStabilizationBeta` (0..1), `globalDamping`.
- Spawn points & bounds for culling/reset.

---

## 4) Simulation Loop (Fixed dt)

For each frame (or step):
1. **External forces**: gravity to all non-pinned particles; controllers add thrust forces to designated particles.
2. **Integrate (symplectic Euler)**: `v += dt * force/m ; p += dt * v`. Clear forces after.
3. **XPBD Sub-iterations** (repeat `xpbdIterations`, e.g., 12):
   - Solve **rods**.
   - Solve **angles**.
   - Solve **motor angles** (using current `target`).
   - Solve **contacts** (particle vs static colliders; push out).
4. **Velocity stabilization**: `v = (p_new − p_prev)/dt * beta + v * (1-beta)`.
5. **Friction pass (velocity-level)** for particles in contact with the most-penetrating collider.
6. **Damping/clamps** and remove out-of-bounds particles/contraptions.

**Determinism knobs:** fixed `dt`, deterministic math ops, CPU path for reference runs.

---

## 5) Numerical Methods

### XPBD Scalar Pattern
For constraint `C(x) = 0` (or `≥ 0` for contacts):
```
α  = compliance / dt^2            // 0 for rigid
w  = Σ invMass[i] * |∂C/∂x_i|^2   // effective inverse mass along gradient
Δλ = -(C + α λ) / (w + α)
Δx_i = invMass[i] * Δλ * ∂C/∂x_i
λ ← λ + Δλ
```
- `λ` reset at start of the step; accumulate over sub-iterations.

### Distance (rod) Constraint
- `C = |p_i − p_j| − L0`
- Gradient `n = (p_i − p_j)/|…|`; apply to i (+n) and j (−n).

### Angle Constraint (i–j–k ≈ θ0)
- Angle at vertex `j`. Use 2D gradients derived from normalized edge vectors.
- Handle wrapping; keep `θ` in [−π, π].

### Motorized Angle (servo)
- Identical to angle constraint, but `theta0 = target(t)` supplied each step.
- Compliance small (e.g., `1e−6`) to avoid jitter; tune separately from rods.

### Contact (particle vs collider)
- Use **signed distance** `φ(p)` and **unit normal** `n(p)` from collider SDF.
- Constraint `C = φ(p) ≥ 0`. If `C < 0`, perform XPBD pushout along `n`.
- Inflate collider by particle radius: subtract `radius_i` from `φ`.

#### SDF/Closest-Point Helpers
- Circle: distance to center; `n = normalize(p − c)`.
- Capsule: closest point on segment; distance to that point; `n = normalize(p − q)`.
- OBB: transform `p` to box local frame via axes `u` and `v` (perp of `u)`; compute inside/outside distances as usual; pick normal from face or corner gradient; convert back to world.

### Friction (velocity-level, Coulomb-like)
- Re-evaluate contact normal `n` for a particle.
- Split velocity `v = v_n n + v_t`. If penetrating previously (within tolerance):
  - Allowed tangential speed change `≤ μ * |v_n|` per step.
  - Shrink `v_t` magnitude accordingly: `v_t ← v_t * max(0, 1 − μ |v_n| / |v_t|)`.
- This is a practical scheme that works well with XPBD position corrections.

---

## 6) Rocket Example (5‑Particle Template)

Indices: `0=top`, `1=bottom`, `2=leftFoot`, `3=rightFoot`, `4=engine`.

**Rods**: `(0–1)`, `(0–2)`, `(1–2)`, `(0–3)`, `(1–3)`, `(0–4)`, `(1–4)`.

**Angles**: fix leg opening angles, e.g., `∠(2–0–1)=θL`, `∠(3–1–0)=θR`.

**Motor**: `Angle(i=4, j=0, k=1)` with target `φ ∈ [−15°, +15°]` (gimbal). The “engine” thrust can be modeled either as additional velocity kick on particle 4 or as external forces; torque-like effects emerge from off-axis placement and constraints.

**Reward Observations (example 8D)**
- Relative COM `(x−x_pad, y−y_padTop)`; COM velocity `(vx, vy)`.
- Upright vector from `(top − bottom)` normalized → `(ux, uy)`.
- Gimbal command and throttle (if actuated).

**Reward Shaping**
- Step reward: `r = −k_p‖pos_err‖ − k_v( |vx| + w_y|vy| ) − k_a * angle_err^2 − k_u*(Δthrottle^2 + Δgimbal^2) + k_alive`.
- Terminal bonus: success if inside pad & `|vx|, |vy|` below limits & `angle_err ≤ θ_max` → `+R_land`.
- Terminal penalty: crash/out-of-bounds.

---

## 7) Demo: “Funnel” Scene

**Static Colliders**
- Two OBB walls forming a V funnel (symmetric about x-axis), meeting above a flat ground OBB.
- Square landing pad OBB centered at bottom.
- A few capsules and circles as bumpers.

**Spawner**
- Periodically spawn a random contraption at the top: random particle count (5–12), random masses/radii, connect via rods (spanning tree) + a few angles. Optionally include one motorized angle.

**Lifecycle**
- Fixed `dt = 1/240 s`, `xpbdIterations = 12` (tunable), optional `substeps = 1–2`.
- Cull & respawn when out of bounds.
- Log landings (met success condition) and simple stats (time to land, bounces).

---

## 8) Public API (CPU baseline; GPU backend should match)

```csharp
public sealed class SimulationConfig {
  public float Dt = 1f/240f;
  public int XpbdIterations = 12;
  public int Substeps = 1;
  public float GravityX = 0f, GravityY = -9.81f;
  public float ContactCompliance = 1e-8f;
  public float RodCompliance = 0f;
  public float AngleCompliance = 0f;
  public float MotorCompliance = 1e-6f;
  public float FrictionMu = 0.6f;
  public float VelocityStabilizationBeta = 1.0f;
  public float GlobalDamping = 0.01f; // per second
}

public interface IStepper {
  void Step(WorldState world, SimulationConfig cfg);
}

public sealed class WorldState {
  // Particles (SoA under the hood; convenience mutators provided)
  public int ParticleCount { get; }
  public Span<float> PosX { get; }
  public Span<float> PosY { get; }
  public Span<float> VelX { get; }
  public Span<float> VelY { get; }
  public Span<float> InvMass { get; }
  public Span<float> Radius { get; }

  // Constraints
  public List<Rod> Rods { get; }
  public List<Angle> Angles { get; }
  public List<MotorAngle> Motors { get; }

  // Static colliders
  public List<CircleCol> Circles { get; }
  public List<CapsuleCol> Capsules { get; }
  public List<OBBCol> Obbs { get; }
}

public static class RewardModel {
  public static float StepReward(in WorldState w, in RewardParams p, out bool terminal, out float terminalReward);
}
```

**Renderer (demo only)**
- `SimpleRenderer.Draw(WorldState)` that shows: particles (radius), rods (lines), angle arcs (optional), colliders (wireframe), COM marker, and landing pad highlight.

---

## 9) GPU Backend (Optional ILGPU)

- Each constraint/contact becomes an **AutoGrouped** kernel operating over its array.
- Keep SoA buffers in device memory; one host→device upload at init; per-step only tiny control updates.
- Kernels:
  1. `IntegrateKernel (per particle)`
  2. `SolveRodsKernel (per rod)`
  3. `SolveAnglesKernel (per angle)`
  4. `SolveMotorsKernel (per motor)`
  5. `SolveContactsKernel (per particle)`
  6. `FrictionKernel (per particle)`
- Synchronize once at end of step; copy back positions only for rendering.
- CPU stepper must remain the reference implementation; GPU is a drop‑in.

---

## 10) Configuration & Defaults

- `dt = 1/240`, `xpbdIterations = 12`, `substeps = 1`.
- Compliance: rods/angles `0`, motor `1e−6`, contacts `1e−8`.
- Friction μ = `0.6` (pad), others `0.3–0.5`.
- Damping `0.01`.
- Radii: feet slightly larger than core particles.

---

## 11) Testing Plan

1. **Unit**
   - Rod preserves length under oscillatory integration.
   - Angle converges to `theta0` from random pose.
   - Contact pushout against each collider type; no tunneling at modest speeds.
   - Friction reduces tangential velocity within Coulomb bound.

2. **Integration**
   - Five-particle rocket lands on square pad with scripted throttle/gimbal.
   - Funnel scene: contraptions are steered into the exit with plausible bounces.

3. **Determinism**
   - CPU stepper produces identical trajectories across runs (fixed seeds).
   - GPU and CPU agree within tolerance over 5 seconds of sim.

---

## 12) Deliverables

- `ParticleLander.Core` — CPU reference implementation
- `ParticleLander.Gpu` — ILGPU backend (optional)
- `ParticleLander.Render` — simple 2D renderer (lines/circles)
- `ParticleLander.Demo` — funnel demo + rocket landing example
- README with build/run instructions

---

## 13) Milestones

1. **M1 — CPU Core**: particles, rods, contacts (circle), demo with ground + pad.
2. **M2 — Colliders**: add capsule & OBB SDFs; friction pass; angle constraint.
3. **M3 — Motors & Rocket**: motorized angle, 5‑particle rocket lands with a simple controller.
4. **M4 — Funnel Demo**: random contraptions, funnel scene, logging & success stats.
5. **M5 — GPU** (optional): migrate kernels to ILGPU and validate against CPU baseline.

---

## 14) Notes & Gotchas

- Use **meters, kilograms, seconds**. Pick consistent scales (gravity ~10 m/s², sizes 0.1–10 m).
- Keep XPBD compliance small but nonzero for contacts to reduce jitter.
- Apply **velocity stabilization**; otherwise XPBD-only systems can accrue drift.
- Prefer SoA arrays for hot loops; copy into simple structs only at API boundaries.
- With only particle–static collisions, cost is linear in (particles × colliders) — manageable at small counts.

