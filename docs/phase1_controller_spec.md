# Phase 1 — The Maneuvering Controller (skill pre-training)

**Status:** implemented — smoke-tested on RTX 4090 (champion 48/100 held-out after 30 gens; trend up).
Files: `ES/DenseTopology.ForRocketController`, `GPU/MegaKernel/DenseRocketControl{Config,StepKernel}.cs`,
`GPU/GPUDenseRocketControlEvaluator.cs`, runner `dotnet run --project Evolvatron.TrainingRunner -- --control`.
**Goal:** Train **one** reusable neural controller that makes the rocket *track motion commands* — rotate, go forward/back, go sideways, hover — with **no pad, no obstacles, no maze**. Freeze it, then drive it from a thin task/guidance policy (Phase 2) across many mazes.

This is the "learn to fly first, solve the puzzle second" decomposition. It is **not** RL on the task; it's a small, dense, command-tracking problem that the existing CEM stack solves directly.

---

## 0. Why this shape

The expensive part of joint learning is forcing one policy to *simultaneously* discover (a) how its actuators move it and (b) where to go. (a) is **task-independent and stationary**; (b) changes per maze. We learn (a) once, freeze it, and reuse it. Bonus: failure becomes *attributable* —

| Symptom | Diagnosis |
|---|---|
| Phase 1 won't converge | vehicle uncontrollable, or tracking reward degenerate |
| Phase 1 fine, Phase 2 fails on **every** maze | capability envelope too weak → **hardware signal** (add a thruster) |
| Phase 2 fails on **one** maze | that maze's reward design is wrong |

Joint learning smears all three into "it didn't work."

---

## 1. The vehicle we're controlling (from the code)

`DenseRocketLandingStepKernel.StepKernel` applies actions exactly as:

```
throttle = clamp(action0, 0, 1)          // one-directional thrust along body axis
gimbal   = clamp(action1, -1, 1)         // direct torque
thrust   = throttle * MaxThrust           // MaxThrust   = 200
body.VelX      += cos(angle) * thrust * InvMass    * dt
body.VelY      += sin(angle) * thrust * InvMass    * dt
body.AngularVel+= gimbal * MaxGimbalTorque * InvInertia * dt   // MaxGimbalTorque = 50
```

So the rocket is the **planar PVTOL / under-actuated lander**:
- Thrust is **only along the body axis** (`up = (cos θ, sin θ)`), magnitude ≥ 0. **No lateral force.**
- A direct torque rotates it.
- `θ = π/2` ⇒ `up = (0,1)` ⇒ pointing straight up.
- Physics: `Dt = 1/120`, gravity `(0, −9.81)`, `GlobalDamping = 0.02/step`, `AngularDamping = 0.1`, 3 rigid bodies (fuselage + 2 legs), 6 solver iterations.

**Consequence that drives the whole design:** to translate sideways the controller must *tilt → thrust → level*. "Sideways" is therefore an **emergent** behavior, not a directly commandable axis. That is exactly the limitation that, if it makes mazes too hard, becomes the "add a lateral thruster" signal.

---

## 2. The command interface (the crux)

**Command = desired velocity in world frame, 2-D: `(cmdVx, cmdVy)`.** Nothing else.

Rationale — this is the *abstract capability* ("where I want to go"), deliberately decoupled from the actuators:

- **Subsumes everything the maze needs.** "Move toward goal at speed v" = `(cmdVx, cmdVy)`. Hover = `(0,0)`. Descend = `(0, −v)`.
- **Heading is left free.** We do **not** command orientation. Commanding heading on an under-actuated vehicle creates infeasible setpoints (can't always hold heading *and* translate). By leaving heading emergent, *every* command inside the achievable speed envelope is feasible, so the tracking reward is never fighting an impossible target.
- **Upright-at-hover falls out for free.** To hold zero velocity against gravity the controller must thrust upward ⇒ must point up. So "command velocity → 0" near the pad naturally produces an upright, settled rocket. Landing = the guidance layer driving `cmd → 0` over the pad; the controller ends upright without being told to.
- **Hardware-transparent.** Add a lateral thruster later → the interface stays 2-D, the maze policies stay byte-for-byte identical, and only the (cheap) controller is retrained. It now satisfies the same commands *directly* instead of by tilting — and mazes that were impossible may suddenly pass.

**Documented escape hatches (do NOT add by default):**
- Add `cmdθ` (desired heading) as a 3rd command **only** if precise landing attitude proves insufficient. Accept that `(cmdVx, cmdVy, cmdθ)` may be jointly infeasible — and that infeasibility *is* the under-actuation telemetry.
- Switch command from velocity to acceleration if the guidance layer needs tighter authority (lower-level interface, guidance does more work).

---

## 3. Observation & output layout

Phase 1 needs a **dynamics-only** observation (no pad-relative position — there is no pad). New layout, distinct from the landing kernel's 8-D obs.

### Controller input — 9 floats (recommended)

| idx | value | scaling | note |
|----|-------|---------|------|
| 0 | `cos θ` (upX) | — | orientation |
| 1 | `sin θ` (upY) | — | orientation |
| 2 | `angVel` | `/10` | body-frame ang. velocity |
| 3 | `errFwd` | `/10` | **velocity error, body frame**: forward component of `(cmd − v)` |
| 4 | `errLat` | `/10` | velocity error, body frame: lateral component of `(cmd − v)` |
| 5 | `speed` | `/10` | `‖v‖` current speed magnitude (helps damping) |
| 6 | `curThrottle` | — | last action (continuity / waggle awareness) |
| 7 | `curGimbal` | — | last action |
| 8 | `gUp` | — | gravity component along body up = `(−g·up)/‖g‖`; lets one controller generalize across gravity |

**Why body-frame error (idx 3–4), not raw world-frame velocity + raw command:** rotating the velocity error into the body frame makes the controller **orientation-equivariant** — "I need to accelerate forward and a little left in *my* frame" is directly actionable (thrust adds forward; torque changes which way forward points). This shrinks the net and generalizes across absolute heading. World-frame is a valid ablation, but body-frame is the recommended default.

`errFwd = (cmd−v)·up`, `errLat = (cmd−v)·right`, where `up=(cosθ,sinθ)`, `right=(sinθ,−cosθ)`.

### Output — 2 floats (unchanged from landing)

| idx | value | post-processing |
|----|-------|-----------------|
| 0 | throttle | `clamp(·,0,1)` |
| 1 | gimbal | `clamp(·,−1,1)` |

`DenseNN.ForwardPass` already squashes outputs with `tanh` → `[−1,1]`; throttle gets clamped to `[0,1]` after.

### Topology

```csharp
// New factory mirroring DenseTopology.ForRocket / ForDPNV
DenseTopology.ForRocketController(new[] { 16, 16 });  // 9→16→16→2 = 466 params
```

All layer widths ≤ 64 (GPU `DenseNN.MaxLayerWidth` hard limit — validated in evaluator ctor).

---

## 4. The tracking reward (the Phase-1 "game")

Per step, after physics:

```
verr   = ‖v − cmd‖                       // world-frame velocity error
track  = 1 − min(verr / VErrScale, 1)    // in [0,1], 1 = perfect tracking; VErrScale ≈ 8 m/s
reward += RewardTrackWeight * track       // RewardTrackWeight ≈ 1.0 per step
reward -= RewardEffortWeight * (dThrottle² + dGimbal²)   // reuse existing "waggle" accumulator; ≈ 0.05
```

Optional shaping (start without, add only if needed):
- `reward -= AngVelPenalty * min(|angVel|/10, 1)` — discourages spinning while tracking.
- Tiny upright bias **only when** `‖cmd‖ < ε`: `reward -= UprightWeight * angleErr` — sharpens hover-to-upright (usually emerges on its own).

**Terminal conditions** (free space — no ground, no pad, no obstacles):
- Airborne tumble: `angleErr > π/2` ⇒ terminate, poison-ish penalty (mirrors landing kernel's mid-air tumble check).
- NaN/Inf after forward pass ⇒ terminate, `fitness = −1e6` (**keep this guard verbatim** from the landing kernel — it's the anti-corruption tripwire).
- `steps ≥ MaxSteps` ⇒ normal end. `MaxSteps ≈ 300` (commands change within the episode, so short is fine).

No `GroundY` collider in Phase 1: spawn in open space, let it fly. (Either omit ground/pad colliders entirely, or push `GroundY` far below the operating region.)

---

## 5. Episode = a *command schedule*, not a single command

A single static command per episode only teaches steady-state holding. To learn the **transitions** the user actually named ("rotate, forward, back, sideways"), each episode runs a **piecewise-constant command schedule**: resample `(cmdVx, cmdVy)` every `K` steps.

- `K ≈ 60–90` steps (~0.5–0.75 s) per segment; 3–5 segments per `MaxSteps=300` episode.
- Commands sampled from the achievable envelope: `‖cmd‖ ≤ CmdSpeedMax` (≈ 6 m/s), uniformly in direction, with a healthy fraction of **`(0,0)` hover** segments and **direction-reversal** segments (forces decelerate + re-accelerate, the hard transitions).
- **Determinism:** precompute the full schedule **on CPU** and upload, exactly like spawn positions are precomputed in `CreateAndUploadRocketTemplate`. Per-world schedule keyed by `(baseSeed + conditionIdx)`. No GPU RNG.

Initial state per condition is also randomized (like spawns): random orientation `θ ∈ π/2 ± SpawnAngleRange`, small `angVel`, small initial velocity. This is the controller analog of multi-position training.

---

## 6. Multi-condition aggregation (reuse the proven pattern)

Mirror `GPUDenseRocketLandingEvaluator.EvaluateMultiSpawn` exactly — it already loops conditions, re-seeds, runs the sim loop, and aggregates. Only the per-condition payload changes:

```
for condition in 0..numConditions:
    seed = baseSeed + condition
    upload initial state + command schedule for this seed
    reset episode state
    run MaxSteps of the fused control kernel
    read per-world accumulated tracking reward

solvedCount[i] = # conditions where mean per-step track ≥ TrackSolveThreshold   (≈ 0.7)
fitness[i]     = solvedCount[i] * MaxSteps + meanTrackReward[i]
```

`solvedCount` is the integer headline metric (the controller analog of "landings"); the `* MaxSteps` term gives CEM the same lexicographic "solve more conditions first, then refine" pressure that works on DPNV and rocket landing.

**Condition count:** start at **16–25** per the project's multi-position doctrine (≥5 minimum, 25 is the DPNV sweet spot). Command variety is the regularizer here — more conditions = broader command envelope coverage = better generalist.

---

## 7. CEM configuration (identical to the working rocket/DPNV setup)

```csharp
var config = new IslandConfig
{
    IslandCount       = 1,
    Strategy          = UpdateStrategyType.CEM,
    InitialSigma      = 0.25f,
    MinSigma          = 0.08f,
    MaxSigma          = 2.0f,
    CEMEliteFraction  = 0.01f,
    CEMSigmaSmoothing = 0.3f,
    CEMMuSmoothing    = 0.2f,
    StagnationThreshold = 9999,   // single island, no reinit
};

var topology  = DenseTopology.ForRocketController(new[] { 16, 16 });
using var eval = new GPUDenseRocketControlEvaluator(topology);   // new evaluator, §8
int gpuCap    = eval.OptimalPopulationSize;                       // SMs * 4 * warpSize
var optimizer = new IslandOptimizer(config, topology, gpuCap);    // Glorot init, unchanged

for (int gen = 0; gen < maxGenerations; gen++)   // ~150–300 gens
{
    var pv = optimizer.GeneratePopulation(rng);
    var (fit, solved, _, _) = eval.EvaluateMultiCondition(
        pv, optimizer.TotalPopulation, numConditions: 20, baseSeed: gen * 100);
    optimizer.Update(fit, pv);
}
var (mu, _) = optimizer.GetBestSolution();   // <-- this is the frozen controller
// serialize mu to controller.bin (same BinaryWriter format as TrainingRunner)
```

No normalization, no weight decay, no pre-activation scaling (all proven harmful — keep the kernel clean).

---

## 8. What to build (new files; nothing existing is modified destructively)

1. **`GPU/MegaKernel/DenseRocketControlStepKernel.cs`** — fork of `DenseRocketLandingStepKernel`:
   - Replace §1 observation build with the 9-D dynamics+command layout (§3). Read this world's current command segment from an uploaded `Commands` buffer (`cmdVx[worldIdx*segments + seg]`, `seg = step / K`).
   - Keep §2 `DenseNN.ForwardPass` **and the NaN/Inf guard verbatim**.
   - Keep §3 action application (`throttle/gimbal → thrust/torque`) and `InlinePhysics.SubStepOneWorld` verbatim.
   - Replace zone/pad/settling logic with the tracking reward (§4) + tumble/NaN/MaxSteps terminals only.
2. **`GPU/GPUDenseRocketControlEvaluator.cs`** — fork of `GPUDenseRocketLandingEvaluator`:
   - Drop zones, obstacles, sensors, pad, settling params.
   - Add command-schedule buffers + CPU precompute (mirror `CreateAndUploadRocketTemplate`'s spawn precompute).
   - `EvaluateMultiCondition(...)` ≈ `EvaluateMultiSpawn(...)` with the §6 aggregation.
   - Keep the GPU-safety scaffolding (`Synchronize` in try/catch, topology width/IO validation, batch-of-10 dispatch).
3. **`ES/DenseTopology.cs`** — add `ForRocketController(int[] hidden)` (input 9, output 2), alongside `ForRocket` / `ForDPNV`.
4. **Runner / test** — mirror `Evolvatron.TrainingRunner/Program.cs`: train, print `solved/total` per gen, evaluate champion across 100 fresh conditions, serialize `controller.bin`.

GPU-parity note: Phase 1 is **GPU-only**, like all rocket evolution. No CPU control env (per CLAUDE.md — GPU is the single source of truth; don't reintroduce a divergent CPU path).

---

## 9. Phase 2 hook (out of scope here — sketch only)

Phase 2 chains two forward passes per step in a `DenseRocketMazeStepKernel`:

```
policyNN(taskObs)          → (cmdVx, cmdVy)     // per-individual weights, evolved
controllerNN(dynObs, cmd)  → (throttle, gimbal) // ONE shared frozen weight set, uploaded once
apply actions → InlinePhysics.SubStepOneWorld
```

- `taskObs` = pad-relative position + sensors (the existing 8/12-D landing obs).
- Controller weights come from `controller.bin`, uploaded once, **shared across all worlds** (constant, not per-individual). Policy weights are per-individual as today.
- "Modify a little" = optional **small additive residual** on `cmd` (or on `throttle/gimbal`), learned in Phase 2, with the base controller frozen. Keeps one-controller-many-mazes reuse; **per-maze full fine-tuning is forbidden** (destroys reuse).

---

## 10. First experiment / acceptance

**Cheapest decisive test before any maze work:** does one controller generalize across the command envelope?

1. Train per §7 (~150 gens, 20 conditions). Watch `solved/total` climb.
2. Champion metric: median per-step tracking reward + `% conditions solved` on **100 held-out conditions** (fresh `baseSeed`).
3. **Visual deliverable:** drive the frozen controller with a hand-authored square-wave command — `up → hover → left → right → descend` — and replay (reuse `PrepareMultiReplay`-style step loop). Watch it tilt-thrust-level. This is the "it learned to fly" screenshot.

**Pass bar (initial):** ≥ 80% of held-out conditions solved at `TrackSolveThreshold = 0.7`, smooth (low-waggle) actuation, upright hover on `cmd=(0,0)`.

**If it fails to reach the envelope edges** (commands saturate, lateral tracking poor): that's the under-actuation ceiling — the documented trigger to (a) widen training / lengthen episodes, or (b) add a lateral thruster and retrain *only this controller* (interface unchanged ⇒ Phase 2 untouched).

---

## Open knobs (decide during implementation, not now)

- `CmdSpeedMax`, `VErrScale`, segment length `K`, hover fraction.
- Body-frame vs world-frame error in obs (recommend body-frame; ablate).
- Whether to include `gUp` (idx 8) — drop it if single-gravity, keep for multi-planet reuse.
- 9→16→16→2 vs a wider/deeper net (stay ≤ 64 width).
