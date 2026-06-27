# Phase 2 — Maze Navigation on a Frozen Maneuvering Controller

Status: **spec / implementing**. Decisions: controller **hard-frozen** (no residual adaptation —
cleanest test of the velocity-command interface); maze layouts **procedurally generated**
(seeded, difficulty = obstacle count). Build order: (1) vertical slice — hierarchical flight to
a goal in open space (gravity on, no obstacles) to validate the two-NN pipeline + frozen-weight
forward pass; (2) add body-frame sensors + procedural obstacles + collision.

Builds directly on Phase 1 (`docs/phase1_controller_spec.md`). Phase 1 produced a
velocity-tracking controller that is excellent **within the rocket's physical envelope**:
given gentle commands (≤3 m/s, slow switches) the plain 16→16 reactive net tracks to
**0.25 m/s settled / 87% held-out**. The hard-command ~2 m/s floor was proven physical
(reward steepness, width, horizon, and Elman recurrence all failed to move it — see
`memory/phase1-controller-physical-floor.md`). Phase 2 therefore **commands the frozen
controller gently** and learns navigation on top.

## 1. Architecture — hierarchical, two networks per step

```
maze obs ──► [MAZE POLICY]  ──► desired velocity (cmdVx, cmdVy), clamped to ±CmdSpeedMax
  (evolved)      NN #1                         │
                                               ▼
dynamics + command ──► [FROZEN CONTROLLER] ──► throttle, gimbal ──► physics substep
   9-D obs (Phase 1)        NN #2 (controller_easy.bin, shared, constant)
```

- **NN #1 (maze policy)** — evolved by CEM, **per-world weights** (population varies).
  Sees the maze; outputs a *gentle* world-frame velocity command.
- **NN #2 (controller)** — **frozen**, loaded once from `controller_easy.bin`, the **same
  shared weight array for every world**. This is the Phase-1 net; its weights never change.
- Two `DenseNN.ForwardPass` calls per step, one per network, in one fused kernel.

The whole point of the task-space command interface: the maze policy never sees throttle or
gimbal. If we later swap the rocket (e.g. add a lateral thruster, #3) and retrain only the
controller, the maze policy is unaffected as long as the command interface (2-D velocity) holds.

## 2. Command interface (maze policy → controller)

- Maze policy raw outputs `(a, b)` ∈ [-1,1] (Tanh). Command = `(a, b) * CmdSpeedMax`.
- `CmdSpeedMax = 3.0` m/s — the envelope where Phase 1 tracks tightly. **Do not exceed**;
  outside it the controller can't follow and a maze failure would be uninterpretable.
- The command updates **every step** (the maze policy is the high-level loop). The Phase-1
  controller was trained on piecewise-constant commands that switch every ~1s; per-step
  commands are smoother than that, so they stay inside the trained regime.

## 3. Controller observation (NN #2 input) — unchanged from Phase 1

Computed in-kernel from the maze command + current dynamics, identical 9-D layout:
`[upX, upY, angVel/10, errFwd/10, errLat/10, speed/10, curThrottle, curGimbal, gUp]`
where `errFwd/errLat` are the body-frame components of `(cmd − comVel)`. Reuse the exact
obs block from `DenseRocketControlStepKernel`.

## 4. Maze observation (NN #1 input)

World-frame navigation signals + body-frame proximity (mirror the landing kernel's sensors):
1. Goal-relative position `(goalX − comX, goalY − comY)` / scale (2)
2. COM velocity `(velX, velY)` / 10 (2)
3. Attitude `upX, upY` (2) — so the policy knows the body frame its sensors live in
4. `K` body-frame distance sensors via `RayVsOBB` against `SharedOBBColliders` (default K=8;
   the landing kernel uses 4 — extend to 8 for finer obstacle sensing)

→ maze InputSize = 6 + K (≈14). OutputSize = 2.

## 5. Reward (navigation)

Dense shaping + sparse events, same structure as the landing reward:
- **Progress**: `+w_prog * (dist_prev − dist_now)` toward goal each step (potential-based)
- **Goal reached** (COM within `GoalRadius`): large `+GoalBonus`, terminal **success**
- **Collision** (contact with collider index ≥ `FirstObstacleIndex`): `−CollisionPenalty`,
  terminal **fail** (reuse landing kernel's obstacle-hit detection)
- **Tumble** (attitude error > π/2): `−TumblePenalty`, terminal
- **Timeout** (MaxSteps): terminal, keep accumulated progress
- Optional small effort/heading penalties later; start minimal.

Solve metric: reached goal before timeout. Aggregate across mazes like Phase 1:
`solvedMazeCount * MaxSteps + meanShapedReward`.

## 6. Multi-maze training (generalization)

Train on **N maze layouts per generation** (the maze analog of multi-position / multi-command).
Each layout = `{start pose, goal position, obstacle set}`, seeded by `baseSeed + mazeIdx`,
shared across the whole population for fair CEM. Start simple (few obstacles, then more).
Goal: one maze policy that generalizes across layouts, exactly as the user envisioned
("use it for several different mazes").

## 7. GPU implementation plan

- **`DenseMazeStepKernel.cs`** (new) — fused: maze obs → NN#1 → command → controller obs →
  NN#2 → actuators → `InlinePhysics.SubStepOneWorld` → progress/goal/collision/terminal.
  Two NaN guards (one per forward pass). Reuse `RayVsOBB`, obstacle-hit logic, COM/attitude
  blocks from the landing + control kernels.
- **`DenseMazeConfig.cs`** (new) — layout params (goal, radius, bonuses, sensor count/range,
  CmdSpeedMax, FirstObstacleIndex) + maze-policy NN layout. Frozen-controller NN layout passed
  alongside (reuse `DenseRocketControlConfig` fields for the controller half).
- **`GPUDenseMazeEvaluator.cs`** (new) — holds **two** weight buffers: per-world maze policy
  (evolved) + a single shared frozen-controller buffer (uploaded once from `controller_easy.bin`).
  `EvaluateMultiMaze(...)`. Reuse `GPUBatchedWorldState`, `SharedOBBColliders`, reset kernel.
- **`TrainingRunner --maze`** — load `--controller controller_easy.bin`, evolve the maze policy.
- **`Demo maze`** — grid replay: rocket + obstacles + goal, draw the commanded vs actual
  velocity arrows (as in the control demo) plus the goal and a success/fail tint.

## 8. Frozen-controller loading

`controller_easy.bin` is `9→16→16→2` (466 params). Validate the loaded length against the
controller topology at startup. Upload once to a shared (non-per-world) device buffer; the
maze kernel indexes it with a fixed base (worldIdx-independent) for NN #2.

## 9. Failure attribution (decision tree from the original vision)

- **Maze policy fails on ALL layouts even with gentle commands** → the command interface is
  too weak for the task → revisit #3 (lateral thruster) to expand the physical envelope, then
  retrain only the controller (maze policy interface unchanged).
- **Fails on one specific maze** → reward/curriculum design for that layout, not the controller.
- **Controller can't hold the commanded velocity in the maze** → command budget too aggressive;
  lower `CmdSpeedMax` / smooth commands. (Shouldn't happen at ≤3 m/s per Phase 1.)

## 10. GPU-safety (per CLAUDE.md)

- NaN/Inf guard after BOTH forward passes; poison fitness + terminal on detection.
- Frozen-controller and maze-policy layer widths ≤ `DenseNN.MaxLayerWidth` (64); validate in ctor.
- `try/catch` around `Synchronize`.
- Obstacles via `SharedOBBColliders`; keep `MaxContactsPerWorld` scaled to obstacle count
  (`24 + obstacles*4`, as the landing evaluator does).
- Mirror any physics change in CPU + GPU paths (N/A here — pure GPU evolution path).
