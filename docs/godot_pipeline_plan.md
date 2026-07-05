# Godot Game Pipeline — Rocket Constructor + Controller Training

**Status:** planning (recon done; awaiting two architecture decisions before build).

## Goal (user's vision)
Wire the training pipeline into the actual game so a level/puzzle plays like this:
1. Bring in a rocket — an **existing** one (may already carry a trained controller → play
   immediately) or **build/modify** one in a rocket constructor.
2. If the rocket is new or its effectors/sensors changed, its controller is invalid →
   **train the controller first**, with live visual feedback.
3. The **player decides when to stop training** (good-enough is good-enough). Clear graphical
   indicators; render periodically so it feels responsive without wasting GPU.
4. Play the level with the trained controller.

## What already exists (recon)
- `Evolvatron.Godot/` — Godot **4.6.3**, .NET 8, C#. Main scene `Scenes/RocketReplay.tscn`
  → `Scripts/RocketReplayScene.cs`. References `Evolvatron.Rigidon` + `Evolvatron.Evolvion`
  and pulls in **ILGPU 1.5.3** → in-process GPU training works inside the game process.
- `RocketReplayScene` is **already a live trainer**: background-thread CEM via
  `GPUDenseRocketLandingEvaluator`, a control panel (reward-weight sliders, spawn width,
  rollouts, obstacle, **Go** button), HUD with generation/fitness/landing-rate, and it
  replays the latest generation's **top-N** individuals, cycling episodes.
- The rocket is **hardcoded** (`CreateReplayRocket`: body + 2 legs). No constructor.
- Replay is on **CPU** (`CpuDenseNN` + `CPUStepper`).

So ~70% of the "live training with feedback" machinery already exists — for the *old
single-stage landing task with a fixed rocket*. The work is to (a) generalize the rocket,
(b) point it at the right training pipeline, and (c) fix issues below.

## Known issues to fix
1. **CPU/GPU replay parity (important).** The scene trains on GPU but renders via a CPU
   reimplementation of the NN + physics. The project's own CLAUDE.md records that this exact
   divergence ("`RocketEnvironment` was deleted… controllers trained on GPU failed on CPU
   due to float math differences") caused hours of bugs and that **GPU is the single source
   of truth**. A player watching a CPU replay may misjudge "is it good enough to stop." → Fix
   by rendering from the **GPU evaluator's own replay path** (positions read back to CPU only
   for drawing), the same pattern `GPUDenseMazeEvaluator`/`GPUDenseRocketControlEvaluator`
   already use.
2. **Hardcoded rocket in 3+ places.** `RigidBodyRocketTemplate`, the Godot `CreateReplayRocket`,
   and each GPU evaluator's `CreateAndUploadRocketTemplate` each hand-build the same 3-body
   rocket. A constructor needs **one** `RocketSpec` (bodies, geoms, joints, thrust, gimbal,
   sensor rays) that every consumer reads. Removes duplication and is the constructor's output.
3. **Single GPU context.** Training (GPU) + a separate GPU replay = two ILGPU contexts on one
   GPU, which our GPU-safety notes flag as a driver-race risk. The render path must share the
   training context (see render cadence below).

## Render cadence (the "don't waste GPU" question)
Single GPU context on the training thread. Every **K generations**, snapshot the champion,
run a **short GPU replay** of it into trajectory buffers, and hand those to the render thread
to **play back** (the render thread draws/interpolates; it does not step physics). The next K
generations train while the captured trajectory animates. This gives smooth, **parity-correct**
viz, one context (no race), bounded GPU overhead, and is exactly the "train K, render 1" idea.
K is a slider (default chosen to feel responsive, ~1–4). A **Stop / Use this controller** button
freezes training and saves the champion to the rocket.

## Decisions (locked 2026-06-27)
- **A = Two-stage hierarchy.** A rocket owns a reusable **velocity controller** (9-D dynamics+
  command → throttle/gimbal), trained once per rocket and level-agnostic. A level owns a
  **navigation policy** (goal+sensors → velocity command) trained on top of the frozen
  controller. Both already exist as GPU evaluators (`GPUDenseRocketControlEvaluator`,
  `GPUDenseMazeEvaluator`). Bringing an old rocket carries its controller (skip controller
  training); a new level still trains a nav policy.
- **B = Freeform component editor.** A 2D canvas to place bodies/rods/joints/motors/sensors,
  edit their properties, test-drive in sim, and save. Larger build — phased below.

## Hard constraint: GPU mega-kernel topology is fixed at 3 bodies
`InlinePhysics.cs` hardcodes **3 rigid bodies per world** (cos/sin cache + ternary body lookups;
see CLAUDE.md GPU-safety note #3), and the evaluators allocate `BodiesPerWorld=3, GeomsPerWorld=19,
JointsPerWorld=2`. A freeform rocket has a *variable* body/geom/joint count, so training an
arbitrary spec on GPU requires generalizing the kernel. Plan: **padding to fixed maxima**
(`MaxBodies`, `MaxGeoms`, `MaxJoints`), kernel loops with bounds and skips inactive bodies
(`InvMass=0`). This is GPU-safety-critical work — do it carefully, after the editor exists, and
keep the CPU path (Rigidon already supports arbitrary topology) as the reference.

## Controller-state model (locked 2026-06-27, supersedes any "test-drive"/PID idea)
The editor must NOT fake flight with a hand-coded PID — that hides the game's premise (the player
trains the controller). Honest model:
- A rocket **owns a controller** (the stage-1 velocity controller) that is either **trained or absent**.
  A freshly built or **edited** rocket (topology / sensors / thrusters / joints changed) → controller
  **invalidated**, weights discarded, **training score reset**. The score is a property of the
  (rocket-spec ⇄ controller) pair.
- The controller carries a **training score** (fitness / quality), shown in the UI.
- **"Fly to Target" runs the ACTUAL controller network.** No controller (or untrained) ⇒ the real
  random-weight NN ⇒ visible **chaos**, shown plainly. The target → velocity-command nav layer is
  trivial/unlearned; the *learned* part that stabilises & tracks is the trained controller — never a PID.
- The player gets a **Train** action: train the controller (in-process CEM, reuse the
  `RocketReplayScene` live-trainer machinery), watch the score climb, **stop when good enough**, save
  the controller onto the rocket. An incoming rocket that already carries a trained controller skips
  training and flies immediately.
- Editing a rocket that HAS a trained controller must warn that it will reset the controller.

## Phased build order
- **P0 — `RocketSpec` foundation (CPU, low risk).** Serializable rocket definition (bodies,
  geoms, joints, actuators, sensors) in a canonical rest pose. Factory: spec → Rigidon
  `WorldState` rocket (CPU) and → GPU template arrays. Round-trip + factory tests. Re-express
  the current 3-body rocket as a spec to prove parity.
- **P1 — Freeform editor (Godot, CPU preview).** Canvas editor producing/editing a `RocketSpec`,
  with a CPU-sim "test drive" (drop it, manual thrust/gimbal, watch it). Save/load JSON. No GPU
  needed — full creative freedom here. **Headline deliverable.**
- **P2 — Generalize the GPU controller trainer to a `RocketSpec`** via the padding approach above.
  Validate trained-on-GPU == behaves-in-CPU-preview within tolerance on the stock rocket first.
- **P3 — Live trainer rework:** GPU-replay parity fix, K-gen render cadence, **Stop/Save
  controller** button; train a constructed rocket's controller end to end.
- **P4 — Pipeline + persistence:** save rocket+controller; on level load detect a compatible
  controller → play vs. train; per-level nav-policy training on the frozen controller; play.

## Open log
- 2026-06-27: Decisions locked (two-stage + freeform editor). Mapped construction surface.
- 2026-06-27: **P0 DONE.** `Evolvatron.Rigidon/Rockets/`: `RocketSpec` (+ Body/Geom/Joint/
  Thruster/Sensor specs, JSON, Validate, ValidateGpuLimits), `RocketSpecLibrary.StockRocket()`
  (reproduces the trained 3-body rocket), `RocketSpecFactory` (`ToCpuWorld`, `ToGpuWorld`
  padding-safe). Tests `RocketSpecTests` (5/5 pass): topology, GPU limits, JSON round-trip,
  CPU-layout parity, GPU geom-packing/striding parity. Stock rocket = 3 bodies / 19 geoms /
  2 joints / 1 gimbaled thruster / 8 sensors, ActuatorDof=2.
- 2026-06-27: **P1a DONE (compiles; needs visual confirmation in Godot).** `Scenes/RocketEditor.tscn`
  + `Scripts/RocketEditorScene.cs`: loads a `RocketSpec` (stock), renders bodies/joints/thruster/
  sensors, **test-drives** it with CPUStepper under manual control (Up/W thrust, A/D gimbal) over a
  ground plane, and saves/loads the spec as JSON (`user://rocket_spec.json`). Made the editor the
  Godot main scene (was `RocketReplay.tscn`; restore for the live trainer — proper menu is P4).
  Cannot self-verify visuals (no display) — user to run and confirm.
- 2026-06-27: **P1b DONE (compiles; runs clean headless).** Reworked `RocketEditorScene.cs` into an
  actual interactive editor: toolbar (Add Body / Add Sensor / Delete Selected / Test Drive / Reset /
  Save / Load), **click-to-select** a body (yellow highlight rings), **drag to move** a body (updates
  `BodySpec.X/Y`, rebuilds), delete with joint/thruster/sensor reindex. Visibility hardened: dark
  clear color, reference grid + ground/Y axes + origin marker, `Camera2D.MakeCurrent()` + initial
  position. Verified by running the scene under Godot **headless** (`--quit-after`): `_Ready`/`_Process`
  throw nothing, exit 0, no script errors. `_Draw` still needs a real window — user to confirm visuals.
  Godot binary: `C:\Dev\Godot\4.6.3-stable\...\Godot_v4.6.3-stable_mono_win64.exe`.
  - Known limitation: dragging a body does NOT re-anchor its joints yet (joint anchors are fixed in
    body-local frame), so moving a jointed body visibly stretches the joint. Joint re-anchoring + a
    property inspector (mass/inertia/radius, add-geom) are the next editor increment.
- 2026-06-27: **Engine upgraded 4.6.3 → 4.7-stable.** Installed Godot 4.7 mono to
  `%LOCALAPPDATA%\Programs\Godot\` (standard per-user spot; Program Files needs admin), removed the
  old `C:\Dev\Godot\` install. Bumped `Evolvatron.Godot.csproj` to `Godot.NET.Sdk/4.7.0` and
  `project.godot` features to "4.7"; reimported headless (no migration errors); full build 0/0.
  Added a `--shot=<abs.png>` screenshot harness to `RocketEditorScene` and **rendered the editor**
  windowed — confirmed visually: stock 3-body rocket, toolbar, 8 sensor rays, grid/axes, readouts.
  Fixed two label overlaps (title vs status, counts vs toolbar). Editor is verified rendering, not
  just compiling.
- 2026-06-27: **Dropped the PID autopilot; wired the REAL controller (P3 core).** Per user: no faking
  flight with a hand-coded controller. `RocketEditorScene` now models the honest controller lifecycle:
  - **Controller = trained-or-absent**, with a **score**, bound to a structural **signature** of the spec.
    Any edit (add/move/delete body, add sensor, load) that changes the signature **invalidates the
    controller and resets the score** (`OnSpecChanged`).
  - **Fly to Target** runs the ACTUAL net via `CpuDenseNN.ForwardPass` with the goal-relative
    observation layout ported verbatim from `RocketReplayScene` (`obs[0]=(comX−targetX)/20`, …). The
    **target is the pad** → move it, the rocket re-homes. **No controller ⇒ real random-weight net ⇒
    chaos**, labeled "UNTRAINED — CHAOS". Not a PID.
  - **Train Controller (G)** runs in-process CEM (`GPUDenseRocketLandingEvaluator` + `IslandOptimizer`,
    CEM cfg from the proven trainer) on a background thread; the champion is committed live each gen so
    that **flying while training shows the controller improving in real time**. Gated to the **stock
    rocket** (`CanTrain`); a constructed rocket shows "training needs P2".
  - **Verified headless:** `--train` → GPU evaluator inits on the 4090, fit 46→676 over 10 gens with
    landings appearing; `--fly` (untrained) → CpuDenseNN+physics run clean, exit 0, no NaN. Build 0/0.
  - Known: training the stock rocket to ~84% takes ~100+ gens (~minutes); editing-during-training isn't
    blocked yet; fly geometry vs training spawn distribution may need tuning for far/lateral targets.
- Next: P2 (RocketSpec → GPU control evaluator; headless-verifiable) — make
  `GPUDenseRocketControlEvaluator` build its rocket from `RocketSpecLibrary.StockRocket()` via
  `RocketSpecFactory.ToGpuWorld` and confirm training parity. Needs no display, so it proceeds in
  parallel with the user's visual check of the editor.

## Open log
- (now) Recon complete; plan written; asking A/B before building.
