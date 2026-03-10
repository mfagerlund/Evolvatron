# Game Integration Plan: Editor World → GPU Training

## Architecture Context

**Godot is the final client.** The current TypeScript web editor and any web-based
visualization are temporary scaffolding. In production, the Godot client and the
C# training backend run **in the same process** — no HTTP API, no REST, no SSE.

**What's permanent:**
- `Evolvatron.Rigidon` — physics engine (CPU + GPU)
- `Evolvatron.Evolvion` — evolutionary framework (CEM, dense NN, GPU evaluators)
- `SimWorld` data model and JSON format — the contract between editor and trainer
- `SimWorldLoader` — JSON → C# objects
- GPU kernels, `MegaKernelConfig`, blittable structs

**What's temporary:**
- TypeScript web editor (replaced by Godot scene editor)
- Any web API or HTTP layer
- Any web-based training visualization

Design decisions should never be shaped by the temporary UI. The JSON format
is the bridge: Godot will export the same `SimWorld` JSON that the web editor
does today.

---

## IRON LAW: No ID-Based Lookups

**Use direct object references. Always. Everywhere.**

IDs are acceptable at exactly two boundaries:
1. **Serialization** — writing to JSON/disk (you need a flat format)
2. **GPU buffers** — array indices in GPU memory (hardware constraint)

Everywhere else — in C# runtime objects, in TypeScript editor state, in any
in-memory graph — **hold a direct reference to the object you mean**. If you
need to find something, you should already have a reference to it. If a
relationship exists, model it as a field holding the related object, not a
string you grep for in a list.

When deserializing, resolve IDs to references **once, immediately**, then
throw the IDs away. The runtime data model uses objects pointing to objects.
Period.

```csharp
// WRONG — hunting through lists by ID
var zone = world.DangerZones.First(z => z.Id == someId); // NO

// RIGHT — direct reference, resolved at load time
rewardZone.DangerZone  // already the object you need
```

If you catch yourself writing `.Find(x => x.Id ==` or `.Where(x => x.Id ==`
in runtime code, stop. You have a design problem. Fix the data model so the
reference is direct.

---

## Current State

**What exists and works:**
- TypeScript editor exports `SimWorld` JSON (landing pad, spawn, obstacles,
  checkpoints, speed zones, danger zones, attractors, reward weights)
- C# `SimWorld` types + `SimWorldLoader.FromJson()` load and validate the JSON
- `GPUDenseRocketLandingEvaluator.Configure(SimWorld)` wires editor data into GPU
- `GPUDenseRocketLandingEvaluator` accepts `DenseTopology` + flat param vectors,
  runs CEM training on GPU with fused mega-kernel
- Obstacles flow as `List<GPUOBBCollider>` → GPU shared collider buffer
- CEM proven: 83-84% landing rate on basic + obstacle scenarios
- Phase 1 tested: JSON → Configure → 10 CEM gens → nonzero fitness

**What's missing:**
- Fitness function is hardcoded in `DenseRocketLandingStepKernel.ComputeFitness`
- Reward weights from editor are ignored
- No checkpoints, zones, or attractors on GPU
- No standalone training runner (file-watch → train loop)

---

## Phase 1: C# World Model + JSON Ingest ✅ DONE

**Goal**: Load the editor's `SimWorld` JSON into C# objects that directly
configure the evaluator. No training changes yet — just prove the data flows.

- `Evolvatron.Evolvion/World/SimWorld.cs` — C# types mirroring editor export
- `Evolvatron.Evolvion/World/SimWorldLoader.cs` — deserialize, degrees→radians,
  sort checkpoints, validate
- `GPUDenseRocketLandingEvaluator.Configure(SimWorld)` — single entry point
- Tests: 4 unit (deserialization, angles, sorting, validation) + 2 integration
  (property mapping, 10-gen training run)

---

## Phase 2: Parameterized Fitness Function ✅ DONE

**Goal**: Make the GPU kernel's fitness function respond to player-designed
reward weights instead of hardcoded constants.

### 2.1 Extend MegaKernelConfig

Add reward weight fields to the blittable struct:

```csharp
// In MegaKernelConfig (already a blittable struct passed to GPU)
public float RewardPositionWeight;     // editor: positionWeight
public float RewardVelocityWeight;     // editor: velocityWeight
public float RewardAngleWeight;        // editor: angleWeight
public float RewardAngVelWeight;       // editor: angularVelocityWeight
public float RewardControlWeight;      // editor: controlEffortWeight
```

### 2.2 Update ComputeFitness in GPU Kernel

Replace hardcoded `20f`, `5f` constants with config fields:

```csharp
// Before (hardcoded):
fitness += 20f * closeBonus;
fitness -= 5f * sp;

// After (parameterized):
fitness += config.RewardPositionWeight * closeBonus;
fitness -= config.RewardVelocityWeight * sp;
```

### 2.3 Wire Through Evaluator

`SimWorld.RewardWeights` → evaluator properties → `MegaKernelConfig` fields
→ GPU kernel reads them. Same pattern as existing physics params.

### Test

Train same scenario with different reward weight profiles:
- High position weight, zero velocity → agent should hover near pad
- High velocity penalty → agent should descend slowly
- Verify meaningfully different champion behaviors

---

## Phase 3: Reward Zones on GPU ✅ DONE

**Goal**: Checkpoints, danger zones, speed zones, and attractors evaluated
per-step in the GPU kernel, producing shaped reward that guides evolution.

### 3.1 GPU Reward Zone Structs

Blittable structs for each zone type, uploaded as GPU array buffers
(same pattern as `GPUOBBCollider` for obstacles):

```csharp
[StructLayout(LayoutKind.Sequential)]
public struct GPUCheckpoint
{
    public float X, Y, Radius;
    public int Order;           // sequence constraint
    public float RewardBonus;
    public float InfluenceRadius;
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUDangerZone
{
    public float CX, CY, HalfExtentX, HalfExtentY;
    public float PenaltyPerStep;
    public int IsLethal;        // 1 = terminal on contact
    public float InfluenceRadius;
}

// Similarly: GPUSpeedZone, GPUAttractor
```

### 3.2 Add Zone Counts to MegaKernelConfig

```csharp
public int CheckpointCount;
public int DangerZoneCount;
public int SpeedZoneCount;
public int AttractorCount;
```

### 3.3 Kernel Integration

Add zone evaluation to `DenseRocketLandingStepKernel.StepKernel`, after
physics step and before terminal check:

```
For each zone type:
  Loop through zone array (count from config)
  Test rocket CoM against zone geometry
  Accumulate per-step reward/penalty in episode buffer
  Handle terminal zones (lethal danger zones)
  Track checkpoint sequence in per-world state
```

### 3.4 Episode State Extension

Add per-world accumulators:
- `float[] ZoneRewardAccum` — running sum of zone rewards
- `int[] CheckpointProgress` — bitmask or counter of reached checkpoints
- `byte[] EnteredAttractor` — per-attractor flags for one-time contact bonus

Fold `ZoneRewardAccum` into final fitness alongside the base fitness.

### 3.5 DenseNN Views Extension

New `ArrayView` parameters on the kernel for zone buffers (checkpoints,
danger zones, speed zones, attractors). Follows existing pattern of
`PhysicsViews`, `EpisodeViews`.

### Test

Create scenario with:
- Funnel obstacles + 2 checkpoints above the funnel
- Danger zone on one side
- Attractor near landing pad
- Verify champions route through checkpoints and avoid danger zone

---

## Phase 4: Training Runner (File-Watch) ✅ DONE

**Goal**: A standalone console app that watches a `SimWorld` JSON file, runs
CEM training when it changes, and prints progress to stdout. No HTTP, no API.

This is temporary scaffolding — in production, Godot calls `Configure()` and
runs the training loop directly in-process.

### 4.1 Console App (`Evolvatron.TrainingRunner/`)

```
dotnet run --project Evolvatron.TrainingRunner -- world.json
```

- Loads `world.json` via `SimWorldLoader.FromJson()`
- Configures evaluator via `Configure(world)`
- Runs CEM training loop, prints per-generation stats
- Watches file for changes: on modification, stops current run, reloads, restarts
- Ctrl+C for clean shutdown

### 4.2 TrainingSession

```csharp
class TrainingSession
{
    SimWorld World;
    GPUDenseRocketLandingEvaluator Evaluator;
    IslandOptimizer Optimizer;
    CancellationToken Cancel;

    // Runs training loop, reports progress per generation
    void Run();
}
```

- One session at a time (single GPU)
- CancellationToken for clean abort on file change or Ctrl+C
- Console output: gen number, best fitness, landing rate, elapsed time

### 4.3 Champion Export

After training completes or is stopped, write champion params to a file
(flat float array or JSON). This is what Godot will eventually load to
replay the trained agent.

### Test

Export default world from editor → save as `world.json` → run training
runner → watch it train → edit `world.json` → verify it reloads and restarts.

---

## Phase Summary

| Phase | What | Depends On | Core Deliverable |
|-------|------|-----------|-----------------|
| 1 ✅ | C# world model + JSON ingest | Nothing | `SimWorldLoader` + `evaluator.Configure(world)` |
| 2 ✅ | Parameterized fitness weights | Phase 1 | Player reward weights affect training |
| 3 ✅ | Reward zones on GPU | Phase 2 | Checkpoints, zones, attractors in kernel |
| 4 ✅ | Training runner (file-watch) | Phase 1 | Console app: JSON file → CEM training |

Phases 2-3 (fitness) and Phase 4 (runner) are independent after Phase 1.
They can be developed in parallel.

**The game loop becomes testable at Phase 4** — design a world in the editor,
save JSON, training runner picks it up and trains. Phases 2-3 make training
*responsive* to the player's reward design.

**Godot integration** replaces Phase 4's file-watching with direct in-process
calls: the Godot editor builds a `SimWorld`, calls `Configure()`, and runs
the training loop on a background thread. The core types and evaluator API
are identical — only the triggering mechanism changes.
