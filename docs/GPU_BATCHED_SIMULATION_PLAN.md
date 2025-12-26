# GPU-Batched Physics Simulation Plan for Evolvatron

## Executive Summary

This plan outlines the implementation of GPU-batched physics simulation to enable massive parallelism (1000+ agents) for evolutionary training. The core challenge is extending the current single-world GPU physics to batch N independent simulations, each with their own dynamic environment state (e.g., collectable targets).

## The Core Problem

**Current Flow (CPU):**
```
For each individual:
  Create WorldState + CPUStepper
  Reset environment (creates rocket, targets)
  For each timestep:
    Neural net forward → actions
    Apply actions to rocket
    Step physics
    Check target collisions
    Compute reward
  Return fitness
```

**Desired Flow (GPU):**
```
Create BatchedWorldState for N agents
Reset all N environments (N rockets, N*M targets)
For each timestep:
  Batch neural net forward → N actions
  Apply N actions to N rockets
  Step N physics worlds in parallel
  Check N*M target collisions
  Compute N rewards
Aggregate N fitness values
```

## Target Architecture

```
+------------------------------------------------------------------+
|                    GPU BATCHED SIMULATION                         |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+    +----------------------------+          |
|  |  GPUBatchedWorld  |    |  GPUBatchedEnvironment     |          |
|  +-------------------+    +----------------------------+          |
|  | N copies of:      |    | Per-agent state:           |          |
|  | - Rigid bodies    |    | - Target positions (x M)   |          |
|  | - Geoms           |    | - Targets active (bitmask) |          |
|  | - Joints          |    | - Cumulative reward        |          |
|  | - Colliders       |    | - Step counter             |          |
|  | - Constraints     |    | - Terminal flag            |          |
|  +--------+----------+    +-------------+--------------+          |
|           |                             |                         |
|           v                             v                         |
|  +--------------------------------------------------+             |
|  |           GPUBatchedStepper Kernels              |             |
|  +--------------------------------------------------+             |
|  | BatchedApplyGravity(N * bodiesPerWorld)          |             |
|  | BatchedIntegrate(N * bodiesPerWorld)             |             |
|  | BatchedDetectContacts(N * bodiesPerWorld * ...)  |             |
|  | BatchedSolveContacts(N * contactsPerWorld)       |             |
|  | BatchedSolveJoints(N * jointsPerWorld)           |             |
|  +--------------------------------------------------+             |
|           |                                                       |
|           v                                                       |
|  +--------------------------------------------------+             |
|  |           GPUBatchedEnvironment Kernels          |             |
|  +--------------------------------------------------+             |
|  | BatchedApplyActions(N) - thrust, gimbal          |             |
|  | BatchedCheckTargets(N * M) - collision tests     |             |
|  | BatchedComputeRewards(N) - shaping rewards       |             |
|  | BatchedGetObservations(N) - for neural net       |             |
|  +--------------------------------------------------+             |
|                                                                   |
+------------------------------------------------------------------+
```

**Memory Layout (Structure of Arrays, batched):**

For N=1024 agents, each with 2 rigid bodies, 8 geoms, 10 targets:

- Rigid Bodies: `[world0_rb0, world0_rb1, world1_rb0, world1_rb1, ...]` → 2048 entries
- Geoms: `[world0_geom0..7, world1_geom0..7, ...]` → 8192 entries
- Targets: `[world0_target0..9, world1_target0..9, ...]` → 10240 entries

Key insight: `worldIdx = globalIdx / itemsPerWorld`

---

## Implementation Phases

### Phase 1: Foundation - Batched Data Structures

**Goal**: Create GPU memory structures for N simultaneous simulations.

**Tasks:**

1.1. Create `GPUBatchedWorldConfig` struct
```csharp
public struct GPUBatchedWorldConfig
{
    public int WorldCount;           // N simulations
    public int RigidBodiesPerWorld;  // Fixed per template (e.g., 2 for rocket)
    public int GeomsPerWorld;        // Fixed per template
    public int JointsPerWorld;       // Fixed per template
    public int CollidersPerWorld;    // Static colliders (shared)
    public int TargetsPerWorld;      // Max dynamic targets per agent
}
```

1.2. Create `GPUBatchedWorldState` class
- Batched rigid bodies: `[world0_rb0, world0_rb1, world1_rb0, ...]`
- Shared static colliders (same arena for all worlds)
- Per-world constraint data

1.3. Create `GPUBatchedEnvironmentState` class
- Per-world targets (positions, active flags)
- Per-world episode state (reward, steps, terminal)
- Per-world observations/actions buffers

1.4. Implement upload/download helpers

**Verification**: Unit tests for index computation, template upload

---

### Phase 2: Batched Physics Kernels

**Goal**: Adapt physics kernels to process N worlds in parallel.

**Key Insight**: Each kernel receives `globalIdx`, computes `worldIdx = globalIdx / itemsPerWorld`

**Tasks:**

2.1. Create `GPUBatchedPhysicsKernels` static class
```csharp
public static void BatchedApplyRigidBodyGravityKernel(
    Index1D globalIdx,
    ArrayView<GPURigidBody> allBodies,
    int bodiesPerWorld,
    float gx, float gy, float dt)
{
    int worldIdx = globalIdx / bodiesPerWorld;
    int localRbIdx = globalIdx % bodiesPerWorld;
    // ...
}
```

2.2. Batched integration kernels
2.3. Batched contact detection (shared static colliders)
2.4. Batched constraint solvers
2.5. Create `GPUBatchedStepper` class

**Verification**: 10 identical worlds produce identical results, match CPUStepper

---

### Phase 3: Batched Environment Logic

**Goal**: Handle dynamic environment elements (targets, rewards) on GPU.

**Tasks:**

3.1. Target initialization kernel (random positions per world)
3.2. Target collision kernel (deactivate on hit, add reward)
3.3. Observation extraction kernel (nearest target, rocket state)
3.4. Action application kernel (thrust, gimbal)
3.5. Terminal condition kernel (OOB, flipped, max steps)

**Verification**: Target collision matches CPU, rewards accumulate correctly

---

### Phase 4: Integration with Evolvion

**Goal**: Connect batched physics+environment with batched neural network evaluation.

**Tasks:**

4.1. Create `GPUBatchedTargetChaseEnvironment` class
4.2. Extend `GPUEvolvionState` for batched physics
4.3. Create `EvaluateWithBatchedPhysics` in GPUEvaluator
4.4. Extend `GPUFitnessEvaluator` for TargetChaseEnvironment

**Verification**: Batched vs CPU fitness values match

---

### Phase 5: Demo and Optimization

**Goal**: Demonstrate 1000+ agents and optimize performance.

**Tasks:**

5.1. Create `BatchedTargetChaseDemo` (visualize top performers)
5.2. Performance profiling and optimization
5.3. Memory optimization (pooling, precision)
5.4. Scaling tests (N=100, 1000, 10000)

**Target**: 1000 agents, 300 steps < 5 seconds

---

## Key Technical Challenges

| Challenge | Solution |
|-----------|----------|
| Index computation complexity | Pre-compute strides in config struct |
| Variable contact count | Allocate max contacts, use `IsValid` flag |
| Warm-starting contacts | Skip initially, add per-world cache later |
| Random number generation | `LCG(baseSeed + worldIdx * 1000 + counter)` |
| Shared vs per-world colliders | Store static colliders once, share across worlds |

---

## What Can Be Done Incrementally

**New, independent code:**
- Phase 1 (data structures)
- Phase 2 (physics kernels)
- Phase 3 (environment kernels)
- Phase 5 (demo)

**Requires careful integration:**
- Phase 4: Extend GPUEvaluator with new method, don't break existing

**Reuse existing:**
- Neural network kernels (extend for worldIdx mapping)
- GPU memory patterns from GPUEvolvionState
- Physics algorithms (adapt indices)

---

## Estimated Timeline

| Phase | Duration | Risk |
|-------|----------|------|
| Phase 1: Data Structures | 1 week | Low |
| Phase 2: Physics Kernels | 2 weeks | Medium |
| Phase 3: Environment Kernels | 1 week | Low |
| Phase 4: Integration | 1 week | Medium |
| Phase 5: Demo/Optimization | 1 week | Low |

**Total: ~6 weeks**

---

## Critical Files

1. `GPUStepper.cs` - Pattern for batched stepper
2. `GPUEvaluator.cs` - Extend with batched physics method
3. `GPUWorldState.cs` - Pattern for batched memory management
4. `TargetChaseEnvironment.cs` - Reference for GPU kernel behavior
5. `GPURigidBodyContactKernels.cs` - Adapt for batching
