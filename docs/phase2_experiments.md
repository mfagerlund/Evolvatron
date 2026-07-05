# Phase-2 Maze Navigation — Experiment Campaign

**Goal:** Establish solid conclusions about the two-phase hierarchy (frozen velocity
controller + evolved maze policy): how far it scales, whether the sensors matter, what
the failure modes are, and whether the solve-rate gap is a training-budget or a
capability ceiling.

**Setup (fixed unless noted):**
- Frozen controller: `scratch/controller_easy.bin` (9→16→16→2, 466 params) — NEVER retrained
- Maze policy: 14→24→24→2 (with 8 sensors) trained by CEM, pop 16384, 12 mazes/gen
- Physics: Dt=1/120, 6 solver iters, MaxSteps=600, CmdSpeedMax=3, GoalRadius=0.75
- GPU: single RTX 4090 → **training runs are strictly sequential** (concurrent ILGPU
  contexts risk a driver race). Subagents used only for CPU-only analysis.
- Champion = single best individual, evaluated on held-out mazes (unseen seeds).

**Failure-reason codes** (written by the kernel to `SettledSteps`, read at eval):
`1=goal 2=collision 3=tumble 4=NaN 5=timeout`.

---

## Status log
- (in progress) Adding failure-mode instrumentation before running experiments.

---

## A. Difficulty scaling (8 sensors, 80 gens unless noted)
How does champion held-out solve rate degrade as obstacle count rises?

| Obstacles | Gens | Champion (held-out /100) | collide% | timeout% | tumble% | notes |
|-----------|------|--------------------------|----------|----------|---------|-------|
| 0         | 60   | 100/100                  | —        | —        | —       | open space (prior run) |
| 3         | 100  | 93/100                   | TBD      | TBD      | TBD     | prior run; breakdown pending |
| 5         | 80   | TBD                      |          |          |         | |
| 8         | 80   | TBD                      |          |          |         | |
| 12        | 80   | TBD                      |          |          |         | |

## B. Sensor ablation (5 obstacles, 80 gens)
Do the raycasts actually drive avoidance, or is goal-direction enough?

| Sensors | Maze input | Champion (held-out /100) | collide% | notes |
|---------|-----------|--------------------------|----------|-------|
| 0       | 6         | TBD                      |          | blind — control |
| 4       | 10        | TBD                      |          | cardinal only |
| 8       | 14        | TBD                      |          | (= A @ 5 obstacles) |

## C. Training budget (3 obstacles, 8 sensors)
Is 93% a compute limit or a capability ceiling?

| Gens | Champion (held-out /100) | notes |
|------|--------------------------|-------|
| 100  | 93/100                   | prior |
| 250  | TBD                      | long run |

## D. Maze policy capacity (8 obstacles, 8 sensors, 80 gens) — optional
Does a bigger policy help on hard mazes?

| Hidden | Params | Champion (held-out /100) | notes |
|--------|--------|--------------------------|-------|
| 24,24  | 1010   | TBD (= A @ 8 obstacles)  | |
| 48,48  | ~3000  | TBD                      | |

---

## Conclusions
_(filled in as results land)_
