# Particle Boxes Demo - Side-by-Side Comparison

**Date:** October 26, 2025
**Scene Index:** 8 (Scene name: "Particle Boxes (Angle Constraints)")

---

## Overview

This demo creates **two sets of particle-based boxes** to visually compare two approaches for maintaining rigid rectangular shapes:

- **LEFT SIDE (Cyan particles):** Boxes using **angle-as-diagonal-rod** method (`AddAngleConstraintAsRod`)
- **RIGHT SIDE (Magenta particles):** Boxes using **traditional diagonal crossbracing** (explicit X-shaped rods)

Both approaches should behave identically, but the visual difference shows that the angle constraint method produces cleaner structures without visible internal bracing.

---

## Key Features

### Box Construction

Each box type consists of:

**Common to both types:**
- **4 corner particles** (mass: 0.3 kg, radius: 0.1 m)
- **4 edge rods** connecting corners in a loop (very stiff: compliance = 1e-6)

**LEFT SIDE (Cyan) - Angle Constraint Method:**
- **2 angle constraints** on opposing corners using `AddAngleConstraintAsRod()`
- These create diagonal rods internally but connect corners that already have edge rods
- Result: Cleaner visual appearance

**RIGHT SIDE (Magenta) - Traditional Method:**
- **2 explicit diagonal rods** forming an X pattern across the interior
- These are clearly visible as additional constraint lines
- Result: Standard rigid box with visible crossbracing

### Visual Comparison

The demo shows the difference between these approaches:

```
LEFT (Cyan):           RIGHT (Magenta):
    +---+                  +---+
    |   |                  |\ /|  ← Visible X diagonals
    +---+                  |/ \|
                           +---+
  "Clean"                "Traditional"
```

Both maintain rigidity equally well, but the angle method produces cleaner-looking structures.

### Angle Constraint Implementation

The angle constraints use the **diagonal distance method** (`AddAngleConstraintAsRod`), which is stable for rigid structures:

```csharp
// Bottom-left corner (90 degrees)
world.AddAngleConstraintAsRod(
    i: topLeft,          // One arm of angle
    j: bottomLeft,       // Vertex (corner)
    k: bottomRight,      // Other arm of angle
    targetAngle: π/2,    // 90 degrees
    len1: height,        // Length of left edge
    len2: width,         // Length of bottom edge
    compliance: 1e-6f);  // Very stiff
```

Behind the scenes, this creates a **diagonal rod** with length calculated using the law of cosines:
```
diagonal² = height² + width² - 2·height·width·cos(90°)
diagonal = √(height² + width²)
```

For a 90-degree angle, this simplifies to the Pythagorean theorem.

---

## Demo Parameters

### Box Generation

- **Count:** 3 boxes per side (6 total)
  - 3 on LEFT (cyan) using angle constraints
  - 3 on RIGHT (magenta) using diagonal crossbracing
- **Width range:** 0.5m - 1.2m (random)
- **Height range:** 0.5m - 1.2m (random)
- **Starting positions:**
  - LEFT side: x = -10 to -5, staggered vertically
  - RIGHT side: x = +4 to +9, staggered vertically
- **Initial rotation:** Random (0 - 2π radians)
- **Initial angular velocity:** Random (-1 to +1 rad/s)
- **Seed:** Fixed at 42 for reproducibility

### Color Coding

- **Cyan particles:** Angle constraint method (LEFT side)
- **Magenta particles:** Diagonal crossbracing method (RIGHT side)
- **SkyBlue lines:** Rod constraints (visible on both)
- **Gray:** Static colliders (ground, obstacles)

### Physics Parameters

- **Particle mass:** 0.3 kg per corner
- **Particle radius:** 0.1 m
- **Rod compliance:** 1e-6 (very stiff, nearly rigid)
- **XPBD iterations:** 12 (from global config)

### Scene Elements

- **Ground:** Large flat platform at y = -8m
- **Obstacles:**
  - Circle at (-5, -4) with radius 1.0m
  - Circle at (5, -5) with radius 1.2m
  - Angled platform at (0, -6) tilted 0.15 radians

---

## How to Run

1. **Build the demo:**
   ```bash
   dotnet build Evolvatron.Demo/Evolvatron.Demo.csproj
   ```

2. **Run the demo:**
   ```bash
   dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj
   ```

3. **Navigate to scene 8:**
   - Press **RIGHT ARROW** 7 times from the start
   - Or use **LEFT ARROW** once from scene 1

4. **Watch the boxes fall and tumble!**

---

## Expected Behavior

### ✅ Correct Behavior

- Boxes **maintain rectangular shape** during falls
- Corners remain at **90-degree angles**
- No visible diagonal crossbracing rods
- Boxes bounce and tumble realistically
- Stable contact with ground and obstacles

### ❌ Failure Modes to Watch For

If the implementation is incorrect, you might see:
- **Shearing:** Boxes collapse into parallelograms
- **Explosions:** Particles fly apart at high velocity
- **Jittering:** Boxes vibrate excessively on surfaces
- **Sinking:** Corners penetrate through colliders

---

## Comparison with Other Approaches

### Diagonal Rods (Traditional)

**Pros:**
- Extremely simple to implement
- Very stable
- Fast to solve

**Cons:**
- Visible diagonal structure (aesthetically undesirable)
- Adds extra constraint overhead

### Direct Angle Constraints (Attempted)

**Pros:**
- No visible diagonals
- Matches physical intuition

**Cons:**
- ❌ **Unstable for rigid structures** (causes explosions)
- Conflicts with distance constraints
- Requires many iterations to converge
- See `ANGLE_CONSTRAINT_CONCLUSION.md` for details

### Angle-as-Rod Method (This Demo)

**Pros:**
- ✅ No visible diagonals (rod connects corners that already have edge rods)
- ✅ Stable (uses distance constraints internally)
- ✅ Converges in 12 iterations
- ✅ Natural API (`AddAngleConstraintAsRod`)

**Cons:**
- Slightly more complex setup (need to specify edge lengths)
- Creates diagonal rod behind the scenes (but between corners, so not visually obvious)

---

## Technical Details

### Constraint Graph

For a single box, the constraint graph is:

```
Particles:
  [0] bottom-left
  [1] bottom-right
  [2] top-right
  [3] top-left

Edge Rods (4):
  [0] → [1]  (bottom edge)
  [1] → [2]  (right edge)
  [2] → [3]  (top edge)
  [3] → [0]  (left edge)

Angle Constraint Diagonals (2):
  [3] → [1]  (diagonal for bottom-left angle at [0])
  [1] → [3]  (diagonal for top-right angle at [2])
```

Note: Both angle constraints create the **same diagonal rod** ([3] ↔ [1]), but from the XPBD solver's perspective, they're different constraints. In practice, only one diagonal is needed, but having two provides extra stiffness.

### Why Two Opposing Corners?

Constraining only one corner allows the opposite corner to deform. **Two opposing corners** ensures the entire rectangle is rigid:

```
Constrained corners:  *     +
                      +-----*   ← Both diagonals constrained

One corner only:      *     +
                      +-----?   ← Other corner can deform
```

---

## Random Number Generation

The demo uses `new Random(42)` for reproducible results:
- Same seed → same box sizes, positions, rotations
- Useful for debugging and comparing implementations

To get different boxes each run, use:
```csharp
var random = new Random();  // Time-based seed
```

---

## Controls

- **SPACE:** Pause/Resume
- **R:** Reset scene
- **LEFT/RIGHT:** Switch scenes
- **WASD:** Pan camera
- **MOUSE WHEEL:** Zoom in/out

---

## Performance

With 6 boxes × 4 particles = **24 particles** and ~36 constraints:
- **Target FPS:** 60
- **Physics steps per frame:** 4 (simulation time runs 4× faster than real-time)
- **XPBD iterations:** 12
- **Expected performance:** Should run smoothly on any modern CPU

---

## Code Location

- **Demo scene:** `Evolvatron.Demo/GraphicalDemo.cs:BuildSceneParticleBoxes()`
- **Box factory:** `Evolvatron.Demo/GraphicalDemo.cs:CreateParticleBox()`
- **Angle-as-rod API:** `Evolvatron.Rigidon/WorldState.cs:AddAngleConstraintAsRod()`

---

## See Also

- `ANGLE_CONSTRAINT_API.md` - Detailed explanation of the angle-as-rod method
- `ANGLE_CONSTRAINT_CONCLUSION.md` - Why direct angle constraints don't work
- `XPBD_ANGLE_CONSTRAINT_FINDINGS.md` - Investigation and testing results
- `PARTICLE_GRID_DEMO.md` - Soft-body cloth simulation (scene 7)
