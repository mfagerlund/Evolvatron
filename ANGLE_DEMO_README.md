# Angle Constraint Visual Demo

**Created:** October 26, 2025

## What This Demo Shows

A real-time visual demonstration of XPBD angle constraints with **30 L-shaped structures** falling onto a platform. This demo lets you **see** how well the corrected angle gradient formula works under realistic physics conditions.

## Running the Demo

```bash
dotnet run --project Evolvatron.Demo/Evolvatron.Demo.csproj angles
```

## What You'll See

### The Shapes
- **30 L-shaped structures** (each made of 3 particles + 2 rods + 1 angle constraint)
- **Every 3rd shape has a 90° angle** (10 shapes total)
- **Other shapes have random angles** between 30° and 150°
- **All shapes start with random rotations** (challenging the solver)

### Visual Indicators

**Color coding:**
- **White lines** = Rods (edges) maintaining correct length
- **Red lines** = Rods with length errors (structure deforming)
- **Sky blue circles** = Particles (vertices)
- **Green arcs** = Angles within tolerance (<3°)
- **Yellow arcs** = Angles with moderate error (3-8°)
- **Red arcs** = Angles with large error (>8°)

### Stats Display

Top-left corner shows:
- Current simulation step
- Pause state
- Particle count
- Angle constraint count
- **Error status breakdown:**
  - Good (<3°): How many angles are nearly perfect
  - OK (3-8°): How many angles have small errors
  - Bad (>8°): How many angles are failing

## Controls

- **SPACE** - Pause/resume simulation
- **R** - Reset and respawn all shapes
- **ESC** - Exit

## What to Watch For

### ✅ Expected Good Behavior (90° angles)

The 90-degree L-shapes should:
- Fall and land without exploding
- Maintain their right angles (green arcs)
- Settle into stable resting positions
- Show minimal deformation

### ⚠️ Known Issues (Non-90° angles)

Non-90-degree shapes may:
- Show angle drift (yellow/red arcs)
- Take longer to stabilize
- Have larger errors after landing
- Still converge but not perfectly

## Technical Details

### Physics Settings
- **Timestep:** 1/60s (60 Hz)
- **Iterations:** 20 (production setting, not test's 40)
- **Rod compliance:** 0 (rigid)
- **Angle compliance:** 0 (rigid)
- **Contact compliance:** 1e-8 (nearly rigid)
- **Friction:** 0.5

### Scene Setup
- **Platform:** 30m wide, 1m tall
- **Spawn area:** 3 rows × 10 columns
- **Spacing:** 2.5m between shapes
- **Initial height:** 5-13m above ground

## What This Demonstrates

### Proof that 90° Angles Work

If you see the 90° L-shapes (every 3rd one) maintaining green arcs and stable configurations, this **proves** the corrected angle gradient formula is working correctly.

### Visualization of Non-90° Issues

The random-angle shapes provide a stress test showing which angles work well and which need more investigation.

### Real-Time Debugging

Watch the arc colors change in real-time to see:
- How quickly angles converge
- Which orientations are challenging
- How angles behave during dynamic motion vs at rest

## Expected Results

Based on our testing:

**90° Angles (every 3rd shape):**
- ✅ Should show **green arcs** (good angles)
- ✅ Should be **stable** (no explosions)
- ✅ Should **settle at rest** within a few seconds

**Other Angles:**
- ⚠️ May show **yellow/red arcs** (angle errors)
- ⚠️ May take **longer to stabilize**
- ⚠️ Some may show **persistent drift**

## Interpreting the Results

### If 90° shapes all stay green:
✅ **Gradient fix is working!** The angle constraint solver is correct.

### If 90° shapes turn yellow/red:
❌ Something is wrong - check configuration or investigate further.

### If non-90° shapes have errors:
⚠️ This is expected - those angles need more tuning (iterations, compliance, etc.)

## Comparison to Previous State

**Before the gradient fix:**
- ALL angles (even 90°) would fail
- Structures would explode or collapse
- Angle errors would be 100°+
- Nothing would stabilize

**After the gradient fix:**
- 90° angles maintain stability
- Structures settle without exploding
- Angle errors for 90° are <3°
- Non-90° angles show improvement (but not perfect)

## Files

- **Demo code:** `Evolvatron.Demo/AngleConstraintDemo.cs`
- **Solver:** `Evolvatron.Rigidon/Physics/XPBDSolver.cs` (corrected gradients)
- **Tests:** `Evolvatron.Tests/AngleConstraintStressTests.cs`

## Next Steps

Use this demo to:
1. **Verify the fix** - Confirm 90° angles work
2. **Tune parameters** - Experiment with iterations, compliance, etc.
3. **Test edge cases** - Try different angles, masses, configurations
4. **Debug visually** - See exactly where/when angles fail

## Fun Experiments

Try modifying the code to:
- Change the angle range (e.g., only 45°, 60°, 90°, 120°, 135°)
- Increase iterations (40, 60, 80) and see improvement
- Add more complex shapes (T-shapes, crosses, stars)
- Test different spawn heights
- Vary the masses of particles
- Add more challenging terrain (slopes, steps, etc.)

---

**Enjoy watching physics in action!** 🎮🔬
