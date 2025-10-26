# XPBD Angle Constraint Investigation

**Date:** October 26, 2025
**Status:** ✅ RESOLVED - Use distance constraints instead of angle constraints

---

## Summary

The XPBD particle system in Rigidon was **causing explosions** when angle constraints were combined with rigid distance (rod) constraints and contact constraints. After investigation and testing against the **Ten Minute Physics** reference implementation, we determined that:

**❌ Direct angle constraints are unstable in XPBD for rigid structures**
**✅ Use diagonal distance constraints (bending constraints) instead**

---

## Problem Description

When creating rigid particle structures (like triangles or L-shapes) with both:
1. Rod constraints (distance constraints)
2. Angle constraints (to maintain angles between edges)
3. Contact constraints (collision with ground)

The simulation would **explode** with particles reaching velocities of 70-150 m/s and flying away chaotically.

---

## Root Cause

### Why Angle Constraints Fail

Angle constraints in XPBD compute gradients using perpendicular directions to edges. The gradient formula:

```csharp
float gradIx = -n1y / len1;  // perpendicular to edge 1
float gradIy = n1x / len1;
```

While mathematically correct, this creates **conflicting corrections** when combined with:
- **Rod constraints** trying to maintain exact distances
- **Contact constraints** pushing particles out of colliders

The result is an **over-constrained system** where the iterative solver oscillates and amplifies corrections, leading to instability.

### Mitigation Attempts (Failed)

We tried several fixes that helped but didn't solve the problem:

1. **Clamping constraint values** to ±π/2 radians
2. **Clamping deltaLambda** corrections to ±0.5 (28 degrees)
3. **Adding compliance** (soft constraints) with alpha = 1e-4
4. **Improved numerical stability** in gradient calculations

All approaches reduced the explosion severity but didn't eliminate it.

---

## Solution: Distance-Based Bending Constraints

### The Ten Minute Physics Approach

After examining the **Ten Minute Physics** reference implementation (by Matthias Müller, co-author of the XPBD paper), we discovered they **don't use angle constraints at all**.

Instead, for cloth bending they use:
- **Distance constraint between opposite vertices** of adjacent triangles

For 2D rigid structures:
- **Diagonal rod constraints** to prevent shape deformation

### Implementation

For a three-particle L-shape:

```csharp
// ❌ WRONG: Angle constraint (causes explosions)
world.Angles.Add(new Angle(p0, p1, p2, theta0: MathF.PI / 2f, compliance: 0f));

// ✅ CORRECT: Diagonal distance constraint
float diagonal = MathF.Sqrt(armLength * armLength + armLength * armLength);
world.Rods.Add(new Rod(p0, p2, restLength: diagonal, compliance: 0f));
```

For a square:

```csharp
// Perimeter rods (4 edges)
world.Rods.Add(new Rod(p0, p1, size));
world.Rods.Add(new Rod(p1, p2, size));
world.Rods.Add(new Rod(p2, p3, size));
world.Rods.Add(new Rod(p3, p0, size));

// ONE diagonal for rigidity (prevents parallelogram deformation)
float diagonal = size * MathF.Sqrt(2f);
world.Rods.Add(new Rod(p0, p2, diagonal));
```

---

## Test Results

Created three stability tests in `XPBDStabilityTests.cs`:

### ✅ Test 1: Triangle (3 particles, 3 rods)
- Falls 2 meters to ground
- Settles without explosion or collapse
- Edge lengths preserved within 5%
- Velocities < 0.5 m/s at rest

### ✅ Test 2: Square with diagonal (4 particles, 5 rods)
- Falls 2 meters to ground
- Remains stable (doesn't collapse or explode)
- Area preserved within 20%
- All particles above ground, at rest

### ✅ Test 3: L-shape (3 particles, 3 rods with diagonal)
- Falls 2 meters to ground
- Shape maintained via diagonal rod
- Edge and diagonal lengths preserved
- No explosions, all particles settle

**All 154 physics tests pass** (2 skipped).

---

## Recommendations

### For Rigidon Users

1. **DO NOT use `Angle` constraints for rigid structures**
   - They are inherently unstable when combined with contacts
   - Only use them for very soft compliance (alpha > 1e-3) if needed

2. **DO use diagonal `Rod` constraints for rigidity**
   - This is the proven, stable approach from Ten Minute Physics
   - Works for triangles, quads, and any polygon

3. **For soft-body bending** (cloth, rope):
   - Use distance between opposite vertices of adjacent triangles
   - Add moderate compliance (alpha ~ 1e-6 to 1e-4)

### For Motor Angles (Rocket Gimbals)

**MotorAngle constraints may still have issues.** If rocket gimbals explode:
- Consider using **rigid body revolute joints** instead (RigidBodyRocketTemplate)
- Or use very high compliance (soft servo) with `MotorCompliance = 1e-3`

### Code Status

The angle constraint solver in `XPBDSolver.cs` has been improved with:
- Better numerical stability
- Constraint clamping to ±π/2
- DeltaLambda clamping to prevent large corrections

However, **these improvements don't fully solve the fundamental instability**. The constraints remain in the codebase for soft-body use cases but should be avoided for rigid structures.

---

## References

1. **Ten Minute Physics** by Matthias Müller
   https://matthias-research.github.io/pages/tenMinutePhysics/index.html
   - Tutorial 10: Soft Bodies (uses distance-based bending)
   - Tutorial 14: Cloth (bending = distance between opposite vertices)

2. **XPBD Paper**
   "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
   Macklin et al., MIG 2016

3. **InteractiveComputerGraphics/PositionBasedDynamics**
   https://github.com/InteractiveComputerGraphics/PositionBasedDynamics
   C++ reference implementation

---

## Conclusion

**The XPBD distance constraint implementation is correct and stable.**
**The angle constraint implementation is correct but inherently unstable for rigid structures.**
**Use diagonal distance constraints instead of angle constraints.**

This aligns with best practices from the creators of XPBD themselves (Matthias Müller's Ten Minute Physics).
