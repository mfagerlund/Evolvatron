# Angle Constraint Drop Test Results

**Date:** October 26, 2025
**Test File:** `Evolvatron.Tests/AngleConstraintDropTests.cs`

## Overview

Created comprehensive drop tests for angle constraints based on the JavaScript demo code provided. These tests verify whether angle constraints can maintain target angles when two rigid sticks (connected by rods and an angle constraint) fall and settle on the ground.

## Test Scenarios

### Test 1: Two Sticks at 60°
- **Setup**: 3 particles, 2 rigid rods, 1 angle constraint (60°)
- **Target Angle**: 60° (π/3 radians)
- **Expected**: Angle maintained within ~3° during fall and rest
- **Result**: ❌ **FAILED**
  - Rod length violated before angle check
  - Structure collapsed during simulation
  - Actual final rod length: -2.09 m (expected ~1.0 m)

### Test 2: L-Shape at 90°
- **Setup**: 3 particles forming L-shape, 90° angle constraint
- **Target Angle**: 90° (π/2 radians)
- **Expected**: Right angle maintained
- **Result**: ❌ **FAILED**
  - Final angle: -37.0°
  - **Error: 127.0°** (completely wrong!)

### Test 3: V-Shape at 120°
- **Setup**: 3 particles forming V-shape, obtuse angle constraint
- **Target Angle**: 120° (2π/3 radians)
- **Expected**: Obtuse angle maintained
- **Result**: ❌ **FAILED**
  - Final angle: -83.8°
  - **Error: 156.2°** (completely wrong!)

## Test Configuration

The tests used high-quality solver settings to give angle constraints the best chance:

```csharp
Dt = 1/60 s                    // Standard 60 Hz physics
XpbdIterations = 40            // Very high (production uses 12)
RodCompliance = 0              // Perfectly rigid rods
AngleCompliance = 2e-6         // Nearly rigid angle (small compliance)
ContactCompliance = 1e-8       // Nearly rigid contacts
FrictionMu = 0.5               // Moderate friction
```

Even with **40 XPBD iterations** (more than 3× the production value of 12), angle constraints completely failed.

## Why the Tests Failed

As documented in `ANGLE_CONSTRAINT_CONCLUSION.md`, angle constraints fail when combined with:

1. **Distance constraints** (rigid rods) on the same particles
2. **Contact constraints** (collision with ground)
3. **Multiple XPBD iterations** that amplify the conflict

The three constraint types fight each other:
- Distance constraints push along edges → maintain rod lengths
- Angle constraints push perpendicular to edges → maintain angle
- Contact constraints push upward → prevent penetration
- Result: **Divergence instead of convergence**

## Comparison with Diagonal Rod Approach

The `XPBDStabilityTests.cs` file contains equivalent tests using **diagonal rods instead of angle constraints**:

| Test | Approach | Result |
|------|----------|--------|
| L-shape (90°) | Angle constraint | ❌ Failed (127° error) |
| L-shape (90°) | Diagonal rod | ✅ **Passed** (stable, maintains shape) |
| Triangle | Angle constraint | ❌ Would fail |
| Triangle | 3 edge rods (inherently rigid) | ✅ **Passed** |

### Why Diagonal Rods Work

Diagonal rods maintain angles **indirectly through geometry**:
- For 90° angle with arms of length L: diagonal = L√2
- For 120° angle: use law of cosines to compute diagonal
- No perpendicular corrections → no conflict with edge rods
- All constraints solve in compatible directions

## Conclusion

These drop tests **confirm** the findings in `ANGLE_CONSTRAINT_CONCLUSION.md`:

### ❌ DO NOT use `Angle` constraints for rigid structures
- Fail to maintain target angles (errors > 100°)
- Cause structural collapse (rod lengths violated)
- Require excessive iterations (40+ still fails)
- Fundamentally incompatible with distance + contact constraints

### ✅ DO use diagonal distance constraints
- Maintain angles perfectly through geometry
- Stable with 12 iterations (production setting)
- Compatible with all other constraint types
- Industry-standard approach (Ten Minute Physics, etc.)

## API Recommendations

Use these helper methods from `WorldState` instead of direct `Angle` constraints:

```csharp
// Preferred: Specify angle and edge lengths explicitly
world.AddAngleConstraintAsRod(
    i: p0,
    j: p1,  // vertex
    k: p2,
    targetAngle: MathF.PI / 2f,  // 90 degrees
    len1: armLength,
    len2: armLength,
    compliance: 0f);

// Or: Read current positions automatically
world.AddAngleConstraintAsRodFromCurrentPositions(p0, p1, p2, compliance: 0f);
```

Both methods add a diagonal **Rod** constraint that maintains the angle through geometry, avoiding the instability of direct angle constraints.

## Test Code Location

All test code is in: `Evolvatron.Tests/AngleConstraintDropTests.cs`

To run these tests:
```bash
dotnet test Evolvatron.Tests/Evolvatron.Tests.csproj --filter "FullyQualifiedName~AngleConstraintDropTests"
```

Expected result: **All 3 tests fail** (as documented above)

---

## References

- **ANGLE_CONSTRAINT_CONCLUSION.md** - Detailed analysis of why angle constraints fail
- **XPBDStabilityTests.cs** - Passing tests using diagonal rod approach
- **JavaScript demo** (provided by user) - Interactive angle constraint visualization
- **Ten Minute Physics** - Uses distance-based bending, not angle constraints
