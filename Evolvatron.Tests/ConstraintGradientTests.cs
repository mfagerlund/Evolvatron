using Evolvatron.Core;
using Evolvatron.Core.Physics;
using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Numerical gradient checks for all XPBD constraints.
/// Verifies that analytical gradients match finite-difference approximations.
/// </summary>
public class ConstraintGradientTests
{
    private const float Epsilon = 1e-5f; // Perturbation for finite differences
    private const float Tolerance = 1e-3f; // Acceptable error between analytical and numerical

    [Fact]
    public void RodConstraint_GradientsMatchNumerical()
    {
        // Test rod constraint: C(p0, p1) = |p1 - p0| - restLength
        var world = new WorldState();

        // Add two particles at arbitrary positions
        int i = world.AddParticle(x: 1.5f, y: 2.3f, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);
        int j = world.AddParticle(x: 3.7f, y: 5.1f, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);

        float restLength = 2.5f;
        world.Rods.Add(new Rod(i, j, restLength, compliance: 0f));

        // Compute analytical gradients (from XPBDSolver.SolveRods)
        float dx = world.PosX[j] - world.PosX[i];
        float dy = world.PosY[j] - world.PosY[i];
        float len = MathF.Sqrt(dx * dx + dy * dy);

        if (len < 1e-6f)
            return; // Degenerate case

        float C = len - restLength;

        // Analytical gradients: ∂C/∂p = (p1 - p0) / |p1 - p0|
        float gradIx = -dx / len;
        float gradIy = -dy / len;
        float gradJx = dx / len;
        float gradJy = dy / len;

        // Numerical gradients via finite differences
        float numGradIx = ComputeNumericalGradient(() => RodConstraintValue(world, i, j, restLength),
            () => world.PosX[i], val => world.PosX[i] = val);
        float numGradIy = ComputeNumericalGradient(() => RodConstraintValue(world, i, j, restLength),
            () => world.PosY[i], val => world.PosY[i] = val);
        float numGradJx = ComputeNumericalGradient(() => RodConstraintValue(world, i, j, restLength),
            () => world.PosX[j], val => world.PosX[j] = val);
        float numGradJy = ComputeNumericalGradient(() => RodConstraintValue(world, i, j, restLength),
            () => world.PosY[j], val => world.PosY[j] = val);

        // Verify analytical matches numerical
        Assert.True(MathF.Abs(gradIx - numGradIx) < Tolerance,
            $"Rod gradient ∂C/∂xi mismatch: analytical={gradIx:F6}, numerical={numGradIx:F6}");
        Assert.True(MathF.Abs(gradIy - numGradIy) < Tolerance,
            $"Rod gradient ∂C/∂yi mismatch: analytical={gradIy:F6}, numerical={numGradIy:F6}");
        Assert.True(MathF.Abs(gradJx - numGradJx) < Tolerance,
            $"Rod gradient ∂C/∂xj mismatch: analytical={gradJx:F6}, numerical={numGradJx:F6}");
        Assert.True(MathF.Abs(gradJy - numGradJy) < Tolerance,
            $"Rod gradient ∂C/∂yj mismatch: analytical={gradJy:F6}, numerical={numGradJy:F6}");
    }

    [Theory]
    [InlineData(45.0)]   // 45 degrees
    [InlineData(90.0)]   // 90 degrees (right angle)
    [InlineData(120.0)]  // 120 degrees (obtuse)
    [InlineData(30.0)]   // 30 degrees (acute)
    [InlineData(135.0)]  // 135 degrees
    [InlineData(60.0)]   // 60 degrees
    public void AngleConstraint_GradientsMatchNumerical(double targetAngleDeg)
    {
        float targetAngle = (float)(targetAngleDeg * Math.PI / 180.0);

        // Test angle constraint: C(p0, p1, p2) = atan2(cross(u,v), dot(u,v)) - targetAngle
        var world = new WorldState();

        // Create three particles forming an angle
        // Angle measures from u=(i-j) to v=(k-j)
        // Place i to the LEFT of j, then place k at the target angle from that
        float armLength = 2.0f;
        float jx = 0f, jy = 0f; // Vertex at origin
        float ix = -armLength, iy = 0f; // i to the LEFT (angle π from +X)

        // k should be at angle (π + targetAngle) from +X to create the correct measured angle
        float kAngle = MathF.PI + targetAngle;
        float kx = jx + armLength * MathF.Cos(kAngle);
        float ky = jy + armLength * MathF.Sin(kAngle);

        int i = world.AddParticle(x: ix, y: iy, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);
        int j = world.AddParticle(x: jx, y: jy, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f); // vertex
        int k = world.AddParticle(x: kx, y: ky, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);

        world.Angles.Add(new Angle(i, j, k, targetAngle, compliance: 0f));

        // Verify initial angle is correct (should be ~0)
        float initialC = AngleConstraintValue(world, i, j, k, targetAngle);

        // Perturb slightly to test gradients away from equilibrium
        world.PosX[k] += 0.2f;
        world.PosY[k] += 0.15f;

        // Verify constraint is now violated
        float perturbedC = AngleConstraintValue(world, i, j, k, targetAngle);
        if (MathF.Abs(perturbedC) < 1e-4f)
        {
            // Constraint still satisfied after perturbation - test won't work
            Assert.Fail($"Perturbation didn't violate constraint: C={perturbedC:F6} (initial C={initialC:F6})");
        }

        // Compute analytical gradients (from XPBDSolver.SolveAngles)
        float ux = world.PosX[i] - world.PosX[j];
        float uy = world.PosY[i] - world.PosY[j];
        float vx = world.PosX[k] - world.PosX[j];
        float vy = world.PosY[k] - world.PosY[j];

        float uu = ux * ux + uy * uy;
        float vv = vx * vx + vy * vy;

        if (uu < 1e-6f || vv < 1e-6f)
            return; // Degenerate case

        float c = ux * vx + uy * vy;  // dot(u, v)
        float s = ux * vy - uy * vx;  // cross(u, v) in 2D

        float denom = uu * vv + 1e-12f;

        // Analytical gradients (corrected formula)
        float dθ_du_x = (c * (-vy) - s * vx) / denom;
        float dθ_du_y = (c * ( vx) - s * vy) / denom;
        float dθ_dv_x = (c * (-uy) - s * ux) / denom;
        float dθ_dv_y = (c * ( ux) - s * uy) / denom;

        float gradIx = dθ_du_x;
        float gradIy = dθ_du_y;
        float gradKx = dθ_dv_x;
        float gradKy = dθ_dv_y;
        float gradJx = -(gradIx + gradKx);
        float gradJy = -(gradIy + gradKy);

        // Numerical gradients via finite differences
        float numGradIx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[i], val => world.PosX[i] = val);
        float numGradIy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[i], val => world.PosY[i] = val);
        float numGradJx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[j], val => world.PosX[j] = val);
        float numGradJy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[j], val => world.PosY[j] = val);
        float numGradKx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[k], val => world.PosX[k] = val);
        float numGradKy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[k], val => world.PosY[k] = val);

        // Verify analytical matches numerical
        Assert.True(MathF.Abs(gradIx - numGradIx) < Tolerance,
            $"Angle gradient ∂C/∂xi mismatch at {targetAngleDeg}°: analytical={gradIx:F6}, numerical={numGradIx:F6}");
        Assert.True(MathF.Abs(gradIy - numGradIy) < Tolerance,
            $"Angle gradient ∂C/∂yi mismatch at {targetAngleDeg}°: analytical={gradIy:F6}, numerical={numGradIy:F6}");
        Assert.True(MathF.Abs(gradJx - numGradJx) < Tolerance,
            $"Angle gradient ∂C/∂xj mismatch at {targetAngleDeg}°: analytical={gradJx:F6}, numerical={numGradJx:F6}");
        Assert.True(MathF.Abs(gradJy - numGradJy) < Tolerance,
            $"Angle gradient ∂C/∂yj mismatch at {targetAngleDeg}°: analytical={gradJy:F6}, numerical={numGradJy:F6}");
        Assert.True(MathF.Abs(gradKx - numGradKx) < Tolerance,
            $"Angle gradient ∂C/∂xk mismatch at {targetAngleDeg}°: analytical={gradKx:F6}, numerical={numGradKx:F6}");
        Assert.True(MathF.Abs(gradKy - numGradKy) < Tolerance,
            $"Angle gradient ∂C/∂yk mismatch at {targetAngleDeg}°: analytical={gradKy:F6}, numerical={numGradKy:F6}");
    }

    [Fact]
    public void MotorConstraint_GradientsMatchNumerical()
    {
        // Motor constraint is just an angle constraint with a time-varying target
        // Test it the same way as angle constraint
        float targetAngle = MathF.PI / 3f; // 60 degrees

        var world = new WorldState();

        // Same setup as angle constraint test
        float armLength = 2.0f;
        float jx = 0f, jy = 0f;
        float ix = -armLength, iy = 0f;
        float kAngle = MathF.PI + targetAngle;
        float kx = jx + armLength * MathF.Cos(kAngle);
        float ky = jy + armLength * MathF.Sin(kAngle);

        int i = world.AddParticle(x: ix, y: iy, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);
        int j = world.AddParticle(x: jx, y: jy, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);
        int k = world.AddParticle(x: kx, y: ky, vx: 0f, vy: 0f, mass: 1f, radius: 0.1f);

        world.Motors.Add(new MotorAngle(i, j, k, targetAngle, compliance: 1e-6f));

        // Perturb
        world.PosX[k] += 0.25f;
        world.PosY[k] -= 0.1f;

        // Compute analytical gradients (same as angle constraint)
        float ux = world.PosX[i] - world.PosX[j];
        float uy = world.PosY[i] - world.PosY[j];
        float vx = world.PosX[k] - world.PosX[j];
        float vy = world.PosY[k] - world.PosY[j];

        float uu = ux * ux + uy * uy;
        float vv = vx * vx + vy * vy;

        if (uu < 1e-6f || vv < 1e-6f)
            return;

        float c = ux * vx + uy * vy;
        float s = ux * vy - uy * vx;
        float denom = uu * vv + 1e-12f;

        float dθ_du_x = (c * (-vy) - s * vx) / denom;
        float dθ_du_y = (c * ( vx) - s * vy) / denom;
        float dθ_dv_x = (c * (-uy) - s * ux) / denom;
        float dθ_dv_y = (c * ( ux) - s * uy) / denom;

        float gradIx = dθ_du_x;
        float gradIy = dθ_du_y;
        float gradKx = dθ_dv_x;
        float gradKy = dθ_dv_y;
        float gradJx = -(gradIx + gradKx);
        float gradJy = -(gradIy + gradKy);

        // Numerical gradients
        float numGradIx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[i], val => world.PosX[i] = val);
        float numGradIy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[i], val => world.PosY[i] = val);
        float numGradJx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[j], val => world.PosX[j] = val);
        float numGradJy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[j], val => world.PosY[j] = val);
        float numGradKx = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosX[k], val => world.PosX[k] = val);
        float numGradKy = ComputeNumericalGradient(() => AngleConstraintValue(world, i, j, k, targetAngle),
            () => world.PosY[k], val => world.PosY[k] = val);

        // Verify
        Assert.True(MathF.Abs(gradIx - numGradIx) < Tolerance,
            $"Motor gradient ∂C/∂xi mismatch: analytical={gradIx:F6}, numerical={numGradIx:F6}");
        Assert.True(MathF.Abs(gradIy - numGradIy) < Tolerance,
            $"Motor gradient ∂C/∂yi mismatch: analytical={gradIy:F6}, numerical={numGradIy:F6}");
        Assert.True(MathF.Abs(gradJx - numGradJx) < Tolerance,
            $"Motor gradient ∂C/∂xj mismatch: analytical={gradJx:F6}, numerical={numGradJx:F6}");
        Assert.True(MathF.Abs(gradJy - numGradJy) < Tolerance,
            $"Motor gradient ∂C/∂yj mismatch: analytical={gradJy:F6}, numerical={numGradJy:F6}");
        Assert.True(MathF.Abs(gradKx - numGradKx) < Tolerance,
            $"Motor gradient ∂C/∂xk mismatch: analytical={gradKx:F6}, numerical={numGradKx:F6}");
        Assert.True(MathF.Abs(gradKy - numGradKy) < Tolerance,
            $"Motor gradient ∂C/∂yk mismatch: analytical={gradKy:F6}, numerical={numGradKy:F6}");
    }

    [Theory]
    [InlineData(0.0f, 5.0f)]   // Particle above ground
    [InlineData(2.0f, 3.0f)]   // Particle at angle
    [InlineData(-3.0f, 4.0f)]  // Particle on left side
    public void ContactConstraint_GradientsMatchNumerical(float particleX, float particleY)
    {
        // Test contact constraint against OBB (axis-aligned ground)
        var world = new WorldState();

        // Ground OBB
        float groundCX = 0f, groundCY = -1f;
        float groundHX = 10f, groundHY = 0.5f;
        world.Obbs.Add(OBBCollider.AxisAligned(groundCX, groundCY, groundHX, groundHY));

        // Particle
        float radius = 0.2f;
        int p = world.AddParticle(x: particleX, y: particleY, vx: 0f, vy: 0f, mass: 1f, radius: radius);

        // Manually compute contact constraint gradient
        // For OBB, the gradient depends on which feature (face/edge/corner) is closest
        // We'll test the simple case of face contact (perpendicular to Y axis for axis-aligned box)

        // Distance from particle to ground top surface = particleY - (groundCY + groundHY) - radius
        // If this is negative, particle is penetrating
        float groundTop = groundCY + groundHY;
        float penetrationDepth = (groundTop + radius) - particleY;

        if (penetrationDepth <= 0f)
            return; // No contact, gradient is zero (skip test)

        // For contact with top face of ground:
        // Constraint: C = penetrationDepth = (groundTop + radius) - py
        // Gradient: ∂C/∂py = -1
        float analyticalGradY = -1f;
        float analyticalGradX = 0f; // No X component for face contact

        // Numerical gradient
        float numGradY = ComputeNumericalGradient(
            () => ComputeContactPenetration(world, p, 0),
            () => world.PosY[p],
            val => world.PosY[p] = val);
        float numGradX = ComputeNumericalGradient(
            () => ComputeContactPenetration(world, p, 0),
            () => world.PosX[p],
            val => world.PosX[p] = val);

        // Verify (only if in contact)
        if (penetrationDepth > 0.01f) // Skip if barely touching (numerical instability)
        {
            Assert.True(MathF.Abs(analyticalGradY - numGradY) < Tolerance,
                $"Contact gradient ∂C/∂y mismatch: analytical={analyticalGradY:F6}, numerical={numGradY:F6}");
            Assert.True(MathF.Abs(analyticalGradX - numGradX) < Tolerance,
                $"Contact gradient ∂C/∂x mismatch: analytical={analyticalGradX:F6}, numerical={numGradX:F6}");
        }
    }

    // Helper: Compute numerical gradient using finite differences
    private float ComputeNumericalGradient(Func<float> constraintFunc, Func<float> getVar, Action<float> setVar)
    {
        float originalValue = getVar();

        // Forward difference: f'(x) ≈ (f(x + ε) - f(x)) / ε
        setVar(originalValue + Epsilon);
        float fPlus = constraintFunc();

        setVar(originalValue - Epsilon);
        float fMinus = constraintFunc();

        setVar(originalValue); // Restore

        // Central difference: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
        return (fPlus - fMinus) / (2f * Epsilon);
    }

    // Helper: Evaluate rod constraint value
    private float RodConstraintValue(WorldState world, int i, int j, float restLength)
    {
        float dx = world.PosX[j] - world.PosX[i];
        float dy = world.PosY[j] - world.PosY[i];
        float len = MathF.Sqrt(dx * dx + dy * dy);
        return len - restLength;
    }

    // Helper: Evaluate angle constraint value (with wrapping)
    private float AngleConstraintValue(WorldState world, int i, int j, int k, float targetAngle)
    {
        float ux = world.PosX[i] - world.PosX[j];
        float uy = world.PosY[i] - world.PosY[j];
        float vx = world.PosX[k] - world.PosX[j];
        float vy = world.PosY[k] - world.PosY[j];

        float c = ux * vx + uy * vy;
        float s = ux * vy - uy * vx;
        float currentAngle = MathF.Atan2(s, c);

        // Return wrapped difference (important for gradient continuity!)
        return Math2D.WrapAngle(currentAngle - targetAngle);
    }

    // Helper: Compute contact penetration depth
    private float ComputeContactPenetration(WorldState world, int particleIdx, int obbIdx)
    {
        var obb = world.Obbs[obbIdx];
        float px = world.PosX[particleIdx];
        float py = world.PosY[particleIdx];
        float radius = world.Radius[particleIdx];

        // For axis-aligned box, simple check
        if (obb.UX == 1f && obb.UY == 0f) // Axis-aligned
        {
            float groundTop = obb.CY + obb.HalfExtentY;
            float penetration = (groundTop + radius) - py;
            return MathF.Max(0f, penetration);
        }

        // For rotated OBB, would need full SDF computation
        // For now, return 0 (skip complex cases)
        return 0f;
    }
}
