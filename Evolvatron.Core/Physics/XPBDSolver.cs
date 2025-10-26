using System;

namespace Evolvatron.Core.Physics;

/// <summary>
/// XPBD (Extended Position Based Dynamics) constraint solvers.
/// Each solver modifies particle positions to satisfy constraints.
/// </summary>
public static class XPBDSolver
{
    private const float Epsilon = 1e-9f;

    #region Rod (Distance) Constraint

    /// <summary>
    /// Solves all rod constraints once.
    /// C = |p_i - p_j| - L0
    /// </summary>
    public static void SolveRods(WorldState world, float dt, float compliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var rods = world.Rods;

        float alpha = compliance / (dt * dt);

        for (int idx = 0; idx < rods.Count; idx++)
        {
            var rod = rods[idx];
            int i = rod.I;
            int j = rod.J;

            // Skip if both particles are pinned
            if (invMass[i] == 0f && invMass[j] == 0f)
                continue;

            // Constraint vector
            float dx = posX[i] - posX[j];
            float dy = posY[i] - posY[j];
            float len = MathF.Sqrt(dx * dx + dy * dy);

            if (len < Epsilon)
                continue;

            // Constraint value
            float C = len - rod.RestLength;

            // Gradient (unit direction)
            float nx = dx / len;
            float ny = dy / len;

            // Effective inverse mass
            float w = invMass[i] + invMass[j];
            if (w < Epsilon)
                continue;

            // XPBD correction
            float deltaLambda = -(C + alpha * rod.Lambda) / (w + alpha);

            // Update positions
            posX[i] += invMass[i] * deltaLambda * nx;
            posY[i] += invMass[i] * deltaLambda * ny;
            posX[j] -= invMass[j] * deltaLambda * nx;
            posY[j] -= invMass[j] * deltaLambda * ny;

            // Update lambda
            rod.Lambda += deltaLambda;
            rods[idx] = rod;
        }
    }

    #endregion

    #region Angle Constraint

    /// <summary>
    /// Solves all angle constraints once.
    /// Maintains angle at vertex j between edges (i-j) and (k-j).
    /// </summary>
    public static void SolveAngles(WorldState world, float dt, float compliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var angles = world.Angles;

        float alpha = compliance / (dt * dt);

        for (int idx = 0; idx < angles.Count; idx++)
        {
            var angle = angles[idx];
            int i = angle.I;
            int j = angle.J;
            int k = angle.K;

            // Skip if all particles are pinned
            if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
                continue;

            // Edge vectors from j
            float e1x = posX[i] - posX[j];
            float e1y = posY[i] - posY[j];
            float e2x = posX[k] - posX[j];
            float e2y = posY[k] - posY[j];

            float len1 = MathF.Sqrt(e1x * e1x + e1y * e1y);
            float len2 = MathF.Sqrt(e2x * e2x + e2y * e2y);

            if (len1 < Epsilon || len2 < Epsilon)
                continue;

            // Normalize edges
            e1x /= len1;
            e1y /= len1;
            e2x /= len2;
            e2y /= len2;

            // Current angle
            float currentAngle = Math2D.AngleBetween(e1x, e1y, e2x, e2y);

            // Constraint value (wrapped to [-π, π])
            float C = Math2D.WrapAngle(currentAngle - angle.Theta0);

            if (MathF.Abs(C) < Epsilon)
                continue;

            // Gradients (derived from angle formula)
            // ∂angle/∂p_i ≈ perp(e1) / len1
            // ∂angle/∂p_k ≈ -perp(e2) / len2
            // ∂angle/∂p_j = -(∂angle/∂p_i + ∂angle/∂p_k)

            float gradIx = -e1y / len1;
            float gradIy = e1x / len1;

            float gradKx = e2y / len2;
            float gradKy = -e2x / len2;

            float gradJx = -(gradIx + gradKx);
            float gradJy = -(gradIy + gradKy);

            // Effective inverse mass
            float w = invMass[i] * (gradIx * gradIx + gradIy * gradIy)
                    + invMass[j] * (gradJx * gradJx + gradJy * gradJy)
                    + invMass[k] * (gradKx * gradKx + gradKy * gradKy);

            if (w < Epsilon)
                continue;

            // XPBD correction
            float deltaLambda = -(C + alpha * angle.Lambda) / (w + alpha);

            // Update positions
            posX[i] += invMass[i] * deltaLambda * gradIx;
            posY[i] += invMass[i] * deltaLambda * gradIy;
            posX[j] += invMass[j] * deltaLambda * gradJx;
            posY[j] += invMass[j] * deltaLambda * gradJy;
            posX[k] += invMass[k] * deltaLambda * gradKx;
            posY[k] += invMass[k] * deltaLambda * gradKy;

            // Update lambda
            angle.Lambda += deltaLambda;
            angles[idx] = angle;
        }
    }

    #endregion

    #region Motor Angle Constraint

    /// <summary>
    /// Solves all motor angle constraints once.
    /// Similar to angle constraint but uses Target instead of Theta0.
    /// </summary>
    public static void SolveMotors(WorldState world, float dt, float compliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var motors = world.Motors;

        float alpha = compliance / (dt * dt);

        for (int idx = 0; idx < motors.Count; idx++)
        {
            var motor = motors[idx];
            int i = motor.I;
            int j = motor.J;
            int k = motor.K;

            // Skip if all particles are pinned
            if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
                continue;

            // Edge vectors from j
            float e1x = posX[i] - posX[j];
            float e1y = posY[i] - posY[j];
            float e2x = posX[k] - posX[j];
            float e2y = posY[k] - posY[j];

            float len1 = MathF.Sqrt(e1x * e1x + e1y * e1y);
            float len2 = MathF.Sqrt(e2x * e2x + e2y * e2y);

            if (len1 < Epsilon || len2 < Epsilon)
                continue;

            // Normalize edges
            e1x /= len1;
            e1y /= len1;
            e2x /= len2;
            e2y /= len2;

            // Current angle
            float currentAngle = Math2D.AngleBetween(e1x, e1y, e2x, e2y);

            // Constraint value (target comes from motor)
            float C = Math2D.WrapAngle(currentAngle - motor.Target);

            if (MathF.Abs(C) < Epsilon)
                continue;

            // Gradients
            float gradIx = -e1y / len1;
            float gradIy = e1x / len1;

            float gradKx = e2y / len2;
            float gradKy = -e2x / len2;

            float gradJx = -(gradIx + gradKx);
            float gradJy = -(gradIy + gradKy);

            // Effective inverse mass
            float w = invMass[i] * (gradIx * gradIx + gradIy * gradIy)
                    + invMass[j] * (gradJx * gradJx + gradJy * gradJy)
                    + invMass[k] * (gradKx * gradKx + gradKy * gradKy);

            if (w < Epsilon)
                continue;

            // XPBD correction
            float deltaLambda = -(C + alpha * motor.Lambda) / (w + alpha);

            // Update positions
            posX[i] += invMass[i] * deltaLambda * gradIx;
            posY[i] += invMass[i] * deltaLambda * gradIy;
            posX[j] += invMass[j] * deltaLambda * gradJx;
            posY[j] += invMass[j] * deltaLambda * gradJy;
            posX[k] += invMass[k] * deltaLambda * gradKx;
            posY[k] += invMass[k] * deltaLambda * gradKy;

            // Update lambda
            motor.Lambda += deltaLambda;
            motors[idx] = motor;
        }
    }

    #endregion

    #region Contact Constraint

    /// <summary>
    /// Solves contacts for all particles against all static colliders.
    /// Uses inequality constraint: C = φ(p) ≥ 0 (where φ is signed distance).
    /// </summary>
    public static void SolveContacts(WorldState world, float dt, float compliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var radius = world.Radius;

        float alpha = compliance / (dt * dt);

        // Check each particle against all colliders
        for (int i = 0; i < world.ParticleCount; i++)
        {
            if (invMass[i] == 0f) // Skip pinned particles
                continue;

            float px = posX[i];
            float py = posY[i];
            float r = radius[i];

            // Check circles
            foreach (var circle in world.Circles)
            {
                Math2D.CircleSDF(px, py, circle, out float phi, out float nx, out float ny);
                phi -= r; // Inflate collider by particle radius

                if (phi < 0f) // Penetrating
                {
                    SolveContactConstraint(ref posX[i], ref posY[i], invMass[i], phi, nx, ny, alpha);
                }
            }

            // Check capsules
            foreach (var capsule in world.Capsules)
            {
                Math2D.CapsuleSDF(px, py, capsule, out float phi, out float nx, out float ny);
                phi -= r;

                if (phi < 0f)
                {
                    SolveContactConstraint(ref posX[i], ref posY[i], invMass[i], phi, nx, ny, alpha);
                }
            }

            // Check OBBs
            foreach (var obb in world.Obbs)
            {
                Math2D.OBBSDF(px, py, obb, out float phi, out float nx, out float ny);
                phi -= r;

                if (phi < 0f)
                {
                    SolveContactConstraint(ref posX[i], ref posY[i], invMass[i], phi, nx, ny, alpha);
                }
            }
        }
    }

    private static void SolveContactConstraint(
        ref float px, ref float py,
        float invMass,
        float phi, float nx, float ny,
        float alpha)
    {
        // Constraint C = phi (negative means penetration)
        // We want C >= 0
        float C = phi;

        // Gradient is just the normal (∂phi/∂p = n)
        float w = invMass; // Only one particle moves

        if (w < Epsilon)
            return;

        // XPBD correction (lambda starts at 0 each contact; no accumulation needed for one-shot)
        float deltaLambda = -(C + alpha * 0f) / (w + alpha);

        // Ensure we only push out (no pulling in)
        if (deltaLambda < 0f)
            deltaLambda = 0f;

        // Update position
        px += invMass * deltaLambda * nx;
        py += invMass * deltaLambda * ny;
    }

    #endregion

    /// <summary>
    /// Resets all constraint lambdas to zero at the start of a step.
    /// </summary>
    public static void ResetLambdas(WorldState world)
    {
        var rods = world.Rods;
        for (int i = 0; i < rods.Count; i++)
        {
            var rod = rods[i];
            rod.Lambda = 0f;
            rods[i] = rod;
        }

        var angles = world.Angles;
        for (int i = 0; i < angles.Count; i++)
        {
            var angle = angles[i];
            angle.Lambda = 0f;
            angles[i] = angle;
        }

        var motors = world.Motors;
        for (int i = 0; i < motors.Count; i++)
        {
            var motor = motors[i];
            motor.Lambda = 0f;
            motors[i] = motor;
        }
    }
}
