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
    /// Uses per-rod compliance if set, otherwise uses the global compliance parameter.
    /// </summary>
    public static void SolveRods(WorldState world, float dt, float globalCompliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var rods = world.Rods;

        for (int idx = 0; idx < rods.Count; idx++)
        {
            var rod = rods[idx];
            int i = rod.I;
            int j = rod.J;

            // Use per-rod compliance if specified, otherwise use global
            float compliance = rod.Compliance > 0f ? rod.Compliance : globalCompliance;
            float alpha = compliance / (dt * dt);

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
    /// Uses stable gradient formulation from XPBD literature (Müller et al).
    /// </summary>
    public static void SolveAngles(WorldState world, float dt, float globalCompliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var angles = world.Angles;

        for (int idx = 0; idx < angles.Count; idx++)
        {
            var angle = angles[idx];
            int i = angle.I;
            int j = angle.J;
            int k = angle.K;

            // Use per-angle compliance if specified, otherwise use global
            float compliance = angle.Compliance > 0f ? angle.Compliance : globalCompliance;
            float alpha = compliance / (dt * dt);

            // Skip if all particles are pinned
            if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
                continue;

            // Edge vectors from j (p0-p1 and p2-p1 in the math notation)
            float ux = posX[i] - posX[j];
            float uy = posY[i] - posY[j];
            float vx = posX[k] - posX[j];
            float vy = posY[k] - posY[j];

            float lenUSq = ux * ux + uy * uy;
            float lenVSq = vx * vx + vy * vy;

            if (lenUSq < Epsilon * Epsilon || lenVSq < Epsilon * Epsilon)
                continue;

            float lenU = MathF.Sqrt(lenUSq);
            float lenV = MathF.Sqrt(lenVSq);

            // Normalize to unit vectors
            float unx = ux / lenU;
            float uny = uy / lenU;
            float vnx = vx / lenV;
            float vny = vy / lenV;

            // Compute current angle: θ = atan2(|u×v|, u·v)
            // In 2D: u×v = ux*vy - uy*vx (scalar z-component)
            float cross = unx * vny - uny * vnx;  // u×v (normalized)
            float dot = unx * vnx + uny * vny;     // u·v (normalized)

            float currentAngle = MathF.Atan2(cross, dot);

            // Constraint value (wrapped to [-π, π])
            float C = Math2D.WrapAngle(currentAngle - angle.Theta0);

            if (MathF.Abs(C) < Epsilon)
                continue;

            // Stable gradients using perpendicular formulation:
            // For 2D, the cross product magnitude |u×v| is just |cross|
            // n = (u×v) / |u×v| is the out-of-plane normal (always ±z in 2D)
            // u_perp = (n×u) / |u×v| and v_perp = (n×v) / |u×v|
            //
            // In 2D this simplifies to:
            // ∂θ/∂p_i = u_perp / |p_i - p_j|
            // ∂θ/∂p_k = -v_perp / |p_k - p_j|
            //
            // u_perp in 2D = perpendicular to u = (-uy, ux) normalized by |u×v|
            // But since u is already normalized: u_perp ≈ (-uny, unx)

            // More stable: use the formula from the paper
            // ∂θ/∂p0 = u_perp / |p0-p1| where u_perp is perpendicular to u
            float crossMag = MathF.Abs(cross) + Epsilon;

            // Gradients (perpendicular to edges, scaled by edge length and cross product)
            // This formulation is stable even when angles are small
            float gradIx = -uny / lenU;
            float gradIy = unx / lenU;

            float gradKx = vny / lenV;
            float gradKy = -vnx / lenV;

            float gradJx = -(gradIx + gradKx);
            float gradJy = -(gradIy + gradKy);

            // Effective inverse mass (sum of w_i * |∇C_i|²)
            float w = invMass[i] * (gradIx * gradIx + gradIy * gradIy)
                    + invMass[j] * (gradJx * gradJx + gradJy * gradJy)
                    + invMass[k] * (gradKx * gradKx + gradKy * gradKy);

            if (w < Epsilon)
                continue;

            // XPBD correction: Δλ = -(C + α·λ) / (Σw_i|∇C_i|² + α)
            float deltaLambda = -(C + alpha * angle.Lambda) / (w + alpha);

            // Update positions: p_i ← p_i + w_i · Δλ · ∇C_i
            posX[i] += invMass[i] * deltaLambda * gradIx;
            posY[i] += invMass[i] * deltaLambda * gradIy;
            posX[j] += invMass[j] * deltaLambda * gradJx;
            posY[j] += invMass[j] * deltaLambda * gradJy;
            posX[k] += invMass[k] * deltaLambda * gradKx;
            posY[k] += invMass[k] * deltaLambda * gradKy;

            // Update accumulated lambda
            angle.Lambda += deltaLambda;
            angles[idx] = angle;
        }
    }

    #endregion

    #region Motor Angle Constraint

    /// <summary>
    /// Solves all motor angle constraints once.
    /// Similar to angle constraint but uses Target instead of Theta0.
    /// Motors are servo-actuators that drive to a target angle.
    /// </summary>
    public static void SolveMotors(WorldState world, float dt, float globalCompliance)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var invMass = world.InvMass;
        var motors = world.Motors;

        for (int idx = 0; idx < motors.Count; idx++)
        {
            var motor = motors[idx];
            int i = motor.I;
            int j = motor.J;
            int k = motor.K;

            // Use per-motor compliance if specified, otherwise use global
            float compliance = motor.Compliance > 0f ? motor.Compliance : globalCompliance;
            float alpha = compliance / (dt * dt);

            // Skip if all particles are pinned
            if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
                continue;

            // Edge vectors from j
            float ux = posX[i] - posX[j];
            float uy = posY[i] - posY[j];
            float vx = posX[k] - posX[j];
            float vy = posY[k] - posY[j];

            float lenUSq = ux * ux + uy * uy;
            float lenVSq = vx * vx + vy * vy;

            if (lenUSq < Epsilon * Epsilon || lenVSq < Epsilon * Epsilon)
                continue;

            float lenU = MathF.Sqrt(lenUSq);
            float lenV = MathF.Sqrt(lenVSq);

            // Normalize to unit vectors
            float unx = ux / lenU;
            float uny = uy / lenU;
            float vnx = vx / lenV;
            float vny = vy / lenV;

            // Compute current angle
            float cross = unx * vny - uny * vnx;
            float dot = unx * vnx + uny * vny;
            float currentAngle = MathF.Atan2(cross, dot);

            // Constraint value (target comes from motor)
            float C = Math2D.WrapAngle(currentAngle - motor.Target);

            if (MathF.Abs(C) < Epsilon)
                continue;

            // Stable gradients
            float gradIx = -uny / lenU;
            float gradIy = unx / lenU;

            float gradKx = vny / lenV;
            float gradKy = -vnx / lenV;

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
