namespace Evolvatron.Core.Physics;

/// <summary>
/// Symplectic Euler integrator for particle dynamics.
/// v += dt * (force / mass)
/// p += dt * v
/// </summary>
public static class Integrator
{
    /// <summary>
    /// Applies gravity and external forces to all non-pinned particles.
    /// </summary>
    public static void ApplyGravity(WorldState world, float gravityX, float gravityY)
    {
        var forceX = world.ForceX;
        var forceY = world.ForceY;
        var invMass = world.InvMass;

        for (int i = 0; i < world.ParticleCount; i++)
        {
            if (invMass[i] > 0f) // Skip pinned particles
            {
                float mass = 1f / invMass[i];
                forceX[i] += mass * gravityX;
                forceY[i] += mass * gravityY;
            }
        }
    }

    /// <summary>
    /// Integrates velocity from forces: v += dt * force / mass.
    /// Then integrates position from velocity: p += dt * v.
    /// Clears forces after integration.
    /// </summary>
    public static void Integrate(WorldState world, float dt)
    {
        var posX = world.PosX;
        var posY = world.PosY;
        var velX = world.VelX;
        var velY = world.VelY;
        var forceX = world.ForceX;
        var forceY = world.ForceY;
        var invMass = world.InvMass;

        for (int i = 0; i < world.ParticleCount; i++)
        {
            if (invMass[i] > 0f) // Skip pinned particles
            {
                // v += dt * (F / m) = dt * F * invMass
                velX[i] += dt * forceX[i] * invMass[i];
                velY[i] += dt * forceY[i] * invMass[i];

                // p += dt * v
                posX[i] += dt * velX[i];
                posY[i] += dt * velY[i];
            }
        }

        world.ClearForces();
    }

    /// <summary>
    /// Applies gravity to all rigid bodies.
    /// </summary>
    public static void ApplyGravityToRigidBodies(WorldState world, float gravityX, float gravityY, float dt)
    {
        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            if (rb.InvMass > 0f) // Skip static rigid bodies
            {
                rb.VelX += gravityX * dt;
                rb.VelY += gravityY * dt;
                world.RigidBodies[i] = rb;
            }
        }
    }

    /// <summary>
    /// Integrates rigid body linear and angular velocity and position.
    /// Uses symplectic Euler: v += dt * a, p += dt * v, angle += dt * omega.
    /// </summary>
    public static void IntegrateRigidBodies(WorldState world, float dt)
    {
        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            if (rb.InvMass == 0f) continue; // Skip static rigid bodies

            // Linear integration
            rb.X += dt * rb.VelX;
            rb.Y += dt * rb.VelY;

            // Angular integration
            rb.Angle += dt * rb.AngularVel;

            world.RigidBodies[i] = rb;
        }
    }

    /// <summary>
    /// Saves current rigid body positions and angles for velocity stabilization.
    /// </summary>
    public static void SaveRigidBodyPreviousState(WorldState world,
        out (float x, float y, float angle)[] prevState)
    {
        prevState = new (float, float, float)[world.RigidBodies.Count];
        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            prevState[i] = (rb.X, rb.Y, rb.Angle);
        }
    }

    /// <summary>
    /// Applies velocity stabilization for rigid bodies.
    /// </summary>
    public static void StabilizeRigidBodyVelocities(WorldState world,
        (float x, float y, float angle)[] prevState, float dt, float beta)
    {
        if (beta <= 0f) return;

        float invDt = 1f / dt;
        float oneMinusBeta = 1f - beta;

        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            if (rb.InvMass == 0f) continue;

            var prev = prevState[i];

            // Corrected velocities from position change
            float correctedVx = (rb.X - prev.x) * invDt;
            float correctedVy = (rb.Y - prev.y) * invDt;
            float correctedOmega = (rb.Angle - prev.angle) * invDt;

            // Blend
            rb.VelX = correctedVx * beta + rb.VelX * oneMinusBeta;
            rb.VelY = correctedVy * beta + rb.VelY * oneMinusBeta;
            rb.AngularVel = correctedOmega * beta + rb.AngularVel * oneMinusBeta;

            world.RigidBodies[i] = rb;
        }
    }

    /// <summary>
    /// Applies damping to rigid bodies (linear and angular).
    /// </summary>
    public static void ApplyRigidBodyDamping(WorldState world, float linearDamping, float angularDamping, float dt)
    {
        float linearFactor = MathF.Max(0f, 1f - linearDamping * dt);
        float angularFactor = MathF.Max(0f, 1f - angularDamping * dt);

        for (int i = 0; i < world.RigidBodies.Count; i++)
        {
            var rb = world.RigidBodies[i];
            if (linearDamping > 0f)
            {
                rb.VelX *= linearFactor;
                rb.VelY *= linearFactor;
            }
            if (angularDamping > 0f)
            {
                rb.AngularVel *= angularFactor;
            }
            world.RigidBodies[i] = rb;
        }
    }

    /// <summary>
    /// Applies angular damping to particles by dampening rotational motion
    /// around rod constraints. For each rod, computes angular velocity and applies damping.
    /// </summary>
    public static void ApplyParticleAngularDamping(WorldState world, float angularDamping, float dt)
    {
        if (angularDamping <= 0f) return;

        float factor = MathF.Max(0f, 1f - angularDamping * dt);
        var posX = world.PosX;
        var posY = world.PosY;
        var velX = world.VelX;
        var velY = world.VelY;
        var invMass = world.InvMass;

        // For each rod, dampen the rotational component of velocity
        foreach (var rod in world.Rods)
        {
            int i = rod.I;
            int j = rod.J;

            if (invMass[i] == 0f && invMass[j] == 0f) continue;

            // Center of mass
            float mi = invMass[i] > 0f ? 1f / invMass[i] : 0f;
            float mj = invMass[j] > 0f ? 1f / invMass[j] : 0f;
            float totalMass = mi + mj;
            if (totalMass < 1e-6f) continue;

            float cmX = (posX[i] * mi + posX[j] * mj) / totalMass;
            float cmY = (posY[i] * mi + posY[j] * mj) / totalMass;

            // Relative positions from center of mass
            float riX = posX[i] - cmX;
            float riY = posY[i] - cmY;
            float rjX = posX[j] - cmX;
            float rjY = posY[j] - cmY;

            // Angular velocity (omega = (r × v) / r²)
            float Li = riX * velY[i] - riY * velX[i]; // Angular momentum contribution from i
            float Lj = rjX * velY[j] - rjY * velX[j]; // Angular momentum contribution from j
            float L = Li + Lj;

            float I = mi * (riX * riX + riY * riY) + mj * (rjX * rjX + rjY * rjY); // Moment of inertia
            if (I < 1e-6f) continue;

            float omega = L / I;

            // Apply angular damping
            float omegaDamped = omega * factor;
            float deltaOmega = omegaDamped - omega;

            // Convert back to velocity change (v_tangent = omega × r)
            if (invMass[i] > 0f)
            {
                float dvix = -riY * deltaOmega;
                float dviy = riX * deltaOmega;
                velX[i] += dvix;
                velY[i] += dviy;
            }

            if (invMass[j] > 0f)
            {
                float dvjx = -rjY * deltaOmega;
                float dvjy = rjX * deltaOmega;
                velX[j] += dvjx;
                velY[j] += dvjy;
            }
        }
    }
}
