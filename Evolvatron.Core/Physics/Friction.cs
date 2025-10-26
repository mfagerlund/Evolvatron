using System;

namespace Evolvatron.Core.Physics;

/// <summary>
/// Velocity-level Coulomb-like friction for particles in contact with colliders.
/// Applied after XPBD position corrections.
/// </summary>
public static class Friction
{
    private const float Epsilon = 1e-9f;
    private const float PenetrationTolerance = 0.01f; // Consider in contact if within 1cm

    /// <summary>
    /// Applies friction to all particles that are in contact with colliders.
    /// </summary>
    public static void ApplyFriction(WorldState world, float frictionMu)
    {
        if (frictionMu <= 0f)
            return;

        var posX = world.PosX;
        var posY = world.PosY;
        var velX = world.VelX;
        var velY = world.VelY;
        var invMass = world.InvMass;
        var radius = world.Radius;

        for (int i = 0; i < world.ParticleCount; i++)
        {
            if (invMass[i] == 0f) // Skip pinned particles
                continue;

            float px = posX[i];
            float py = posY[i];
            float r = radius[i];

            // Find the most penetrating collider (or closest if near contact)
            float minPhi = float.MaxValue;
            float contactNx = 0f;
            float contactNy = 0f;
            bool inContact = false;

            // Check circles
            foreach (var circle in world.Circles)
            {
                Math2D.CircleSDF(px, py, circle, out float phi, out float nx, out float ny);
                phi -= r;

                if (phi < minPhi)
                {
                    minPhi = phi;
                    contactNx = nx;
                    contactNy = ny;
                }
            }

            // Check capsules
            foreach (var capsule in world.Capsules)
            {
                Math2D.CapsuleSDF(px, py, capsule, out float phi, out float nx, out float ny);
                phi -= r;

                if (phi < minPhi)
                {
                    minPhi = phi;
                    contactNx = nx;
                    contactNy = ny;
                }
            }

            // Check OBBs
            foreach (var obb in world.Obbs)
            {
                Math2D.OBBSDF(px, py, obb, out float phi, out float nx, out float ny);
                phi -= r;

                if (phi < minPhi)
                {
                    minPhi = phi;
                    contactNx = nx;
                    contactNy = ny;
                }
            }

            // Only apply friction if in contact (penetrating or very close)
            if (minPhi < PenetrationTolerance)
            {
                inContact = true;
            }

            if (!inContact)
                continue;

            // Current velocity
            float vx = velX[i];
            float vy = velY[i];

            // Decompose velocity into normal and tangent components
            float vn = vx * contactNx + vy * contactNy; // Normal component (scalar)
            float vtx = vx - vn * contactNx; // Tangent component (vector)
            float vty = vy - vn * contactNy;

            float vtMag = MathF.Sqrt(vtx * vtx + vty * vty);

            if (vtMag < Epsilon)
                continue; // No tangential motion

            // Coulomb friction: max tangential impulse is μ * |normal impulse|
            // For velocity-level: reduce tangential velocity by at most μ * |vn|
            float maxTangentialReduction = frictionMu * MathF.Abs(vn);

            if (maxTangentialReduction >= vtMag)
            {
                // Full friction: stop tangential motion
                velX[i] = vn * contactNx;
                velY[i] = vn * contactNy;
            }
            else
            {
                // Partial friction: reduce tangential velocity
                float reductionFactor = (vtMag - maxTangentialReduction) / vtMag;
                vtx *= reductionFactor;
                vty *= reductionFactor;

                velX[i] = vn * contactNx + vtx;
                velY[i] = vn * contactNy + vty;
            }
        }
    }
}
