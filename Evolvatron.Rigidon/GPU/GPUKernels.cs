using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using System;

namespace Evolvatron.Core.GPU;

/// <summary>
/// GPU kernels for XPBD physics simulation.
/// Each kernel operates on device memory in parallel.
/// </summary>
public static class GPUKernels
{
    private const float Epsilon = 1e-9f;

    #region Integration Kernels

    /// <summary>
    /// Applies gravity to all particles.
    /// </summary>
    public static void ApplyGravityKernel(
        Index1D index,
        ArrayView<float> forceX,
        ArrayView<float> forceY,
        ArrayView<float> invMass,
        float gravityX,
        float gravityY)
    {
        if (invMass[index] > 0f)
        {
            float mass = 1f / invMass[index];
            forceX[index] += mass * gravityX;
            forceY[index] += mass * gravityY;
        }
    }

    /// <summary>
    /// Integrates particle velocities and positions (symplectic Euler).
    /// </summary>
    public static void IntegrateKernel(
        Index1D index,
        ArrayView<float> posX,
        ArrayView<float> posY,
        ArrayView<float> velX,
        ArrayView<float> velY,
        ArrayView<float> forceX,
        ArrayView<float> forceY,
        ArrayView<float> invMass,
        float dt)
    {
        if (invMass[index] > 0f)
        {
            // v += dt * F * invMass
            velX[index] += dt * forceX[index] * invMass[index];
            velY[index] += dt * forceY[index] * invMass[index];

            // p += dt * v
            posX[index] += dt * velX[index];
            posY[index] += dt * velY[index];
        }

        // Clear forces
        forceX[index] = 0f;
        forceY[index] = 0f;
    }

    #endregion

    #region XPBD Constraint Kernels

    /// <summary>
    /// Solves rod (distance) constraints.
    /// </summary>
    public static void SolveRodsKernel(
        Index1D index,
        ArrayView<GPURod> rods,
        ArrayView<float> posX,
        ArrayView<float> posY,
        ArrayView<float> invMass,
        float dt,
        float compliance)
    {
        var rod = rods[index];
        int i = rod.I;
        int j = rod.J;

        if (invMass[i] == 0f && invMass[j] == 0f)
            return;

        float dx = posX[i] - posX[j];
        float dy = posY[i] - posY[j];
        float len = XMath.Sqrt(dx * dx + dy * dy);

        if (len < Epsilon)
            return;

        float C = len - rod.RestLength;
        float nx = dx / len;
        float ny = dy / len;

        float w = invMass[i] + invMass[j];
        if (w < Epsilon)
            return;

        float alpha = compliance / (dt * dt);
        float deltaLambda = -(C + alpha * rod.Lambda) / (w + alpha);

        // Update positions (atomic operations)
        Atomic.Add(ref posX[i], invMass[i] * deltaLambda * nx);
        Atomic.Add(ref posY[i], invMass[i] * deltaLambda * ny);
        Atomic.Add(ref posX[j], -invMass[j] * deltaLambda * nx);
        Atomic.Add(ref posY[j], -invMass[j] * deltaLambda * ny);

        // Update lambda
        rod.Lambda += deltaLambda;
        rods[index] = rod;
    }

    /// <summary>
    /// Solves angle constraints.
    /// </summary>
    public static void SolveAnglesKernel(
        Index1D index,
        ArrayView<GPUAngle> angles,
        ArrayView<float> posX,
        ArrayView<float> posY,
        ArrayView<float> invMass,
        float dt,
        float compliance)
    {
        var angle = angles[index];
        int i = angle.I;
        int j = angle.J;
        int k = angle.K;

        if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
            return;

        // Edge vectors (unnormalized)
        float ux = posX[i] - posX[j];
        float uy = posY[i] - posY[j];
        float vx = posX[k] - posX[j];
        float vy = posY[k] - posY[j];

        float uu = ux * ux + uy * uy;
        float vv = vx * vx + vy * vy;

        if (uu < Epsilon || vv < Epsilon)
            return;

        float c = ux * vx + uy * vy;
        float s = ux * vy - uy * vx;
        float currentAngle = XMath.Atan2(s, c);

        // Constraint value (wrap to [-π, π])
        float C = currentAngle - angle.Theta0;
        while (C > XMath.PI) C -= 2f * XMath.PI;
        while (C < -XMath.PI) C += 2f * XMath.PI;

        if (XMath.Abs(C) < Epsilon)
            return;

        float denom = uu * vv + 1e-12f;

        // Gradients (match CPU solver)
        float gradIx = (c * vy - s * vx) / denom;
        float gradIy = (-c * vx - s * vy) / denom;
        float gradKx = (c * (-uy) - s * ux) / denom;
        float gradKy = (c * ux - s * uy) / denom;
        float gradJx = -(gradIx + gradKx);
        float gradJy = -(gradIy + gradKy);

        float w = invMass[i] * (gradIx * gradIx + gradIy * gradIy)
                + invMass[j] * (gradJx * gradJx + gradJy * gradJy)
                + invMass[k] * (gradKx * gradKx + gradKy * gradKy);

        if (w < Epsilon)
            return;

        float alpha = compliance / (dt * dt);
        float deltaLambda = -(C + alpha * angle.Lambda) / (w + alpha);

        // Update positions
        Atomic.Add(ref posX[i], invMass[i] * deltaLambda * gradIx);
        Atomic.Add(ref posY[i], invMass[i] * deltaLambda * gradIy);
        Atomic.Add(ref posX[j], invMass[j] * deltaLambda * gradJx);
        Atomic.Add(ref posY[j], invMass[j] * deltaLambda * gradJy);
        Atomic.Add(ref posX[k], invMass[k] * deltaLambda * gradKx);
        Atomic.Add(ref posY[k], invMass[k] * deltaLambda * gradKy);

        angle.Lambda += deltaLambda;
        angles[index] = angle;
    }

    /// <summary>
    /// Solves motor angle constraints.
    /// </summary>
    public static void SolveMotorsKernel(
        Index1D index,
        ArrayView<GPUMotorAngle> motors,
        ArrayView<float> posX,
        ArrayView<float> posY,
        ArrayView<float> invMass,
        float dt,
        float compliance)
    {
        var motor = motors[index];
        int i = motor.I;
        int j = motor.J;
        int k = motor.K;

        if (invMass[i] == 0f && invMass[j] == 0f && invMass[k] == 0f)
            return;

        // Same as angle constraint but uses Target
        float ux = posX[i] - posX[j];
        float uy = posY[i] - posY[j];
        float vx = posX[k] - posX[j];
        float vy = posY[k] - posY[j];

        float uu = ux * ux + uy * uy;
        float vv = vx * vx + vy * vy;

        if (uu < Epsilon || vv < Epsilon)
            return;

        float c = ux * vx + uy * vy;
        float s = ux * vy - uy * vx;
        float currentAngle = XMath.Atan2(s, c);

        float C = currentAngle - motor.Target;
        while (C > XMath.PI) C -= 2f * XMath.PI;
        while (C < -XMath.PI) C += 2f * XMath.PI;

        if (XMath.Abs(C) < Epsilon)
            return;

        float denom = uu * vv + 1e-12f;

        float gradIx = (c * vy - s * vx) / denom;
        float gradIy = (-c * vx - s * vy) / denom;
        float gradKx = (c * (-uy) - s * ux) / denom;
        float gradKy = (c * ux - s * uy) / denom;
        float gradJx = -(gradIx + gradKx);
        float gradJy = -(gradIy + gradKy);

        float w = invMass[i] * (gradIx * gradIx + gradIy * gradIy)
                + invMass[j] * (gradJx * gradJx + gradJy * gradJy)
                + invMass[k] * (gradKx * gradKx + gradKy * gradKy);

        if (w < Epsilon)
            return;

        float alpha = compliance / (dt * dt);
        float deltaLambda = -(C + alpha * motor.Lambda) / (w + alpha);

        Atomic.Add(ref posX[i], invMass[i] * deltaLambda * gradIx);
        Atomic.Add(ref posY[i], invMass[i] * deltaLambda * gradIy);
        Atomic.Add(ref posX[j], invMass[j] * deltaLambda * gradJx);
        Atomic.Add(ref posY[j], invMass[j] * deltaLambda * gradJy);
        Atomic.Add(ref posX[k], invMass[k] * deltaLambda * gradKx);
        Atomic.Add(ref posY[k], invMass[k] * deltaLambda * gradKy);

        motor.Lambda += deltaLambda;
        motors[index] = motor;
    }

    #endregion

    #region Contact and Friction Kernels

    /// <summary>
    /// Solves contacts for all particles against all colliders.
    /// </summary>
    public static void SolveContactsKernel(
        Index1D index,
        ArrayView<float> posX,
        ArrayView<float> posY,
        ArrayView<float> invMass,
        ArrayView<float> radius,
        ArrayView<GPUCircleCollider> circles,
        ArrayView<GPUCapsuleCollider> capsules,
        ArrayView<GPUOBBCollider> obbs,
        int circleCount,
        int capsuleCount,
        int obbCount,
        float dt,
        float compliance)
    {
        if (invMass[index] == 0f)
            return;

        float px = posX[index];
        float py = posY[index];
        float r = radius[index];
        float alpha = compliance / (dt * dt);

        // Check circles
        for (int c = 0; c < circleCount; c++)
        {
            var circle = circles[c];
            float phi, nx, ny;
            CircleSDF(px, py, circle, out phi, out nx, out ny);
            phi -= r;

            if (phi < 0f)
            {
                SolveContact(ref px, ref py, invMass[index], phi, nx, ny, alpha);
            }
        }

        // Check capsules
        for (int c = 0; c < capsuleCount; c++)
        {
            var capsule = capsules[c];
            float phi, nx, ny;
            CapsuleSDF(px, py, capsule, out phi, out nx, out ny);
            phi -= r;

            if (phi < 0f)
            {
                SolveContact(ref px, ref py, invMass[index], phi, nx, ny, alpha);
            }
        }

        // Check OBBs
        for (int c = 0; c < obbCount; c++)
        {
            var obb = obbs[c];
            float phi, nx, ny;
            OBBSDF(px, py, obb, out phi, out nx, out ny);
            phi -= r;

            if (phi < 0f)
            {
                SolveContact(ref px, ref py, invMass[index], phi, nx, ny, alpha);
            }
        }

        posX[index] = px;
        posY[index] = py;
    }

    private static void SolveContact(
        ref float px, ref float py,
        float invMass,
        float phi, float nx, float ny,
        float alpha)
    {
        float C = phi;
        float w = invMass;

        if (w < Epsilon)
            return;

        float deltaLambda = -(C + alpha * 0f) / (w + alpha);
        if (deltaLambda < 0f)
            deltaLambda = 0f;

        px += invMass * deltaLambda * nx;
        py += invMass * deltaLambda * ny;
    }

    #endregion

    #region Post-Processing Kernels

    /// <summary>
    /// Stabilizes velocities from position changes during XPBD solving.
    /// v = beta * (pos - prevPos) * invDt + (1 - beta) * vel
    /// Optionally clamps velocity magnitude to prevent energy injection.
    /// </summary>
    public static void VelocityStabilizationKernel(
        Index1D index,
        ArrayView<float> posX, ArrayView<float> posY,
        ArrayView<float> prevPosX, ArrayView<float> prevPosY,
        ArrayView<float> velX, ArrayView<float> velY,
        ArrayView<float> invMass,
        float invDt, float beta, float maxVelocity)
    {
        // Only process dynamic particles
        if (invMass[index] == 0f)
            return;

        float correctedVx = (posX[index] - prevPosX[index]) * invDt;
        float correctedVy = (posY[index] - prevPosY[index]) * invDt;

        float oneMinusBeta = 1f - beta;
        float vx = correctedVx * beta + velX[index] * oneMinusBeta;
        float vy = correctedVy * beta + velY[index] * oneMinusBeta;

        // Clamp velocity magnitude if maxVelocity > 0, with energy dissipation
        if (maxVelocity > 0f)
        {
            float velSq = vx * vx + vy * vy;
            float maxVelSq = maxVelocity * maxVelocity;
            if (velSq > maxVelSq)
            {
                // Dissipate energy: scale down to 50% of max velocity
                float scale = (maxVelocity * 0.5f) / XMath.Sqrt(velSq);
                vx *= scale;
                vy *= scale;
            }
        }

        velX[index] = vx;
        velY[index] = vy;
    }

    private const float FrictionPenetrationTolerance = 0.01f; // Consider in contact if within 1cm

    /// <summary>
    /// Applies velocity-level Coulomb friction to particles in contact with colliders.
    /// </summary>
    public static void FrictionKernel(
        Index1D index,
        ArrayView<float> posX, ArrayView<float> posY,
        ArrayView<float> velX, ArrayView<float> velY,
        ArrayView<float> invMass, ArrayView<float> radius,
        ArrayView<GPUCircleCollider> circles,
        ArrayView<GPUCapsuleCollider> capsules,
        ArrayView<GPUOBBCollider> obbs,
        int circleCount, int capsuleCount, int obbCount,
        float mu)
    {
        // Skip pinned particles
        if (invMass[index] == 0f)
            return;

        if (mu <= 0f)
            return;

        float px = posX[index];
        float py = posY[index];
        float r = radius[index];

        // Find the most penetrating collider (or closest if near contact)
        float minPhi = float.MaxValue;
        float contactNx = 0f;
        float contactNy = 0f;

        // Check circles
        for (int c = 0; c < circleCount; c++)
        {
            var circle = circles[c];
            float phi, nx, ny;
            CircleSDF(px, py, circle, out phi, out nx, out ny);
            phi -= r;

            if (phi < minPhi)
            {
                minPhi = phi;
                contactNx = nx;
                contactNy = ny;
            }
        }

        // Check capsules
        for (int c = 0; c < capsuleCount; c++)
        {
            var capsule = capsules[c];
            float phi, nx, ny;
            CapsuleSDF(px, py, capsule, out phi, out nx, out ny);
            phi -= r;

            if (phi < minPhi)
            {
                minPhi = phi;
                contactNx = nx;
                contactNy = ny;
            }
        }

        // Check OBBs
        for (int c = 0; c < obbCount; c++)
        {
            var obb = obbs[c];
            float phi, nx, ny;
            OBBSDF(px, py, obb, out phi, out nx, out ny);
            phi -= r;

            if (phi < minPhi)
            {
                minPhi = phi;
                contactNx = nx;
                contactNy = ny;
            }
        }

        // Only apply friction if in contact (penetrating or very close)
        if (minPhi >= FrictionPenetrationTolerance)
            return;

        // Current velocity
        float vx = velX[index];
        float vy = velY[index];

        // Decompose velocity into normal and tangent components
        float vn = vx * contactNx + vy * contactNy; // Normal component (scalar)
        float vtx = vx - vn * contactNx; // Tangent component (vector)
        float vty = vy - vn * contactNy;

        float vtMag = XMath.Sqrt(vtx * vtx + vty * vty);

        if (vtMag < Epsilon)
            return; // No tangential motion

        // Coulomb friction: max tangential impulse is mu * |normal impulse|
        // For velocity-level: reduce tangential velocity by at most mu * |vn|
        float maxTangentialReduction = mu * XMath.Abs(vn);

        if (maxTangentialReduction >= vtMag)
        {
            // Full friction: stop tangential motion
            velX[index] = vn * contactNx;
            velY[index] = vn * contactNy;
        }
        else
        {
            // Partial friction: reduce tangential velocity
            float reductionFactor = (vtMag - maxTangentialReduction) / vtMag;
            vtx *= reductionFactor;
            vty *= reductionFactor;

            velX[index] = vn * contactNx + vtx;
            velY[index] = vn * contactNy + vty;
        }
    }

    /// <summary>
    /// Applies global velocity damping to all dynamic particles.
    /// vel *= dampingFactor where dampingFactor = max(0, 1 - damping * dt)
    /// </summary>
    public static void DampingKernel(
        Index1D index,
        ArrayView<float> velX, ArrayView<float> velY,
        ArrayView<float> invMass,
        float dampingFactor)
    {
        // Only apply to dynamic particles (invMass > 0)
        if (invMass[index] == 0f)
            return;

        velX[index] *= dampingFactor;
        velY[index] *= dampingFactor;
    }

    #endregion

    #region SDF Functions (GPU versions)

    private static void CircleSDF(
        float px, float py,
        GPUCircleCollider circle,
        out float phi, out float nx, out float ny)
    {
        float dx = px - circle.CX;
        float dy = py - circle.CY;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        if (dist < Epsilon)
        {
            phi = -circle.Radius;
            nx = 1f;
            ny = 0f;
        }
        else
        {
            phi = dist - circle.Radius;
            nx = dx / dist;
            ny = dy / dist;
        }
    }

    private static void CapsuleSDF(
        float px, float py,
        GPUCapsuleCollider capsule,
        out float phi, out float nx, out float ny)
    {
        float ax = capsule.CX - capsule.UX * capsule.HalfLength;
        float ay = capsule.CY - capsule.UY * capsule.HalfLength;
        float bx = capsule.CX + capsule.UX * capsule.HalfLength;
        float by = capsule.CY + capsule.UY * capsule.HalfLength;

        float abx = bx - ax;
        float aby = by - ay;
        float apx = px - ax;
        float apy = py - ay;

        float t = (apx * abx + apy * aby) / (abx * abx + aby * aby);
        t = XMath.Clamp(t, 0f, 1f);

        float qx = ax + t * abx;
        float qy = ay + t * aby;

        float dx = px - qx;
        float dy = py - qy;
        float dist = XMath.Sqrt(dx * dx + dy * dy);

        if (dist < Epsilon)
        {
            phi = -capsule.Radius;
            nx = -capsule.UY;
            ny = capsule.UX;
        }
        else
        {
            phi = dist - capsule.Radius;
            nx = dx / dist;
            ny = dy / dist;
        }
    }

    private static void OBBSDF(
        float px, float py,
        GPUOBBCollider obb,
        out float phi, out float nx, out float ny)
    {
        float dx = px - obb.CX;
        float dy = py - obb.CY;

        float localX = dx * obb.UX + dy * obb.UY;
        float localY = dx * (-obb.UY) + dy * obb.UX;

        float qx = XMath.Clamp(localX, -obb.HalfExtentX, obb.HalfExtentX);
        float qy = XMath.Clamp(localY, -obb.HalfExtentY, obb.HalfExtentY);

        float ex = localX - qx;
        float ey = localY - qy;
        float distSq = ex * ex + ey * ey;

        if (distSq < Epsilon)
        {
            float distX = obb.HalfExtentX - XMath.Abs(localX);
            float distY = obb.HalfExtentY - XMath.Abs(localY);

            if (distX < distY)
            {
                phi = -distX;
                float signX = localX >= 0f ? 1f : -1f;
                nx = signX * obb.UX;
                ny = signX * obb.UY;
            }
            else
            {
                phi = -distY;
                float signY = localY >= 0f ? 1f : -1f;
                nx = -signY * obb.UY;
                ny = signY * obb.UX;
            }
        }
        else
        {
            phi = XMath.Sqrt(distSq);
            float localNx = ex / phi;
            float localNy = ey / phi;
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
    }

    #endregion
}
