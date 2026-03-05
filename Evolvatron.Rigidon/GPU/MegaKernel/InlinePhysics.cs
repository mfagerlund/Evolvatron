using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.MegaKernel;

/// <summary>
/// Inline physics substep for a single world.
/// Runs the complete physics pipeline sequentially on one GPU thread.
/// Ported from GPUBatchedPhysicsKernels, GPUBatchedContactKernels, GPUBatchedJointKernels.
/// No atomics needed — one thread owns the entire world.
/// </summary>
public static class InlinePhysics
{
    private const float Epsilon = 1e-9f;
    private const float Baumgarte = 0.2f;
    private const float LinearSlop = 0.01f;
    private const float MaxLinearCorrection = 0.2f;
    private const float JointLinearSlop = 0.005f;
    private const float AngularSlop = 2f * XMath.PI / 180f;

    /// <summary>
    /// Run the complete physics substep for one world.
    /// Gravity → Save → Integrate → Geoms → Detect → WarmStart → InitJoints →
    /// N×(SolveContacts + SolveJoints) → PositionSolve → StoreCache → Damp
    /// </summary>
    public static void SubStepOneWorld(
        PhysicsViews pv,
        int worldIdx,
        MegaKernelConfig cfg)
    {
        int bodyBase = worldIdx * cfg.BodiesPerWorld;
        int geomBase = worldIdx * cfg.GeomsPerWorld;
        int jointBase = worldIdx * cfg.JointsPerWorld;
        int contactBase = worldIdx * cfg.MaxContactsPerWorld;
        float dt = cfg.Dt;

        // === 1. Apply gravity ===
        for (int i = 0; i < cfg.BodiesPerWorld; i++)
        {
            int idx = bodyBase + i;
            var body = pv.Bodies[idx];
            if (body.InvMass <= 0f) continue;
            body.VelX += cfg.GravityX * dt;
            body.VelY += cfg.GravityY * dt;
            pv.Bodies[idx] = body;
        }

        // === 2. Save previous positions ===
        for (int i = 0; i < cfg.BodiesPerWorld; i++)
        {
            int idx = bodyBase + i;
            var body = pv.Bodies[idx];
            body.PrevX = body.X;
            body.PrevY = body.Y;
            body.PrevAngle = body.Angle;
            pv.Bodies[idx] = body;
        }

        // === 3. Integrate positions ===
        for (int i = 0; i < cfg.BodiesPerWorld; i++)
        {
            int idx = bodyBase + i;
            var body = pv.Bodies[idx];
            if (body.InvMass <= 0f) continue;
            body.X += body.VelX * dt;
            body.Y += body.VelY * dt;
            body.Angle += body.AngularVel * dt;
            pv.Bodies[idx] = body;
        }

        // === 3b. Cache cos/sin for all 3 bodies (valid until position solve modifies angles) ===
        float cos0 = XMath.Cos(pv.Bodies[bodyBase].Angle);
        float sin0 = XMath.Sin(pv.Bodies[bodyBase].Angle);
        float cos1 = XMath.Cos(pv.Bodies[bodyBase + 1].Angle);
        float sin1 = XMath.Sin(pv.Bodies[bodyBase + 1].Angle);
        float cos2 = XMath.Cos(pv.Bodies[bodyBase + 2].Angle);
        float sin2 = XMath.Sin(pv.Bodies[bodyBase + 2].Angle);

        // === 4. Update geom world positions ===
        for (int i = 0; i < cfg.GeomsPerWorld; i++)
        {
            int gIdx = geomBase + i;
            var geom = pv.Geoms[gIdx];
            int bIdx = bodyBase + geom.BodyIndex;
            var body = pv.Bodies[bIdx];

            float cos = geom.BodyIndex == 0 ? cos0 : (geom.BodyIndex == 1 ? cos1 : cos2);
            float sin = geom.BodyIndex == 0 ? sin0 : (geom.BodyIndex == 1 ? sin1 : sin2);
            geom.WorldX = body.X + geom.LocalX * cos - geom.LocalY * sin;
            geom.WorldY = body.Y + geom.LocalX * sin + geom.LocalY * cos;
            pv.Geoms[gIdx] = geom;
        }

        // === 5. Clear contact count ===
        int contactCount = 0;

        // === 6. Detect OBB contacts ===
        for (int gi = 0; gi < cfg.GeomsPerWorld; gi++)
        {
            int gIdx = geomBase + gi;
            var geom = pv.Geoms[gIdx];
            int bIdx = bodyBase + geom.BodyIndex;
            var body = pv.Bodies[bIdx];
            if (body.InvMass <= 0f) continue;

            for (int ci = 0; ci < cfg.SharedColliderCount; ci++)
            {
                var obb = pv.SharedOBBColliders[ci];

                float phi, nx, ny;
                CircleVsOBBSDF(geom.WorldX, geom.WorldY, geom.Radius, obb, out phi, out nx, out ny);

                if (phi >= 0f) continue;
                if (contactCount >= cfg.MaxContactsPerWorld) break;

                int cIdx = contactBase + contactCount;

                float contactX = geom.WorldX - nx * geom.Radius;
                float contactY = geom.WorldY - ny * geom.Radius;
                float rx = contactX - body.X;
                float ry = contactY - body.Y;

                float rnCross = rx * ny - ry * nx;
                float normalMass = body.InvMass + body.InvInertia * rnCross * rnCross;
                normalMass = normalMass > Epsilon ? 1f / normalMass : 0f;

                float tx = -ny;
                float ty = nx;
                float rtCross = rx * ty - ry * tx;
                float tangentMass = body.InvMass + body.InvInertia * rtCross * rtCross;
                tangentMass = tangentMass > Epsilon ? 1f / tangentMass : 0f;

                float velocityBias = 0f;
                if (phi < -LinearSlop)
                    velocityBias = (Baumgarte / dt) * (-phi - LinearSlop);

                pv.Contacts[cIdx] = new GPUContactConstraint
                {
                    RigidBodyIndex = geom.BodyIndex,
                    NormalX = nx, NormalY = ny,
                    TangentX = tx, TangentY = ty,
                    ContactX = contactX, ContactY = contactY,
                    RA_X = rx, RA_Y = ry,
                    Separation = phi,
                    NormalMass = normalMass,
                    TangentMass = tangentMass,
                    VelocityBias = velocityBias,
                    NormalImpulse = 0f,
                    TangentImpulse = 0f,
                    Friction = cfg.FrictionMu,
                    Restitution = cfg.Restitution,
                    GeomIndex = gi - body.GeomStartIndex,
                    ColliderType = 2,
                    ColliderIndex = ci,
                    IsValid = 1
                };
                contactCount++;
            }
            if (contactCount >= cfg.MaxContactsPerWorld) break;
        }
        pv.ContactCounts[worldIdx] = contactCount;

        // === 7. Warm-start from cached impulses ===
        int cacheSize = cfg.MaxContactsPerWorld;
        int cacheStart = worldIdx * cacheSize;
        for (int c = 0; c < contactCount; c++)
        {
            int cIdx = contactBase + c;
            var contact = pv.Contacts[cIdx];
            if (contact.IsValid == 0) continue;

            for (int i = cacheStart; i < cacheStart + cacheSize; i++)
            {
                var cached = pv.ContactCache[i];
                if (cached.IsValid != 0 &&
                    cached.RigidBodyIndex == contact.RigidBodyIndex &&
                    cached.GeomIndex == contact.GeomIndex &&
                    cached.ColliderType == contact.ColliderType &&
                    cached.ColliderIndex == contact.ColliderIndex)
                {
                    contact.NormalImpulse = cached.NormalImpulse;
                    contact.TangentImpulse = cached.TangentImpulse;

                    int bIdx = bodyBase + contact.RigidBodyIndex;
                    var body = pv.Bodies[bIdx];

                    float px = contact.NormalX * contact.NormalImpulse + contact.TangentX * contact.TangentImpulse;
                    float py = contact.NormalY * contact.NormalImpulse + contact.TangentY * contact.TangentImpulse;

                    body.VelX += body.InvMass * px;
                    body.VelY += body.InvMass * py;
                    body.AngularVel += body.InvInertia * (contact.RA_X * py - contact.RA_Y * px);
                    pv.Bodies[bIdx] = body;

                    pv.Contacts[cIdx] = contact;
                    break;
                }
            }
        }

        // === 8. Initialize joint constraints ===
        for (int j = 0; j < cfg.JointsPerWorld; j++)
        {
            int jIdx = jointBase + j;
            var joint = pv.Joints[jIdx];
            var bodyA = pv.Bodies[joint.BodyA];
            var bodyB = pv.Bodies[joint.BodyB];

            if (bodyA.InvMass == 0f && bodyB.InvMass == 0f)
            {
                pv.JointConstraints[jIdx] = new GPUJointConstraint
                {
                    BodyAIndex = joint.BodyA, BodyBIndex = joint.BodyB,
                    Mass11 = 0f, Mass12 = 0f, Mass21 = 0f, Mass22 = 0f,
                    EnableLimits = 0, EnableMotor = 0
                };
                continue;
            }

            var constraint = new GPUJointConstraint
            {
                BodyAIndex = joint.BodyA, BodyBIndex = joint.BodyB,
                RA_X = joint.LocalAnchorAX, RA_Y = joint.LocalAnchorAY,
                RB_X = joint.LocalAnchorBX, RB_Y = joint.LocalAnchorBY,
                EnableLimits = joint.EnableLimits,
                LowerAngle = joint.LowerAngle, UpperAngle = joint.UpperAngle,
                ReferenceAngle = joint.ReferenceAngle,
                EnableMotor = joint.EnableMotor,
                MotorSpeed = joint.MotorSpeed,
                MaxMotorTorque = joint.MaxMotorTorque,
                ImpulseX = 0f, ImpulseY = 0f,
                AngleLimitImpulse = 0f, MotorImpulse = 0f
            };

            int localA = joint.BodyA - bodyBase;
            int localB = joint.BodyB - bodyBase;
            float cosA = localA == 0 ? cos0 : (localA == 1 ? cos1 : cos2);
            float sinA = localA == 0 ? sin0 : (localA == 1 ? sin1 : sin2);
            float rAX_w = constraint.RA_X * cosA - constraint.RA_Y * sinA;
            float rAY_w = constraint.RA_X * sinA + constraint.RA_Y * cosA;

            float cosB = localB == 0 ? cos0 : (localB == 1 ? cos1 : cos2);
            float sinB = localB == 0 ? sin0 : (localB == 1 ? sin1 : sin2);
            float rBX_w = constraint.RB_X * cosB - constraint.RB_Y * sinB;
            float rBY_w = constraint.RB_X * sinB + constraint.RB_Y * cosB;

            float k11 = bodyA.InvMass + bodyB.InvMass;
            float k22 = bodyA.InvMass + bodyB.InvMass;
            k11 += bodyA.InvInertia * rAY_w * rAY_w + bodyB.InvInertia * rBY_w * rBY_w;
            k22 += bodyA.InvInertia * rAX_w * rAX_w + bodyB.InvInertia * rBX_w * rBX_w;
            float k12 = -bodyA.InvInertia * rAX_w * rAY_w - bodyB.InvInertia * rBX_w * rBY_w;

            float det = k11 * k22 - k12 * k12;
            if (XMath.Abs(det) > Epsilon)
            {
                float invDet = 1f / det;
                constraint.Mass11 = k22 * invDet;
                constraint.Mass12 = -k12 * invDet;
                constraint.Mass21 = -k12 * invDet;
                constraint.Mass22 = k11 * invDet;
            }

            if (constraint.EnableLimits != 0)
            {
                float angMass = bodyA.InvInertia + bodyB.InvInertia;
                constraint.AngleLimitMass = angMass > Epsilon ? 1f / angMass : 0f;
            }

            if (constraint.EnableMotor != 0)
            {
                float motorMass = bodyA.InvInertia + bodyB.InvInertia;
                constraint.MotorMass = motorMass > Epsilon ? 1f / motorMass : 0f;
            }

            pv.JointConstraints[jIdx] = constraint;
        }

        // === 9. Iterative solve: contacts + joints ===
        for (int iter = 0; iter < cfg.SolverIterations; iter++)
        {
            // Solve contact velocities (Gauss-Seidel)
            for (int c = 0; c < contactCount; c++)
            {
                int cIdx = contactBase + c;
                var contact = pv.Contacts[cIdx];
                if (contact.IsValid == 0) continue;

                int bIdx = bodyBase + contact.RigidBodyIndex;
                var body = pv.Bodies[bIdx];
                if (body.InvMass <= 0f) continue;

                // Friction (tangent)
                float vx = body.VelX - body.AngularVel * contact.RA_Y;
                float vy = body.VelY + body.AngularVel * contact.RA_X;
                float vt = vx * contact.TangentX + vy * contact.TangentY;
                float lambda = -contact.TangentMass * vt;

                float maxFriction = contact.Friction * contact.NormalImpulse;
                float oldTI = contact.TangentImpulse;
                float newTI = oldTI + lambda;
                newTI = XMath.Max(-maxFriction, XMath.Min(newTI, maxFriction));
                lambda = newTI - oldTI;
                contact.TangentImpulse = newTI;

                float px = contact.TangentX * lambda;
                float py = contact.TangentY * lambda;
                body.VelX += body.InvMass * px;
                body.VelY += body.InvMass * py;
                body.AngularVel += body.InvInertia * (contact.RA_X * py - contact.RA_Y * px);

                // Normal
                vx = body.VelX - body.AngularVel * contact.RA_Y;
                vy = body.VelY + body.AngularVel * contact.RA_X;
                float vn = vx * contact.NormalX + vy * contact.NormalY;
                lambda = -contact.NormalMass * (vn - contact.VelocityBias);

                float oldNI = contact.NormalImpulse;
                float newNI = XMath.Max(oldNI + lambda, 0f);
                lambda = newNI - oldNI;
                contact.NormalImpulse = newNI;

                px = contact.NormalX * lambda;
                py = contact.NormalY * lambda;
                body.VelX += body.InvMass * px;
                body.VelY += body.InvMass * py;
                body.AngularVel += body.InvInertia * (contact.RA_X * py - contact.RA_Y * px);

                pv.Bodies[bIdx] = body;
                pv.Contacts[cIdx] = contact;
            }

            // Solve joint velocities (Gauss-Seidel)
            for (int j = 0; j < cfg.JointsPerWorld; j++)
            {
                int jIdx = jointBase + j;
                var constraint = pv.JointConstraints[jIdx];

                if (constraint.Mass11 == 0f && constraint.Mass22 == 0f &&
                    constraint.MotorMass == 0f && constraint.AngleLimitMass == 0f)
                    continue;

                var bodyA = pv.Bodies[constraint.BodyAIndex];
                var bodyB = pv.Bodies[constraint.BodyBIndex];

                int lA = constraint.BodyAIndex - bodyBase;
                int lB = constraint.BodyBIndex - bodyBase;
                float cosA = lA == 0 ? cos0 : (lA == 1 ? cos1 : cos2);
                float sinA = lA == 0 ? sin0 : (lA == 1 ? sin1 : sin2);
                float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
                float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

                float cosB = lB == 0 ? cos0 : (lB == 1 ? cos1 : cos2);
                float sinB = lB == 0 ? sin0 : (lB == 1 ? sin1 : sin2);
                float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
                float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

                // Motor
                if (constraint.EnableMotor != 0 && constraint.MotorMass > 0f)
                {
                    float angVel = bodyB.AngularVel - bodyA.AngularVel;
                    float motorImpulse = -constraint.MotorMass * (angVel - constraint.MotorSpeed);
                    float oldMI = constraint.MotorImpulse;
                    float maxImpulse = constraint.MaxMotorTorque * dt;
                    float newMI = XMath.Max(-maxImpulse, XMath.Min(oldMI + motorImpulse, maxImpulse));
                    motorImpulse = newMI - oldMI;
                    constraint.MotorImpulse = newMI;
                    bodyA.AngularVel -= bodyA.InvInertia * motorImpulse;
                    bodyB.AngularVel += bodyB.InvInertia * motorImpulse;
                }

                // Angle limits
                if (constraint.EnableLimits != 0 && constraint.AngleLimitMass > 0f)
                {
                    float angle = bodyB.Angle - bodyA.Angle - constraint.ReferenceAngle;
                    angle -= 2f * XMath.PI * XMath.Floor((angle + XMath.PI) / (2f * XMath.PI));

                    float limitImpulse = 0f;
                    if (angle < constraint.LowerAngle)
                    {
                        limitImpulse = -constraint.AngleLimitMass * (angle - constraint.LowerAngle);
                        float oldImp = constraint.AngleLimitImpulse;
                        float newImp = XMath.Max(oldImp + limitImpulse, 0f);
                        limitImpulse = newImp - oldImp;
                        constraint.AngleLimitImpulse = newImp;
                    }
                    else if (angle > constraint.UpperAngle)
                    {
                        limitImpulse = -constraint.AngleLimitMass * (angle - constraint.UpperAngle);
                        float oldImp = constraint.AngleLimitImpulse;
                        float newImp = XMath.Min(oldImp + limitImpulse, 0f);
                        limitImpulse = newImp - oldImp;
                        constraint.AngleLimitImpulse = newImp;
                    }
                    bodyA.AngularVel -= bodyA.InvInertia * limitImpulse;
                    bodyB.AngularVel += bodyB.InvInertia * limitImpulse;
                }

                // Position constraint (velocity level)
                float vAX = bodyA.VelX - bodyA.AngularVel * rAY;
                float vAY = bodyA.VelY + bodyA.AngularVel * rAX;
                float vBX = bodyB.VelX - bodyB.AngularVel * rBY;
                float vBY = bodyB.VelY + bodyB.AngularVel * rBX;

                float relVelX = vBX - vAX;
                float relVelY = vBY - vAY;

                float lambdaX = -(constraint.Mass11 * relVelX + constraint.Mass12 * relVelY);
                float lambdaY = -(constraint.Mass21 * relVelX + constraint.Mass22 * relVelY);

                constraint.ImpulseX += lambdaX;
                constraint.ImpulseY += lambdaY;

                bodyA.VelX -= bodyA.InvMass * lambdaX;
                bodyA.VelY -= bodyA.InvMass * lambdaY;
                bodyA.AngularVel -= bodyA.InvInertia * (rAX * lambdaY - rAY * lambdaX);

                bodyB.VelX += bodyB.InvMass * lambdaX;
                bodyB.VelY += bodyB.InvMass * lambdaY;
                bodyB.AngularVel += bodyB.InvInertia * (rBX * lambdaY - rBY * lambdaX);

                pv.Bodies[constraint.BodyAIndex] = bodyA;
                pv.Bodies[constraint.BodyBIndex] = bodyB;
                pv.JointConstraints[jIdx] = constraint;
            }
        }

        // === 10. Solve joint position constraints ===
        for (int j = 0; j < cfg.JointsPerWorld; j++)
        {
            int jIdx = jointBase + j;
            var constraint = pv.JointConstraints[jIdx];

            if (constraint.Mass11 == 0f && constraint.Mass22 == 0f && constraint.AngleLimitMass == 0f)
                continue;

            var bodyA = pv.Bodies[constraint.BodyAIndex];
            var bodyB = pv.Bodies[constraint.BodyBIndex];

            float cosA = XMath.Cos(bodyA.Angle);
            float sinA = XMath.Sin(bodyA.Angle);
            float rAX = constraint.RA_X * cosA - constraint.RA_Y * sinA;
            float rAY = constraint.RA_X * sinA + constraint.RA_Y * cosA;

            float cosB = XMath.Cos(bodyB.Angle);
            float sinB = XMath.Sin(bodyB.Angle);
            float rBX = constraint.RB_X * cosB - constraint.RB_Y * sinB;
            float rBY = constraint.RB_X * sinB + constraint.RB_Y * cosB;

            float errX = bodyB.X + rBX - bodyA.X - rAX;
            float errY = bodyB.Y + rBY - bodyA.Y - rAY;
            float errLen = XMath.Sqrt(errX * errX + errY * errY);

            if (errLen > JointLinearSlop)
            {
                float correction = XMath.Min(errLen, MaxLinearCorrection);
                float scale = correction / errLen;
                errX *= scale;
                errY *= scale;

                float impX = -(constraint.Mass11 * errX + constraint.Mass12 * errY);
                float impY = -(constraint.Mass21 * errX + constraint.Mass22 * errY);

                bodyA.X -= bodyA.InvMass * impX;
                bodyA.Y -= bodyA.InvMass * impY;
                bodyA.Angle -= bodyA.InvInertia * (rAX * impY - rAY * impX);

                bodyB.X += bodyB.InvMass * impX;
                bodyB.Y += bodyB.InvMass * impY;
                bodyB.Angle += bodyB.InvInertia * (rBX * impY - rBY * impX);
            }

            if (constraint.EnableLimits != 0 && constraint.AngleLimitMass > 0f)
            {
                float angle = bodyB.Angle - bodyA.Angle - constraint.ReferenceAngle;
                angle -= 2f * XMath.PI * XMath.Floor((angle + XMath.PI) / (2f * XMath.PI));

                float angleError = 0f;
                if (angle < constraint.LowerAngle - AngularSlop)
                    angleError = angle - constraint.LowerAngle;
                else if (angle > constraint.UpperAngle + AngularSlop)
                    angleError = angle - constraint.UpperAngle;

                if (XMath.Abs(angleError) > Epsilon)
                {
                    float angleImpulse = -constraint.AngleLimitMass * angleError;
                    bodyA.Angle -= bodyA.InvInertia * angleImpulse;
                    bodyB.Angle += bodyB.InvInertia * angleImpulse;
                }
            }

            pv.Bodies[constraint.BodyAIndex] = bodyA;
            pv.Bodies[constraint.BodyBIndex] = bodyB;
        }

        // === 11. Store contacts to cache ===
        for (int c = 0; c < cfg.MaxContactsPerWorld; c++)
        {
            int cIdx = contactBase + c;
            if (c >= contactCount)
            {
                pv.ContactCache[cIdx] = new GPUCachedContactImpulse { IsValid = 0 };
                continue;
            }
            var contact = pv.Contacts[cIdx];
            if (contact.IsValid == 0)
            {
                pv.ContactCache[cIdx] = new GPUCachedContactImpulse { IsValid = 0 };
                continue;
            }
            pv.ContactCache[cIdx] = new GPUCachedContactImpulse
            {
                RigidBodyIndex = contact.RigidBodyIndex,
                ColliderType = contact.ColliderType,
                ColliderIndex = contact.ColliderIndex,
                GeomIndex = contact.GeomIndex,
                NormalImpulse = contact.NormalImpulse,
                TangentImpulse = contact.TangentImpulse,
                IsValid = 1
            };
        }

        // === 12. Damp velocities ===
        // NOTE: No velocity stabilization for rigid bodies (impulse solver handles velocities)
        if (cfg.GlobalDamping > 0f || cfg.AngularDamping > 0f)
        {
            float linearFactor = 1f - cfg.GlobalDamping * dt;
            float angularFactor = 1f - cfg.AngularDamping * dt;
            if (linearFactor < 0f) linearFactor = 0f;
            if (angularFactor < 0f) angularFactor = 0f;

            for (int i = 0; i < cfg.BodiesPerWorld; i++)
            {
                int idx = bodyBase + i;
                var body = pv.Bodies[idx];
                if (body.InvMass <= 0f) continue;
                body.VelX *= linearFactor;
                body.VelY *= linearFactor;
                body.AngularVel *= angularFactor;
                pv.Bodies[idx] = body;
            }
        }
    }

    #region SDF Helpers

    private static void CircleVsOBBSDF(
        float px, float py, float radius,
        GPUOBBCollider obb,
        out float phi, out float nx, out float ny)
    {
        float dx = px - obb.CX;
        float dy = py - obb.CY;

        float localX = dx * obb.UX + dy * obb.UY;
        float localY = -dx * obb.UY + dy * obb.UX;

        float clampedX = XMath.Clamp(localX, -obb.HalfExtentX, obb.HalfExtentX);
        float clampedY = XMath.Clamp(localY, -obb.HalfExtentY, obb.HalfExtentY);

        float diffX = localX - clampedX;
        float diffY = localY - clampedY;
        float dist = XMath.Sqrt(diffX * diffX + diffY * diffY);

        if (dist < Epsilon)
        {
            float distToRight = obb.HalfExtentX - localX;
            float distToLeft = localX + obb.HalfExtentX;
            float distToTop = obb.HalfExtentY - localY;
            float distToBottom = localY + obb.HalfExtentY;

            float minDist = distToRight;
            float localNx = 1f;
            float localNy = 0f;

            if (distToLeft < minDist)
            {
                minDist = distToLeft;
                localNx = -1f;
                localNy = 0f;
            }
            if (distToTop < minDist)
            {
                minDist = distToTop;
                localNx = 0f;
                localNy = 1f;
            }
            if (distToBottom < minDist)
            {
                minDist = distToBottom;
                localNx = 0f;
                localNy = -1f;
            }

            phi = -minDist - radius;
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
        else
        {
            phi = dist - radius;
            float localNx = diffX / dist;
            float localNy = diffY / dist;
            nx = localNx * obb.UX - localNy * obb.UY;
            ny = localNx * obb.UY + localNy * obb.UX;
        }
    }

    #endregion
}
