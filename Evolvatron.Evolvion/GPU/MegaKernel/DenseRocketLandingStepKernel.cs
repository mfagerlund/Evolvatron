using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Fused step kernel for rocket landing using dense NN forward pass.
/// Physics identical to RocketLandingStepKernel. Only the NN call differs:
/// uses DenseNN.ForwardPass instead of InlineNN.ForwardPassOneWorld.
/// One thread per world, one kernel launch per step.
/// </summary>
public static class DenseRocketLandingStepKernel
{
    private const float Pi = 3.14159265358979f;

    public static void StepKernel(
        Index1D worldIdx,
        PhysicsViews physics,
        DenseNNViews nn,
        EpisodeViews episode,
        MegaKernelConfig config,
        DenseRocketNNConfig nnConfig,
        ZoneViews zones,
        ArrayView<float> observations,
        ArrayView<float> actions)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int bodiesPerWorld = config.BodiesPerWorld;
        int bodyBase = worldIdx * bodiesPerWorld;

        // === 1. Get observations (8D base + optional sensors) ===
        int obsBase = worldIdx * config.InputSize;

        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            comX += b.X * mass;
            comY += b.Y * mass;
            velX += b.VelX * mass;
            velY += b.VelY * mass;
            totalMass += mass;
        }
        if (totalMass > 0f)
        {
            float inv = 1f / totalMass;
            comX *= inv; comY *= inv;
            velX *= inv; velY *= inv;
        }

        var body0 = physics.Bodies[bodyBase];
        float upX = XMath.Cos(body0.Angle);
        float upY = XMath.Sin(body0.Angle);

        observations[obsBase + 0] = (comX - config.PadX) / 20f;
        observations[obsBase + 1] = (comY - config.PadY) / 20f;
        observations[obsBase + 2] = velX / 10f;
        observations[obsBase + 3] = velY / 10f;
        observations[obsBase + 4] = upX;
        observations[obsBase + 5] = upY;
        observations[obsBase + 6] = episode.CurrentGimbal[worldIdx];
        observations[obsBase + 7] = episode.CurrentThrottle[worldIdx];

        // === 1b. Distance sensors (body-frame raycasts) ===
        if (config.SensorCount >= 4)
        {
            float maxRange = config.MaxSensorRange;
            float d0 = maxRange, d1 = maxRange, d2 = maxRange, d3 = maxRange;

            for (int ci = 0; ci < config.SharedColliderCount; ci++)
            {
                var obb = physics.SharedOBBColliders[ci];
                float t;

                t = RayVsOBB(comX, comY, upX, upY, maxRange, obb);
                if (t < d0) d0 = t;

                t = RayVsOBB(comX, comY, -upX, -upY, maxRange, obb);
                if (t < d1) d1 = t;

                t = RayVsOBB(comX, comY, -upY, upX, maxRange, obb);
                if (t < d2) d2 = t;

                t = RayVsOBB(comX, comY, upY, -upX, maxRange, obb);
                if (t < d3) d3 = t;
            }

            observations[obsBase + 8] = d0 / maxRange;
            observations[obsBase + 9] = d1 / maxRange;
            observations[obsBase + 10] = d2 / maxRange;
            observations[obsBase + 11] = d3 / maxRange;
        }

        // === 2. Dense NN forward pass ===
        DenseNN.ForwardPass(
            nn.Weights, nn.Biases, nn.LayerSizes,
            observations, actions, worldIdx,
            nnConfig.NumLayers, nnConfig.TotalWeightsPerNet, nnConfig.TotalBiasesPerNet,
            config.InputSize, config.OutputSize);

        // === 3. Apply actions ===
        int actionBase = worldIdx * config.OutputSize;
        float throttle = actions[actionBase + 0];
        float gimbal = actions[actionBase + 1];

        throttle = XMath.Max(0f, XMath.Min(1f, throttle));
        gimbal = XMath.Max(-1f, XMath.Min(1f, gimbal));

        float prevThrottle = episode.CurrentThrottle[worldIdx];
        float prevGimbal = episode.CurrentGimbal[worldIdx];
        episode.CurrentThrottle[worldIdx] = throttle;
        episode.CurrentGimbal[worldIdx] = gimbal;

        var body = physics.Bodies[bodyBase];
        float cos = XMath.Cos(body.Angle);
        float sin = XMath.Sin(body.Angle);
        float thrust = throttle * config.MaxThrust;
        float dt = config.Dt;

        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;
        body.AngularVel += gimbal * config.MaxGimbalTorque * body.InvInertia * dt;
        physics.Bodies[bodyBase] = body;

        // === 4. Physics substep (reuses InlinePhysics with MegaKernelConfig) ===
        InlinePhysics.SubStepOneWorld(physics, worldIdx, config);

        // === 5. Post-physics state ===
        int steps = episode.StepCounters[worldIdx] + 1;
        episode.StepCounters[worldIdx] = steps;

        float comX2 = 0f, comY2 = 0f, velX2 = 0f, velY2 = 0f, totalMass2 = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            comX2 += b.X * mass;
            comY2 += b.Y * mass;
            velX2 += b.VelX * mass;
            velY2 += b.VelY * mass;
            totalMass2 += mass;
        }
        if (totalMass2 > 0f)
        {
            float inv = 1f / totalMass2;
            comX2 *= inv; comY2 *= inv;
            velX2 *= inv; velY2 *= inv;
        }

        var body0post = physics.Bodies[bodyBase];
        float upX2 = XMath.Cos(body0post.Angle);
        float upY2 = XMath.Sin(body0post.Angle);
        float angleErr = XMath.Abs(XMath.Atan2(upX2, upY2));

        float dThrottle = throttle - prevThrottle;
        float dGimbal = gimbal - prevGimbal;
        float stepWaggle = dThrottle * dThrottle + dGimbal * dGimbal;
        episode.WaggleAccum[worldIdx] += stepWaggle;
        float waggle = episode.WaggleAccum[worldIdx];

        float errX = comX2 - config.PadX;
        float errY = comY2 - config.PadY;

        float angVel = body0post.AngularVel;

        // === 6. Zone evaluation (after physics, before terminal checks) ===
        if (config.CheckpointCount > 0 || config.DangerZoneCount > 0 ||
            config.SpeedZoneCount > 0 || config.AttractorCount > 0)
        {
            float speed2 = XMath.Sqrt(velX2 * velX2 + velY2 * velY2);
            float zoneReward = EvaluateZones(
                worldIdx, comX2, comY2, speed2, config, zones);
            zones.ZoneRewardAccum[worldIdx] += zoneReward;
        }

        float zoneAccum = zones.ZoneRewardAccum[worldIdx];

        // === 7. Terminal checks (settling-based) ===

        float speed = XMath.Sqrt(velX2 * velX2 + velY2 * velY2);
        bool nearPad = XMath.Abs(errX) < config.PadHalfWidth && XMath.Abs(errY) < 2f;

        // Check for any physics contacts this step
        int contactBase2 = worldIdx * config.MaxContactsPerWorld;
        int nContacts = physics.ContactCounts[worldIdx];
        bool hasContact = false;
        bool hitObstacle = false;
        for (int c = 0; c < nContacts; c++)
        {
            var contact = physics.Contacts[contactBase2 + c];
            if (contact.IsValid != 0)
            {
                hasContact = true;
                if (contact.ColliderIndex >= config.FirstObstacleIndex)
                    hitObstacle = true;
            }
        }

        // Obstacle collision — always fatal (regardless of speed)
        if (hitObstacle && config.ObstacleDeathEnabled != 0)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                zoneAccum, config);
            return;
        }

        // High-speed impact — crash on hard contact (ground, pad, or obstacle)
        if (hasContact && speed > config.MaxLandingVel)
        {
            episode.IsTerminal[worldIdx] = 1;
            float padCrashBonus = 0f;
            if (nearPad)
            {
                float speedRatio = config.MaxLandingVel / XMath.Max(speed, 0.01f);
                if (speedRatio > 1f) speedRatio = 1f;
                padCrashBonus = config.LandingBonus * 0.5f * speedRatio;
            }
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                zoneAccum, config) + padCrashBonus;
            return;
        }

        // Low-speed contact — keep simulating (rocket settles physically)

        // Mid-air extreme tumbling (only when airborne — let grounded rockets tip)
        if (!hasContact && angleErr > Pi * 0.75f)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                zoneAccum, config);
            return;
        }

        // Out of bounds
        float dist = XMath.Sqrt(errX * errX + errY * errY);
        if (dist > 50f || comY2 < config.GroundY - 10f || comY2 > config.SpawnHeight + 30f)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                zoneAccum, config);
            return;
        }

        // Settling detection — track consecutive frames at rest
        bool atRest = speed < config.SettleSpeedThreshold
                   && XMath.Abs(angVel) < config.SettleAngVelThreshold;

        if (atRest && hasContact)
        {
            int settled = episode.SettledSteps[worldIdx] + 1;
            episode.SettledSteps[worldIdx] = settled;

            if (settled >= config.SettleStepsRequired)
            {
                episode.IsTerminal[worldIdx] = 1;

                if (nearPad && angleErr < config.SettleTipAngle)
                {
                    // Successful landing — settled upright on pad
                    episode.HasLanded[worldIdx] = 1;
                    episode.FitnessValues[worldIdx] = ComputeFitness(
                        steps, comX2, comY2, velX2, velY2, angleErr, angVel, 1, waggle,
                        zoneAccum, config);
                }
                else
                {
                    // Settled but tipped over or off-pad — crash
                    float padCrashBonus = 0f;
                    if (nearPad)
                    {
                        padCrashBonus = config.LandingBonus * 0.3f;
                    }
                    episode.FitnessValues[worldIdx] = ComputeFitness(
                        steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                        zoneAccum, config) + padCrashBonus;
                }
                return;
            }
        }
        else
        {
            episode.SettledSteps[worldIdx] = 0;
        }

        // Max steps
        if (steps >= config.MaxSteps)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, comX2, comY2, velX2, velY2, angleErr, angVel, 0, waggle,
                zoneAccum, config);
        }
    }

    private static float EvaluateZones(
        int worldIdx, float comX, float comY, float speed,
        MegaKernelConfig config, ZoneViews zones)
    {
        float reward = 0f;

        // --- Checkpoints (circles, sequential, one-time bonus + proximity shaping) ---
        int cpProgress = zones.CheckpointProgress[worldIdx];
        for (int ci = 0; ci < config.CheckpointCount; ci++)
        {
            var cp = zones.Checkpoints[ci];
            if (cp.Order != cpProgress) continue;

            float dx = comX - cp.X;
            float dy = comY - cp.Y;
            float dist = XMath.Sqrt(dx * dx + dy * dy);

            if (dist < cp.Radius)
            {
                reward += cp.RewardBonus;
                cpProgress++;
                zones.CheckpointProgress[worldIdx] = cpProgress;
            }
            else if (dist < cp.Radius + cp.InfluenceRadius)
            {
                float closeness = 1f - (dist - cp.Radius) / cp.InfluenceRadius;
                reward += closeness * cp.RewardBonus * 0.01f;
            }
            break;
        }

        // --- Danger zones (AABB, per-step penalty + proximity warning) ---
        for (int di = 0; di < config.DangerZoneCount; di++)
        {
            var dz = zones.DangerZones[di];
            float dx = XMath.Abs(comX - dz.X) - dz.HalfExtentX;
            float dy = XMath.Abs(comY - dz.Y) - dz.HalfExtentY;

            if (dx < 0f && dy < 0f)
            {
                reward -= dz.PenaltyPerStep;
                if (dz.IsLethal != 0)
                    reward -= 1000f;
            }
            else
            {
                float edgeDist = XMath.Max(dx, dy);
                if (edgeDist > 0f && edgeDist < dz.InfluenceRadius)
                {
                    float closeness = 1f - edgeDist / dz.InfluenceRadius;
                    reward -= closeness * dz.PenaltyPerStep * 0.1f;
                }
            }
        }

        // --- Speed zones (AABB, reward per step if speed below max) ---
        for (int si = 0; si < config.SpeedZoneCount; si++)
        {
            var sz = zones.SpeedZones[si];
            float dx = XMath.Abs(comX - sz.X) - sz.HalfExtentX;
            float dy = XMath.Abs(comY - sz.Y) - sz.HalfExtentY;

            if (dx < 0f && dy < 0f && speed <= sz.MaxSpeed)
                reward += sz.RewardPerStep;
        }

        // --- Attractors (AABB + influence, one-time contact bonus + proximity shaping) ---
        int attractorMask = zones.AttractorContacted[worldIdx];
        for (int ai = 0; ai < config.AttractorCount; ai++)
        {
            var att = zones.Attractors[ai];
            float dx = XMath.Abs(comX - att.X) - att.HalfExtentX;
            float dy = XMath.Abs(comY - att.Y) - att.HalfExtentY;

            if (dx < 0f && dy < 0f)
            {
                if ((attractorMask & (1 << ai)) == 0)
                {
                    reward += att.ContactBonus;
                    attractorMask |= (1 << ai);
                }
            }

            float edgeDist = XMath.Max(XMath.Max(dx, dy), 0f);
            if (edgeDist < att.InfluenceRadius)
            {
                float closeness = 1f - edgeDist / att.InfluenceRadius;
                reward += closeness * att.Magnitude * 0.001f;
            }
        }
        zones.AttractorContacted[worldIdx] = attractorMask;

        return reward;
    }

    private static float ComputeFitness(
        int steps, float comX, float comY,
        float velX, float velY, float angleErr, float angVel,
        byte hasLanded, float waggle, float zoneReward,
        MegaKernelConfig config)
    {
        float errX = comX - config.PadX;
        float errY = comY - config.PadY;
        float distToPad = XMath.Sqrt(errX * errX + errY * errY);
        float speed = XMath.Sqrt(velX * velX + velY * velY);
        float survivalFrac = (float)steps / config.MaxSteps;

        float fitness = config.RewardSurvivalWeight * survivalFrac;

        float closeBonus = 1f - distToPad / 30f;
        if (closeBonus < 0f) closeBonus = 0f;
        fitness += config.RewardPositionWeight * closeBonus;

        float sp = speed / 20f;
        if (sp > 1f) sp = 1f;
        fitness -= config.RewardVelocityWeight * sp;

        float ap = angleErr / Pi;
        if (ap > 1f) ap = 1f;
        fitness -= config.RewardAngleWeight * ap;

        float avp = XMath.Abs(angVel) / 10f;
        if (avp > 1f) avp = 1f;
        fitness -= config.RewardAngVelWeight * avp;

        if (hasLanded != 0)
        {
            fitness += config.LandingBonus;
            fitness += (config.MaxSteps - steps) * config.HasteBonus;
        }

        fitness -= waggle * config.WagglePenalty;

        fitness += zoneReward;

        return fitness;
    }

    private static float RayVsOBB(
        float ox, float oy, float dx, float dy, float maxRange,
        GPUOBBCollider obb)
    {
        float relX = ox - obb.CX;
        float relY = oy - obb.CY;

        float localOX = relX * obb.UX + relY * obb.UY;
        float localOY = -relX * obb.UY + relY * obb.UX;
        float localDX = dx * obb.UX + dy * obb.UY;
        float localDY = -dx * obb.UY + dy * obb.UX;

        float tMin = 0f;
        float tMax = maxRange;

        if (XMath.Abs(localDX) < 1e-8f)
        {
            if (localOX < -obb.HalfExtentX || localOX > obb.HalfExtentX)
                return maxRange;
        }
        else
        {
            float invD = 1f / localDX;
            float t1 = (-obb.HalfExtentX - localOX) * invD;
            float t2 = (obb.HalfExtentX - localOX) * invD;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = XMath.Max(tMin, t1);
            tMax = XMath.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        if (XMath.Abs(localDY) < 1e-8f)
        {
            if (localOY < -obb.HalfExtentY || localOY > obb.HalfExtentY)
                return maxRange;
        }
        else
        {
            float invD = 1f / localDY;
            float t1 = (-obb.HalfExtentY - localOY) * invD;
            float t2 = (obb.HalfExtentY - localOY) * invD;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = XMath.Max(tMin, t1);
            tMax = XMath.Min(tMax, t2);
            if (tMin > tMax) return maxRange;
        }

        return tMin > 0f ? tMin : maxRange;
    }
}
