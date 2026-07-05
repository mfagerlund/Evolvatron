using ILGPU;
using ILGPU.Algorithms;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Fused step kernel for the goal-relative POSE-REACHING controller. Physics is identical to
/// <see cref="DenseRocketLandingStepKernel"/> (same InlinePhysics substep) — only the observation
/// layout, reward, and success criterion differ.
///
/// Observation (10D): targetX-relative, targetY-relative, velX, velY, cos/sin(bodyAngle),
/// cos/sin(targetAngle), curGimbal, curThrottle. The target pose (PadX, PadY, TargetAngle) is a
/// shared per-spawn condition — randomized across spawns, exactly like multi-position training.
///
/// Reward = per-step proximity shaping (position, and — gated by proximity — alignment + slow),
/// plus a terminal bonus when the rocket HELD the target pose (within pos/angle/speed tolerance)
/// for PoseHoldSteps consecutive frames. "Hit" (reused HasLanded flag) drives the hit% metric.
///
/// Buffer reuse: WaggleAccum = accumulated shaping, SettledSteps = consecutive in-tolerance steps,
/// HasLanded = held-the-pose flag.
/// </summary>
public static class DenseRocketPoseStepKernel
{
    private const float Pi = 3.14159265358979f;

    public static void StepKernel(
        Index1D worldIdx,
        PhysicsViews physics,
        DenseNNViews nn,
        EpisodeViews episode,
        MegaKernelConfig config,
        DenseRocketNNConfig nnConfig,
        ArrayView<float> observations,
        ArrayView<float> actions)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int bodiesPerWorld = config.BodiesPerWorld;
        int bodyBase = worldIdx * bodiesPerWorld;

        // === 1. Observations (10D pose layout) ===
        int obsBase = worldIdx * config.InputSize;

        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            comX += b.X * mass; comY += b.Y * mass;
            velX += b.VelX * mass; velY += b.VelY * mass;
            totalMass += mass;
        }
        if (totalMass > 0f)
        {
            float inv = 1f / totalMass;
            comX *= inv; comY *= inv; velX *= inv; velY *= inv;
        }

        var body0 = physics.Bodies[bodyBase];
        float cosB = XMath.Cos(body0.Angle);
        float sinB = XMath.Sin(body0.Angle);

        observations[obsBase + 0] = (comX - config.PadX) / 20f;
        observations[obsBase + 1] = (comY - config.PadY) / 20f;
        observations[obsBase + 2] = velX / 10f;
        observations[obsBase + 3] = velY / 10f;
        observations[obsBase + 4] = cosB;
        observations[obsBase + 5] = sinB;
        observations[obsBase + 6] = XMath.Cos(config.TargetAngle);
        observations[obsBase + 7] = XMath.Sin(config.TargetAngle);
        observations[obsBase + 8] = episode.CurrentGimbal[worldIdx];
        observations[obsBase + 9] = episode.CurrentThrottle[worldIdx];

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

        if (float.IsNaN(throttle) || float.IsInfinity(throttle) ||
            float.IsNaN(gimbal) || float.IsInfinity(gimbal))
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = -1e6f;
            return;
        }

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

        // === 4. Physics substep ===
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
            comX2 += b.X * mass; comY2 += b.Y * mass;
            velX2 += b.VelX * mass; velY2 += b.VelY * mass;
            totalMass2 += mass;
        }
        if (totalMass2 > 0f)
        {
            float inv = 1f / totalMass2;
            comX2 *= inv; comY2 *= inv; velX2 *= inv; velY2 *= inv;
        }

        var body0post = physics.Bodies[bodyBase];
        float angDiff = body0post.Angle - config.TargetAngle;
        float angErr = XMath.Abs(XMath.Atan2(XMath.Sin(angDiff), XMath.Cos(angDiff)));

        float dxT = comX2 - config.PadX;
        float dyT = comY2 - config.PadY;
        float dist2 = dxT * dxT + dyT * dyT;
        float speed = XMath.Sqrt(velX2 * velX2 + velY2 * velY2);

        // === 6. Per-step proximity shaping (alignment + slow gated by proximity) ===
        float prox = 1f / (1f + dist2 * 0.25f);
        float align = 0.5f * (1f + XMath.Cos(angDiff));
        float slow = 1f - XMath.Min(1f, speed / 3f);
        float shape = config.PosePosWeight * prox
                    + config.PoseAngleWeight * prox * align
                    + config.PoseVelWeight * prox * slow;

        // small control-effort penalty (reuses WagglePenalty)
        float dThrottle = throttle - prevThrottle;
        float dGimbal = gimbal - prevGimbal;
        shape -= (dThrottle * dThrottle + dGimbal * dGimbal) * config.WagglePenalty;

        episode.WaggleAccum[worldIdx] += shape;

        // === 7. Hit detection (held the pose) ===
        bool inTol = dist2 < config.PoseHitRadius * config.PoseHitRadius
                  && angErr < config.PoseHitAngle
                  && speed < config.PoseHitSpeed;
        int hold = episode.SettledSteps[worldIdx];
        if (inTol)
        {
            hold++;
            episode.SettledSteps[worldIdx] = hold;
            if (hold >= config.PoseHoldSteps)
            {
                episode.HasLanded[worldIdx] = 1;
                episode.IsTerminal[worldIdx] = 1;
                episode.FitnessValues[worldIdx] = episode.WaggleAccum[worldIdx]
                    + config.PoseHitBonus + (config.MaxSteps - steps) * config.HasteBonus;
                return;
            }
        }
        else
        {
            episode.SettledSteps[worldIdx] = 0;
        }

        // === 8. Out of bounds ===
        if (dist2 > 60f * 60f || comY2 < config.GroundY - 10f || comY2 > config.SpawnHeight + 30f)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = episode.WaggleAccum[worldIdx];
            return;
        }

        // === 9. Max steps ===
        if (steps >= config.MaxSteps)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = episode.WaggleAccum[worldIdx];
        }
    }
}
