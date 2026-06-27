using ILGPU;
using ILGPU.Algorithms;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Phase-1 maneuvering controller step kernel (see docs/phase1_controller_spec.md).
///
/// Same vehicle, actuators, and physics as DenseRocketLandingStepKernel — only the
/// observation, reward, and terminal logic differ:
///   - Observation: 9-D dynamics + world-frame velocity command (no pad, no sensors).
///   - Reward: track the commanded velocity (cmdVx, cmdVy) for the current schedule segment.
///   - Terminals: airborne tumble, NaN guard, MaxSteps. No ground/pad/obstacles.
///
/// One thread per world, one kernel launch per step.
/// </summary>
public static class DenseRocketControlStepKernel
{
    private const float HalfPi = 1.5707963267949f;

    public static void StepKernel(
        Index1D worldIdx,
        PhysicsViews physics,
        DenseNNViews nn,
        EpisodeViews episode,
        MegaKernelConfig config,
        DenseRocketControlConfig ctrl,
        ControlViews control,
        ArrayView<float> observations,
        ArrayView<float> actions)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int bodiesPerWorld = config.BodiesPerWorld;
        int bodyBase = worldIdx * bodiesPerWorld;
        int stepsSoFar = episode.StepCounters[worldIdx];

        // === Current command segment (piecewise-constant schedule) ===
        int seg = stepsSoFar / ctrl.SegmentLength;
        if (seg >= ctrl.SegmentsPerEpisode) seg = ctrl.SegmentsPerEpisode - 1;
        int cmdIdx = worldIdx * ctrl.SegmentsPerEpisode + seg;
        float cmdVx = control.CmdVx[cmdIdx];
        float cmdVy = control.CmdVy[cmdIdx];

        // === COM velocity (mass-weighted, world frame) ===
        float velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            velX += b.VelX * mass;
            velY += b.VelY * mass;
            totalMass += mass;
        }
        if (totalMass > 0f) { float inv = 1f / totalMass; velX *= inv; velY *= inv; }

        var body0 = physics.Bodies[bodyBase];
        float upX = XMath.Cos(body0.Angle);
        float upY = XMath.Sin(body0.Angle);
        float rightX = upY;    // right = (sinθ, -cosθ): body up rotated -90°
        float rightY = -upX;

        // Velocity error in body frame (orientation-equivariant)
        float errWX = cmdVx - velX;
        float errWY = cmdVy - velY;
        float errFwd = errWX * upX + errWY * upY;
        float errLat = errWX * rightX + errWY * rightY;
        float speed = XMath.Sqrt(velX * velX + velY * velY);
        float angVel = body0.AngularVel;

        float gmag = XMath.Sqrt(config.GravityX * config.GravityX + config.GravityY * config.GravityY);
        float gUp = gmag > 1e-6f ? -(config.GravityX * upX + config.GravityY * upY) / gmag : 0f;

        int obsBase = worldIdx * config.InputSize;
        observations[obsBase + 0] = upX;
        observations[obsBase + 1] = upY;
        observations[obsBase + 2] = angVel / 10f;
        observations[obsBase + 3] = errFwd / 10f;
        observations[obsBase + 4] = errLat / 10f;
        observations[obsBase + 5] = speed / 10f;
        observations[obsBase + 6] = episode.CurrentThrottle[worldIdx];
        observations[obsBase + 7] = episode.CurrentGimbal[worldIdx];
        observations[obsBase + 8] = gUp;

        // === Elman recurrence: previous context → trailing input slots ===
        if (ctrl.ContextSize > 0)
        {
            int ctxBase = worldIdx * ctrl.ContextSize;
            int ctxObsStart = obsBase + config.InputSize - ctrl.ContextSize;
            for (int c = 0; c < ctrl.ContextSize; c++)
                observations[ctxObsStart + c] = control.ContextBuffer[ctxBase + c];
        }

        // === Dense NN forward pass ===
        DenseNN.ForwardPass(
            nn.Weights, nn.Biases, nn.LayerSizes,
            observations, actions, worldIdx,
            ctrl.NumLayers, ctrl.TotalWeightsPerNet, ctrl.TotalBiasesPerNet,
            config.InputSize, config.OutputSize);

        int actionBase = worldIdx * config.OutputSize;
        float throttle = actions[actionBase + 0];
        float gimbal = actions[actionBase + 1];

        // NaN/Inf guard — defense-in-depth before NaN enters physics → memory corruption.
        if (float.IsNaN(throttle) || float.IsInfinity(throttle) ||
            float.IsNaN(gimbal) || float.IsInfinity(gimbal))
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = -1e6f;
            control.TrackRewardAccum[worldIdx] = -1e6f;
            return;
        }

        throttle = XMath.Max(0f, XMath.Min(1f, throttle));
        gimbal = XMath.Max(-1f, XMath.Min(1f, gimbal));

        float prevThrottle = episode.CurrentThrottle[worldIdx];
        float prevGimbal = episode.CurrentGimbal[worldIdx];
        episode.CurrentThrottle[worldIdx] = throttle;
        episode.CurrentGimbal[worldIdx] = gimbal;

        // === Elman recurrence: trailing outputs → context (bounded + NaN-guarded before feedback) ===
        if (ctrl.ContextSize > 0)
        {
            int ctxBase = worldIdx * ctrl.ContextSize;
            int ctxOutStart = actionBase + config.OutputSize - ctrl.ContextSize;
            for (int c = 0; c < ctrl.ContextSize; c++)
            {
                float cv = actions[ctxOutStart + c];
                if (float.IsNaN(cv) || float.IsInfinity(cv)) cv = 0f;
                control.ContextBuffer[ctxBase + c] = XMath.Max(-1f, XMath.Min(1f, cv));
            }
        }

        // === Apply actions (body-axis thrust + gimbal torque) — identical to landing ===
        var body = physics.Bodies[bodyBase];
        float cos = XMath.Cos(body.Angle);
        float sin = XMath.Sin(body.Angle);
        float thrust = throttle * config.MaxThrust;
        float dt = config.Dt;
        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;
        body.AngularVel += gimbal * config.MaxGimbalTorque * body.InvInertia * dt;
        physics.Bodies[bodyBase] = body;

        // === Physics substep (free space — SharedColliderCount = 0) ===
        InlinePhysics.SubStepOneWorld(physics, worldIdx, config);

        int steps = stepsSoFar + 1;
        episode.StepCounters[worldIdx] = steps;

        // === Post-physics COM velocity ===
        float velX2 = 0f, velY2 = 0f, totalMass2 = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = physics.Bodies[bodyBase + i];
            if (b.InvMass <= 0f) continue;
            float mass = 1f / b.InvMass;
            velX2 += b.VelX * mass;
            velY2 += b.VelY * mass;
            totalMass2 += mass;
        }
        if (totalMass2 > 0f) { float inv = 1f / totalMass2; velX2 *= inv; velY2 *= inv; }

        var body0post = physics.Bodies[bodyBase];
        float upX2 = XMath.Cos(body0post.Angle);
        float upY2 = XMath.Sin(body0post.Angle);
        float angleErr = XMath.Abs(XMath.Atan2(upX2, upY2)); // deviation from upright
        float angVel2 = body0post.AngularVel;

        // === Tracking reward (against the command tracked this step) ===
        float dvx = velX2 - cmdVx;
        float dvy = velY2 - cmdVy;
        float verr = XMath.Sqrt(dvx * dvx + dvy * dvy);
        float track = 1f - XMath.Min(verr / ctrl.VErrScale, 1f);

        float dThrottle = throttle - prevThrottle;
        float dGimbal = gimbal - prevGimbal;
        float effort = dThrottle * dThrottle + dGimbal * dGimbal;
        episode.WaggleAccum[worldIdx] += effort;

        float avp = XMath.Min(XMath.Abs(angVel2) / 10f, 1f);

        float stepReward = ctrl.RewardTrackWeight * track
                         - ctrl.RewardEffortWeight * effort
                         - ctrl.AngVelPenalty * avp;
        control.TrackRewardAccum[worldIdx] += stepReward;

        // === Terminals ===

        // Airborne tumble (everything is airborne in free space)
        if (angleErr > HalfPi)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = control.TrackRewardAccum[worldIdx] - ctrl.TumblePenalty;
            return;
        }

        // Max steps
        if (steps >= config.MaxSteps)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = control.TrackRewardAccum[worldIdx];
        }
    }
}
