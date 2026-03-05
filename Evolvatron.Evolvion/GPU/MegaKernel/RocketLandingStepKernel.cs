using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.MegaKernel;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Fused step kernel for rocket landing evaluation.
/// One thread per world, one kernel launch per step.
/// Combines: observations → NN forward pass → actions → physics substep → terminal check + inline fitness.
/// Replaces ~35 separate kernel dispatches with 1.
///
/// Terminal worlds are skipped entirely (early return), saving all compute for dead rockets.
/// Fitness is computed inline at the moment of termination, not post-mortem.
/// </summary>
public static class RocketLandingStepKernel
{
    private const float Pi = 3.14159265358979f;

    /// <summary>
    /// The fused kernel entry point. One thread per world, called once per simulation step.
    /// </summary>
    public static void StepKernel(
        Index1D worldIdx,
        PhysicsViews physics,
        NNViews nn,
        EpisodeViews episode,
        MegaKernelConfig config,
        ArrayView<float> observations,
        ArrayView<float> actions)
    {
        if (episode.IsTerminal[worldIdx] != 0) return;

        int bodiesPerWorld = config.BodiesPerWorld;
        int bodyBase = worldIdx * bodiesPerWorld;
        float dt = config.Dt;

        // === 1. Get observations (8D) ===
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

        // === 2. NN forward pass ===
        InlineNN.ForwardPassOneWorld(nn, observations, actions, worldIdx, config);

        // === 3. Apply actions ===
        int actionBase = worldIdx * config.OutputSize;
        float throttle = actions[actionBase + 0];
        float gimbal = actions[actionBase + 1];

        throttle = XMath.Max(0f, XMath.Min(1f, throttle));
        gimbal = XMath.Max(-1f, XMath.Min(1f, gimbal));

        episode.CurrentThrottle[worldIdx] = throttle;
        episode.CurrentGimbal[worldIdx] = gimbal;

        var body = physics.Bodies[bodyBase];
        float cos = XMath.Cos(body.Angle);
        float sin = XMath.Sin(body.Angle);
        float thrust = throttle * config.MaxThrust;

        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;
        body.AngularVel += gimbal * config.MaxGimbalTorque * body.InvInertia * dt;
        physics.Bodies[bodyBase] = body;

        // === 4. Physics substep ===
        InlinePhysics.SubStepOneWorld(physics, worldIdx, config);

        // === 5. Terminal check + inline fitness ===
        int steps = episode.StepCounters[worldIdx] + 1;
        episode.StepCounters[worldIdx] = steps;

        // Recompute COM/vel after physics
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

        float errX = comX2 - config.PadX;
        float errY = comY2 - config.PadY;

        // Landing success
        bool nearPad = XMath.Abs(errX) < config.PadHalfWidth && XMath.Abs(errY) < 2f;
        bool lowVel = XMath.Abs(velX2) < config.MaxLandingVel && XMath.Abs(velY2) < config.MaxLandingVel;
        bool upright = angleErr < config.MaxLandingAngle;

        if (nearPad && lowVel && upright)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.HasLanded[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, config.MaxSteps, comX2, comY2, config.PadX, config.PadY,
                velX2, velY2, angleErr, 1, config.LandingBonus);
            return;
        }

        // Crash
        if (XMath.Abs(velY2) > 15f || XMath.Abs(velX2) > 10f || angleErr > Pi * 0.4f)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, config.MaxSteps, comX2, comY2, config.PadX, config.PadY,
                velX2, velY2, angleErr, 0, config.LandingBonus);
            return;
        }

        // Out of bounds
        float dist = XMath.Sqrt(errX * errX + errY * errY);
        if (dist > 50f || comY2 < config.GroundY - 10f || comY2 > config.SpawnHeight + 30f)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, config.MaxSteps, comX2, comY2, config.PadX, config.PadY,
                velX2, velY2, angleErr, 0, config.LandingBonus);
            return;
        }

        // Max steps
        if (steps >= config.MaxSteps)
        {
            episode.IsTerminal[worldIdx] = 1;
            episode.FitnessValues[worldIdx] = ComputeFitness(
                steps, config.MaxSteps, comX2, comY2, config.PadX, config.PadY,
                velX2, velY2, angleErr, 0, config.LandingBonus);
        }
    }

    private static float ComputeFitness(
        int steps, int maxSteps,
        float comX, float comY, float padX, float padY,
        float velX, float velY, float angleErr,
        byte hasLanded, float landingBonus)
    {
        float errX = comX - padX;
        float errY = comY - padY;
        float distToPad = XMath.Sqrt(errX * errX + errY * errY);
        float speed = XMath.Sqrt(velX * velX + velY * velY);
        float survivalFrac = (float)steps / maxSteps;

        float fitness = 20f * survivalFrac;
        float closeBonus = 1f - distToPad / 30f;
        if (closeBonus < 0f) closeBonus = 0f;
        fitness += 20f * closeBonus;
        float sp = speed / 20f;
        if (sp > 1f) sp = 1f;
        fitness -= 5f * sp;
        float ap = angleErr / Pi;
        if (ap > 1f) ap = 1f;
        fitness -= 5f * ap;

        if (hasLanded != 0)
            fitness += landingBonus;

        return fitness;
    }
}
