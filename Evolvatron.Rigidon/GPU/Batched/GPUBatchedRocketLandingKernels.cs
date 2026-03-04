using ILGPU;
using ILGPU.Algorithms;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// ILGPU kernels for batched rocket landing environment.
/// Handles observations, actions, terminal conditions, and fitness
/// for 1000+ parallel rocket landing simulations.
///
/// Rocket structure: 3 rigid bodies (body[0] + 2 legs), body[0] angle PI/2 = upright.
/// Observations (8D): [relPosX, relPosY, velX, velY, upX, upY, gimbal, throttle]
/// Actions (2D): [throttle 0..1, gimbal -1..1]
/// </summary>
public static class GPUBatchedRocketLandingKernels
{
    private const float Pi = 3.14159265358979f;

    /// <summary>
    /// Extract 8D landing observations for neural network input.
    /// Matches RocketEnvironment.GetObservations() exactly.
    /// One thread per world.
    /// </summary>
    public static void GetLandingObservationsKernel(
        Index1D worldIdx,
        ArrayView<float> observations,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> currentThrottle,
        ArrayView<float> currentGimbal,
        ArrayView<byte> isTerminal,
        int bodiesPerWorld,
        float padX, float padY)
    {
        int obsBase = worldIdx * 8;

        if (isTerminal[worldIdx] != 0)
        {
            for (int i = 0; i < 8; i++)
                observations[obsBase + i] = 0f;
            return;
        }

        int bodyBase = worldIdx * bodiesPerWorld;

        // Compute COM and velocity (mass-weighted across all rocket bodies)
        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = bodies[bodyBase + i];
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

        // Up vector from body[0] (local +X axis in world space)
        var body = bodies[bodyBase];
        float upX = XMath.Cos(body.Angle);
        float upY = XMath.Sin(body.Angle);

        observations[obsBase + 0] = (comX - padX) / 20f;
        observations[obsBase + 1] = (comY - padY) / 20f;
        observations[obsBase + 2] = velX / 10f;
        observations[obsBase + 3] = velY / 10f;
        observations[obsBase + 4] = upX;
        observations[obsBase + 5] = upY;
        observations[obsBase + 6] = currentGimbal[worldIdx];
        observations[obsBase + 7] = currentThrottle[worldIdx];
    }

    /// <summary>
    /// Apply throttle + gimbal actions to the rocket body.
    /// Matches RigidBodyRocketTemplate.ApplyThrust/ApplyGimbal.
    /// One thread per world.
    /// </summary>
    public static void ApplyRocketActionsKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<float> actions,
        ArrayView<float> currentThrottle,
        ArrayView<float> currentGimbal,
        ArrayView<byte> isTerminal,
        int bodiesPerWorld,
        float maxThrust, float maxGimbalTorque, float dt)
    {
        if (isTerminal[worldIdx] != 0) return;

        int actionBase = worldIdx * 2;
        float throttle = actions[actionBase + 0];
        float gimbal = actions[actionBase + 1];

        // Clamp (Tanh output maps [-1,1], throttle needs [0,1])
        throttle = XMath.Max(0f, XMath.Min(1f, throttle));
        gimbal = XMath.Max(-1f, XMath.Min(1f, gimbal));

        currentThrottle[worldIdx] = throttle;
        currentGimbal[worldIdx] = gimbal;

        // Apply thrust along body's local +X direction (up when angle=PI/2)
        int bodyGlobalIdx = worldIdx * bodiesPerWorld;
        var body = bodies[bodyGlobalIdx];

        float cos = XMath.Cos(body.Angle);
        float sin = XMath.Sin(body.Angle);
        float thrust = throttle * maxThrust;

        body.VelX += cos * thrust * body.InvMass * dt;
        body.VelY += sin * thrust * body.InvMass * dt;

        // Apply gimbal as angular impulse
        body.AngularVel += gimbal * maxGimbalTorque * body.InvInertia * dt;

        bodies[bodyGlobalIdx] = body;
    }

    /// <summary>
    /// Check terminal conditions: landed, crashed, out of bounds, max steps.
    /// Matches RocketEnvironment.ComputeReward terminal logic.
    /// One thread per world.
    /// </summary>
    public static void CheckLandingTerminalKernel(
        Index1D worldIdx,
        ArrayView<GPURigidBody> bodies,
        ArrayView<int> stepCounters,
        ArrayView<byte> isTerminal,
        ArrayView<byte> hasLanded,
        int bodiesPerWorld,
        int maxSteps,
        float padX, float padY, float padHalfWidth,
        float maxLandingVel, float maxLandingAngle,
        float groundY, float spawnHeight)
    {
        if (isTerminal[worldIdx] != 0) return;

        int steps = stepCounters[worldIdx] + 1;
        stepCounters[worldIdx] = steps;

        int bodyBase = worldIdx * bodiesPerWorld;

        // COM and velocity
        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = bodies[bodyBase + i];
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

        var body = bodies[bodyBase];
        float upX = XMath.Cos(body.Angle);
        float upY = XMath.Sin(body.Angle);
        float angleErr = XMath.Abs(XMath.Atan2(upX, upY));

        float errX = comX - padX;
        float errY = comY - padY;

        // 1. Landing success
        bool nearPad = XMath.Abs(errX) < padHalfWidth && XMath.Abs(errY) < 2f;
        bool lowVel = XMath.Abs(velX) < maxLandingVel && XMath.Abs(velY) < maxLandingVel;
        bool upright = angleErr < maxLandingAngle;

        if (nearPad && lowVel && upright)
        {
            isTerminal[worldIdx] = 1;
            hasLanded[worldIdx] = 1;
            return;
        }

        // 2. Crash (high velocity or flipped)
        if (XMath.Abs(velY) > 15f || XMath.Abs(velX) > 10f || angleErr > Pi * 0.4f)
        {
            isTerminal[worldIdx] = 1;
            return;
        }

        // 3. Out of bounds
        float dist = XMath.Sqrt(errX * errX + errY * errY);
        if (dist > 50f || comY < groundY - 10f || comY > spawnHeight + 30f)
        {
            isTerminal[worldIdx] = 1;
            return;
        }

        // 4. Max steps
        if (steps >= maxSteps)
            isTerminal[worldIdx] = 1;
    }

    /// <summary>
    /// Compute terminal-state fitness. Matches RocketEnvironment.GetFinalFitness().
    /// One thread per world.
    /// </summary>
    public static void ComputeLandingFitnessKernel(
        Index1D worldIdx,
        ArrayView<float> fitnessValues,
        ArrayView<GPURigidBody> bodies,
        ArrayView<int> stepCounters,
        ArrayView<byte> hasLanded,
        int bodiesPerWorld,
        int maxSteps,
        float padX, float padY,
        float landingBonus)
    {
        int bodyBase = worldIdx * bodiesPerWorld;

        float comX = 0f, comY = 0f, velX = 0f, velY = 0f, totalMass = 0f;
        for (int i = 0; i < bodiesPerWorld; i++)
        {
            var b = bodies[bodyBase + i];
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

        var body = bodies[bodyBase];
        float upX = XMath.Cos(body.Angle);
        float upY = XMath.Sin(body.Angle);
        float angleErr = XMath.Abs(XMath.Atan2(upX, upY));

        float errX = comX - padX;
        float errY = comY - padY;
        float distToPad = XMath.Sqrt(errX * errX + errY * errY);
        float speed = XMath.Sqrt(velX * velX + velY * velY);
        float survivalFrac = (float)stepCounters[worldIdx] / maxSteps;

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

        if (hasLanded[worldIdx] != 0)
            fitness += landingBonus;

        fitnessValues[worldIdx] = fitness;
    }

    /// <summary>
    /// Reset per-world state for a new episode. One thread per world.
    /// </summary>
    public static void ResetLandingStateKernel(
        Index1D worldIdx,
        ArrayView<int> stepCounters,
        ArrayView<byte> isTerminal,
        ArrayView<byte> hasLanded,
        ArrayView<float> currentThrottle,
        ArrayView<float> currentGimbal,
        ArrayView<float> fitnessValues)
    {
        stepCounters[worldIdx] = 0;
        isTerminal[worldIdx] = 0;
        hasLanded[worldIdx] = 0;
        currentThrottle[worldIdx] = 0f;
        currentGimbal[worldIdx] = 0f;
        fitnessValues[worldIdx] = 0f;
    }
}
