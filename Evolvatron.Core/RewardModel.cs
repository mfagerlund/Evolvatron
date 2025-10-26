using System;

namespace Evolvatron.Core;

/// <summary>
/// Parameters for reward shaping in the landing task.
/// </summary>
public struct RewardParams
{
    /// <summary>Landing pad center X coordinate.</summary>
    public float PadX;

    /// <summary>Landing pad center Y coordinate.</summary>
    public float PadY;

    /// <summary>Landing pad half-width.</summary>
    public float PadHalfWidth;

    /// <summary>Landing pad half-height.</summary>
    public float PadHalfHeight;

    /// <summary>Weight for position error penalty.</summary>
    public float K_Position;

    /// <summary>Weight for horizontal velocity penalty.</summary>
    public float K_VelocityX;

    /// <summary>Weight for vertical velocity penalty (asymmetric - upward is worse).</summary>
    public float K_VelocityY;

    /// <summary>Weight for angle error penalty.</summary>
    public float K_Angle;

    /// <summary>Weight for control effort penalty (throttle + gimbal changes).</summary>
    public float K_Control;

    /// <summary>Per-step reward for staying alive.</summary>
    public float K_Alive;

    /// <summary>Terminal bonus for successful landing.</summary>
    public float R_Land;

    /// <summary>Terminal penalty for crash/out-of-bounds.</summary>
    public float R_Crash;

    /// <summary>Maximum velocity for successful landing (m/s).</summary>
    public float MaxLandingVelocity;

    /// <summary>Maximum angle error for successful landing (radians).</summary>
    public float MaxLandingAngle;

    /// <summary>Maximum distance from pad for successful landing (meters).</summary>
    public float MaxLandingDistance;

    /// <summary>Creates default reward parameters.</summary>
    public static RewardParams Default()
    {
        return new RewardParams
        {
            PadX = 0f,
            PadY = -4f,
            PadHalfWidth = 2f,
            PadHalfHeight = 0.5f,
            K_Position = 0.1f,
            K_VelocityX = 0.5f,
            K_VelocityY = 0.3f,
            K_Angle = 1.0f,
            K_Control = 0.01f,
            K_Alive = 0.01f,
            R_Land = 100f,
            R_Crash = -100f,
            MaxLandingVelocity = 2f,
            MaxLandingAngle = 15f * MathF.PI / 180f,
            MaxLandingDistance = 3f
        };
    }
}

/// <summary>
/// Observation vector for RL agents.
/// </summary>
public struct RocketObservation
{
    /// <summary>COM position relative to pad (X, Y).</summary>
    public float RelPosX, RelPosY;

    /// <summary>COM velocity (X, Y).</summary>
    public float VelX, VelY;

    /// <summary>Upright vector components (normalized).</summary>
    public float UpX, UpY;

    /// <summary>Current gimbal command.</summary>
    public float Gimbal;

    /// <summary>Current throttle command.</summary>
    public float Throttle;

    /// <summary>Converts to array for ML frameworks.</summary>
    public float[] ToArray()
    {
        return new[] { RelPosX, RelPosY, VelX, VelY, UpX, UpY, Gimbal, Throttle };
    }
}

/// <summary>
/// Reward model for the rocket landing task.
/// Computes step rewards and terminal states for RL training.
/// </summary>
public static class RewardModel
{
    /// <summary>
    /// Computes the observation vector for a rocket.
    /// </summary>
    public static RocketObservation GetObservation(
        WorldState world,
        int[] rocketIndices,
        in RewardParams rparams,
        float gimbal,
        float throttle)
    {
        // Get rocket state
        Templates.RocketTemplate.GetCenterOfMass(world, rocketIndices, out float comX, out float comY);
        Templates.RocketTemplate.GetVelocity(world, rocketIndices, out float velX, out float velY);
        Templates.RocketTemplate.GetUpVector(world, rocketIndices, out float upX, out float upY);

        return new RocketObservation
        {
            RelPosX = comX - rparams.PadX,
            RelPosY = comY - rparams.PadY,
            VelX = velX,
            VelY = velY,
            UpX = upX,
            UpY = upY,
            Gimbal = gimbal,
            Throttle = throttle
        };
    }

    /// <summary>
    /// Computes step reward and checks for terminal conditions.
    /// </summary>
    /// <param name="world">World state</param>
    /// <param name="rocketIndices">Rocket particle indices</param>
    /// <param name="rparams">Reward parameters</param>
    /// <param name="prevThrottle">Previous throttle command</param>
    /// <param name="prevGimbal">Previous gimbal command</param>
    /// <param name="throttle">Current throttle command</param>
    /// <param name="gimbal">Current gimbal command</param>
    /// <param name="terminal">Output: is episode terminal?</param>
    /// <param name="terminalReward">Output: terminal reward (if any)</param>
    /// <returns>Step reward</returns>
    public static float StepReward(
        WorldState world,
        int[] rocketIndices,
        in RewardParams rparams,
        float prevThrottle,
        float prevGimbal,
        float throttle,
        float gimbal,
        out bool terminal,
        out float terminalReward)
    {
        terminal = false;
        terminalReward = 0f;

        // Get rocket state
        Templates.RocketTemplate.GetCenterOfMass(world, rocketIndices, out float comX, out float comY);
        Templates.RocketTemplate.GetVelocity(world, rocketIndices, out float velX, out float velY);
        Templates.RocketTemplate.GetUpVector(world, rocketIndices, out float upX, out float upY);

        // Position error relative to pad
        float errX = comX - rparams.PadX;
        float errY = comY - rparams.PadY;
        float positionError = MathF.Sqrt(errX * errX + errY * errY);

        // Angle error (want upright: ux=0, uy=1)
        float angleErr = MathF.Abs(MathF.Atan2(upX, upY));

        // Control effort (penalize large changes)
        float dThrottle = throttle - prevThrottle;
        float dGimbal = gimbal - prevGimbal;
        float controlPenalty = dThrottle * dThrottle + dGimbal * dGimbal;

        // Step reward (negative penalties + alive bonus)
        float stepReward = -rparams.K_Position * positionError
                         - rparams.K_VelocityX * MathF.Abs(velX)
                         - rparams.K_VelocityY * MathF.Abs(velY)
                         - rparams.K_Angle * angleErr * angleErr
                         - rparams.K_Control * controlPenalty
                         + rparams.K_Alive;

        // Check terminal conditions
        // 1. Success: inside pad, low velocity, upright
        bool insidePad = MathF.Abs(errX) < rparams.PadHalfWidth &&
                         MathF.Abs(errY) < rparams.PadHalfHeight * 2f;
        bool lowVelocity = MathF.Abs(velX) < rparams.MaxLandingVelocity &&
                           MathF.Abs(velY) < rparams.MaxLandingVelocity;
        bool upright = angleErr < rparams.MaxLandingAngle;
        bool nearPad = positionError < rparams.MaxLandingDistance;

        if (insidePad && lowVelocity && upright && nearPad)
        {
            terminal = true;
            terminalReward = rparams.R_Land;
            return stepReward;
        }

        // 2. Crash: high impact velocity or extreme angle
        bool highImpact = MathF.Abs(velY) > 15f || MathF.Abs(velX) > 10f;
        bool flipped = angleErr > MathF.PI * 0.4f;

        if (highImpact || flipped)
        {
            terminal = true;
            terminalReward = rparams.R_Crash;
            return stepReward;
        }

        // 3. Out of bounds (too far from pad)
        if (positionError > 50f || comY < rparams.PadY - 20f || comY > rparams.PadY + 30f)
        {
            terminal = true;
            terminalReward = rparams.R_Crash;
            return stepReward;
        }

        return stepReward;
    }

    /// <summary>
    /// Checks if a rocket has successfully landed.
    /// </summary>
    public static bool IsLanded(
        WorldState world,
        int[] rocketIndices,
        in RewardParams rparams)
    {
        Templates.RocketTemplate.GetCenterOfMass(world, rocketIndices, out float comX, out float comY);
        Templates.RocketTemplate.GetVelocity(world, rocketIndices, out float velX, out float velY);
        Templates.RocketTemplate.GetUpVector(world, rocketIndices, out float upX, out float upY);

        float errX = comX - rparams.PadX;
        float errY = comY - rparams.PadY;
        float angleErr = MathF.Abs(MathF.Atan2(upX, upY));

        bool insidePad = MathF.Abs(errX) < rparams.PadHalfWidth &&
                         MathF.Abs(errY) < rparams.PadHalfHeight * 2f;
        bool lowVelocity = MathF.Abs(velX) < rparams.MaxLandingVelocity &&
                           MathF.Abs(velY) < rparams.MaxLandingVelocity;
        bool upright = angleErr < rparams.MaxLandingAngle;

        return insidePad && lowVelocity && upright;
    }
}
