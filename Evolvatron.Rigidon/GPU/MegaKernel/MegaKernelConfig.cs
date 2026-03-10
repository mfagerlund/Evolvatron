using System.Runtime.InteropServices;

namespace Evolvatron.Core.GPU.MegaKernel;

/// <summary>
/// All scalar configuration for the fused step kernel.
/// Blittable struct passed as a single kernel parameter.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct MegaKernelConfig
{
    // World layout
    public int BodiesPerWorld;
    public int GeomsPerWorld;
    public int JointsPerWorld;
    public int SharedColliderCount;
    public int MaxContactsPerWorld;

    // Physics
    public float Dt;
    public float GravityX;
    public float GravityY;
    public float FrictionMu;
    public float Restitution;
    public float GlobalDamping;
    public float AngularDamping;
    public int SolverIterations;

    // NN layout
    public int TotalNodes;
    public int TotalEdges;
    public int NumRows;
    public int InputSize;
    public int OutputSize;

    // Landing task
    public int MaxSteps;
    public float MaxThrust;
    public float MaxGimbalTorque;
    public float PadX;
    public float PadY;
    public float PadHalfWidth;
    public float MaxLandingVel;
    public float MaxLandingAngle;
    public float GroundY;
    public float SpawnHeight;
    public float LandingBonus;
    public float HasteBonus;      // (maxSteps - landingStep) * hasteBonus on successful landing

    // Sensors
    public int SensorCount;
    public float MaxSensorRange;

    // Behaviour shaping
    public float WagglePenalty;     // per-step penalty weight for throttle/gimbal changes
    public int ObstacleDeathEnabled; // 1 = contact with collider index >= FirstObstacleIndex is terminal
    public int FirstObstacleIndex;  // collider indices below this are safe (ground, pad)

    // Reward weights (parameterized fitness — editor-controlled)
    public float RewardSurvivalWeight;   // scales survival-fraction bonus
    public float RewardPositionWeight;   // scales closeness-to-pad bonus
    public float RewardVelocityWeight;   // scales speed penalty
    public float RewardAngleWeight;      // scales angle-error penalty
    public float RewardAngVelWeight;     // scales angular-velocity penalty

    // Reward zone counts (Phase 3)
    public int CheckpointCount;
    public int DangerZoneCount;
    public int SpeedZoneCount;
    public int AttractorCount;
}
