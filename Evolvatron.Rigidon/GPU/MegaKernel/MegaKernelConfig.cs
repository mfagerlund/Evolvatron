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

    // Sensors
    public int SensorCount;
    public float MaxSensorRange;

    // Behaviour shaping
    public float WagglePenalty;     // per-step penalty weight for throttle/gimbal changes
    public int ObstacleDeathEnabled; // 1 = contact with any collider index > 0 is terminal
}
