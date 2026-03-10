using System.Runtime.InteropServices;

namespace Evolvatron.Evolvion.GPU.MegaKernel;

/// <summary>
/// Blittable GPU structs for reward zones. Uploaded as shared buffers
/// (same pattern as GPUOBBCollider for obstacles).
/// </summary>

[StructLayout(LayoutKind.Sequential)]
public struct GPUCheckpoint
{
    public float X, Y, Radius;
    public int Order;
    public float RewardBonus;
    public float InfluenceRadius;
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUDangerZone
{
    public float X, Y, HalfExtentX, HalfExtentY;
    public float PenaltyPerStep;
    public int IsLethal;
    public float InfluenceRadius;
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUSpeedZone
{
    public float X, Y, HalfExtentX, HalfExtentY;
    public float MaxSpeed;
    public float RewardPerStep;
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUAttractor
{
    public float X, Y, HalfExtentX, HalfExtentY;
    public float Magnitude;
    public float InfluenceRadius;
    public float ContactBonus;
}
