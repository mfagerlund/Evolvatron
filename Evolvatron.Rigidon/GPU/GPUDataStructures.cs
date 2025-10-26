using System.Runtime.InteropServices;

namespace Evolvatron.Core.GPU;

/// <summary>
/// GPU-compatible constraint and collider structures.
/// Must be blittable (no managed references) for GPU transfer.
/// </summary>

[StructLayout(LayoutKind.Sequential)]
public struct GPURod
{
    public int I;
    public int J;
    public float RestLength;
    public float Compliance;
    public float Lambda;

    public GPURod(Rod rod)
    {
        I = rod.I;
        J = rod.J;
        RestLength = rod.RestLength;
        Compliance = rod.Compliance;
        Lambda = rod.Lambda;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUAngle
{
    public int I;
    public int J;
    public int K;
    public float Theta0;
    public float Compliance;
    public float Lambda;

    public GPUAngle(Angle angle)
    {
        I = angle.I;
        J = angle.J;
        K = angle.K;
        Theta0 = angle.Theta0;
        Compliance = angle.Compliance;
        Lambda = angle.Lambda;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUMotorAngle
{
    public int I;
    public int J;
    public int K;
    public float Target;
    public float Compliance;
    public float Lambda;

    public GPUMotorAngle(MotorAngle motor)
    {
        I = motor.I;
        J = motor.J;
        K = motor.K;
        Target = motor.Target;
        Compliance = motor.Compliance;
        Lambda = motor.Lambda;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUCircleCollider
{
    public float CX;
    public float CY;
    public float Radius;

    public GPUCircleCollider(CircleCollider circle)
    {
        CX = circle.CX;
        CY = circle.CY;
        Radius = circle.Radius;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUCapsuleCollider
{
    public float CX;
    public float CY;
    public float UX;
    public float UY;
    public float HalfLength;
    public float Radius;

    public GPUCapsuleCollider(CapsuleCollider capsule)
    {
        CX = capsule.CX;
        CY = capsule.CY;
        UX = capsule.UX;
        UY = capsule.UY;
        HalfLength = capsule.HalfLength;
        Radius = capsule.Radius;
    }
}

[StructLayout(LayoutKind.Sequential)]
public struct GPUOBBCollider
{
    public float CX;
    public float CY;
    public float UX;
    public float UY;
    public float HalfExtentX;
    public float HalfExtentY;

    public GPUOBBCollider(OBBCollider obb)
    {
        CX = obb.CX;
        CY = obb.CY;
        UX = obb.UX;
        UY = obb.UY;
        HalfExtentX = obb.HalfExtentX;
        HalfExtentY = obb.HalfExtentY;
    }
}
