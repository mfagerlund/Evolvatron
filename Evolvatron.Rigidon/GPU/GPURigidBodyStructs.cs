using System.Runtime.InteropServices;
using Evolvatron.Core.Physics;

namespace Evolvatron.Core.GPU;

/// <summary>
/// GPU-compatible rigid body state.
/// Stores position, rotation, velocities, and mass properties for a single rigid body.
/// Circle geometries are stored separately in GPURigidBodyGeom array, indexed by GeomStartIndex/GeomCount.
/// Must be blittable (no managed references) for GPU transfer.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPURigidBody
{
    /// <summary>Center of mass X position in world space.</summary>
    public float X;

    /// <summary>Center of mass Y position in world space.</summary>
    public float Y;

    /// <summary>Rotation angle in radians.</summary>
    public float Angle;

    /// <summary>Linear velocity X component (m/s).</summary>
    public float VelX;

    /// <summary>Linear velocity Y component (m/s).</summary>
    public float VelY;

    /// <summary>Angular velocity (rad/s).</summary>
    public float AngularVel;

    /// <summary>Inverse mass (1/kg). Zero for static bodies.</summary>
    public float InvMass;

    /// <summary>Inverse moment of inertia (1/(kg*m^2)). Zero for static bodies.</summary>
    public float InvInertia;

    /// <summary>Starting index into the GPURigidBodyGeom array.</summary>
    public int GeomStartIndex;

    /// <summary>Number of circle geometries attached to this body.</summary>
    public int GeomCount;

    /// <summary>
    /// Creates a GPU-compatible rigid body from a CPU RigidBody.
    /// </summary>
    /// <param name="rb">The CPU rigid body to convert.</param>
    public GPURigidBody(RigidBody rb)
    {
        X = rb.X;
        Y = rb.Y;
        Angle = rb.Angle;
        VelX = rb.VelX;
        VelY = rb.VelY;
        AngularVel = rb.AngularVel;
        InvMass = rb.InvMass;
        InvInertia = rb.InvInertia;
        GeomStartIndex = rb.GeomStartIndex;
        GeomCount = rb.GeomCount;
    }

    /// <summary>
    /// Converts this GPU struct back to a CPU RigidBody.
    /// </summary>
    public RigidBody ToCPU()
    {
        return new RigidBody
        {
            X = X,
            Y = Y,
            Angle = Angle,
            VelX = VelX,
            VelY = VelY,
            AngularVel = AngularVel,
            InvMass = InvMass,
            InvInertia = InvInertia,
            GeomStartIndex = GeomStartIndex,
            GeomCount = GeomCount
        };
    }
}

/// <summary>
/// GPU-compatible circle geometry attached to a rigid body.
/// Position is relative to the parent body's center of mass in local space.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPURigidBodyGeom
{
    /// <summary>X offset from rigid body center in local space.</summary>
    public float LocalX;

    /// <summary>Y offset from rigid body center in local space.</summary>
    public float LocalY;

    /// <summary>Circle radius.</summary>
    public float Radius;

    /// <summary>
    /// Creates a GPU-compatible geom from a CPU RigidBodyGeom.
    /// </summary>
    /// <param name="geom">The CPU geometry to convert.</param>
    public GPURigidBodyGeom(RigidBodyGeom geom)
    {
        LocalX = geom.LocalX;
        LocalY = geom.LocalY;
        Radius = geom.Radius;
    }

    /// <summary>
    /// Converts this GPU struct back to a CPU RigidBodyGeom.
    /// </summary>
    public RigidBodyGeom ToCPU()
    {
        return new RigidBodyGeom(LocalX, LocalY, Radius);
    }
}

/// <summary>
/// GPU-compatible revolute joint definition.
/// Connects two rigid bodies at anchor points, allowing rotation but constraining position.
/// Supports optional angle limits and motor.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPURevoluteJoint
{
    /// <summary>Index of first connected body in the rigid body array.</summary>
    public int BodyA;

    /// <summary>Index of second connected body in the rigid body array.</summary>
    public int BodyB;

    /// <summary>Anchor point X in body A's local space.</summary>
    public float LocalAnchorAX;

    /// <summary>Anchor point Y in body A's local space.</summary>
    public float LocalAnchorAY;

    /// <summary>Anchor point X in body B's local space.</summary>
    public float LocalAnchorBX;

    /// <summary>Anchor point Y in body B's local space.</summary>
    public float LocalAnchorBY;

    /// <summary>Initial relative angle between bodies (radians).</summary>
    public float ReferenceAngle;

    /// <summary>Whether angle limits are enabled (0 = false, 1 = true).</summary>
    public byte EnableLimits;

    /// <summary>Minimum allowed relative angle (radians).</summary>
    public float LowerAngle;

    /// <summary>Maximum allowed relative angle (radians).</summary>
    public float UpperAngle;

    /// <summary>Whether the motor is enabled (0 = false, 1 = true).</summary>
    public byte EnableMotor;

    /// <summary>Target angular velocity for motor (rad/s).</summary>
    public float MotorSpeed;

    /// <summary>Maximum torque the motor can apply (N*m).</summary>
    public float MaxMotorTorque;

    /// <summary>
    /// Creates a GPU-compatible joint from a CPU RevoluteJoint.
    /// </summary>
    /// <param name="joint">The CPU joint to convert.</param>
    public GPURevoluteJoint(RevoluteJoint joint)
    {
        BodyA = joint.BodyA;
        BodyB = joint.BodyB;
        LocalAnchorAX = joint.LocalAnchorAX;
        LocalAnchorAY = joint.LocalAnchorAY;
        LocalAnchorBX = joint.LocalAnchorBX;
        LocalAnchorBY = joint.LocalAnchorBY;
        ReferenceAngle = joint.ReferenceAngle;
        EnableLimits = joint.EnableLimits ? (byte)1 : (byte)0;
        LowerAngle = joint.LowerAngle;
        UpperAngle = joint.UpperAngle;
        EnableMotor = joint.EnableMotor ? (byte)1 : (byte)0;
        MotorSpeed = joint.MotorSpeed;
        MaxMotorTorque = joint.MaxMotorTorque;
    }

    /// <summary>
    /// Converts this GPU struct back to a CPU RevoluteJoint.
    /// </summary>
    public RevoluteJoint ToCPU()
    {
        return new RevoluteJoint(BodyA, BodyB, LocalAnchorAX, LocalAnchorAY, LocalAnchorBX, LocalAnchorBY)
        {
            ReferenceAngle = ReferenceAngle,
            EnableLimits = EnableLimits != 0,
            LowerAngle = LowerAngle,
            UpperAngle = UpperAngle,
            EnableMotor = EnableMotor != 0,
            MotorSpeed = MotorSpeed,
            MaxMotorTorque = MaxMotorTorque
        };
    }
}

/// <summary>
/// GPU-compatible contact constraint for impulse-based rigid body collision solving.
/// Contains all data needed to resolve a single contact between a rigid body and static collider.
/// Flattened structure optimized for GPU access patterns.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPUContactConstraint
{
    /// <summary>Index of the rigid body in the rigid body array.</summary>
    public int RigidBodyIndex;

    /// <summary>Contact normal X (points from static collider into rigid body).</summary>
    public float NormalX;

    /// <summary>Contact normal Y (points from static collider into rigid body).</summary>
    public float NormalY;

    /// <summary>Friction tangent X (perpendicular to normal).</summary>
    public float TangentX;

    /// <summary>Friction tangent Y (perpendicular to normal).</summary>
    public float TangentY;

    /// <summary>World-space contact point X.</summary>
    public float ContactX;

    /// <summary>World-space contact point Y.</summary>
    public float ContactY;

    /// <summary>Vector from body center to contact point X (world space).</summary>
    public float RA_X;

    /// <summary>Vector from body center to contact point Y (world space).</summary>
    public float RA_Y;

    /// <summary>Penetration depth (negative = penetrating).</summary>
    public float Separation;

    /// <summary>Effective mass for normal impulse calculation.</summary>
    public float NormalMass;

    /// <summary>Effective mass for friction impulse calculation.</summary>
    public float TangentMass;

    /// <summary>Velocity bias for Baumgarte position stabilization.</summary>
    public float VelocityBias;

    /// <summary>Accumulated normal impulse (for warm-starting and clamping).</summary>
    public float NormalImpulse;

    /// <summary>Accumulated tangent impulse (for Coulomb friction cone).</summary>
    public float TangentImpulse;

    /// <summary>Coefficient of friction (Coulomb).</summary>
    public float Friction;

    /// <summary>Coefficient of restitution (bounciness, 0-1).</summary>
    public float Restitution;

    /// <summary>Valid contact flag (1 = valid, 0 = invalid).</summary>
    public byte IsValid;
 
    /// <summary>Index of the geom within the rigid body that created this contact.</summary>
    public int GeomIndex;

    /// <summary>Type of static collider (0=Circle, 1=Capsule, 2=OBB).</summary>
    public int ColliderType;

    /// <summary>Index of the static collider.</summary>
    public int ColliderIndex;

    /// <summary>
    /// Creates a GPU-compatible contact constraint from a CPU ContactConstraint.
    /// Uses the first contact point (Point1) from the CPU constraint.
    /// </summary>
    /// <param name="constraint">The CPU contact constraint to convert.</param>
    public GPUContactConstraint(ContactConstraint constraint)
    {
        RigidBodyIndex = constraint.RigidBodyIndex;
        NormalX = constraint.NormalX;
        NormalY = constraint.NormalY;
        TangentX = constraint.TangentX;
        TangentY = constraint.TangentY;
        ContactX = constraint.Point1.WorldX;
        ContactY = constraint.Point1.WorldY;
        RA_X = constraint.Point1.RA_X;
        RA_Y = constraint.Point1.RA_Y;
        Separation = constraint.Point1.Separation;
        NormalMass = constraint.Point1.NormalMass;
        TangentMass = constraint.Point1.TangentMass;
        VelocityBias = constraint.Point1.VelocityBias;
        NormalImpulse = constraint.Point1.NormalImpulse;
        TangentImpulse = constraint.Point1.TangentImpulse;
        Friction = constraint.Friction;
        Restitution = constraint.Restitution;
        IsValid = 1;
    }
}

/// <summary>
/// GPU-compatible joint constraint solver working data.
/// Contains precomputed effective masses and accumulated impulses for revolute joints.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPUJointConstraint
{
    /// <summary>Index of body A in the rigid body array.</summary>
    public int BodyAIndex;

    /// <summary>Index of body B in the rigid body array.</summary>
    public int BodyBIndex;

    /// <summary>Anchor offset from body A center X (local space, transformed each step).</summary>
    public float RA_X;

    /// <summary>Anchor offset from body A center Y (local space, transformed each step).</summary>
    public float RA_Y;

    /// <summary>Anchor offset from body B center X (local space, transformed each step).</summary>
    public float RB_X;

    /// <summary>Anchor offset from body B center Y (local space, transformed each step).</summary>
    public float RB_Y;

    /// <summary>Effective mass matrix element [1,1] (inverse of K matrix).</summary>
    public float Mass11;

    /// <summary>Effective mass matrix element [1,2] (inverse of K matrix).</summary>
    public float Mass12;

    /// <summary>Effective mass matrix element [2,1] (inverse of K matrix).</summary>
    public float Mass21;

    /// <summary>Effective mass matrix element [2,2] (inverse of K matrix).</summary>
    public float Mass22;

    /// <summary>Accumulated position impulse X.</summary>
    public float ImpulseX;

    /// <summary>Accumulated position impulse Y.</summary>
    public float ImpulseY;

    /// <summary>Effective mass for angle limit constraint.</summary>
    public float AngleLimitMass;

    /// <summary>Accumulated angle limit impulse.</summary>
    public float AngleLimitImpulse;

    /// <summary>Effective mass for motor constraint.</summary>
    public float MotorMass;

    /// <summary>Accumulated motor impulse.</summary>
    public float MotorImpulse;

    /// <summary>Minimum allowed relative angle (radians).</summary>
    public float LowerAngle;

    /// <summary>Maximum allowed relative angle (radians).</summary>
    public float UpperAngle;

    /// <summary>Initial relative angle between bodies.</summary>
    public float ReferenceAngle;

    /// <summary>Target angular velocity for motor (rad/s).</summary>
    public float MotorSpeed;

    /// <summary>Maximum torque the motor can apply (N*m).</summary>
    public float MaxMotorTorque;

    /// <summary>Whether angle limits are enabled (0 = false, 1 = true).</summary>
    public byte EnableLimits;

    /// <summary>Whether the motor is enabled (0 = false, 1 = true).</summary>
    public byte EnableMotor;

    /// <summary>
    /// Creates a GPU-compatible joint constraint from a CPU RevoluteJointConstraint.
    /// </summary>
    /// <param name="constraint">The CPU joint constraint to convert.</param>
    public GPUJointConstraint(RevoluteJointConstraint constraint)
    {
        BodyAIndex = constraint.BodyAIndex;
        BodyBIndex = constraint.BodyBIndex;
        RA_X = constraint.RA_X;
        RA_Y = constraint.RA_Y;
        RB_X = constraint.RB_X;
        RB_Y = constraint.RB_Y;
        Mass11 = constraint.Mass11;
        Mass12 = constraint.Mass12;
        Mass21 = constraint.Mass21;
        Mass22 = constraint.Mass22;
        ImpulseX = constraint.ImpulseX;
        ImpulseY = constraint.ImpulseY;
        AngleLimitMass = constraint.AngleLimitMass;
        AngleLimitImpulse = constraint.AngleLimitImpulse;
        MotorMass = constraint.MotorMass;
        MotorImpulse = constraint.MotorImpulse;
        LowerAngle = constraint.LowerAngle;
        UpperAngle = constraint.UpperAngle;
        ReferenceAngle = constraint.ReferenceAngle;
        MotorSpeed = constraint.MotorSpeed;
        MaxMotorTorque = constraint.MaxMotorTorque;
        EnableLimits = constraint.EnableLimits ? (byte)1 : (byte)0;
        EnableMotor = constraint.EnableMotor ? (byte)1 : (byte)0;
    }

    /// <summary>
    /// Converts this GPU struct back to a CPU RevoluteJointConstraint.
    /// </summary>
    public RevoluteJointConstraint ToCPU()
    {
        return new RevoluteJointConstraint
        {
            BodyAIndex = BodyAIndex,
            BodyBIndex = BodyBIndex,
            RA_X = RA_X,
            RA_Y = RA_Y,
            RB_X = RB_X,
            RB_Y = RB_Y,
            Mass11 = Mass11,
            Mass12 = Mass12,
            Mass21 = Mass21,
            Mass22 = Mass22,
            ImpulseX = ImpulseX,
            ImpulseY = ImpulseY,
            AngleLimitMass = AngleLimitMass,
            AngleLimitImpulse = AngleLimitImpulse,
            MotorMass = MotorMass,
            MotorImpulse = MotorImpulse,
            LowerAngle = LowerAngle,
            UpperAngle = UpperAngle,
            ReferenceAngle = ReferenceAngle,
            MotorSpeed = MotorSpeed,
            MaxMotorTorque = MaxMotorTorque,
            EnableLimits = EnableLimits != 0,
            EnableMotor = EnableMotor != 0
        };
    }
}

/// <summary>
/// GPU-compatible cached contact impulse for warm-starting.
/// Stores the accumulated impulses from a contact for use in the next frame.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPUCachedContactImpulse
{
    /// <summary>Index of the rigid body.</summary>
    public int RigidBodyIndex;

    /// <summary>Type of static collider (0=Circle, 1=Capsule, 2=OBB).</summary>
    public int ColliderType;

    /// <summary>Index of the static collider.</summary>
    public int ColliderIndex;

    /// <summary>Index of the geom within the rigid body.</summary>
    public int GeomIndex;

    /// <summary>Cached normal impulse from previous frame.</summary>
    public float NormalImpulse;

    /// <summary>Cached tangent impulse from previous frame.</summary>
    public float TangentImpulse;

    /// <summary>Valid flag (1 = valid entry, 0 = empty slot).</summary>
    public byte IsValid;
}
