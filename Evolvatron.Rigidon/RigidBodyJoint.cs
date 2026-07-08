namespace Evolvatron.Core;

/// <summary>
/// Revolute joint (hinge) connecting two rigid bodies at a common point.
/// Allows rotation but constrains the anchor points to coincide.
/// Based on Box2D's revolute joint implementation.
/// </summary>
public struct RevoluteJoint
{
    public int BodyA;           // First rigid body index
    public int BodyB;           // Second rigid body index

    // Anchor points in local space
    public float LocalAnchorAX;
    public float LocalAnchorAY;
    public float LocalAnchorBX;
    public float LocalAnchorBY;

    // Enable angle limits
    public bool EnableLimits;
    public float LowerAngle;    // Minimum relative angle (radians)
    public float UpperAngle;    // Maximum relative angle (radians)
    public float ReferenceAngle; // Initial angle offset between bodies

    // Enable motor
    public bool EnableMotor;
    public float MotorSpeed;    // Target angular velocity (rad/s)
    public float MaxMotorTorque; // Maximum torque the motor can apply

    // Diagnostic read-back (output): the motor impulse the solver actually applied on the last Step, written
    // back after the velocity solve. |AppliedMotorImpulse| ≤ MaxMotorTorque·dt, and applied torque =
    // AppliedMotorImpulse / dt. Does not affect the simulation — it is populated for inspection only (e.g. a
    // "how saturated is this motor?" gauge). CPU stepper only; 0 until the first Step.
    public float AppliedMotorImpulse;

    public RevoluteJoint(int bodyA, int bodyB, float anchorAX, float anchorAY, float anchorBX, float anchorBY)
    {
        BodyA = bodyA;
        BodyB = bodyB;
        LocalAnchorAX = anchorAX;
        LocalAnchorAY = anchorAY;
        LocalAnchorBX = anchorBX;
        LocalAnchorBY = anchorBY;
        EnableLimits = false;
        LowerAngle = 0f;
        UpperAngle = 0f;
        ReferenceAngle = 0f;
        EnableMotor = false;
        MotorSpeed = 0f;
        MaxMotorTorque = 0f;
        AppliedMotorImpulse = 0f;
    }
}

/// <summary>
/// Joint constraint solver data (computed during initialization).
/// </summary>
public struct RevoluteJointConstraint
{
    public int JointIndex;      // Source index into WorldState.RevoluteJoints (constraints skip static-static
                                // joints, so this is not the list position) — used to write diagnostics back.
    public int BodyAIndex;
    public int BodyBIndex;

    // Local anchor points
    public float RA_X, RA_Y;  // Anchor A in local space
    public float RB_X, RB_Y;  // Anchor B in local space

    // World-space anchor offsets (local anchor rotated by body angle), cached during initialization.
    // Body angles are constant across the velocity-constraint iterations of a substep, so these are
    // computed once and reused instead of recomputing sin/cos every iteration.
    public float RA_WorldX, RA_WorldY;
    public float RB_WorldX, RB_WorldY;

    // Effective mass for position constraint (2x2 matrix)
    public float K11, K12, K21, K22; // K matrix
    public float Mass11, Mass12, Mass21, Mass22; // Inverse of K

    // Accumulated impulses for warm starting
    public float ImpulseX;
    public float ImpulseY;

    // Angle limit constraint data
    public bool EnableLimits;
    public float LowerAngle;
    public float UpperAngle;
    public float ReferenceAngle;
    public float AngleLimitImpulse;
    public float AngleLimitMass; // Effective mass for angle constraint

    // Motor constraint data
    public bool EnableMotor;
    public float MotorSpeed;
    public float MaxMotorTorque;
    public float MotorImpulse;
    public float MotorMass; // Effective mass for motor
}
