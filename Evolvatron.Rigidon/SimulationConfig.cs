namespace Evolvatron.Core;

/// <summary>
/// Configuration parameters for the XPBD particle simulation.
/// All values use SI units: meters, kilograms, seconds.
/// </summary>
public sealed class SimulationConfig
{
    /// <summary>
    /// Fixed timestep in seconds (default: 1/240 = ~4.17ms).
    /// </summary>
    public float Dt { get; set; } = 1f / 240f;

    /// <summary>
    /// Number of XPBD constraint solving iterations per substep (default: 12).
    /// Higher values = more rigid constraints but more computation.
    /// </summary>
    public int XpbdIterations { get; set; } = 12;

    /// <summary>
    /// Number of substeps per Step call (default: 1).
    /// Can increase for better stability at larger Dt values.
    /// </summary>
    public int Substeps { get; set; } = 1;

    /// <summary>
    /// Gravity acceleration in X direction (m/s², default: 0).
    /// </summary>
    public float GravityX { get; set; } = 0f;

    /// <summary>
    /// Gravity acceleration in Y direction (m/s², default: -9.81).
    /// Negative Y is typically "down" in screen space.
    /// </summary>
    public float GravityY { get; set; } = -9.81f;

    /// <summary>
    /// Compliance for contact constraints (default: 1e-8).
    /// Small nonzero value reduces jitter while maintaining rigidity.
    /// α = compliance / dt² in XPBD formulation.
    /// </summary>
    public float ContactCompliance { get; set; } = 1e-8f;

    /// <summary>
    /// Compliance for rod (distance) constraints (default: 0 = rigid).
    /// Increase for soft/elastic rods.
    /// </summary>
    public float RodCompliance { get; set; } = 0f;

    /// <summary>
    /// Compliance for angle constraints (default: 0 = rigid).
    /// Increase for flexible joints.
    /// </summary>
    public float AngleCompliance { get; set; } = 0f;

    /// <summary>
    /// Compliance for motorized angle constraints (default: 1e-6).
    /// Small value prevents jitter in servo motors.
    /// </summary>
    public float MotorCompliance { get; set; } = 1e-6f;

    /// <summary>
    /// Coefficient of friction (μ) for Coulomb friction model (default: 0.6).
    /// Determines tangential impulse relative to normal impulse.
    /// </summary>
    public float FrictionMu { get; set; } = 0.6f;

    /// <summary>
    /// Coefficient of restitution (bounciness) for rigid body contacts (default: 0.0).
    /// 0 = perfectly inelastic (no bounce), 1 = perfectly elastic (full bounce).
    /// </summary>
    public float Restitution { get; set; } = 0.0f;

    /// <summary>
    /// Velocity stabilization factor (0..1, default: 1.0).
    /// β=1: full velocity correction from position changes.
    /// β=0: no correction (may cause drift).
    /// v_new = (p_new - p_old)/dt * β + v_old * (1-β)
    /// </summary>
    public float VelocityStabilizationBeta { get; set; } = 1.0f;

    /// <summary>
    /// Global velocity damping per second (default: 0.01).
    /// Applied as: v *= (1 - damping * dt).
    /// Helps stabilize simulation and dissipate energy.
    /// </summary>
    public float GlobalDamping { get; set; } = 0.01f;

    /// <summary>
    /// Angular damping per second (default: 0.1).
    /// Applied to rotational motion to dissipate spinning energy.
    /// Higher values more aggressively dampen rotation.
    /// Applied as: angular_velocity *= (1 - angularDamping * dt).
    /// </summary>
    public float AngularDamping { get; set; } = 0.1f;

    /// <summary>
    /// Maximum velocity magnitude for particles (m/s, default: 10).
    /// Velocity stabilization clamps corrected velocities to this limit.
    /// Prevents energy injection from large XPBD position corrections.
    /// Set to 0 or negative to disable clamping.
    /// </summary>
    public float MaxVelocity { get; set; } = 10f;

    /// <summary>
    /// Creates a deep copy of this configuration.
    /// </summary>
    public SimulationConfig Clone()
    {
        return new SimulationConfig
        {
            Dt = Dt,
            XpbdIterations = XpbdIterations,
            Substeps = Substeps,
            GravityX = GravityX,
            GravityY = GravityY,
            ContactCompliance = ContactCompliance,
            RodCompliance = RodCompliance,
            AngleCompliance = AngleCompliance,
            MotorCompliance = MotorCompliance,
            FrictionMu = FrictionMu,
            Restitution = Restitution,
            VelocityStabilizationBeta = VelocityStabilizationBeta,
            GlobalDamping = GlobalDamping,
            AngularDamping = AngularDamping,
            MaxVelocity = MaxVelocity
        };
    }
}
