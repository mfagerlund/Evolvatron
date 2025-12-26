using Evolvatron.Core.Physics;
using System.Collections.Generic;

namespace Evolvatron.Core;

/// <summary>
/// CPU-based reference implementation of the XPBD particle simulation stepper.
/// Executes all simulation phases in sequence.
/// </summary>
public sealed class CPUStepper : IStepper
{
    /// <summary>
    /// Cache for warm-starting rigid body contacts between frames.
    /// </summary>
    private readonly Dictionary<ContactId, CachedContactImpulse> _contactCache = new();

    /// <summary>
    /// Advances the simulation by one step.
    /// Each step may contain multiple substeps for stability.
    /// </summary>
    public void Step(WorldState world, SimulationConfig cfg)
    {
        for (int substep = 0; substep < cfg.Substeps; substep++)
        {
            SubStep(world, cfg);
        }
    }

    private void SubStep(WorldState world, SimulationConfig cfg)
    {
        float dt = cfg.Dt;

        // === PARTICLES ===

        // 1. External forces (gravity)
        Integrator.ApplyGravity(world, cfg.GravityX, cfg.GravityY);

        // Controllers would add forces here (e.g., thrust)

        // 2. Integrate velocity and position (symplectic Euler)
        world.SavePreviousPositions(); // For velocity stabilization
        Integrator.Integrate(world, dt);

        // === RIGID BODIES ===

        // 1. Apply gravity to rigid bodies
        Integrator.ApplyGravityToRigidBodies(world, cfg.GravityX, cfg.GravityY, dt);

        // 2. Save previous state for velocity stabilization
        Integrator.SaveRigidBodyPreviousState(world, out var rbPrevState);

        // 3. Integrate rigid bodies
        Integrator.IntegrateRigidBodies(world, dt);

        // === CONSTRAINT SOLVING ===

        // 3. XPBD constraint solving iterations (for particles)
        // Reset lambdas once per substep (NOT per iteration - critical for XPBD stability)
        XPBDSolver.ResetLambdas(world);

        for (int iter = 0; iter < cfg.XpbdIterations; iter++)
        {
            // Solve particle constraints in order of importance:
            // 1. Structural (rods) - maintain edge lengths
            // 2. Shape (angles) - maintain angles
            // 3. Positional (contacts) - prevent penetration
            // 4. Actuation (motors) - apply control
            XPBDSolver.SolveRods(world, dt, cfg.RodCompliance);
            XPBDSolver.SolveAngles(world, dt, cfg.AngleCompliance);
            XPBDSolver.SolveContacts(world, dt, cfg.ContactCompliance);
            XPBDSolver.SolveMotors(world, dt, cfg.MotorCompliance);
        }

        // 4. Impulse-based contact solving (for rigid bodies)
        // Initialize contact constraints and compute effective masses
        var rigidBodyContacts = ImpulseContactSolver.InitializeConstraints(world, dt, cfg.FrictionMu, cfg.Restitution);

        // Apply cached impulses from previous frame to initialize solver state
        ImpulseContactSolver.ApplyWarmStarting(rigidBodyContacts, _contactCache);

        // Warm-start with cached impulses (improves convergence)
        ImpulseContactSolver.WarmStart(world, rigidBodyContacts);

        // Initialize joint constraints
        var jointConstraints = RigidBodyJointSolver.InitializeConstraints(world, dt);

        // Solve velocity constraints (sequential impulse)
        for (int iter = 0; iter < cfg.XpbdIterations; iter++)
        {
            ImpulseContactSolver.SolveVelocityConstraints(world, rigidBodyContacts);
            RigidBodyJointSolver.SolveVelocityConstraints(world, jointConstraints);
        }

        // Solve position constraints for stability
        RigidBodyJointSolver.SolvePositionConstraints(world, jointConstraints);

        // Store impulses for next frame's warm-start
        ImpulseContactSolver.StoreImpulses(rigidBodyContacts, _contactCache);

        // === POST-PROCESSING ===

        // 5. Velocity stabilization (correct velocities from position changes in particle solver)
        // MaxVelocity clamping prevents energy injection from large XPBD corrections
        world.StabilizeVelocities(dt, cfg.VelocityStabilizationBeta, cfg.MaxVelocity);
        // Note: Rigid body velocity stabilization not needed with impulse solver

        // 6. Friction pass for particles (velocity-level)
        Friction.ApplyFriction(world, cfg.FrictionMu);
        // Note: Rigid body friction is now handled in ImpulseContactSolver

        // 7. Global damping (linear velocity)
        world.ApplyDamping(cfg.GlobalDamping, dt);
        Integrator.ApplyRigidBodyDamping(world, cfg.GlobalDamping, cfg.AngularDamping, dt);

        // 8. Angular damping for particles
        Integrator.ApplyParticleAngularDamping(world, cfg.AngularDamping, dt);

        // 9. Optional: cull out-of-bounds particles/contraptions (not implemented yet)
    }
}
