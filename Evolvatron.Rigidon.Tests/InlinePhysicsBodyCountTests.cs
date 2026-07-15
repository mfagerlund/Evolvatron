using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.MegaKernel;
using ILGPU;
using ILGPU.Runtime;
using Xunit;

namespace Evolvatron.Rigidon.Tests;

/// <summary>
/// Regression guard for the 3-body cos/sin cache in InlinePhysics.
///
/// The old code cached cos/sin for exactly three bodies and then did
///     geom.BodyIndex == 0 ? cos0 : (geom.BodyIndex == 1 ? cos1 : cos2)
/// while the surrounding loops honoured cfg.BodiesPerWorld. So any geom on body 3+ silently
/// used BODY 2's angle to place itself — corrupt physics with no crash and no exception.
/// A swarm world has 7-16 bodies, which is how this got found.
///
/// These tests run on ILGPU's CPU accelerator so they guard the fix on machines without a GPU
/// (the dev laptop has none). The kernel code under test is the same code CUDA compiles.
/// </summary>
public sealed class InlinePhysicsBodyCountTests : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;

    public InlinePhysicsBodyCountTests()
    {
        _context = Context.Create(b => b.Default().EnableAlgorithms());
        _accelerator = _context.GetPreferredDevice(preferCPU: true).CreateAccelerator(_context);
    }

    /// <summary>
    /// Every body gets a distinct angle and a geom offset along its local +X. After a substep,
    /// each geom's world position must reflect ITS OWN body's angle.
    ///
    /// Under the old 3-body cache, bodies 3..N-1 all placed their geoms using body 2's angle,
    /// so this fails for N > 3 and passes for N <= 3.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]   // the old ceiling — passed before the fix too
    [InlineData(4)]   // first body the old cache got wrong
    [InlineData(7)]   // a 6-agent swarm + payload
    [InlineData(16)]  // two competing teams of 8
    public void GeomsUseTheirOwnBodyAngle_ForAnyBodyCount(int bodyCount)
    {
        const float offset = 1.0f;

        var bodies = new GPURigidBody[bodyCount];
        var geoms = new GPURigidBodyGeom[bodyCount];

        for (int i = 0; i < bodyCount; i++)
        {
            // Distinct, well-separated angles so a wrong one can't coincidentally match.
            float angle = i * 0.37f + 0.11f;
            bodies[i] = new GPURigidBody
            {
                X = i * 10f, Y = 0f, Angle = angle,
                VelX = 0f, VelY = 0f, AngularVel = 0f,
                InvMass = 1f, InvInertia = 1f,
                GeomStartIndex = i, GeomCount = 1,
            };
            geoms[i] = new GPURigidBodyGeom
            {
                LocalX = offset, LocalY = 0f, Radius = 0.1f, BodyIndex = i,
            };
        }

        var (outBodies, outGeoms) = RunOneSubstep(bodies, geoms, bodyCount);

        for (int i = 0; i < bodyCount; i++)
        {
            float a = outBodies[i].Angle;
            float expectedX = outBodies[i].X + offset * MathF.Cos(a);
            float expectedY = outBodies[i].Y + offset * MathF.Sin(a);

            Assert.True(MathF.Abs(outGeoms[i].WorldX - expectedX) < 1e-4f,
                $"body {i}/{bodyCount}: geom WorldX {outGeoms[i].WorldX} != {expectedX} " +
                $"(angle {a}). Geom is being placed with another body's angle.");
            Assert.True(MathF.Abs(outGeoms[i].WorldY - expectedY) < 1e-4f,
                $"body {i}/{bodyCount}: geom WorldY {outGeoms[i].WorldY} != {expectedY} " +
                $"(angle {a}). Geom is being placed with another body's angle.");
        }
    }

    /// <summary>
    /// Bodies must rotate independently: giving each a different angular velocity and stepping
    /// must leave every body at its own integrated angle. Guards against any future shared-cache
    /// or shared-index regression.
    /// </summary>
    [Fact]
    public void BodiesRotateIndependently_BeyondThree()
    {
        const int bodyCount = 8;
        const float dt = 1f / 120f;

        var bodies = new GPURigidBody[bodyCount];
        var geoms = new GPURigidBodyGeom[bodyCount];
        var expected = new float[bodyCount];

        for (int i = 0; i < bodyCount; i++)
        {
            float angVel = 0.5f + i * 0.25f;   // all distinct
            bodies[i] = new GPURigidBody
            {
                X = i * 10f, Y = 0f, Angle = 0f,
                VelX = 0f, VelY = 0f, AngularVel = angVel,
                InvMass = 1f, InvInertia = 1f,
                GeomStartIndex = i, GeomCount = 1,
            };
            geoms[i] = new GPURigidBodyGeom
            {
                LocalX = 0f, LocalY = 0f, Radius = 0.1f, BodyIndex = i,
            };
            expected[i] = angVel * dt;   // no damping in this config
        }

        var (outBodies, _) = RunOneSubstep(bodies, geoms, bodyCount, dt);

        for (int i = 0; i < bodyCount; i++)
            Assert.True(MathF.Abs(outBodies[i].Angle - expected[i]) < 1e-5f,
                $"body {i}: angle {outBodies[i].Angle} != expected {expected[i]}");
    }

    private (GPURigidBody[], GPURigidBodyGeom[]) RunOneSubstep(
        GPURigidBody[] bodies, GPURigidBodyGeom[] geoms, int bodyCount, float dt = 1f / 120f)
    {
        var cfg = new MegaKernelConfig
        {
            BodiesPerWorld = bodyCount,
            GeomsPerWorld = bodyCount,
            JointsPerWorld = 0,
            SharedColliderCount = 0,
            MaxContactsPerWorld = 1,
            Dt = dt,
            GravityX = 0f, GravityY = 0f,
            FrictionMu = 0f, Restitution = 0f,
            GlobalDamping = 0f, AngularDamping = 0f,
            SolverIterations = 1,
        };

        using var dBodies = _accelerator.Allocate1D(bodies);
        using var dGeoms = _accelerator.Allocate1D(geoms);
        using var dJoints = _accelerator.Allocate1D<GPURevoluteJoint>(1);
        using var dJointC = _accelerator.Allocate1D<GPUJointConstraint>(1);
        using var dContacts = _accelerator.Allocate1D<GPUContactConstraint>(1);
        using var dCache = _accelerator.Allocate1D<GPUCachedContactImpulse>(1);
        using var dCounts = _accelerator.Allocate1D<int>(1);
        using var dColliders = _accelerator.Allocate1D<GPUOBBCollider>(1);
        dCounts.MemSetToZero();

        var pv = new PhysicsViews
        {
            Bodies = dBodies.View,
            Geoms = dGeoms.View,
            Joints = dJoints.View,
            JointConstraints = dJointC.View,
            Contacts = dContacts.View,
            ContactCache = dCache.View,
            ContactCounts = dCounts.View,
            SharedOBBColliders = dColliders.View,
        };

        var kernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, PhysicsViews, MegaKernelConfig>(
            static (Index1D w, PhysicsViews p, MegaKernelConfig c) =>
                InlinePhysics.SubStepOneWorld(p, w, c));

        kernel(1, pv, cfg);
        _accelerator.Synchronize();

        return (dBodies.GetAsArray1D(), dGeoms.GetAsArray1D());
    }

    public void Dispose()
    {
        _accelerator.Dispose();
        _context.Dispose();
    }
}
