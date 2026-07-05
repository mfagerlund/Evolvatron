using System;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.Rockets;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// P0 foundation tests (see docs/godot_pipeline_plan.md): the RocketSpec data model, JSON
/// round-trip, and parity of the spec→engine factories against the stock 3-body rocket.
/// </summary>
public class RocketSpecTests
{
    const float Tol = 1e-4f;

    [Fact]
    public void StockRocket_HasExpectedTopology()
    {
        var spec = RocketSpecLibrary.StockRocket();
        spec.Validate();

        Assert.Equal(3, spec.BodyCount);
        Assert.Equal(2, spec.JointCount);
        Assert.Equal(19, spec.TotalGeoms);          // 5 + 7 + 7
        Assert.Equal(5, spec.Bodies[0].Geoms.Count);
        Assert.Equal(7, spec.Bodies[1].Geoms.Count);
        Assert.Equal(7, spec.Bodies[2].Geoms.Count);
        Assert.Single(spec.Thrusters);
        Assert.Equal(8, spec.SensorCount);
        Assert.Equal(2, spec.ActuatorDofCount);     // 1 throttle + 1 gimbal
    }

    [Fact]
    public void StockRocket_PassesGpuLimits_AtStockConfig()
    {
        var spec = RocketSpecLibrary.StockRocket();
        spec.ValidateGpuLimits(maxBodies: 3, maxGeoms: 19, maxJoints: 2);   // exact fit, must not throw
        Assert.Throws<InvalidOperationException>(() => spec.ValidateGpuLimits(2, 19, 2));  // too many bodies
    }

    [Fact]
    public void JsonRoundTrip_PreservesStructureAndValues()
    {
        var spec = RocketSpecLibrary.StockRocket();
        var restored = RocketSpec.FromJson(spec.ToJson());

        Assert.Equal(spec.BodyCount, restored.BodyCount);
        Assert.Equal(spec.JointCount, restored.JointCount);
        Assert.Equal(spec.TotalGeoms, restored.TotalGeoms);
        Assert.Equal(spec.SensorCount, restored.SensorCount);
        Assert.Equal(spec.Bodies[0].Mass, restored.Bodies[0].Mass, Tol);
        Assert.Equal(spec.Bodies[0].Inertia, restored.Bodies[0].Inertia, Tol);
        Assert.Equal(spec.Bodies[2].Angle, restored.Bodies[2].Angle, Tol);
        Assert.Equal(spec.Joints[0].ReferenceAngle, restored.Joints[0].ReferenceAngle, Tol);
        Assert.Equal(spec.Thrusters[0].MaxThrust, restored.Thrusters[0].MaxThrust, Tol);
        Assert.Equal(spec.Bodies[1].Geoms[3].LocalX, restored.Bodies[1].Geoms[3].LocalX, Tol);
    }

    [Fact]
    public void ToCpuWorld_MatchesStockRocketLayout()
    {
        var spec = RocketSpecLibrary.StockRocket();
        var world = new WorldState();
        const float sx = 3f, sy = 10f;

        var idx = RocketSpecFactory.ToCpuWorld(spec, world, sx, sy);

        Assert.Equal(3, world.RigidBodies.Count);
        Assert.Equal(19, world.RigidBodyGeoms.Count);
        Assert.Equal(2, world.RevoluteJoints.Count);

        // Body 0: fuselage center is bodyHalf (0.75) above the spawn, upright.
        var body0 = world.RigidBodies[idx[0]];
        Assert.Equal(sx, body0.X, Tol);
        Assert.Equal(sy + 0.75f, body0.Y, Tol);
        Assert.Equal(MathF.PI / 2f, body0.Angle, Tol);
        Assert.Equal(1f / 8f, body0.InvMass, Tol);

        // First body geom at local -0.75, radius 0.2.
        var g0 = world.RigidBodyGeoms[body0.GeomStartIndex];
        Assert.Equal(-0.75f, g0.LocalX, Tol);
        Assert.Equal(0.2f, g0.Radius, Tol);

        // Legs: 1.5 kg each.
        Assert.Equal(1f / 1.5f, world.RigidBodies[idx[1]].InvMass, Tol);
        Assert.Equal(1f / 1.5f, world.RigidBodies[idx[2]].InvMass, Tol);

        // Joints connect body 0 to each leg.
        Assert.Equal(idx[0], world.RevoluteJoints[0].BodyA);
        Assert.Equal(idx[1], world.RevoluteJoints[0].BodyB);
        Assert.Equal(idx[2], world.RevoluteJoints[1].BodyB);
    }

    [Fact]
    public void ToGpuWorld_MatchesEvaluatorGeomLayout()
    {
        var spec = RocketSpecLibrary.StockRocket();
        const int bpw = 3, gpw = 19, jpw = 2;
        var bodies = new GPURigidBody[bpw * 2];     // two worlds, to check striding
        var geoms = new GPURigidBodyGeom[gpw * 2];
        var joints = new GPURevoluteJoint[jpw * 2];

        RocketSpecFactory.ToGpuWorld(spec, bodies, geoms, joints, worldIdx: 1, bpw, gpw, jpw, spawnX: 0f, spawnY: 5f);

        int bodyBase = 1 * bpw, geomBase = 1 * gpw, jointBase = 1 * jpw;

        // Contiguous geom packing: body0 → [0,5), body1 → [5,12), body2 → [12,19).
        Assert.Equal(0, bodies[bodyBase + 0].GeomStartIndex);
        Assert.Equal(5, bodies[bodyBase + 0].GeomCount);
        Assert.Equal(5, bodies[bodyBase + 1].GeomStartIndex);
        Assert.Equal(7, bodies[bodyBase + 1].GeomCount);
        Assert.Equal(12, bodies[bodyBase + 2].GeomStartIndex);
        Assert.Equal(7, bodies[bodyBase + 2].GeomCount);

        Assert.Equal(1f / 8f, bodies[bodyBase + 0].InvMass, Tol);

        // Geom BodyIndex tags follow the packing.
        Assert.Equal(0, geoms[geomBase + 0].BodyIndex);
        Assert.Equal(1, geoms[geomBase + 5].BodyIndex);
        Assert.Equal(2, geoms[geomBase + 12].BodyIndex);

        // Joints carry GLOBAL body indices (bodyBase + local).
        Assert.Equal(bodyBase + 0, joints[jointBase + 0].BodyA);
        Assert.Equal(bodyBase + 1, joints[jointBase + 0].BodyB);
        Assert.Equal(bodyBase + 2, joints[jointBase + 1].BodyB);
        Assert.Equal((byte)1, joints[jointBase + 0].EnableMotor);
    }
}
