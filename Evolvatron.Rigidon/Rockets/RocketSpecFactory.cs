using System;
using Evolvatron.Core.GPU;

namespace Evolvatron.Core.Rockets;

/// <summary>
/// Builds a <see cref="RocketSpec"/> into the physics engines. The canonical rest pose is rotated by
/// <c>tilt</c> about the rocket origin and translated to (spawnX, spawnY). Body 0 receives the initial
/// angular velocity; all bodies share the initial linear velocity (matching the existing builders).
/// </summary>
public static class RocketSpecFactory
{
    /// <summary>Build the rocket into a CPU <see cref="WorldState"/>. Returns the new body indices in spec order.</summary>
    public static int[] ToCpuWorld(RocketSpec spec, WorldState world,
        float spawnX, float spawnY, float tilt = 0f, float velX = 0f, float velY = 0f, float angVel = 0f)
    {
        float cosT = MathF.Cos(tilt), sinT = MathF.Sin(tilt);
        var indices = new int[spec.Bodies.Count];

        for (int bi = 0; bi < spec.Bodies.Count; bi++)
        {
            var b = spec.Bodies[bi];
            int geomStart = world.RigidBodyGeoms.Count;
            foreach (var g in b.Geoms)
                world.RigidBodyGeoms.Add(new RigidBodyGeom(g.LocalX, g.LocalY, g.Radius));

            float wx = spawnX + (b.X * cosT - b.Y * sinT);
            float wy = spawnY + (b.X * sinT + b.Y * cosT);
            indices[bi] = world.RigidBodies.Count;
            world.RigidBodies.Add(new RigidBody(wx, wy, b.Angle + tilt, b.Mass, b.Inertia, geomStart, b.Geoms.Count)
            {
                VelX = velX,
                VelY = velY,
                AngularVel = bi == 0 ? angVel : 0f
            });
        }

        foreach (var j in spec.Joints)
        {
            world.RevoluteJoints.Add(new RevoluteJoint(
                indices[j.BodyA], indices[j.BodyB], j.AnchorAX, j.AnchorAY, j.AnchorBX, j.AnchorBY)
            {
                ReferenceAngle = j.ReferenceAngle,
                EnableLimits = j.EnableLimits,
                LowerAngle = j.LowerAngle,
                UpperAngle = j.UpperAngle,
                EnableMotor = j.EnableMotor,
                MotorSpeed = j.MotorSpeed,
                MaxMotorTorque = j.MaxMotorTorque
            });
        }

        return indices;
    }

    /// <summary>
    /// Fill the strided per-world GPU arrays for one world. Active bodies occupy local indices
    /// [0, spec.BodyCount); any remaining slots up to bodiesPerWorld are written inactive (InvMass=0).
    /// Geoms are packed contiguously; each body's GeomStartIndex is the LOCAL offset within the world
    /// (what InlinePhysics expects: geoms[worldIdx*geomsPerWorld + GeomStartIndex + g]).
    /// </summary>
    public static void ToGpuWorld(RocketSpec spec,
        GPURigidBody[] bodies, GPURigidBodyGeom[] geoms, GPURevoluteJoint[] joints,
        int worldIdx, int bodiesPerWorld, int geomsPerWorld, int jointsPerWorld,
        float spawnX, float spawnY, float tilt = 0f, float velX = 0f, float velY = 0f, float angVel = 0f)
    {
        if (spec.BodyCount > bodiesPerWorld)
            throw new InvalidOperationException($"spec bodies {spec.BodyCount} > bodiesPerWorld {bodiesPerWorld}");
        if (spec.TotalGeoms > geomsPerWorld)
            throw new InvalidOperationException($"spec geoms {spec.TotalGeoms} > geomsPerWorld {geomsPerWorld}");
        if (spec.JointCount > jointsPerWorld)
            throw new InvalidOperationException($"spec joints {spec.JointCount} > jointsPerWorld {jointsPerWorld}");

        float cosT = MathF.Cos(tilt), sinT = MathF.Sin(tilt);
        int bodyBase = worldIdx * bodiesPerWorld;
        int geomBase = worldIdx * geomsPerWorld;
        int jointBase = worldIdx * jointsPerWorld;

        int localGeom = 0;
        for (int bi = 0; bi < spec.Bodies.Count; bi++)
        {
            var b = spec.Bodies[bi];
            int geomStart = localGeom;
            for (int g = 0; g < b.Geoms.Count; g++)
            {
                var gs = b.Geoms[g];
                geoms[geomBase + localGeom] = new GPURigidBodyGeom
                {
                    LocalX = gs.LocalX, LocalY = gs.LocalY, Radius = gs.Radius, BodyIndex = bi
                };
                localGeom++;
            }

            float wx = spawnX + (b.X * cosT - b.Y * sinT);
            float wy = spawnY + (b.X * sinT + b.Y * cosT);
            float angle = b.Angle + tilt;
            bodies[bodyBase + bi] = new GPURigidBody
            {
                X = wx, Y = wy, Angle = angle,
                VelX = velX, VelY = velY, AngularVel = bi == 0 ? angVel : 0f,
                PrevX = wx, PrevY = wy, PrevAngle = angle,
                InvMass = 1f / b.Mass, InvInertia = 1f / b.Inertia,
                GeomStartIndex = geomStart, GeomCount = b.Geoms.Count
            };
        }

        // Pad unused body slots inactive so the kernel's fixed-count loop skips them safely.
        for (int bi = spec.Bodies.Count; bi < bodiesPerWorld; bi++)
            bodies[bodyBase + bi] = new GPURigidBody { InvMass = 0f, InvInertia = 0f, GeomStartIndex = 0, GeomCount = 0 };

        for (int ji = 0; ji < spec.Joints.Count; ji++)
        {
            var j = spec.Joints[ji];
            joints[jointBase + ji] = new GPURevoluteJoint
            {
                BodyA = bodyBase + j.BodyA, BodyB = bodyBase + j.BodyB,
                LocalAnchorAX = j.AnchorAX, LocalAnchorAY = j.AnchorAY,
                LocalAnchorBX = j.AnchorBX, LocalAnchorBY = j.AnchorBY,
                ReferenceAngle = j.ReferenceAngle,
                EnableLimits = (byte)(j.EnableLimits ? 1 : 0),
                LowerAngle = j.LowerAngle, UpperAngle = j.UpperAngle,
                EnableMotor = (byte)(j.EnableMotor ? 1 : 0),
                MotorSpeed = j.MotorSpeed, MaxMotorTorque = j.MaxMotorTorque
            };
        }
    }
}
