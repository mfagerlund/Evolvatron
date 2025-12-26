using ILGPU;
using ILGPU.Runtime;
using System;

namespace Evolvatron.Core.GPU;

/// <summary>
/// GPU-side representation of WorldState using device memory buffers.
/// Manages host-to-device and device-to-host transfers.
/// Supports both particle physics (XPBD) and rigid body physics (sequential impulse).
/// </summary>
public sealed class GPUWorldState : IDisposable
{
    private readonly Accelerator _accelerator;

    // Particle data buffers (device memory)
    public MemoryBuffer1D<float, Stride1D.Dense> PosX { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> PosY { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> VelX { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> VelY { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> InvMass { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> Radius { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> ForceX { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> ForceY { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> PrevPosX { get; private set; }
    public MemoryBuffer1D<float, Stride1D.Dense> PrevPosY { get; private set; }

    // Constraint buffers
    public MemoryBuffer1D<GPURod, Stride1D.Dense> Rods { get; private set; }
    public MemoryBuffer1D<GPUAngle, Stride1D.Dense> Angles { get; private set; }
    public MemoryBuffer1D<GPUMotorAngle, Stride1D.Dense> Motors { get; private set; }

    // Collider buffers
    public MemoryBuffer1D<GPUCircleCollider, Stride1D.Dense> Circles { get; private set; }
    public MemoryBuffer1D<GPUCapsuleCollider, Stride1D.Dense> Capsules { get; private set; }
    public MemoryBuffer1D<GPUOBBCollider, Stride1D.Dense> OBBs { get; private set; }

    // Rigid body buffers
    public MemoryBuffer1D<GPURigidBody, Stride1D.Dense> RigidBodies { get; private set; }
    public MemoryBuffer1D<GPURigidBodyGeom, Stride1D.Dense> RigidBodyGeoms { get; private set; }
    public MemoryBuffer1D<GPURevoluteJoint, Stride1D.Dense> RevoluteJoints { get; private set; }
    public MemoryBuffer1D<GPUJointConstraint, Stride1D.Dense> JointConstraints { get; private set; }
    public MemoryBuffer1D<GPUContactConstraint, Stride1D.Dense> ContactConstraints { get; private set; }

    public int ParticleCount { get; private set; }
    public int RodCount { get; private set; }
    public int AngleCount { get; private set; }
    public int MotorCount { get; private set; }
    public int CircleCount { get; private set; }
    public int CapsuleCount { get; private set; }
    public int OBBCount { get; private set; }

    // Rigid body counts
    public int RigidBodyCount { get; private set; }
    public int RigidBodyGeomCount { get; private set; }
    public int RevoluteJointCount { get; private set; }
    public int MaxContactConstraints { get; private set; }
    public int MaxGeomsPerBody { get; private set; }

    public GPUWorldState(Accelerator accelerator, int particleCapacity = 512, int rigidBodyCapacity = 64)
    {
        _accelerator = accelerator;

        // Allocate particle buffers
        PosX = accelerator.Allocate1D<float>(particleCapacity);
        PosY = accelerator.Allocate1D<float>(particleCapacity);
        VelX = accelerator.Allocate1D<float>(particleCapacity);
        VelY = accelerator.Allocate1D<float>(particleCapacity);
        InvMass = accelerator.Allocate1D<float>(particleCapacity);
        Radius = accelerator.Allocate1D<float>(particleCapacity);
        ForceX = accelerator.Allocate1D<float>(particleCapacity);
        ForceY = accelerator.Allocate1D<float>(particleCapacity);
        PrevPosX = accelerator.Allocate1D<float>(particleCapacity);
        PrevPosY = accelerator.Allocate1D<float>(particleCapacity);

        // Allocate constraint buffers (initial size)
        Rods = accelerator.Allocate1D<GPURod>(256);
        Angles = accelerator.Allocate1D<GPUAngle>(128);
        Motors = accelerator.Allocate1D<GPUMotorAngle>(32);

        // Allocate collider buffers
        Circles = accelerator.Allocate1D<GPUCircleCollider>(64);
        Capsules = accelerator.Allocate1D<GPUCapsuleCollider>(64);
        OBBs = accelerator.Allocate1D<GPUOBBCollider>(64);

        // Allocate rigid body buffers
        RigidBodies = accelerator.Allocate1D<GPURigidBody>(rigidBodyCapacity);
        RigidBodyGeoms = accelerator.Allocate1D<GPURigidBodyGeom>(rigidBodyCapacity * 4); // Assume max 4 geoms per body
        RevoluteJoints = accelerator.Allocate1D<GPURevoluteJoint>(rigidBodyCapacity);
        JointConstraints = accelerator.Allocate1D<GPUJointConstraint>(rigidBodyCapacity);

        // Contact constraints: bodies * maxGeoms * (circles + capsules + obbs)
        // Initial allocation, will grow if needed
        MaxGeomsPerBody = 4;
        MaxContactConstraints = rigidBodyCapacity * MaxGeomsPerBody * 64; // Max 64 colliders
        ContactConstraints = accelerator.Allocate1D<GPUContactConstraint>(MaxContactConstraints);
    }

    /// <summary>
    /// Uploads world state from CPU to GPU.
    /// </summary>
    public void UploadFrom(WorldState world)
    {
        ParticleCount = world.ParticleCount;

        // Upload particle data (only copy actual particle count, not full capacity)
        PosX.View.SubView(0, ParticleCount).CopyFromCPU(world.PosX.ToArray());
        PosY.View.SubView(0, ParticleCount).CopyFromCPU(world.PosY.ToArray());
        VelX.View.SubView(0, ParticleCount).CopyFromCPU(world.VelX.ToArray());
        VelY.View.SubView(0, ParticleCount).CopyFromCPU(world.VelY.ToArray());
        InvMass.View.SubView(0, ParticleCount).CopyFromCPU(world.InvMass.ToArray());
        Radius.View.SubView(0, ParticleCount).CopyFromCPU(world.Radius.ToArray());
        ForceX.View.SubView(0, ParticleCount).CopyFromCPU(world.ForceX.ToArray());
        ForceY.View.SubView(0, ParticleCount).CopyFromCPU(world.ForceY.ToArray());
        PrevPosX.View.SubView(0, ParticleCount).CopyFromCPU(world.PrevPosX.ToArray());
        PrevPosY.View.SubView(0, ParticleCount).CopyFromCPU(world.PrevPosY.ToArray());

        // Upload constraints
        RodCount = world.Rods.Count;
        if (RodCount > Rods.Length)
            Rods = _accelerator.Allocate1D<GPURod>(RodCount * 2);
        if (RodCount > 0)
            Rods.View.SubView(0, RodCount).CopyFromCPU(ConvertRods(world.Rods));

        AngleCount = world.Angles.Count;
        if (AngleCount > Angles.Length)
            Angles = _accelerator.Allocate1D<GPUAngle>(AngleCount * 2);
        if (AngleCount > 0)
            Angles.View.SubView(0, AngleCount).CopyFromCPU(ConvertAngles(world.Angles));

        MotorCount = world.Motors.Count;
        if (MotorCount > Motors.Length)
            Motors = _accelerator.Allocate1D<GPUMotorAngle>(MotorCount * 2);
        if (MotorCount > 0)
            Motors.View.SubView(0, MotorCount).CopyFromCPU(ConvertMotors(world.Motors));

        // Upload colliders
        CircleCount = world.Circles.Count;
        if (CircleCount > Circles.Length)
            Circles = _accelerator.Allocate1D<GPUCircleCollider>(CircleCount * 2);
        if (CircleCount > 0)
            Circles.View.SubView(0, CircleCount).CopyFromCPU(ConvertCircles(world.Circles));

        CapsuleCount = world.Capsules.Count;
        if (CapsuleCount > Capsules.Length)
            Capsules = _accelerator.Allocate1D<GPUCapsuleCollider>(CapsuleCount * 2);
        if (CapsuleCount > 0)
            Capsules.View.SubView(0, CapsuleCount).CopyFromCPU(ConvertCapsules(world.Capsules));

        OBBCount = world.Obbs.Count;
        if (OBBCount > OBBs.Length)
            OBBs = _accelerator.Allocate1D<GPUOBBCollider>(OBBCount * 2);
        if (OBBCount > 0)
            OBBs.View.SubView(0, OBBCount).CopyFromCPU(ConvertOBBs(world.Obbs));

        // Upload rigid bodies
        RigidBodyCount = world.RigidBodies.Count;
        if (RigidBodyCount > RigidBodies.Length)
            RigidBodies = _accelerator.Allocate1D<GPURigidBody>(RigidBodyCount * 2);
        if (RigidBodyCount > 0)
            RigidBodies.View.SubView(0, RigidBodyCount).CopyFromCPU(ConvertRigidBodies(world.RigidBodies));

        // Upload rigid body geoms
        RigidBodyGeomCount = world.RigidBodyGeoms.Count;
        if (RigidBodyGeomCount > RigidBodyGeoms.Length)
            RigidBodyGeoms = _accelerator.Allocate1D<GPURigidBodyGeom>(RigidBodyGeomCount * 2);
        if (RigidBodyGeomCount > 0)
            RigidBodyGeoms.View.SubView(0, RigidBodyGeomCount).CopyFromCPU(ConvertRigidBodyGeoms(world.RigidBodyGeoms));

        // Upload revolute joints
        RevoluteJointCount = world.RevoluteJoints.Count;
        if (RevoluteJointCount > RevoluteJoints.Length)
            RevoluteJoints = _accelerator.Allocate1D<GPURevoluteJoint>(RevoluteJointCount * 2);
        if (RevoluteJointCount > 0)
            RevoluteJoints.View.SubView(0, RevoluteJointCount).CopyFromCPU(ConvertRevoluteJoints(world.RevoluteJoints));

        // Ensure joint constraints buffer is sized correctly
        if (RevoluteJointCount > JointConstraints.Length)
            JointConstraints = _accelerator.Allocate1D<GPUJointConstraint>(RevoluteJointCount * 2);

        // Calculate max geoms per body for contact detection
        MaxGeomsPerBody = 1;
        foreach (var rb in world.RigidBodies)
        {
            if (rb.GeomCount > MaxGeomsPerBody)
                MaxGeomsPerBody = rb.GeomCount;
        }

        // Ensure contact constraints buffer is sized correctly
        int requiredContacts = RigidBodyCount * MaxGeomsPerBody * (CircleCount + CapsuleCount + OBBCount);
        if (requiredContacts > ContactConstraints.Length)
        {
            ContactConstraints.Dispose();
            MaxContactConstraints = requiredContacts * 2;
            ContactConstraints = _accelerator.Allocate1D<GPUContactConstraint>(MaxContactConstraints);
        }
    }

    /// <summary>
    /// Downloads world state from GPU to CPU.
    /// </summary>
    public void DownloadTo(WorldState world)
    {
        // Download particle data
        var posXArray = PosX.GetAsArray1D();
        var posYArray = PosY.GetAsArray1D();
        var velXArray = VelX.GetAsArray1D();
        var velYArray = VelY.GetAsArray1D();
        var prevPosXArray = PrevPosX.GetAsArray1D();
        var prevPosYArray = PrevPosY.GetAsArray1D();

        for (int i = 0; i < ParticleCount; i++)
        {
            world.PosX[i] = posXArray[i];
            world.PosY[i] = posYArray[i];
            world.VelX[i] = velXArray[i];
            world.VelY[i] = velYArray[i];
            world.PrevPosX[i] = prevPosXArray[i];
            world.PrevPosY[i] = prevPosYArray[i];
        }

        // Download constraint lambdas
        var rods = Rods.GetAsArray1D();
        for (int i = 0; i < RodCount; i++)
        {
            var rod = world.Rods[i];
            rod.Lambda = rods[i].Lambda;
            world.Rods[i] = rod;
        }

        var angles = Angles.GetAsArray1D();
        for (int i = 0; i < AngleCount; i++)
        {
            var angle = world.Angles[i];
            angle.Lambda = angles[i].Lambda;
            world.Angles[i] = angle;
        }

        var motors = Motors.GetAsArray1D();
        for (int i = 0; i < MotorCount; i++)
        {
            var motor = world.Motors[i];
            motor.Lambda = motors[i].Lambda;
            world.Motors[i] = motor;
        }

        // Download rigid bodies
        if (RigidBodyCount > 0)
        {
            var rigidBodies = RigidBodies.GetAsArray1D();
            for (int i = 0; i < RigidBodyCount; i++)
            {
                world.RigidBodies[i] = rigidBodies[i].ToCPU();
            }
        }
    }

    // Conversion helpers
    private GPURod[] ConvertRods(System.Collections.Generic.List<Rod> rods)
    {
        var result = new GPURod[rods.Count];
        for (int i = 0; i < rods.Count; i++)
            result[i] = new GPURod(rods[i]);
        return result;
    }

    private GPUAngle[] ConvertAngles(System.Collections.Generic.List<Angle> angles)
    {
        var result = new GPUAngle[angles.Count];
        for (int i = 0; i < angles.Count; i++)
            result[i] = new GPUAngle(angles[i]);
        return result;
    }

    private GPUMotorAngle[] ConvertMotors(System.Collections.Generic.List<MotorAngle> motors)
    {
        var result = new GPUMotorAngle[motors.Count];
        for (int i = 0; i < motors.Count; i++)
            result[i] = new GPUMotorAngle(motors[i]);
        return result;
    }

    private GPUCircleCollider[] ConvertCircles(System.Collections.Generic.List<CircleCollider> circles)
    {
        var result = new GPUCircleCollider[circles.Count];
        for (int i = 0; i < circles.Count; i++)
            result[i] = new GPUCircleCollider(circles[i]);
        return result;
    }

    private GPUCapsuleCollider[] ConvertCapsules(System.Collections.Generic.List<CapsuleCollider> capsules)
    {
        var result = new GPUCapsuleCollider[capsules.Count];
        for (int i = 0; i < capsules.Count; i++)
            result[i] = new GPUCapsuleCollider(capsules[i]);
        return result;
    }

    private GPUOBBCollider[] ConvertOBBs(System.Collections.Generic.List<OBBCollider> obbs)
    {
        var result = new GPUOBBCollider[obbs.Count];
        for (int i = 0; i < obbs.Count; i++)
            result[i] = new GPUOBBCollider(obbs[i]);
        return result;
    }

    private GPURigidBody[] ConvertRigidBodies(System.Collections.Generic.List<RigidBody> bodies)
    {
        var result = new GPURigidBody[bodies.Count];
        for (int i = 0; i < bodies.Count; i++)
            result[i] = new GPURigidBody(bodies[i]);
        return result;
    }

    private GPURigidBodyGeom[] ConvertRigidBodyGeoms(System.Collections.Generic.List<RigidBodyGeom> geoms)
    {
        var result = new GPURigidBodyGeom[geoms.Count];
        for (int i = 0; i < geoms.Count; i++)
            result[i] = new GPURigidBodyGeom(geoms[i]);
        return result;
    }

    private GPURevoluteJoint[] ConvertRevoluteJoints(System.Collections.Generic.List<RevoluteJoint> joints)
    {
        var result = new GPURevoluteJoint[joints.Count];
        for (int i = 0; i < joints.Count; i++)
            result[i] = new GPURevoluteJoint(joints[i]);
        return result;
    }

    public void Dispose()
    {
        PosX?.Dispose();
        PosY?.Dispose();
        VelX?.Dispose();
        VelY?.Dispose();
        InvMass?.Dispose();
        Radius?.Dispose();
        ForceX?.Dispose();
        ForceY?.Dispose();
        PrevPosX?.Dispose();
        PrevPosY?.Dispose();
        Rods?.Dispose();
        Angles?.Dispose();
        Motors?.Dispose();
        Circles?.Dispose();
        Capsules?.Dispose();
        OBBs?.Dispose();
        RigidBodies?.Dispose();
        RigidBodyGeoms?.Dispose();
        RevoluteJoints?.Dispose();
        JointConstraints?.Dispose();
        ContactConstraints?.Dispose();
    }
}
