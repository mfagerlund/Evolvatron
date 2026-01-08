using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// Manages GPU memory for N parallel physics worlds.
/// All arrays are batched: [world0_item0, world0_item1, world1_item0, world1_item1, ...]
/// </summary>
public class GPUBatchedWorldState : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly GPUBatchedWorldConfig _config;
    private bool _disposed;

    // Batched rigid bodies: N * RigidBodiesPerWorld entries
    public MemoryBuffer1D<GPURigidBody, Stride1D.Dense> RigidBodies { get; private set; } = null!;

    // Batched geoms: N * GeomsPerWorld entries
    public MemoryBuffer1D<GPURigidBodyGeom, Stride1D.Dense> Geoms { get; private set; } = null!;

    // Batched joints: N * JointsPerWorld entries
    public MemoryBuffer1D<GPURevoluteJoint, Stride1D.Dense> Joints { get; private set; } = null!;

    // Batched joint constraints (solver state)
    public MemoryBuffer1D<GPUJointConstraint, Stride1D.Dense> JointConstraints { get; private set; } = null!;

    // Batched contact constraints: N * MaxContactsPerWorld entries
    public MemoryBuffer1D<GPUContactConstraint, Stride1D.Dense> ContactConstraints { get; private set; } = null!;
    public MemoryBuffer1D<int, Stride1D.Dense> ContactCounts { get; private set; } = null!; // per-world count

    // Shared static colliders (same for all worlds)
    public MemoryBuffer1D<GPUOBBCollider, Stride1D.Dense> SharedOBBColliders { get; private set; } = null!;

    public GPUBatchedWorldConfig Config => _config;

    public GPUBatchedWorldState(Accelerator accelerator, GPUBatchedWorldConfig config)
    {
        _accelerator = accelerator;
        _config = config;
        AllocateBuffers();
    }

    private void AllocateBuffers()
    {
        RigidBodies = _accelerator.Allocate1D<GPURigidBody>(_config.TotalRigidBodies);
        Geoms = _accelerator.Allocate1D<GPURigidBodyGeom>(_config.TotalGeoms);
        Joints = _accelerator.Allocate1D<GPURevoluteJoint>(_config.TotalJoints);
        JointConstraints = _accelerator.Allocate1D<GPUJointConstraint>(_config.TotalJoints);
        ContactConstraints = _accelerator.Allocate1D<GPUContactConstraint>(_config.TotalContacts);
        ContactCounts = _accelerator.Allocate1D<int>(_config.WorldCount);
        SharedOBBColliders = _accelerator.Allocate1D<GPUOBBCollider>(_config.SharedColliderCount);
    }

    /// <summary>
    /// Upload a rocket template to ALL worlds (same initial state).
    /// </summary>
    public void UploadRocketTemplate(
        GPURigidBody[] templateBodies,
        GPURigidBodyGeom[] templateGeoms,
        GPURevoluteJoint[] templateJoints)
    {
        // Expand template to all worlds
        var allBodies = new GPURigidBody[_config.TotalRigidBodies];
        var allGeoms = new GPURigidBodyGeom[_config.TotalGeoms];
        var allJoints = new GPURevoluteJoint[_config.TotalJoints];

        for (int w = 0; w < _config.WorldCount; w++)
        {
            for (int i = 0; i < templateBodies.Length && i < _config.RigidBodiesPerWorld; i++)
            {
                allBodies[_config.GetRigidBodyIndex(w, i)] = templateBodies[i];
            }
            for (int i = 0; i < templateGeoms.Length && i < _config.GeomsPerWorld; i++)
            {
                allGeoms[_config.GetGeomIndex(w, i)] = templateGeoms[i];
            }
            for (int i = 0; i < templateJoints.Length && i < _config.JointsPerWorld; i++)
            {
                allJoints[_config.GetJointIndex(w, i)] = templateJoints[i];
            }
        }

        RigidBodies.CopyFromCPU(allBodies);
        Geoms.CopyFromCPU(allGeoms);
        Joints.CopyFromCPU(allJoints);
    }

    /// <summary>
    /// Upload shared static colliders (arena walls).
    /// </summary>
    public void UploadSharedColliders(GPUOBBCollider[] colliders)
    {
        SharedOBBColliders.CopyFromCPU(colliders);
    }

    /// <summary>
    /// Download rigid body state for a specific world.
    /// </summary>
    public GPURigidBody[] DownloadWorldBodies(int worldIdx)
    {
        var allBodies = RigidBodies.GetAsArray1D();
        var worldBodies = new GPURigidBody[_config.RigidBodiesPerWorld];
        for (int i = 0; i < _config.RigidBodiesPerWorld; i++)
        {
            worldBodies[i] = allBodies[_config.GetRigidBodyIndex(worldIdx, i)];
        }
        return worldBodies;
    }

    /// <summary>
    /// Download all rigid bodies (for debugging/visualization).
    /// </summary>
    public GPURigidBody[] DownloadAllBodies()
    {
        return RigidBodies.GetAsArray1D();
    }

    /// <summary>
    /// Reset contact counts to zero for all worlds.
    /// </summary>
    public void ClearContactCounts()
    {
        ContactCounts.MemSetToZero();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        RigidBodies?.Dispose();
        Geoms?.Dispose();
        Joints?.Dispose();
        JointConstraints?.Dispose();
        ContactConstraints?.Dispose();
        ContactCounts?.Dispose();
        SharedOBBColliders?.Dispose();
    }
}
