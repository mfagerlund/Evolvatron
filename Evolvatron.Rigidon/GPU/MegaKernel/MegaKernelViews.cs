using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;

namespace Evolvatron.Core.GPU.MegaKernel;

/// <summary>
/// Bundles physics ArrayViews into a single kernel parameter.
/// Bypasses ILGPU's 16-parameter Action limit.
/// ArrayView is a value type so structs containing them are valid kernel parameters.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct PhysicsViews
{
    public ArrayView<GPURigidBody> Bodies;
    public ArrayView<GPURigidBodyGeom> Geoms;
    public ArrayView<GPURevoluteJoint> Joints;
    public ArrayView<GPUJointConstraint> JointConstraints;
    public ArrayView<GPUContactConstraint> Contacts;
    public ArrayView<GPUCachedContactImpulse> ContactCache;
    public ArrayView<int> ContactCounts;
    public ArrayView<GPUOBBCollider> SharedOBBColliders;
}
