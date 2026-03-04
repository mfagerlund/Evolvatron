using Evolvatron.Core;

namespace Evolvatron.Evolvion.TrajectoryOptimization;

/// <summary>
/// Captures and restores the dynamic state of a WorldState for rollback during optimization.
/// Only stores what changes during simulation — positions, velocities, angles.
/// Static geometry, constraints, and topology are unchanged.
/// </summary>
public sealed class WorldStateSnapshot
{
    // Particle dynamic state
    private float[] _posX = Array.Empty<float>();
    private float[] _posY = Array.Empty<float>();
    private float[] _velX = Array.Empty<float>();
    private float[] _velY = Array.Empty<float>();
    private int _particleCount;

    // Rigid body dynamic state
    private RigidBodyState[] _rigidBodies = Array.Empty<RigidBodyState>();
    private int _rigidBodyCount;

    private struct RigidBodyState
    {
        public float X, Y, Angle;
        public float VelX, VelY, AngularVel;
    }

    public static WorldStateSnapshot Capture(WorldState world)
    {
        var snap = new WorldStateSnapshot();
        snap.CaptureFrom(world);
        return snap;
    }

    public void CaptureFrom(WorldState world)
    {
        // Particles
        _particleCount = world.ParticleCount;
        EnsureCapacity(ref _posX, _particleCount);
        EnsureCapacity(ref _posY, _particleCount);
        EnsureCapacity(ref _velX, _particleCount);
        EnsureCapacity(ref _velY, _particleCount);

        world.PosX.CopyTo(_posX.AsSpan());
        world.PosY.CopyTo(_posY.AsSpan());
        world.VelX.CopyTo(_velX.AsSpan());
        world.VelY.CopyTo(_velY.AsSpan());

        // Rigid bodies
        _rigidBodyCount = world.RigidBodies.Count;
        EnsureCapacity(ref _rigidBodies, _rigidBodyCount);
        for (int i = 0; i < _rigidBodyCount; i++)
        {
            var rb = world.RigidBodies[i];
            _rigidBodies[i] = new RigidBodyState
            {
                X = rb.X, Y = rb.Y, Angle = rb.Angle,
                VelX = rb.VelX, VelY = rb.VelY, AngularVel = rb.AngularVel
            };
        }

    }

    public void Restore(WorldState world)
    {
        // Particles
        _posX.AsSpan(0, _particleCount).CopyTo(world.PosX);
        _posY.AsSpan(0, _particleCount).CopyTo(world.PosY);
        _velX.AsSpan(0, _particleCount).CopyTo(world.VelX);
        _velY.AsSpan(0, _particleCount).CopyTo(world.VelY);

        // Rigid bodies
        for (int i = 0; i < _rigidBodyCount; i++)
        {
            var s = _rigidBodies[i];
            var rb = world.RigidBodies[i];
            rb.X = s.X; rb.Y = s.Y; rb.Angle = s.Angle;
            rb.VelX = s.VelX; rb.VelY = s.VelY; rb.AngularVel = s.AngularVel;
            world.RigidBodies[i] = rb;
        }
    }

    private static void EnsureCapacity<T>(ref T[] array, int needed)
    {
        if (array.Length < needed)
            array = new T[needed];
    }
}
