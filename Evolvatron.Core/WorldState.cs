using System;
using System.Collections.Generic;

namespace Evolvatron.Core;

/// <summary>
/// Represents the complete state of the simulation world.
/// Particles use Structure-of-Arrays (SoA) layout for performance.
/// Constraints and colliders are stored as lists of structs.
/// </summary>
public sealed class WorldState
{
    private float[] _posX;
    private float[] _posY;
    private float[] _velX;
    private float[] _velY;
    private float[] _invMass;
    private float[] _radius;
    private float[] _forceX;  // Accumulated forces (cleared each step)
    private float[] _forceY;
    private float[] _prevPosX; // For velocity stabilization
    private float[] _prevPosY;

    private int _particleCount;
    private int _capacity;

    /// <summary>
    /// Number of active particles in the simulation.
    /// </summary>
    public int ParticleCount => _particleCount;

    /// <summary>
    /// Current capacity of particle arrays.
    /// </summary>
    public int Capacity => _capacity;

    // Particle arrays (SoA)
    public Span<float> PosX => _posX.AsSpan(0, _particleCount);
    public Span<float> PosY => _posY.AsSpan(0, _particleCount);
    public Span<float> VelX => _velX.AsSpan(0, _particleCount);
    public Span<float> VelY => _velY.AsSpan(0, _particleCount);
    public Span<float> InvMass => _invMass.AsSpan(0, _particleCount);
    public Span<float> Radius => _radius.AsSpan(0, _particleCount);
    public Span<float> ForceX => _forceX.AsSpan(0, _particleCount);
    public Span<float> ForceY => _forceY.AsSpan(0, _particleCount);
    public Span<float> PrevPosX => _prevPosX.AsSpan(0, _particleCount);
    public Span<float> PrevPosY => _prevPosY.AsSpan(0, _particleCount);

    // Constraints
    public List<Rod> Rods { get; } = new();
    public List<Angle> Angles { get; } = new();
    public List<MotorAngle> Motors { get; } = new();

    // Static colliders
    public List<CircleCollider> Circles { get; } = new();
    public List<CapsuleCollider> Capsules { get; } = new();
    public List<OBBCollider> Obbs { get; } = new();

    // Rigid bodies (can have multiple circle geoms each)
    public List<RigidBody> RigidBodies { get; } = new();
    public List<RigidBodyGeom> RigidBodyGeoms { get; } = new();

    // Rigid body joints
    public List<RevoluteJoint> RevoluteJoints { get; } = new();

    /// <summary>
    /// Creates a world with specified initial particle capacity.
    /// </summary>
    public WorldState(int initialCapacity = 256)
    {
        _capacity = initialCapacity;
        _particleCount = 0;

        _posX = new float[_capacity];
        _posY = new float[_capacity];
        _velX = new float[_capacity];
        _velY = new float[_capacity];
        _invMass = new float[_capacity];
        _radius = new float[_capacity];
        _forceX = new float[_capacity];
        _forceY = new float[_capacity];
        _prevPosX = new float[_capacity];
        _prevPosY = new float[_capacity];
    }

    /// <summary>
    /// Adds a particle to the world. Returns the particle index.
    /// </summary>
    public int AddParticle(float x, float y, float vx, float vy, float mass, float radius)
    {
        if (_particleCount >= _capacity)
        {
            ResizeCapacity(_capacity * 2);
        }

        int idx = _particleCount++;
        _posX[idx] = x;
        _posY[idx] = y;
        _velX[idx] = vx;
        _velY[idx] = vy;
        _invMass[idx] = mass > 0f ? 1f / mass : 0f;
        _radius[idx] = radius;
        _forceX[idx] = 0f;
        _forceY[idx] = 0f;
        _prevPosX[idx] = x;
        _prevPosY[idx] = y;

        return idx;
    }

    /// <summary>
    /// Adds a pinned (static/immovable) particle. InvMass = 0.
    /// </summary>
    public int AddPinnedParticle(float x, float y, float radius)
    {
        return AddParticle(x, y, 0f, 0f, 0f, radius);
    }

    /// <summary>
    /// Removes all particles, constraints, and colliders.
    /// </summary>
    public void Clear()
    {
        _particleCount = 0;
        Rods.Clear();
        Angles.Clear();
        Motors.Clear();
        Circles.Clear();
        Capsules.Clear();
        Obbs.Clear();
        RigidBodies.Clear();
        RigidBodyGeoms.Clear();
        RevoluteJoints.Clear();
    }

    /// <summary>
    /// Clears accumulated forces on all particles.
    /// </summary>
    public void ClearForces()
    {
        Array.Clear(_forceX, 0, _particleCount);
        Array.Clear(_forceY, 0, _particleCount);
    }

    /// <summary>
    /// Saves current positions for velocity stabilization.
    /// </summary>
    public void SavePreviousPositions()
    {
        Array.Copy(_posX, _prevPosX, _particleCount);
        Array.Copy(_posY, _prevPosY, _particleCount);
    }

    /// <summary>
    /// Applies velocity stabilization: v = (p_new - p_prev)/dt * beta + v * (1-beta)
    /// </summary>
    public void StabilizeVelocities(float dt, float beta)
    {
        if (beta <= 0f) return;
        if (beta >= 1f)
        {
            // Full correction
            float invDt = 1f / dt;
            for (int i = 0; i < _particleCount; i++)
            {
                _velX[i] = (_posX[i] - _prevPosX[i]) * invDt;
                _velY[i] = (_posY[i] - _prevPosY[i]) * invDt;
            }
        }
        else
        {
            // Blended correction
            float invDt = 1f / dt;
            float oneMinusBeta = 1f - beta;
            for (int i = 0; i < _particleCount; i++)
            {
                float correctedVx = (_posX[i] - _prevPosX[i]) * invDt;
                float correctedVy = (_posY[i] - _prevPosY[i]) * invDt;
                _velX[i] = correctedVx * beta + _velX[i] * oneMinusBeta;
                _velY[i] = correctedVy * beta + _velY[i] * oneMinusBeta;
            }
        }
    }

    /// <summary>
    /// Applies global damping to all particles: v *= (1 - damping * dt).
    /// </summary>
    public void ApplyDamping(float damping, float dt)
    {
        if (damping <= 0f) return;
        float factor = MathF.Max(0f, 1f - damping * dt);
        for (int i = 0; i < _particleCount; i++)
        {
            _velX[i] *= factor;
            _velY[i] *= factor;
        }
    }

    private void ResizeCapacity(int newCapacity)
    {
        Array.Resize(ref _posX, newCapacity);
        Array.Resize(ref _posY, newCapacity);
        Array.Resize(ref _velX, newCapacity);
        Array.Resize(ref _velY, newCapacity);
        Array.Resize(ref _invMass, newCapacity);
        Array.Resize(ref _radius, newCapacity);
        Array.Resize(ref _forceX, newCapacity);
        Array.Resize(ref _forceY, newCapacity);
        Array.Resize(ref _prevPosX, newCapacity);
        Array.Resize(ref _prevPosY, newCapacity);
        _capacity = newCapacity;
    }
}
