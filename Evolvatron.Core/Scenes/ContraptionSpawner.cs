using System;
using System.Collections.Generic;

namespace Evolvatron.Core.Scenes;

/// <summary>
/// Spawns random contraptions made from particles and constraints.
/// Used for the funnel demo to create varied falling objects.
/// </summary>
public class ContraptionSpawner
{
    private readonly Random _rng;

    public ContraptionSpawner(int seed = 0)
    {
        _rng = seed == 0 ? new Random() : new Random(seed);
    }

    /// <summary>
    /// Spawns a random contraption in the specified area.
    /// Returns the particle indices that make up this contraption.
    /// </summary>
    public List<int> SpawnRandomContraption(
        WorldState world,
        float minX, float maxX,
        float minY, float maxY,
        int minParticles = 5,
        int maxParticles = 12)
    {
        int particleCount = _rng.Next(minParticles, maxParticles + 1);
        List<int> indices = new List<int>(particleCount);

        // Random center point
        float centerX = Lerp(minX, maxX, (float)_rng.NextDouble());
        float centerY = Lerp(minY, maxY, (float)_rng.NextDouble());

        // Random configuration
        float spread = 0.5f + (float)_rng.NextDouble() * 1.5f;
        float minMass = 0.5f;
        float maxMass = 3f;
        float minRadius = 0.08f;
        float maxRadius = 0.15f;

        // Create particles in a cluster
        for (int i = 0; i < particleCount; i++)
        {
            float offsetX = ((float)_rng.NextDouble() - 0.5f) * spread;
            float offsetY = ((float)_rng.NextDouble() - 0.5f) * spread;
            float mass = Lerp(minMass, maxMass, (float)_rng.NextDouble());
            float radius = Lerp(minRadius, maxRadius, (float)_rng.NextDouble());

            int idx = world.AddParticle(
                x: centerX + offsetX,
                y: centerY + offsetY,
                vx: 0f,
                vy: 0f,
                mass: mass,
                radius: radius
            );

            indices.Add(idx);
        }

        // Connect with rods (spanning tree to ensure connectivity)
        ConnectParticlesWithRods(world, indices);

        // Maybe add some angle constraints for rigidity
        if (_rng.NextDouble() > 0.5f && indices.Count >= 3)
        {
            AddAngleConstraints(world, indices, maxAngles: 3);
        }

        // Rarely, add a motor
        if (_rng.NextDouble() > 0.9f && indices.Count >= 3)
        {
            AddMotorConstraint(world, indices);
        }

        return indices;
    }

    /// <summary>
    /// Connects particles with rods using a random spanning tree approach.
    /// </summary>
    private void ConnectParticlesWithRods(WorldState world, List<int> indices)
    {
        if (indices.Count < 2)
            return;

        // Create spanning tree: start with first particle, randomly connect others
        List<int> connected = new List<int> { indices[0] };
        List<int> unconnected = new List<int>(indices);
        unconnected.RemoveAt(0);

        while (unconnected.Count > 0)
        {
            // Pick random particle from connected set
            int fromIdx = connected[_rng.Next(connected.Count)];

            // Pick random particle from unconnected set
            int toPickIdx = _rng.Next(unconnected.Count);
            int toIdx = unconnected[toPickIdx];

            // Add rod
            float dx = world.PosX[fromIdx] - world.PosX[toIdx];
            float dy = world.PosY[fromIdx] - world.PosY[toIdx];
            float restLength = MathF.Sqrt(dx * dx + dy * dy);

            // Add some randomness to compliance
            float compliance = _rng.NextDouble() < 0.8 ? 0f : (float)_rng.NextDouble() * 1e-5f;

            world.Rods.Add(new Rod(fromIdx, toIdx, restLength, compliance));

            // Move to connected
            connected.Add(toIdx);
            unconnected.RemoveAt(toPickIdx);
        }

        // Add a few extra random rods for stability
        int extraRods = _rng.Next(0, Math.Min(3, indices.Count - 1));
        for (int i = 0; i < extraRods; i++)
        {
            int idx1 = indices[_rng.Next(indices.Count)];
            int idx2 = indices[_rng.Next(indices.Count)];

            if (idx1 == idx2)
                continue;

            float dx = world.PosX[idx1] - world.PosX[idx2];
            float dy = world.PosY[idx1] - world.PosY[idx2];
            float restLength = MathF.Sqrt(dx * dx + dy * dy);

            world.Rods.Add(new Rod(idx1, idx2, restLength, compliance: 0f));
        }
    }

    /// <summary>
    /// Adds a few angle constraints for structural rigidity.
    /// </summary>
    private void AddAngleConstraints(WorldState world, List<int> indices, int maxAngles)
    {
        int angleCount = Math.Min(maxAngles, indices.Count / 2);

        for (int n = 0; n < angleCount; n++)
        {
            // Pick three random particles
            int i = indices[_rng.Next(indices.Count)];
            int j = indices[_rng.Next(indices.Count)];
            int k = indices[_rng.Next(indices.Count)];

            if (i == j || j == k || i == k)
                continue;

            // Compute current angle
            float e1x = world.PosX[i] - world.PosX[j];
            float e1y = world.PosY[i] - world.PosY[j];
            float e2x = world.PosX[k] - world.PosX[j];
            float e2y = world.PosY[k] - world.PosY[j];

            float theta = Math2D.AngleBetween(e1x, e1y, e2x, e2y);

            world.Angles.Add(new Angle(i, j, k, theta, compliance: 0f));
        }
    }

    /// <summary>
    /// Adds a single motorized angle constraint (rare, for variety).
    /// </summary>
    private void AddMotorConstraint(WorldState world, List<int> indices)
    {
        if (indices.Count < 3)
            return;

        int i = indices[_rng.Next(indices.Count)];
        int j = indices[_rng.Next(indices.Count)];
        int k = indices[_rng.Next(indices.Count)];

        if (i == j || j == k || i == k)
            return;

        // Compute current angle
        float e1x = world.PosX[i] - world.PosX[j];
        float e1y = world.PosY[i] - world.PosY[j];
        float e2x = world.PosX[k] - world.PosX[j];
        float e2y = world.PosY[k] - world.PosY[j];

        float theta = Math2D.AngleBetween(e1x, e1y, e2x, e2y);

        // Random target variation
        float targetOffset = ((float)_rng.NextDouble() - 0.5f) * 0.5f;
        float target = theta + targetOffset;

        world.Motors.Add(new MotorAngle(i, j, k, target, compliance: 1e-6f));
    }

    private static float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }
}
