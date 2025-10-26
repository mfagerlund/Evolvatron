using System;

namespace Evolvatron.Core.Scenes;

/// <summary>
/// Builds the "funnel" demo scene with converging walls that guide contraptions
/// down to a landing area.
/// </summary>
public static class FunnelSceneBuilder
{
    /// <summary>
    /// Creates a funnel scene with V-shaped walls leading to a landing pad.
    /// </summary>
    /// <param name="world">World to populate</param>
    /// <param name="funnelWidth">Width at top of funnel</param>
    /// <param name="funnelHeight">Height of funnel walls</param>
    /// <param name="funnelAngleDeg">Angle of funnel walls from vertical (degrees)</param>
    /// <param name="groundY">Y position of ground level</param>
    /// <param name="padWidth">Width of landing pad</param>
    public static void BuildFunnelScene(
        WorldState world,
        float funnelWidth = 20f,
        float funnelHeight = 15f,
        float funnelAngleDeg = 30f,
        float groundY = -10f,
        float padWidth = 4f)
    {
        float wallThickness = 0.5f;
        float funnelTop = groundY + funnelHeight;
        float funnelAngleRad = funnelAngleDeg * MathF.PI / 180f;

        // Calculate funnel geometry
        float halfWidthTop = funnelWidth * 0.5f;
        float halfWidthBottom = padWidth * 0.5f;
        float wallSlope = MathF.Tan(funnelAngleRad);

        // Left funnel wall (OBB)
        float leftWallCenterX = -(halfWidthTop + halfWidthBottom) * 0.5f;
        float leftWallCenterY = (funnelTop + groundY) * 0.5f;
        float leftWallLength = funnelHeight / MathF.Cos(funnelAngleRad);

        world.Obbs.Add(OBBCollider.FromAngle(
            cx: leftWallCenterX,
            cy: leftWallCenterY,
            hx: leftWallLength * 0.5f,
            hy: wallThickness * 0.5f,
            angleRad: -funnelAngleRad
        ));

        // Right funnel wall (OBB)
        float rightWallCenterX = (halfWidthTop + halfWidthBottom) * 0.5f;
        float rightWallCenterY = leftWallCenterY;

        world.Obbs.Add(OBBCollider.FromAngle(
            cx: rightWallCenterX,
            cy: rightWallCenterY,
            hx: leftWallLength * 0.5f,
            hy: wallThickness * 0.5f,
            angleRad: funnelAngleRad
        ));

        // Ground (wide OBB)
        world.Obbs.Add(OBBCollider.AxisAligned(
            cx: 0f,
            cy: groundY - 0.5f,
            hx: funnelWidth * 2f,
            hy: 0.5f
        ));

        // Landing pad (highlighted area)
        world.Obbs.Add(OBBCollider.AxisAligned(
            cx: 0f,
            cy: groundY + 0.3f,
            hx: padWidth * 0.5f,
            hy: 0.3f
        ));

        // Add some bumpers for visual interest
        AddBumpers(world, groundY, funnelTop, halfWidthTop);
    }

    /// <summary>
    /// Adds circular and capsule bumpers throughout the funnel.
    /// </summary>
    private static void AddBumpers(WorldState world, float groundY, float funnelTop, float maxX)
    {
        Random rng = new Random(42); // Fixed seed for determinism

        // A few circles
        for (int i = 0; i < 3; i++)
        {
            float x = (float)(rng.NextDouble() * maxX * 1.5 - maxX * 0.75);
            float y = groundY + (funnelTop - groundY) * (0.3f + (float)rng.NextDouble() * 0.6f);
            float radius = 0.3f + (float)rng.NextDouble() * 0.4f;

            world.Circles.Add(new CircleCollider(x, y, radius));
        }

        // A couple of capsule obstacles
        for (int i = 0; i < 2; i++)
        {
            float x = (float)(rng.NextDouble() * maxX * 1.2 - maxX * 0.6);
            float y = groundY + (funnelTop - groundY) * (0.2f + (float)rng.NextDouble() * 0.5f);
            float angle = (float)rng.NextDouble() * MathF.PI;
            float len = 0.8f + (float)rng.NextDouble() * 1.2f;

            float halfLen = len * 0.5f;
            float x1 = x - MathF.Cos(angle) * halfLen;
            float y1 = y - MathF.Sin(angle) * halfLen;
            float x2 = x + MathF.Cos(angle) * halfLen;
            float y2 = y + MathF.Sin(angle) * halfLen;

            world.Capsules.Add(CapsuleCollider.FromEndpoints(x1, y1, x2, y2, radius: 0.2f));
        }
    }

    /// <summary>
    /// Returns the spawn area bounds for the funnel scene.
    /// </summary>
    public static void GetSpawnBounds(
        float funnelWidth,
        float funnelHeight,
        float groundY,
        out float minX, out float maxX,
        out float minY, out float maxY)
    {
        float halfWidth = funnelWidth * 0.4f; // Spawn in narrower area
        minX = -halfWidth;
        maxX = halfWidth;
        minY = groundY + funnelHeight + 2f; // Above funnel
        maxY = minY + 5f;
    }

    /// <summary>
    /// Returns the landing pad bounds.
    /// </summary>
    public static void GetPadBounds(
        float groundY,
        float padWidth,
        out float minX, out float maxX,
        out float minY, out float maxY)
    {
        float halfWidth = padWidth * 0.5f;
        minX = -halfWidth;
        maxX = halfWidth;
        minY = groundY - 0.5f;
        maxY = groundY + 1.5f;
    }

    /// <summary>
    /// Returns culling bounds - contraptions outside this area are removed.
    /// </summary>
    public static void GetCullBounds(
        float funnelWidth,
        float funnelHeight,
        float groundY,
        out float minX, out float maxX,
        out float minY, out float maxY)
    {
        minX = -funnelWidth * 2f;
        maxX = funnelWidth * 2f;
        minY = groundY - 20f;
        maxY = groundY + funnelHeight + 20f;
    }
}
