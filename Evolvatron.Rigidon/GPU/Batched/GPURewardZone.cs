using System.Runtime.InteropServices;

namespace Evolvatron.Core.GPU.Batched;

/// <summary>
/// GPU-compatible reward zone definition.
/// Represents an attractor (positive magnitude) or danger zone (negative magnitude)
/// with distance-based falloff from core zone to influence region.
///
/// Runtime behavior (per-world, per-zone):
/// - ClosestDistance buffer tracks min signed distance ever achieved
/// - Signed distance: negative = inside core, 0 = at core edge, positive = outside
/// - Reward = magnitude * falloff(closestDistance) + contactBonus if closestDistance &lt;= 0
/// - Falloff is quadratic: (1 - t²) where t = normalized distance from core to influence edge
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct GPURewardZone
{
    /// <summary>Center X position in world space.</summary>
    public float CenterX;

    /// <summary>Center Y position in world space.</summary>
    public float CenterY;

    /// <summary>Half-extent X of the core zone.</summary>
    public float HalfExtentX;

    /// <summary>Half-extent Y of the core zone.</summary>
    public float HalfExtentY;

    /// <summary>
    /// Reward magnitude. Positive = attractor, negative = repulsor/danger.
    /// The actual reward is magnitude * falloff(closestDistance).
    /// </summary>
    public float Magnitude;

    /// <summary>
    /// Multiplier on half-extents for the influence falloff region.
    /// E.g., 3.0 means influence extends 3x the core size.
    /// </summary>
    public float InfluenceFactor;

    /// <summary>
    /// One-time bonus/penalty when agent first enters the core zone.
    /// Triggered when closestDistance transitions to &lt;= 0.
    /// </summary>
    public float ContactBonus;
}
