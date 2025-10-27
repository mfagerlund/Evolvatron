using System;
using Xunit;

namespace Evolvatron.Tests;

/// <summary>
/// Finite-difference gradient verification for constraint Jacobians.
/// Following best practices: differentiate the raw angle function, NOT the wrapped constraint.
/// </summary>
public class FiniteDifferenceGradientCheck
{
    private const float Eps = 1e-5f; // Perturbation for finite differences
    private const float RelTolerance = 5e-3f; // Relative error tolerance (0.5%)
    private const float AbsTolerance = 1e-4f; // Absolute error tolerance (for near-zero gradients)

    [Theory]
    [InlineData(2.0f, 0.0f, 1.0f, 1.0f)]      // 45 degree angle
    [InlineData(1.0f, 0.0f, 0.0f, 1.0f)]      // 90 degree angle
    [InlineData(1.0f, 0.0f, -1.0f, 1.0f)]     // 135 degree angle
    [InlineData(1.5f, 0.5f, -0.5f, 1.5f)]     // Random configuration
    [InlineData(0.8f, -1.2f, 1.3f, 0.9f)]     // Another random config
    public void AngleGradients_MatchFiniteDifference(float ux, float uy, float vx, float vy)
    {
        // Compute analytical gradients (from XPBDSolver.SolveAngles)
        float uu = ux * ux + uy * uy;
        float vv = vx * vx + vy * vy;

        if (uu < 1e-6f || vv < 1e-6f)
            return; // Skip degenerate case

        float c = ux * vx + uy * vy;  // dot(u, v)
        float s = ux * vy - uy * vx;  // cross(u, v)

        float denom = uu * vv + 1e-12f;

        // Analytical gradients: ∂θ/∂u and ∂θ/∂v
        float dθ_du_x_analytical = (c * vy - s * vx) / denom;
        float dθ_du_y_analytical = (-c * vx - s * vy) / denom;
        float dθ_dv_x_analytical = (c * (-uy) - s * ux) / denom;
        float dθ_dv_y_analytical = (c * ux - s * uy) / denom;

        // Compute numerical gradients via finite differences (DON'T wrap!)
        var (dθ_du_x_numerical, dθ_du_y_numerical, dθ_dv_x_numerical, dθ_dv_y_numerical) =
            FD_AngleGrads(ux, uy, vx, vy, Eps);

        // Verify analytical matches numerical (use relative error for robustness)
        AssertGradientMatch(dθ_du_x_analytical, dθ_du_x_numerical, "∂θ/∂ux");
        AssertGradientMatch(dθ_du_y_analytical, dθ_du_y_numerical, "∂θ/∂uy");
        AssertGradientMatch(dθ_dv_x_analytical, dθ_dv_x_numerical, "∂θ/∂vx");
        AssertGradientMatch(dθ_dv_y_analytical, dθ_dv_y_numerical, "∂θ/∂vy");
    }

    [Fact]
    public void RodGradients_MatchFiniteDifference()
    {
        // Test rod constraint: C(p0, p1) = |p1 - p0| - restLength
        float p0x = 1.5f, p0y = 2.3f;
        float p1x = 3.7f, p1y = 5.1f;
        float restLength = 2.5f;

        float dx = p1x - p0x;
        float dy = p1y - p0y;
        float len = MathF.Sqrt(dx * dx + dy * dy);

        if (len < 1e-6f)
            return; // Degenerate

        // Analytical gradients: ∂C/∂p = (p1 - p0) / |p1 - p0|
        float gradP0x_analytical = -dx / len;
        float gradP0y_analytical = -dy / len;
        float gradP1x_analytical = dx / len;
        float gradP1y_analytical = dy / len;

        // Numerical gradients using central differences
        float RodConstraint(float px0, float py0, float px1, float py1)
        {
            float ddx = px1 - px0;
            float ddy = py1 - py0;
            return MathF.Sqrt(ddx * ddx + ddy * ddy) - restLength;
        }

        float gradP0x_numerical = (RodConstraint(p0x + Eps, p0y, p1x, p1y) - RodConstraint(p0x - Eps, p0y, p1x, p1y)) / (2f * Eps);
        float gradP0y_numerical = (RodConstraint(p0x, p0y + Eps, p1x, p1y) - RodConstraint(p0x, p0y - Eps, p1x, p1y)) / (2f * Eps);
        float gradP1x_numerical = (RodConstraint(p0x, p0y, p1x + Eps, p1y) - RodConstraint(p0x, p0y, p1x - Eps, p1y)) / (2f * Eps);
        float gradP1y_numerical = (RodConstraint(p0x, p0y, p1x, p1y + Eps) - RodConstraint(p0x, p0y, p1x, p1y - Eps)) / (2f * Eps);

        // Verify
        AssertGradientMatch(gradP0x_analytical, gradP0x_numerical, "Rod ∂C/∂p0x");
        AssertGradientMatch(gradP0y_analytical, gradP0y_numerical, "Rod ∂C/∂p0y");
        AssertGradientMatch(gradP1x_analytical, gradP1x_numerical, "Rod ∂C/∂p1x");
        AssertGradientMatch(gradP1y_analytical, gradP1y_numerical, "Rod ∂C/∂p1y");
    }

    [Theory]
    [InlineData(0.5f, 0.8f)]   // Various angles around the circle
    [InlineData(-1.2f, 1.5f)]
    [InlineData(2.1f, -0.3f)]
    [InlineData(-0.7f, -1.1f)]
    public void AngleGradients_RobustAcrossQuadrants(float ux, float uy)
    {
        // Test with v at various angles relative to u
        float[] vAngles = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f };

        foreach (float angle in vAngles)
        {
            float vx = MathF.Cos(angle);
            float vy = MathF.Sin(angle);

            // Normalize u and v to unit length for this test
            float uLen = MathF.Sqrt(ux * ux + uy * uy);
            float vLen = MathF.Sqrt(vx * vx + vy * vy);
            float unx = ux / uLen;
            float uny = uy / uLen;
            float vnx = vx / vLen;
            float vny = vy / vLen;

            // Scale back to reasonable lengths
            unx *= 1.5f; uny *= 1.5f;
            vnx *= 2.0f; vny *= 2.0f;

            // Compute gradients
            float uu = unx * unx + uny * uny;
            float vv = vnx * vnx + vny * vny;
            float c = unx * vnx + uny * vny;
            float s = unx * vny - uny * vnx;
            float denom = uu * vv + 1e-12f;

            float dθ_du_x_analytical = (c * vny - s * vnx) / denom;
            float dθ_du_y_analytical = (-c * vnx - s * vny) / denom;
            float dθ_dv_x_analytical = (c * (-uny) - s * unx) / denom;
            float dθ_dv_y_analytical = (c * unx - s * uny) / denom;

            var (dθ_du_x_numerical, dθ_du_y_numerical, dθ_dv_x_numerical, dθ_dv_y_numerical) =
                FD_AngleGrads(unx, uny, vnx, vny, Eps);

            AssertGradientMatch(dθ_du_x_analytical, dθ_du_x_numerical, $"∂θ/∂ux @ angle {angle:F2}");
            AssertGradientMatch(dθ_du_y_analytical, dθ_du_y_numerical, $"∂θ/∂uy @ angle {angle:F2}");
            AssertGradientMatch(dθ_dv_x_analytical, dθ_dv_x_numerical, $"∂θ/∂vx @ angle {angle:F2}");
            AssertGradientMatch(dθ_dv_y_analytical, dθ_dv_y_numerical, $"∂θ/∂vy @ angle {angle:F2}");
        }
    }

    /// <summary>
    /// Finite-difference gradient of raw angle function θ = atan2(cross, dot).
    /// IMPORTANT: Does NOT wrap the angle - differentiate the continuous function!
    /// </summary>
    private static (float dθdux, float dθduy, float dθdvx, float dθdvy) FD_AngleGrads(
        float ux, float uy, float vx, float vy, float eps)
    {
        double uxD = ux;
        double uyD = uy;
        double vxD = vx;
        double vyD = vy;
        double epsD = eps;

        static double Theta(double ax, double ay, double bx, double by)
        {
            double c = ax * bx + ay * by;  // dot
            double s = ax * by - ay * bx;  // cross
            return Math.Atan2(s, c);
        }

        double thetaUx = (Theta(uxD + epsD, uyD, vxD, vyD) - Theta(uxD - epsD, uyD, vxD, vyD)) / (2.0 * epsD);
        double thetaUy = (Theta(uxD, uyD + epsD, vxD, vyD) - Theta(uxD, uyD - epsD, vxD, vyD)) / (2.0 * epsD);
        double thetaVx = (Theta(uxD, uyD, vxD + epsD, vyD) - Theta(uxD, uyD, vxD - epsD, vyD)) / (2.0 * epsD);
        double thetaVy = (Theta(uxD, uyD, vxD, vyD + epsD) - Theta(uxD, uyD, vxD, vyD - epsD)) / (2.0 * epsD);

        return ((float)thetaUx, (float)thetaUy, (float)thetaVx, (float)thetaVy);
    }

    /// <summary>
    /// Assert gradient match using relative error (robust for large gradients)
    /// and absolute error (for near-zero gradients).
    /// </summary>
    private void AssertGradientMatch(float analytical, float numerical, string name)
    {
        float absDiff = MathF.Abs(analytical - numerical);
        float absMax = MathF.Max(MathF.Abs(analytical), MathF.Abs(numerical));

        // Use absolute tolerance if both are near zero
        if (absMax < AbsTolerance)
        {
            Assert.True(absDiff < AbsTolerance,
                $"{name}: analytical={analytical:F8}, numerical={numerical:F8}, diff={absDiff:F8} (abs check)");
        }
        else
        {
            // Use relative error otherwise
            float relError = absDiff / absMax;
            Assert.True(relError < RelTolerance,
                $"{name}: analytical={analytical:F8}, numerical={numerical:F8}, relError={relError:F8}");
        }
    }
}
