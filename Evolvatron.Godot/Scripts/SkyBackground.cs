using Godot;
using System;

namespace Evolvatron.Godot;

/// <summary>
/// Screen-space lunar/space backdrop drawn on a low CanvasLayer: a vertical gradient (deep space →
/// lighter horizon), a seeded starfield, and a cratered moon with a soft glow. Static — drawn once.
/// </summary>
public partial class SkyBackground : Node2D
{
    public Vector2 DesignSize = new(1280, 720);

    public override void _Ready() => QueueRedraw();

    public override void _Draw()
    {
        float w = DesignSize.X, h = DesignSize.Y;

        // Gradient: deep blue-black at the top, a slightly warmer/lighter band near the horizon.
        var top = new Color(0.015f, 0.02f, 0.06f);
        var mid = new Color(0.04f, 0.06f, 0.13f);
        var horizon = new Color(0.10f, 0.12f, 0.22f);
        const int bands = 64;
        for (int i = 0; i < bands; i++)
        {
            float t = i / (float)(bands - 1);
            Color c = t < 0.6f ? top.Lerp(mid, t / 0.6f) : mid.Lerp(horizon, (t - 0.6f) / 0.4f);
            DrawRect(new Rect2(0, h * i / bands, w, h / bands + 1f), c);
        }

        // Stars (seeded so they don't twinkle/jump). Brighter, sparser "hero" stars get a tiny cross.
        var rng = new Random(20260627);
        for (int i = 0; i < 220; i++)
        {
            float x = (float)rng.NextDouble() * w;
            float y = (float)rng.NextDouble() * h * 0.92f;
            float b = 0.25f + (float)rng.NextDouble() * 0.75f;
            bool hero = rng.NextDouble() < 0.08;
            float r = hero ? 1.7f : 0.8f + (float)rng.NextDouble() * 0.5f;
            var col = new Color(b, b, b * 1.08f, b);
            DrawCircle(new Vector2(x, y), r, col);
            if (hero)
            {
                var faint = new Color(b, b, b, b * 0.5f);
                DrawLine(new Vector2(x - 3, y), new Vector2(x + 3, y), faint, 0.8f);
                DrawLine(new Vector2(x, y - 3), new Vector2(x, y + 3), faint, 0.8f);
            }
        }

        // Moon: soft glow, lit disc, a darker terminator crescent, and a few craters.
        var mc = new Vector2(w * 0.80f, h * 0.20f);
        float mr = 50f;
        for (int g = 5; g >= 1; g--)
            DrawCircle(mc, mr + g * 6f, new Color(0.65f, 0.72f, 0.95f, 0.025f));
        DrawCircle(mc, mr, new Color(0.86f, 0.87f, 0.93f));
        // Soft terminator: progressively darker discs biased toward the shadow limb. Each is kept
        // fully inside the lit disc (offset + radius == 0.95*mr) so nothing can poke past the rim.
        var shadowDir = new Vector2(0.74f, 0.67f); // toward shadow side (light from upper-left); ~unit length
        (float radius, Color col)[] shade =
        {
            (mr * 0.80f, new Color(0.80f, 0.81f, 0.88f)),
            (mr * 0.62f, new Color(0.72f, 0.73f, 0.82f)),
            (mr * 0.46f, new Color(0.64f, 0.65f, 0.76f)),
        };
        foreach (var (radius, col) in shade)
            DrawCircle(mc + shadowDir * (mr * 0.95f - radius), radius, col);
        var craterRng = new Random(99);
        for (int i = 0; i < 7; i++)
        {
            float a = (float)craterRng.NextDouble() * MathF.Tau;
            float d = (float)craterRng.NextDouble() * mr * 0.72f;
            var p = mc + new Vector2(MathF.Cos(a), MathF.Sin(a)) * d;
            float cr = 2.5f + (float)craterRng.NextDouble() * 6f;
            DrawCircle(p, cr, new Color(0.74f, 0.75f, 0.82f));
        }
    }
}
