using System.Globalization;
using System.Text.RegularExpressions;
using Godot;

namespace Evolvatron.Evolvion.Utilities;

/// <summary>
/// Minimal SVG path decoder for parsing SVG path data.
/// Extracted from Colonel.Framework to remove dependency.
/// </summary>
public static class SvgPathDecoder
{
    public static List<List<Vector2>> DecodePath(string spath, float scale = 1)
    {
        var pathSteps = new Queue<string>();
        var regexObj = new Regex(@"[MLHVZ, ]|-?\d+(\.\d+)?", RegexOptions.IgnoreCase | RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant);
        var matchResult = regexObj.Match(spath);
        while (matchResult.Success)
        {
            if (matchResult.Value != " " && matchResult.Value != ",")
            {
                pathSteps.Enqueue(matchResult.Value);
            }

            matchResult = matchResult.NextMatch();
        }

        var paths = new List<List<Vector2>>();
        List<Vector2>? path = null;
        var currentPos = new Vector2(float.NaN, float.NaN);

        void MoveTo(Vector2 newPos)
        {
            if (path == null)
            {
                throw new InvalidOperationException($"No path is currently active!");
            }

            newPos *= scale;
            if (!RoughlyEquals(currentPos, newPos))
            {
                currentPos = newPos;
                path.Add(currentPos);
            }
        }

        while (pathSteps.Any())
        {
            var command = pathSteps.Dequeue();
            switch (command.ToUpperInvariant())
            {
                case "M":
                    path = new List<Vector2>();
                    paths.Add(path);
                    MoveTo(new Vector2(
                        float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture),
                        float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture)));
                    break;
                case "L":
                    MoveTo(new Vector2(
                        float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture),
                        float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture)));
                    break;
                case "H":
                    MoveTo(new Vector2(
                        currentPos.X + float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture),
                        currentPos.Y));
                    break;
                case "V":
                    MoveTo(new Vector2(
                        currentPos.X,
                        currentPos.Y + float.Parse(pathSteps.Dequeue(), CultureInfo.InvariantCulture)));
                    break;
                case "Z":
                    MoveTo(path![0]);
                    currentPos = new Vector2(float.NaN, float.NaN);
                    path = null;
                    break;
                default:
                    throw new InvalidOperationException($"Unhandled path command: {command}");
            }
        }

        return paths;
    }

    private static bool RoughlyEquals(Vector2 a, Vector2 b, float tolerance = 0.001f)
    {
        return Math.Abs(a.X - b.X) < tolerance && Math.Abs(a.Y - b.Y) < tolerance;
    }
}
