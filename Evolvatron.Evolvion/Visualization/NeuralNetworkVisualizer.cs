using System.Globalization;
using System.Text;

namespace Evolvatron.Evolvion.Visualization;

/// <summary>
/// Simple SVG-based neural network visualizer.
/// Renders network topology with nodes and edges.
/// </summary>
public class NeuralNetworkVisualizer
{
    private readonly StringBuilder _svg = new();
    private const float NodeRadius = 14f;
    private const float LayerSpacing = 140f;
    private const float NodeSpacing = 45f;
    private const float Padding = 50f;

    private static string F(float value) => value.ToString("F1", CultureInfo.InvariantCulture);

    private void AddStyleDefs()
    {
        _svg.AppendLine("  <defs>");

        // Gradient for background
        _svg.AppendLine("    <linearGradient id=\"bgGradient\" x1=\"0%\" y1=\"0%\" x2=\"0%\" y2=\"100%\">");
        _svg.AppendLine("      <stop offset=\"0%\" style=\"stop-color:#f5f7fa;stop-opacity:1\" />");
        _svg.AppendLine("      <stop offset=\"100%\" style=\"stop-color:#e3e8ef;stop-opacity:1\" />");
        _svg.AppendLine("    </linearGradient>");

        // Gradient for input nodes
        _svg.AppendLine("    <radialGradient id=\"inputGrad\">");
        _svg.AppendLine("      <stop offset=\"0%\" style=\"stop-color:#64B5F6;stop-opacity:1\" />");
        _svg.AppendLine("      <stop offset=\"100%\" style=\"stop-color:#1976D2;stop-opacity:1\" />");
        _svg.AppendLine("    </radialGradient>");

        // Gradient for output nodes
        _svg.AppendLine("    <radialGradient id=\"outputGrad\">");
        _svg.AppendLine("      <stop offset=\"0%\" style=\"stop-color:#FFB74D;stop-opacity:1\" />");
        _svg.AppendLine("      <stop offset=\"100%\" style=\"stop-color:#F57C00;stop-opacity:1\" />");
        _svg.AppendLine("    </radialGradient>");

        // Gradient for active hidden nodes
        _svg.AppendLine("    <radialGradient id=\"hiddenActiveGrad\">");
        _svg.AppendLine("      <stop offset=\"0%\" style=\"stop-color:#81C784;stop-opacity:1\" />");
        _svg.AppendLine("      <stop offset=\"100%\" style=\"stop-color:#388E3C;stop-opacity:1\" />");
        _svg.AppendLine("    </radialGradient>");

        // Gradient for inactive hidden nodes
        _svg.AppendLine("    <radialGradient id=\"hiddenInactiveGrad\">");
        _svg.AppendLine("      <stop offset=\"0%\" style=\"stop-color:#E0E0E0;stop-opacity:1\" />");
        _svg.AppendLine("      <stop offset=\"100%\" style=\"stop-color:#9E9E9E;stop-opacity:1\" />");
        _svg.AppendLine("    </radialGradient>");

        // Drop shadow filter
        _svg.AppendLine("    <filter id=\"dropShadow\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">");
        _svg.AppendLine("      <feGaussianBlur in=\"SourceAlpha\" stdDeviation=\"2\"/>");
        _svg.AppendLine("      <feOffset dx=\"1\" dy=\"2\" result=\"offsetblur\"/>");
        _svg.AppendLine("      <feComponentTransfer>");
        _svg.AppendLine("        <feFuncA type=\"linear\" slope=\"0.3\"/>");
        _svg.AppendLine("      </feComponentTransfer>");
        _svg.AppendLine("      <feMerge>");
        _svg.AppendLine("        <feMergeNode/>");
        _svg.AppendLine("        <feMergeNode in=\"SourceGraphic\"/>");
        _svg.AppendLine("      </feMerge>");
        _svg.AppendLine("    </filter>");

        // Arrow marker for directed edges
        _svg.AppendLine("    <marker id=\"arrowPos\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">");
        _svg.AppendLine("      <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#2E7D32\" opacity=\"0.6\"/>");
        _svg.AppendLine("    </marker>");

        _svg.AppendLine("    <marker id=\"arrowNeg\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">");
        _svg.AppendLine("      <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#C62828\" opacity=\"0.6\"/>");
        _svg.AppendLine("    </marker>");

        _svg.AppendLine("  </defs>");
    }

    public class NodePosition
    {
        public int NodeIndex { get; set; }
        public int Row { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
    }

    /// <summary>
    /// Render a neural network topology to SVG.
    /// </summary>
    public string RenderNetwork(
        SpeciesSpec spec,
        Individual? individual = null,
        string title = "Neural Network")
    {
        var positions = ComputeNodePositions(spec);

        float width = spec.RowCounts.Length * LayerSpacing + 2 * Padding;
        float maxNodesInRow = spec.RowCounts.Max();
        float height = maxNodesInRow * NodeSpacing + 2 * Padding;

        _svg.Clear();
        _svg.AppendLine($"<svg width=\"{F(width)}\" height=\"{F(height)}\" xmlns=\"http://www.w3.org/2000/svg\">");
        _svg.AppendLine($"  <title>{title}</title>");

        // Add defs for gradients and filters
        AddStyleDefs();

        // Background with gradient
        _svg.AppendLine($"  <rect width=\"{F(width)}\" height=\"{F(height)}\" fill=\"url(#bgGradient)\"/>");

        // Add title
        _svg.AppendLine($"  <text x=\"{F(width/2)}\" y=\"{F(25f)}\" text-anchor=\"middle\" " +
                      $"font-family=\"Arial, sans-serif\" font-size=\"18\" font-weight=\"bold\" fill=\"#333\">{title}</text>");

        // Render edges first (so they're behind nodes)
        RenderEdges(spec, positions, individual);

        // Render nodes
        RenderNodes(spec, positions, individual);

        // Render labels
        RenderLabels(spec, positions);

        _svg.AppendLine("</svg>");

        return _svg.ToString();
    }

    /// <summary>
    /// Render multiple networks side by side showing mutation progression.
    /// </summary>
    public string RenderMutationProgression(
        List<(SpeciesSpec spec, Individual? individual, string label)> networks)
    {
        if (networks.Count == 0)
            return "<svg></svg>";

        // Calculate dimensions
        float networkWidth = networks[0].spec.RowCounts.Length * LayerSpacing + 2 * Padding;
        float maxNodesInRow = networks.Max(n => n.spec.RowCounts.Max());
        float networkHeight = maxNodesInRow * NodeSpacing + 2 * Padding;

        int cols = Math.Min(networks.Count, 4); // Max 4 per row
        int rows = (int)Math.Ceiling(networks.Count / (float)cols);

        float totalWidth = cols * networkWidth + Padding;
        float totalHeight = rows * (networkHeight + 40) + Padding; // +40 for labels

        _svg.Clear();
        _svg.AppendLine($"<svg width=\"{F(totalWidth)}\" height=\"{F(totalHeight)}\" xmlns=\"http://www.w3.org/2000/svg\">");

        // Add defs for gradients and filters
        AddStyleDefs();

        _svg.AppendLine("  <style>");
        _svg.AppendLine("    .network-label { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; }");
        _svg.AppendLine("  </style>");

        _svg.AppendLine($"  <rect width=\"{F(totalWidth)}\" height=\"{F(totalHeight)}\" fill=\"white\"/>");

        // Render each network in grid
        for (int i = 0; i < networks.Count; i++)
        {
            int col = i % cols;
            int row = i / cols;

            float offsetX = col * networkWidth + Padding;
            float offsetY = row * (networkHeight + 40) + Padding;

            _svg.AppendLine($"  <g transform=\"translate({F(offsetX)}, {F(offsetY)})\">");

            // Label
            _svg.AppendLine($"    <text x=\"{F(networkWidth/2)}\" y=\"{F(-10f)}\" text-anchor=\"middle\" class=\"network-label\">{networks[i].label}</text>");

            // Render mini network
            RenderNetworkAt(networks[i].spec, networks[i].individual, 0, 0, networkWidth, networkHeight);

            _svg.AppendLine("  </g>");
        }

        _svg.AppendLine("</svg>");

        return _svg.ToString();
    }

    private void RenderNetworkAt(SpeciesSpec spec, Individual? individual, float offsetX, float offsetY, float width, float height)
    {
        var positions = ComputeNodePositions(spec);
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);

        _svg.AppendLine($"    <rect x=\"{F(0f)}\" y=\"{F(0f)}\" width=\"{F(width)}\" height=\"{F(height)}\" fill=\"url(#bgGradient)\" stroke=\"#ccc\" stroke-width=\"1\"/>");

        RenderEdgesHelper(spec, positions, individual, activeNodes, "    ", NodeRadius * 0.7f, 1.5f);
        RenderNodesHelper(spec, positions, activeNodes, "    ", NodeRadius * 0.7f);
    }

    private Dictionary<int, NodePosition> ComputeNodePositions(SpeciesSpec spec)
    {
        var positions = new Dictionary<int, NodePosition>();
        int nodeIndex = 0;

        for (int row = 0; row < spec.RowCounts.Length; row++)
        {
            int nodesInRow = spec.RowCounts[row];
            float x = Padding + row * LayerSpacing;

            for (int i = 0; i < nodesInRow; i++)
            {
                // Center nodes vertically
                float totalHeight = nodesInRow * NodeSpacing;
                float startY = Padding + (spec.RowCounts.Max() * NodeSpacing - totalHeight) / 2;
                float y = startY + i * NodeSpacing;

                positions[nodeIndex] = new NodePosition
                {
                    NodeIndex = nodeIndex,
                    Row = row,
                    X = x,
                    Y = y
                };

                nodeIndex++;
            }
        }

        return positions;
    }

    private void RenderEdges(SpeciesSpec spec, Dictionary<int, NodePosition> positions, Individual? individual)
    {
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
        RenderEdgesHelper(spec, positions, individual, activeNodes, "  ", NodeRadius, 3f);
    }

    private void RenderNodes(SpeciesSpec spec, Dictionary<int, NodePosition> positions, Individual? individual)
    {
        var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
        RenderNodesHelper(spec, positions, activeNodes, "  ", NodeRadius);
    }

    private void RenderLabels(SpeciesSpec spec, Dictionary<int, NodePosition> positions)
    {
        // Add row labels
        var rowLabels = new[] { "Inputs", "Hidden", "Hidden", "Hidden", "Hidden", "Outputs" };

        for (int row = 0; row < spec.RowCounts.Length; row++)
        {
            var firstNodeInRow = positions.Values.First(p => p.Row == row);
            string label = row == 0 ? "Inputs" :
                          row == spec.RowCounts.Length - 1 ? "Outputs" :
                          $"Hidden {row}";

            _svg.AppendLine($"  <text x=\"{F(firstNodeInRow.X)}\" y=\"{F(Padding - 10)}\" text-anchor=\"middle\" " +
                          $"font-family=\"Arial\" font-size=\"12\" fill=\"#666\">{label}</text>");
        }

        // Add statistics
        int activeCount = ConnectivityValidator.ComputeActiveNodes(spec).Count(x => x);
        int totalNodes = spec.TotalNodes;
        int edges = spec.Edges.Count;

        _svg.AppendLine($"  <text x=\"{F(Padding)}\" y=\"{F(positions.Values.Max(p => p.Y) + Padding)}\" " +
                      $"font-family=\"Arial\" font-size=\"11\" fill=\"#666\">" +
                      $"Nodes: {activeCount}/{totalNodes} active | Edges: {edges}</text>");
    }

    private string GetWeightColor(float weight)
    {
        if (weight > 0)
        {
            // Positive weights: green shades
            int intensity = (int)Math.Clamp(Math.Abs(weight) * 200, 50, 200);
            return $"rgb(0, {intensity}, 0)";
        }
        else
        {
            // Negative weights: red shades
            int intensity = (int)Math.Clamp(Math.Abs(weight) * 200, 50, 200);
            return $"rgb({intensity}, 0, 0)";
        }
    }

    private void RenderEdgesHelper(SpeciesSpec spec, Dictionary<int, NodePosition> positions, Individual? individual, bool[] activeNodes, string indent, float nodeRadius, float weightScale)
    {
        for (int edgeIdx = 0; edgeIdx < spec.Edges.Count; edgeIdx++)
        {
            var (source, dest) = spec.Edges[edgeIdx];
            var srcPos = positions[source];
            var dstPos = positions[dest];

            bool isActive = activeNodes[source] && activeNodes[dest];

            float weight = individual?.Weights[edgeIdx] ?? 0.5f;
            string color = isActive ? GetWeightColor(weight) : "#888888";
            float thickness = isActive ? Math.Clamp(Math.Abs(weight) * weightScale + 0.5f, 0.5f, 5f) : (weightScale / 2f);
            float opacity = isActive ? 0.6f : 0.7f;

            string marker = isActive ? (weight > 0 ? "url(#arrowPos)" : "url(#arrowNeg)") : "";

            float dx = dstPos.X - srcPos.X;
            float dy = dstPos.Y - srcPos.Y;
            float len = MathF.Sqrt(dx * dx + dy * dy);
            if (len > 0)
            {
                float shortenBy = nodeRadius + (isActive ? 3f : 0f);
                float ratio = (len - shortenBy) / len;
                float x2 = srcPos.X + dx * ratio;
                float y2 = srcPos.Y + dy * ratio;

                if (!isActive)
                {
                    _svg.AppendLine($"{indent}<line x1=\"{F(srcPos.X)}\" y1=\"{F(srcPos.Y)}\" x2=\"{F(x2)}\" y2=\"{F(y2)}\" " +
                                  $"stroke=\"white\" stroke-width=\"{F(thickness + 2.5f)}\" opacity=\"0.9\"/>");
                }

                _svg.AppendLine($"{indent}<line x1=\"{F(srcPos.X)}\" y1=\"{F(srcPos.Y)}\" x2=\"{F(x2)}\" y2=\"{F(y2)}\" " +
                              $"stroke=\"{color}\" stroke-width=\"{F(thickness)}\" opacity=\"{F(opacity)}\" " +
                              (string.IsNullOrEmpty(marker) ? "/>" : $"marker-end=\"{marker}\"/>"));
            }
        }
    }

    private void RenderNodesHelper(SpeciesSpec spec, Dictionary<int, NodePosition> positions, bool[] activeNodes, string indent, float nodeRadius)
    {
        for (int i = 0; i < spec.TotalNodes; i++)
        {
            var pos = positions[i];

            string fillGradient;
            string strokeColor;
            float strokeWidth;

            if (pos.Row == 0)
            {
                fillGradient = "url(#inputGrad)";
                strokeColor = "#0D47A1";
                strokeWidth = 2.5f;
            }
            else if (pos.Row == spec.RowCounts.Length - 1)
            {
                fillGradient = "url(#outputGrad)";
                strokeColor = "#E65100";
                strokeWidth = 2.5f;
            }
            else
            {
                fillGradient = activeNodes[i] ? "url(#hiddenActiveGrad)" : "url(#hiddenInactiveGrad)";
                strokeColor = activeNodes[i] ? "#1B5E20" : "#616161";
                strokeWidth = activeNodes[i] ? 2.0f : 1.5f;
            }

            _svg.AppendLine($"{indent}<circle cx=\"{F(pos.X)}\" cy=\"{F(pos.Y)}\" r=\"{F(nodeRadius)}\" " +
                          $"fill=\"{fillGradient}\" stroke=\"{strokeColor}\" stroke-width=\"{F(strokeWidth)}\" " +
                          $"filter=\"url(#dropShadow)\"/>");

            _svg.AppendLine($"{indent}<circle cx=\"{F(pos.X - 2)}\" cy=\"{F(pos.Y - 2)}\" r=\"{F(nodeRadius * 0.4f)}\" " +
                          $"fill=\"white\" opacity=\"0.4\"/>");
        }
    }

    /// <summary>
    /// Save SVG to file.
    /// </summary>
    public void SaveToFile(string svgContent, string filePath)
    {
        File.WriteAllText(filePath, svgContent);
    }
}
