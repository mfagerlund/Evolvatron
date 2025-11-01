using Evolvatron.Evolvion;

// Demo: InitializeDense() edge counts for 2→6→6→1 network at different densities

var random = new Random(42);

// Test different density levels
var densities = new[] { 1.0f, 0.75f, 0.5f, 0.25f };

Console.WriteLine("Network topology: 2 inputs → 6 hidden → 6 hidden → 1 output");
Console.WriteLine("MaxInDegree: 12");
Console.WriteLine();

foreach (var density in densities)
{
    var topology = new SpeciesBuilder()
        .AddInputRow(2)
        .AddHiddenRow(6, ActivationType.ReLU)
        .AddHiddenRow(6, ActivationType.Tanh)
        .AddOutputRow(1, ActivationType.Tanh)
        .WithMaxInDegree(12)
        .InitializeDense(random, density: density)
        .Build();

    // Count edges per layer
    int layer1Edges = topology.Edges.Count(e => e.Source < 2 && e.Dest < 8);  // input→hidden1
    int layer2Edges = topology.Edges.Count(e => e.Source >= 2 && e.Source < 8 && e.Dest >= 8 && e.Dest < 14);  // hidden1→hidden2
    int layer3Edges = topology.Edges.Count(e => e.Source >= 8 && e.Source < 14 && e.Dest >= 14);  // hidden2→output

    Console.WriteLine($"Density: {density:F2} ({density * 100:F0}%)");
    Console.WriteLine($"  Input→Hidden1:  {layer1Edges} edges (6 nodes × {density:F2} × 2 sources = {6 * density * 2:F1} theoretical)");
    Console.WriteLine($"  Hidden1→Hidden2: {layer2Edges} edges (6 nodes × {density:F2} × 6 sources = {6 * density * 6:F1} theoretical)");
    Console.WriteLine($"  Hidden2→Output:  {layer3Edges} edges (1 node × {density:F2} × 6 sources = {1 * density * 6:F1} theoretical)");
    Console.WriteLine($"  Total edges: {topology.Edges.Count}");
    Console.WriteLine();
}

// Compare with fully-connected theoretical maximum
Console.WriteLine("Theoretical maximum (fully connected, no MaxInDegree limit):");
Console.WriteLine("  Input→Hidden1:  2×6 = 12 edges");
Console.WriteLine("  Hidden1→Hidden2: 6×6 = 36 edges");
Console.WriteLine("  Hidden2→Output:  6×1 = 6 edges");
Console.WriteLine("  Total: 54 edges");
