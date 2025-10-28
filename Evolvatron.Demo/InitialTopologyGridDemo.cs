using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Visualization;

namespace Evolvatron.Demo;

public class InitialTopologyGridDemo
{
    public static void Run()
    {
        Console.WriteLine("=== Initial Topology Grid Demo ===\n");
        Console.WriteLine("Generating 25 newly spawned species/individuals...\n");

        var visualizer = new NeuralNetworkVisualizer();
        var networks = new List<(SpeciesSpec spec, Individual? individual, string label)>();

        for (int i = 0; i < 25; i++)
        {
            var random = new Random(1000 + i);

            var spec = new SpeciesBuilder()
                .AddInputRow(3)
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid)
                .AddHiddenRow(8, ActivationType.ReLU, ActivationType.LeakyReLU)
                .AddOutputRow(2, ActivationType.Tanh)
                .WithMaxInDegree(10)
                .InitializeSparse(random)
                .Build();

            var individual = SpeciesDiversification.InitializeIndividual(spec, random);

            networks.Add((spec, individual, $"Species {i + 1}"));

            var activeNodes = ConnectivityValidator.ComputeActiveNodes(spec);
            int activeHidden = activeNodes.Skip(3).Take(16).Count(x => x);
            Console.WriteLine($"Species {i + 1}: {spec.Edges.Count} edges, {activeHidden} active hidden nodes");
        }

        string svg = visualizer.RenderMutationProgression(networks);
        string outputPath = @"c:\slask\initial_topologies_5x5.svg";
        visualizer.SaveToFile(svg, outputPath);

        Console.WriteLine($"\n✓ Saved 5x5 grid to: {outputPath}");
        Console.WriteLine($"\nStatistics:");
        Console.WriteLine($"  Avg edges: {networks.Average(n => n.spec.Edges.Count):F1}");
        Console.WriteLine($"  Min edges: {networks.Min(n => n.spec.Edges.Count)}");
        Console.WriteLine($"  Max edges: {networks.Max(n => n.spec.Edges.Count)}");
    }
}
