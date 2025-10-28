using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Visualization;

namespace Evolvatron.Demo;

/// <summary>
/// Demo showing how neural networks evolve through mutations.
/// Creates a sequence of networks, applies various mutations,
/// and visualizes the progression.
/// </summary>
public class MutationProgressionDemo
{
    public static void Run()
    {
        Console.WriteLine("=== Mutation Progression Demo ===\n");

        var random = new Random(42);
        var visualizer = new NeuralNetworkVisualizer();

        // Create initial sparse network
        Console.WriteLine("Creating initial sparse network...");
        var spec = CreateInitialNetwork();
        var individual = SpeciesDiversification.InitializeIndividual(spec, random);

        var progression = new List<(SpeciesSpec spec, Individual? individual, string label)>
        {
            (CloneSpec(spec), CloneIndividual(individual), "0. Initial Sparse")
        };

        // Apply series of mutations
        Console.WriteLine("Applying mutations...\n");

        // 1. Add edge mutation
        if (EdgeTopologyMutations.TryEdgeAdd(spec, random))
        {
            individual = AdaptIndividualToNewTopology(individual, spec, random);
            progression.Add((CloneSpec(spec), CloneIndividual(individual), "1. EdgeAdd"));
            Console.WriteLine("✓ EdgeAdd: Added 1 edge");
        }

        // 2. Another edge add
        if (EdgeTopologyMutations.TryEdgeAdd(spec, random))
        {
            individual = AdaptIndividualToNewTopology(individual, spec, random);
            progression.Add((CloneSpec(spec), CloneIndividual(individual), "2. EdgeAdd x2"));
            Console.WriteLine("✓ EdgeAdd: Added another edge");
        }

        // 3. Weight jitter
        MutationOperators.ApplyWeightJitter(individual, 0.3f, random);
        progression.Add((CloneSpec(spec), CloneIndividual(individual), "3. Weight Jitter"));
        Console.WriteLine("✓ Weight Jitter: Perturbed weights");

        // 4. Edge split (if possible)
        if (EdgeTopologyMutations.TryEdgeSplit(spec, random))
        {
            individual = AdaptIndividualToNewTopology(individual, spec, random);
            progression.Add((CloneSpec(spec), CloneIndividual(individual), "4. EdgeSplit"));
            Console.WriteLine("✓ EdgeSplit: Split edge through node");
        }

        // 5. More edges
        for (int i = 0; i < 3; i++)
        {
            EdgeTopologyMutations.TryEdgeAdd(spec, random);
        }
        individual = AdaptIndividualToNewTopology(individual, spec, random);
        progression.Add((CloneSpec(spec), CloneIndividual(individual), "5. EdgeAdd x3"));
        Console.WriteLine("✓ EdgeAdd: Added 3 more edges");

        // 6. Activation swap
        MutationOperators.ApplyActivationSwap(individual, spec, random);
        progression.Add((CloneSpec(spec), CloneIndividual(individual), "6. Activation Swap"));
        Console.WriteLine("✓ Activation Swap: Changed activation function");

        // 7. Edge redirect (if possible)
        if (EdgeTopologyMutations.TryEdgeRedirect(spec, random))
        {
            individual = AdaptIndividualToNewTopology(individual, spec, random);
            progression.Add((CloneSpec(spec), CloneIndividual(individual), "7. EdgeRedirect"));
            Console.WriteLine("✓ EdgeRedirect: Redirected edge");
        }

        // 8. Weight L1 shrink
        MutationOperators.ApplyWeightL1Shrink(individual, 0.1f);
        progression.Add((CloneSpec(spec), CloneIndividual(individual), "8. Weight L1 Shrink"));
        Console.WriteLine("✓ Weight L1 Shrink: Reduced weight magnitudes");

        // Render progression
        Console.WriteLine($"\nGenerating visualization with {progression.Count} steps...");
        string svg = visualizer.RenderMutationProgression(progression);

        string outputPath = @"c:\slask\mutation_progression.svg";
        visualizer.SaveToFile(svg, outputPath);

        Console.WriteLine($"\n✓ Saved to: {outputPath}");
        Console.WriteLine($"\nNetwork statistics:");
        Console.WriteLine($"  Initial edges: {progression[0].spec.Edges.Count}");
        Console.WriteLine($"  Final edges: {progression[^1].spec.Edges.Count}");
        Console.WriteLine($"  Growth: +{progression[^1].spec.Edges.Count - progression[0].spec.Edges.Count} edges");

        // Also render final network in detail
        string detailedSvg = visualizer.RenderNetwork(
            progression[^1].spec,
            progression[^1].individual,
            "Final Network After Mutations");

        string detailPath = @"c:\slask\mutation_final.svg";
        visualizer.SaveToFile(detailedSvg, detailPath);
        Console.WriteLine($"✓ Saved detailed view to: {detailPath}");
    }

    private static SpeciesSpec CreateInitialNetwork()
    {
        var random = new Random(123);
        return new SpeciesBuilder()
            .AddInputRow(3)
            .AddHiddenRow(6, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid)
            .AddHiddenRow(6, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeSparse(random)
            .Build();
    }

    private static Individual AdaptIndividualToNewTopology(Individual old, SpeciesSpec newSpec, Random random)
    {
        // Expand weight array if needed
        if (old.Weights.Length < newSpec.Edges.Count)
        {
            var newWeights = new float[newSpec.Edges.Count];
            Array.Copy(old.Weights, newWeights, old.Weights.Length);

            // Initialize new weights with Glorot
            for (int i = old.Weights.Length; i < newWeights.Length; i++)
            {
                newWeights[i] = MutationOperators.GlorotUniform(3, 3, random);
            }

            old.Weights = newWeights;
        }

        return old;
    }

    private static SpeciesSpec CloneSpec(SpeciesSpec spec)
    {
        return SpeciesDiversification.CloneTopology(spec);
    }

    private static Individual CloneIndividual(Individual individual)
    {
        return new Individual(individual);
    }
}
