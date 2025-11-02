using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using System.Globalization;

// Optuna CLI Evaluation Tool
// Usage: dotnet run -- <param1>=<value1> <param2>=<value2> ...
// Output: Single line with final best fitness (more negative = better for MSE loss)

// Parse command-line arguments into dictionary
var cmdArgs = Environment.GetCommandLineArgs().Skip(1).ToArray();
var parameters = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

foreach (var arg in cmdArgs)
{
    var parts = arg.Split('=', 2);
    if (parts.Length == 2)
    {
        parameters[parts[0]] = parts[1];
    }
}

// Helper to get parameter with default
T GetParam<T>(string name, T defaultValue)
{
    if (!parameters.TryGetValue(name, out var value))
        return defaultValue;

    try
    {
        if (typeof(T) == typeof(int))
            return (T)(object)int.Parse(value);
        if (typeof(T) == typeof(float))
            return (T)(object)float.Parse(value, CultureInfo.InvariantCulture);
        if (typeof(T) == typeof(bool))
            return (T)(object)bool.Parse(value);
        return defaultValue;
    }
    catch
    {
        return defaultValue;
    }
}

// Build EvolutionConfig from parameters
var config = new EvolutionConfig
{
    // Population structure
    SpeciesCount = GetParam("species_count", 20),
    IndividualsPerSpecies = GetParam("individuals_per_species", 40),
    MinSpeciesCount = GetParam("min_species_count", 8),

    // Selection
    Elites = GetParam("elites", 2),
    TournamentSize = GetParam("tournament_size", 16),
    ParentPoolPercentage = GetParam("parent_pool_percentage", 1.0f),

    // Culling thresholds
    GraceGenerations = GetParam("grace_generations", 1),
    StagnationThreshold = GetParam("stagnation_threshold", 6),
    SpeciesDiversityThreshold = GetParam("species_diversity_threshold", 0.08f),
    RelativePerformanceThreshold = GetParam("relative_performance_threshold", 0.7f),

    // Weight mutations
    MutationRates = new MutationRates
    {
        WeightJitter = GetParam("weight_jitter", 0.95f),
        WeightJitterStdDev = GetParam("weight_jitter_stddev", 0.3f),
        WeightReset = GetParam("weight_reset", 0.1f),
        WeightL1Shrink = GetParam("weight_l1_shrink", 0.2f),
        L1ShrinkFactor = GetParam("l1_shrink_factor", 0.9f),
        ActivationSwap = GetParam("activation_swap", 0.10f),
        NodeParamMutate = GetParam("node_param_mutate", 0.0f),
        NodeParamStdDev = GetParam("node_param_stddev", 0.1f)
    },

    // Topology mutations
    EdgeMutations = new EdgeMutationConfig
    {
        EdgeAdd = GetParam("edge_add", 0.05f),
        EdgeDeleteRandom = GetParam("edge_delete_random", 0.01f),
        EdgeSplit = GetParam("edge_split", 0.01f),
        EdgeRedirect = GetParam("edge_redirect", 0.03f),
        EdgeSwap = GetParam("edge_swap", 0.02f),

        WeakEdgePruning = new WeakEdgePruningConfig
        {
            Enabled = GetParam("weak_edge_pruning_enabled", true),
            Threshold = GetParam("weak_edge_pruning_threshold", 0.01f),
            BasePruneRate = GetParam("weak_edge_pruning_base_rate", 0.7f),
            ApplyOnSpeciesBirth = GetParam("weak_edge_pruning_on_birth", true),
            ApplyDuringEvolution = GetParam("weak_edge_pruning_during_evolution", false)
        }
    }
};

// Evaluation configuration (fixed for consistency)
const int seed = 42;
const int generations = 150;
const int numSeeds = 3; // Run multiple seeds for robustness

var results = new List<float>();

for (int s = 0; s < numSeeds; s++)
{
    var evolver = new Evolver(seed + s);
    var random = new Random(seed + s);

    // Create spiral classification topology (2→8→8→1)
    var topology = new SpeciesBuilder()
        .AddInputRow(2)
        .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
        .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
        .AddOutputRow(1, ActivationType.Tanh)
        .WithMaxInDegree(10)
        .InitializeDense(random, density: 0.3f)
        .Build();

    var population = evolver.InitializePopulation(config, topology);
    var environment = new SpiralEnvironment(pointsPerSpiral: 50, noise: 0.0f);
    var evaluator = new SimpleFitnessEvaluator();

    // Evolve
    for (int gen = 0; gen < generations; gen++)
    {
        evaluator.EvaluatePopulation(population, environment, seed: gen + s);
        evolver.StepGeneration(population);
    }

    // Final evaluation
    evaluator.EvaluatePopulation(population, environment, seed: generations + s);
    var finalStats = population.GetStatistics();
    results.Add(finalStats.BestFitness);
}

// Output: mean fitness across seeds (negative MSE - more negative = worse)
float meanFitness = results.Average();
Console.WriteLine(meanFitness.ToString("F6", CultureInfo.InvariantCulture));
