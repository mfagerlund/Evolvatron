using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Benchmarks;
using Evolvatron.Evolvion.Environments;
using System.Globalization;

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

var config = new EvolutionConfig
{
    SpeciesCount = GetParam("species_count", 27),
    IndividualsPerSpecies = GetParam("individuals_per_species", 88),
    MinSpeciesCount = GetParam("min_species_count", 8),

    Elites = GetParam("elites", 4),
    TournamentSize = GetParam("tournament_size", 22),
    ParentPoolPercentage = GetParam("parent_pool_percentage", 0.75f),

    GraceGenerations = GetParam("grace_generations", 1),
    StagnationThreshold = GetParam("stagnation_threshold", 6),
    SpeciesDiversityThreshold = GetParam("species_diversity_threshold", 0.066f),
    RelativePerformanceThreshold = GetParam("relative_performance_threshold", 0.885f),

    MutationRates = new MutationRates
    {
        WeightJitter = GetParam("weight_jitter", 0.972f),
        WeightJitterStdDev = GetParam("weight_jitter_stddev", 0.402f),
        WeightReset = GetParam("weight_reset", 0.137f),
        WeightL1Shrink = GetParam("weight_l1_shrink", 0.090f),
        L1ShrinkFactor = GetParam("l1_shrink_factor", 0.949f),
        ActivationSwap = GetParam("activation_swap", 0.186f),
        NodeParamMutate = GetParam("node_param_mutate", 0.022f),
        NodeParamStdDev = GetParam("node_param_stddev", 0.053f)
    },

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

const int baseSeed = 42;
const int maxGenerations = 100;
const int numSeeds = 5;
const float solveThreshold = -0.10f;

var task = new LandscapeNavigationTask(
    OptimizationLandscapes.Rosenbrock,
    dimensions: 5,
    timesteps: 150,
    stepSize: 0.1f,
    minBound: -2f,
    maxBound: 2f,
    observationType: ObservationType.FullPosition,
    seed: baseSeed);

var solvedCount = 0;
var totalGenerations = 0;
var bestFitnesses = new List<float>();

for (int s = 0; s < numSeeds; s++)
{
    var evolver = new Evolver(baseSeed + s);
    var random = new Random(baseSeed + s);

    var topology = new SpeciesBuilder()
        .AddInputRow(task.GetObservationSize())
        .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.Sigmoid, ActivationType.LeakyReLU)
        .AddHiddenRow(8, ActivationType.ReLU, ActivationType.Tanh, ActivationType.LeakyReLU)
        .AddOutputRow(task.GetDimensions(), ActivationType.Tanh)
        .WithMaxInDegree(10)
        .InitializeDense(random, density: 0.3f)
        .Build();

    var population = evolver.InitializePopulation(config, topology);
    var environment = new LandscapeEnvironment(task);
    var evaluator = new SimpleFitnessEvaluator();

    evaluator.EvaluatePopulation(population, environment, seed: 0);

    bool solved = false;
    int generationsToSolve = maxGenerations;

    for (int gen = 1; gen <= maxGenerations; gen++)
    {
        evaluator.EvaluatePopulation(population, environment, seed: 0);

        var stats = population.GetStatistics();

        if (stats.BestFitness >= solveThreshold)
        {
            solved = true;
            generationsToSolve = gen;
            bestFitnesses.Add(stats.BestFitness);
            break;
        }

        evolver.StepGeneration(population);
    }

    if (solved)
    {
        solvedCount++;
        totalGenerations += generationsToSolve;
    }
    else
    {
        evaluator.EvaluatePopulation(population, environment, seed: 0);
        var finalStats = population.GetStatistics();
        bestFitnesses.Add(finalStats.BestFitness);
    }
}

float solveRate = (float)solvedCount / numSeeds;
float avgFitness = bestFitnesses.Average();

float fitness = (solveRate * 100f) + avgFitness;

Console.WriteLine(fitness.ToString("F6", CultureInfo.InvariantCulture));
