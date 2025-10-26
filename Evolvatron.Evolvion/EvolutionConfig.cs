namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration parameters for evolutionary algorithm.
/// Defines population structure, selection pressure, mutation rates, and culling thresholds.
/// </summary>
public class EvolutionConfig
{
    /// <summary>
    /// Number of species in the population.
    /// Default: 8
    /// </summary>
    public int SpeciesCount { get; set; } = 8;

    /// <summary>
    /// Minimum number of species to maintain (prevents complete collapse).
    /// Default: 4
    /// </summary>
    public int MinSpeciesCount { get; set; } = 4;

    /// <summary>
    /// Number of individuals per species.
    /// Default: 100 (based on XOR hyperparameter sweep: 400 total pop = 4 species × 100)
    /// </summary>
    public int IndividualsPerSpecies { get; set; } = 100;

    /// <summary>
    /// Number of elite individuals preserved unchanged each generation.
    /// Default: 4
    /// </summary>
    public int Elites { get; set; } = 4;

    /// <summary>
    /// Tournament size for selection pressure.
    /// Higher values = stronger selection.
    /// Default: 4
    /// </summary>
    public int TournamentSize { get; set; } = 4;

    /// <summary>
    /// Number of generations a new species is protected from culling.
    /// Default: 3
    /// </summary>
    public int GraceGenerations { get; set; } = 3;

    /// <summary>
    /// Number of generations without improvement before species eligible for culling.
    /// Default: 15
    /// </summary>
    public int StagnationThreshold { get; set; } = 15;

    /// <summary>
    /// Minimum fitness variance required to avoid low-diversity culling.
    /// Default: 0.15
    /// </summary>
    public float SpeciesDiversityThreshold { get; set; } = 0.15f;

    /// <summary>
    /// Relative performance threshold for culling eligibility.
    /// Species with median fitness below this fraction of the best species are eligible.
    /// Default: 0.5 (50%)
    /// </summary>
    public float RelativePerformanceThreshold { get; set; } = 0.5f;

    /// <summary>
    /// Weight initialization method.
    /// Default: "GlorotUniform"
    /// </summary>
    public string WeightInitialization { get; set; } = "GlorotUniform";

    /// <summary>
    /// Mutation rate configuration.
    /// </summary>
    public MutationRates MutationRates { get; set; } = new MutationRates();

    /// <summary>
    /// Edge topology mutation configuration.
    /// </summary>
    public EdgeMutationConfig EdgeMutations { get; set; } = new EdgeMutationConfig();

    /// <summary>
    /// Number of evaluation seeds per individual.
    /// Default: 5
    /// </summary>
    public int SeedsPerIndividual { get; set; } = 5;

    /// <summary>
    /// Fitness aggregation method: "Mean" or "CVaR50".
    /// Default: "CVaR50"
    /// </summary>
    public string FitnessAggregation { get; set; } = "CVaR50";
}

/// <summary>
/// Weight-level mutation rates (per individual).
/// </summary>
public class MutationRates
{
    /// <summary>
    /// Probability of applying Gaussian noise to each weight.
    /// Default: 0.95 (based on XOR hyperparameter sweep)
    /// </summary>
    public float WeightJitter { get; set; } = 0.95f;

    /// <summary>
    /// Standard deviation for weight jitter, as fraction of weight magnitude.
    /// Default: 0.3 (30% of weight - based on XOR hyperparameter sweep)
    /// </summary>
    public float WeightJitterStdDev { get; set; } = 0.3f;

    /// <summary>
    /// Probability of resetting a weight to random value.
    /// Default: 0.1 (based on XOR hyperparameter sweep)
    /// </summary>
    public float WeightReset { get; set; } = 0.1f;

    /// <summary>
    /// Probability of shrinking weight magnitude toward zero.
    /// Default: 0.1
    /// </summary>
    public float WeightL1Shrink { get; set; } = 0.1f;

    /// <summary>
    /// Shrinkage factor for L1 regularization.
    /// Default: 0.9 (reduce magnitude by 10%)
    /// </summary>
    public float L1ShrinkFactor { get; set; } = 0.9f;

    /// <summary>
    /// Probability of swapping activation function.
    /// Default: 0.01
    /// </summary>
    public float ActivationSwap { get; set; } = 0.01f;

    /// <summary>
    /// Probability of mutating node parameters (alpha, beta, etc.).
    /// Default: 0.2
    /// </summary>
    public float NodeParamMutate { get; set; } = 0.2f;

    /// <summary>
    /// Standard deviation for node parameter jitter.
    /// Default: 0.1
    /// </summary>
    public float NodeParamStdDev { get; set; } = 0.1f;
}
