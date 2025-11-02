namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration parameters for evolutionary algorithm.
/// Defines population structure, selection pressure, mutation rates, and culling thresholds.
/// </summary>
public class EvolutionConfig
{
    /// <summary>
    /// Number of species in the population.
    /// Default: 20 (NEAT-style: more species enables topology diversity)
    /// Previous: 4 (Phase 6: 4 species × 200 individuals beats 8 × 100 by +3.3%)
    /// </summary>
    public int SpeciesCount { get; set; } = 20;

    /// <summary>
    /// Minimum number of species to maintain (prevents complete collapse).
    /// Default: 8 (NEAT-style: ensures floor of diversity)
    /// Previous: 2
    /// </summary>
    public int MinSpeciesCount { get; set; } = 8;

    /// <summary>
    /// Number of individuals per species.
    /// Default: 40 (NEAT-style: smaller niches force competition)
    /// Previous: 200 (Phase 6: larger populations per species = more robust evolution)
    /// Total population: SpeciesCount × IndividualsPerSpecies = 800 (unchanged)
    /// </summary>
    public int IndividualsPerSpecies { get; set; } = 40;

    /// <summary>
    /// Number of elite individuals preserved unchanged each generation.
    /// Default: 2 (Phase 2: low elitism = better exploration. More elites hurt performance -0.264 correlation)
    /// </summary>
    public int Elites { get; set; } = 2;

    /// <summary>
    /// Tournament size for selection pressure.
    /// Higher values = stronger selection.
    /// Default: 16 (Phase 2: CRITICAL parameter, +0.743 correlation with improvement)
    /// </summary>
    public int TournamentSize { get; set; } = 16;

    /// <summary>
    /// Percentage of top individuals eligible as parents (0.0 to 1.0).
    /// Only the top X% by fitness can be selected as parents.
    /// Default: 1.0 (100% - all individuals eligible)
    /// </summary>
    public float ParentPoolPercentage { get; set; } = 1.0f;

    /// <summary>
    /// Number of generations a new species is protected from culling.
    /// Default: 1 (allows culling to start at generation 2+)
    /// Previous: 3
    /// </summary>
    public int GraceGenerations { get; set; } = 1;

    /// <summary>
    /// Number of generations without improvement before species eligible for culling.
    /// Default: 6 (NEAT-style: aggressive turnover enables topology exploration)
    /// Previous: 15
    /// </summary>
    public int StagnationThreshold { get; set; } = 6;

    /// <summary>
    /// Minimum fitness variance required to avoid low-diversity culling.
    /// Default: 0.08 (NEAT-style: catches convergence earlier)
    /// Previous: 0.15
    /// </summary>
    public float SpeciesDiversityThreshold { get; set; } = 0.08f;

    /// <summary>
    /// Relative performance threshold for culling eligibility.
    /// Species with median fitness below this fraction of the best species are eligible.
    /// Default: 0.7 (NEAT-style: more aggressive than previous 0.5)
    /// Previous: 0.5 (50%)
    /// </summary>
    public float RelativePerformanceThreshold { get; set; } = 0.7f;

    /// <summary>
    /// Mutation rate configuration.
    /// </summary>
    public MutationRates MutationRates { get; set; } = new MutationRates();

    /// <summary>
    /// Edge topology mutation configuration.
    /// </summary>
    public EdgeMutationConfig EdgeMutations { get; set; } = new EdgeMutationConfig();
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
    /// Default: 0.2 (Phase 7: increased from 0.1, provides +15.1% improvement)
    /// </summary>
    public float WeightL1Shrink { get; set; } = 0.2f;

    /// <summary>
    /// Shrinkage factor for L1 regularization.
    /// Default: 0.9 (reduce magnitude by 10%)
    /// </summary>
    public float L1ShrinkFactor { get; set; } = 0.9f;

    /// <summary>
    /// Probability of swapping activation function.
    /// Default: 0.10 (Phase 7: increased from 0.01, mixed activations provide +33.3% improvement)
    /// </summary>
    public float ActivationSwap { get; set; } = 0.10f;

    /// <summary>
    /// Probability of mutating node parameters (alpha, beta, etc.).
    /// Default: 0.0 (Phase 7: disabled - disabling provides +38.9% improvement)
    /// </summary>
    public float NodeParamMutate { get; set; } = 0.0f;

    /// <summary>
    /// Standard deviation for node parameter jitter.
    /// Default: 0.1
    /// </summary>
    public float NodeParamStdDev { get; set; } = 0.1f;
}
