namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration parameters for evolutionary algorithm.
/// Defines population structure, selection pressure, mutation rates, and culling thresholds.
/// </summary>
public class EvolutionConfig
{
    /// <summary>
    /// Number of species in the population.
    /// Default: 27 (Phase 10 Optuna: optimized via Bayesian search)
    /// Previous: 20 (NEAT-style), 4 (Phase 6)
    /// </summary>
    public int SpeciesCount { get; set; } = 27;

    /// <summary>
    /// Minimum number of species to maintain (prevents complete collapse).
    /// Default: 8 (NEAT-style: ensures floor of diversity)
    /// Previous: 2
    /// </summary>
    public int MinSpeciesCount { get; set; } = 8;

    /// <summary>
    /// Number of individuals per species.
    /// Default: 88 (Phase 10 Optuna: optimized via Bayesian search)
    /// Previous: 40 (NEAT-style), 200 (Phase 6)
    /// Total population: SpeciesCount × IndividualsPerSpecies = 2376
    /// </summary>
    public int IndividualsPerSpecies { get; set; } = 88;

    /// <summary>
    /// Number of elite individuals preserved unchanged each generation.
    /// Default: 4 (Phase 10 Optuna: optimized via Bayesian search)
    /// Previous: 2 (Phase 2: low elitism preferred, but larger populations benefit from more elites)
    /// </summary>
    public int Elites { get; set; } = 4;

    /// <summary>
    /// Tournament size for selection pressure.
    /// Higher values = stronger selection.
    /// Default: 22 (Phase 10 Optuna: optimized via Bayesian search)
    /// Previous: 16 (Phase 2: +0.743 correlation with improvement)
    /// </summary>
    public int TournamentSize { get; set; } = 22;

    /// <summary>
    /// Percentage of top individuals eligible as parents (0.0 to 1.0).
    /// Only the top X% by fitness can be selected as parents.
    /// Default: 0.75 (Phase 10 Optuna: restricts parent pool for stronger selection pressure)
    /// Previous: 1.0 (100% - all individuals eligible)
    /// </summary>
    public float ParentPoolPercentage { get; set; } = 0.75f;

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
    /// Default: 0.066 (Phase 10 Optuna: optimized via Bayesian search)
    /// Previous: 0.08 (NEAT-style), 0.15 (baseline)
    /// </summary>
    public float SpeciesDiversityThreshold { get; set; } = 0.066f;

    /// <summary>
    /// Relative performance threshold for culling eligibility.
    /// Species with median fitness below this fraction of the best species are eligible.
    /// Default: 0.885 (Phase 10 Optuna: very aggressive culling of underperformers)
    /// Previous: 0.7 (NEAT-style), 0.5 (baseline)
    /// </summary>
    public float RelativePerformanceThreshold { get; set; } = 0.885f;

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
    /// Default: 0.972 (Phase 10 Optuna: very high exploration rate)
    /// Previous: 0.95 (XOR sweep)
    /// </summary>
    public float WeightJitter { get; set; } = 0.972f;

    /// <summary>
    /// Standard deviation for weight jitter, as fraction of weight magnitude.
    /// Default: 0.402 (Phase 10 Optuna: higher noise for stronger exploration)
    /// Previous: 0.3 (XOR sweep)
    /// </summary>
    public float WeightJitterStdDev { get; set; } = 0.402f;

    /// <summary>
    /// Probability of resetting a weight to random value.
    /// Default: 0.137 (Phase 10 Optuna: increased for more random exploration)
    /// Previous: 0.1 (XOR sweep)
    /// </summary>
    public float WeightReset { get; set; } = 0.137f;

    /// <summary>
    /// Probability of shrinking weight magnitude toward zero.
    /// Default: 0.090 (Phase 10 Optuna: reduced regularization pressure)
    /// Previous: 0.2 (Phase 7: +15.1% improvement)
    /// </summary>
    public float WeightL1Shrink { get; set; } = 0.090f;

    /// <summary>
    /// Shrinkage factor for L1 regularization.
    /// Default: 0.949 (Phase 10 Optuna: slightly stronger shrinkage when applied)
    /// Previous: 0.9 (reduce magnitude by 10%)
    /// </summary>
    public float L1ShrinkFactor { get; set; } = 0.949f;

    /// <summary>
    /// Probability of swapping activation function.
    /// Default: 0.186 (Phase 10 Optuna: strong activation diversity preference)
    /// Previous: 0.10 (Phase 7: +33.3% improvement)
    /// </summary>
    public float ActivationSwap { get; set; } = 0.186f;

    /// <summary>
    /// Probability of mutating node parameters (alpha, beta, etc.).
    /// Default: 0.022 (Phase 10 Optuna: minimal but non-zero)
    /// Previous: 0.0 (Phase 7: disabled was best, but Optuna found slight benefit)
    /// </summary>
    public float NodeParamMutate { get; set; } = 0.022f;

    /// <summary>
    /// Standard deviation for node parameter jitter.
    /// Default: 0.053 (Phase 10 Optuna: small adjustments when enabled)
    /// Previous: 0.1
    /// </summary>
    public float NodeParamStdDev { get; set; } = 0.053f;
}
