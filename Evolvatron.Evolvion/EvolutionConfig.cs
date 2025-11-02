namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration parameters for evolutionary algorithm.
/// Defines population structure, selection pressure, mutation rates, and culling thresholds.
/// </summary>
public class EvolutionConfig
{
    /// <summary>
    /// Number of species in the population.
    /// Default: 39 (Phase 10 Rosenbrock: Trial 138 - valley navigation champion)
    /// Previous: 27 (Phase 10 spiral), 20 (NEAT-style), 4 (Phase 6)
    /// </summary>
    public int SpeciesCount { get; set; } = 39;

    /// <summary>
    /// Minimum number of species to maintain (prevents complete collapse).
    /// Default: 13 (Phase 10 Rosenbrock: Trial 138 - higher diversity floor)
    /// Previous: 8 (NEAT-style), 2
    /// </summary>
    public int MinSpeciesCount { get; set; } = 13;

    /// <summary>
    /// Number of individuals per species.
    /// Default: 132 (Phase 10 Rosenbrock: Trial 138 - larger population for valley exploration)
    /// Previous: 88 (Phase 10 spiral), 40 (NEAT-style), 200 (Phase 6)
    /// Total population: SpeciesCount × IndividualsPerSpecies = 5148
    /// </summary>
    public int IndividualsPerSpecies { get; set; } = 132;

    /// <summary>
    /// Number of elite individuals preserved unchanged each generation.
    /// Default: 5 (Phase 10 Rosenbrock: Trial 138 - more elites for larger population)
    /// Previous: 4 (Phase 10 spiral), 2 (Phase 2)
    /// </summary>
    public int Elites { get; set; } = 5;

    /// <summary>
    /// Tournament size for selection pressure.
    /// Higher values = stronger selection.
    /// Default: 10 (Phase 10 Rosenbrock: Trial 138 - CRITICAL for valley navigation!)
    /// Previous: 22 (Phase 10 spiral - too high for valleys), 16 (Phase 2)
    /// Lower selection pressure prevents premature convergence in narrow fitness landscapes.
    /// </summary>
    public int TournamentSize { get; set; } = 10;

    /// <summary>
    /// Percentage of top individuals eligible as parents (0.0 to 1.0).
    /// Only the top X% by fitness can be selected as parents.
    /// Default: 0.593 (Phase 10 Rosenbrock: Trial 138 - stronger parent filtering)
    /// Previous: 0.75 (Phase 10 spiral), 1.0 (Phase 2 - all eligible)
    /// </summary>
    public float ParentPoolPercentage { get; set; } = 0.593f;

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
    /// Default: 0.113 (Phase 10 Rosenbrock: Trial 138 - CRITICAL for valley diversity!)
    /// Previous: 0.066 (Phase 10 spiral), 0.08 (NEAT-style), 0.15 (baseline)
    /// Higher threshold prevents species collapse in narrow fitness landscapes.
    /// </summary>
    public float SpeciesDiversityThreshold { get; set; } = 0.113f;

    /// <summary>
    /// Relative performance threshold for culling eligibility.
    /// Species with median fitness below this fraction of the best species are eligible.
    /// Default: 0.627 (Phase 10 Rosenbrock: Trial 138 - more lenient for valley exploration)
    /// Previous: 0.885 (Phase 10 spiral - too aggressive for valleys), 0.7 (NEAT-style), 0.5 (baseline)
    /// </summary>
    public float RelativePerformanceThreshold { get; set; } = 0.627f;

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
    /// Default: 0.812 (Phase 10 Rosenbrock: Trial 138 - reduced for gentler mutations)
    /// Previous: 0.972 (Phase 10 spiral), 0.95 (XOR sweep)
    /// </summary>
    public float WeightJitter { get; set; } = 0.812f;

    /// <summary>
    /// Standard deviation for weight jitter, as fraction of weight magnitude.
    /// Default: 0.058 (Phase 10 Rosenbrock: Trial 138 - CRITICAL! 86% reduction for fine-grained valley navigation)
    /// Previous: 0.402 (Phase 10 spiral - too coarse for valleys), 0.3 (XOR sweep)
    /// Gentle mutations enable precise navigation of narrow fitness valleys.
    /// </summary>
    public float WeightJitterStdDev { get; set; } = 0.058f;

    /// <summary>
    /// Probability of resetting a weight to random value.
    /// Default: 0.212 (Phase 10 Rosenbrock: Trial 138 - increased for escaping local optima)
    /// Previous: 0.137 (Phase 10 spiral), 0.1 (XOR sweep)
    /// </summary>
    public float WeightReset { get; set; } = 0.212f;

    /// <summary>
    /// Probability of shrinking weight magnitude toward zero.
    /// Default: 0.288 (Phase 10 Rosenbrock: Trial 138 - much stronger network simplification)
    /// Previous: 0.090 (Phase 10 spiral), 0.2 (Phase 7)
    /// </summary>
    public float WeightL1Shrink { get; set; } = 0.288f;

    /// <summary>
    /// Shrinkage factor for L1 regularization.
    /// Default: 0.857 (Phase 10 Rosenbrock: Trial 138 - more aggressive shrinkage)
    /// Previous: 0.949 (Phase 10 spiral), 0.9 (Phase 7)
    /// </summary>
    public float L1ShrinkFactor { get; set; } = 0.857f;

    /// <summary>
    /// Probability of swapping activation function.
    /// Default: 0.150 (Phase 10 Rosenbrock: Trial 138 - reduced activation churn)
    /// Previous: 0.186 (Phase 10 spiral), 0.10 (Phase 7)
    /// </summary>
    public float ActivationSwap { get; set; } = 0.150f;

    /// <summary>
    /// Probability of mutating node parameters (alpha, beta, etc.).
    /// Default: 0.072 (Phase 10 Rosenbrock: Trial 138 - increased for node diversity)
    /// Previous: 0.022 (Phase 10 spiral), 0.0 (Phase 7)
    /// </summary>
    public float NodeParamMutate { get; set; } = 0.072f;

    /// <summary>
    /// Standard deviation for node parameter jitter.
    /// Default: 0.222 (Phase 10 Rosenbrock: Trial 138 - much larger when applied)
    /// Previous: 0.053 (Phase 10 spiral), 0.1
    /// Creates punctuated equilibrium: rare but large node parameter changes.
    /// </summary>
    public float NodeParamStdDev { get; set; } = 0.222f;
}
