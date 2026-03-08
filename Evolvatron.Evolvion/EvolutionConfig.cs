namespace Evolvatron.Evolvion;

/// <summary>
/// Configuration parameters for evolutionary algorithm.
/// Defines population structure, selection pressure, mutation rates, and culling thresholds.
/// </summary>
public class EvolutionConfig
{
    /// <summary>
    /// Number of species in the population.
    /// Default: 3 (Phase B2: best tradeoff between topology diversity and generation throughput)
    /// Species provide essential topology exploration — different structures compete to find
    /// what works for the problem. Each species gets full GPU-optimal pop (free throughput).
    /// Phase B2 dp_6_6_6: 3sp=8/10 vs 1sp=7/10 vs 6sp=6/10 (120s budget, 10 seeds)
    /// </summary>
    public int SpeciesCount { get; set; } = 3;

    /// <summary>
    /// Minimum number of species to maintain (prevents complete collapse).
    /// Default: 2 (always keep at least 2 competing topologies for diversity)
    /// </summary>
    public int MinSpeciesCount { get; set; } = 2;

    /// <summary>
    /// Number of individuals per species.
    /// Default: 132 (Phase 10 Rosenbrock: Trial 138 - larger population for valley exploration)
    /// Previous: 88 (Phase 10 spiral), 40 (NEAT-style), 200 (Phase 6)
    /// Total population: SpeciesCount × IndividualsPerSpecies = 5148
    /// </summary>
    public int IndividualsPerSpecies { get; set; } = 132;

    /// <summary>
    /// Number of elite individuals preserved unchanged each generation.
    /// Default: 10 (Sparse Study Phase D: used across all winning configs)
    /// Previous: 5 (Rosenbrock Trial 138), 4 (spiral), 2 (Phase 2)
    /// </summary>
    public int Elites { get; set; } = 10;

    /// <summary>
    /// Tournament size for selection pressure.
    /// Higher values = stronger selection.
    /// Default: 5 (Sparse Study Phase D: lower pressure avoids premature convergence)
    /// Previous: 10 (Rosenbrock Trial 138), 22 (spiral), 16 (Phase 2)
    /// </summary>
    public int TournamentSize { get; set; } = 5;

    /// <summary>
    /// Percentage of top individuals eligible as parents (0.0 to 1.0).
    /// Only the top X% by fitness can be selected as parents.
    /// Default: 0.5 (Sparse Study Phase D: used across all winning configs)
    /// Previous: 0.593 (Rosenbrock Trial 138), 0.75 (spiral), 1.0 (Phase 2)
    /// </summary>
    public float ParentPoolPercentage { get; set; } = 0.5f;

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
    /// Default: 0.15 (Sparse Study: 18x improvement over 0.058 for DPNV, validated across 500+ runs)
    /// Previous: 0.058 (Rosenbrock Trial 138), 0.402 (spiral), 0.3 (XOR sweep)
    /// </summary>
    public float WeightJitterStdDev { get; set; } = 0.15f;

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
