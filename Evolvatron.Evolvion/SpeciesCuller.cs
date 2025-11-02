namespace Evolvatron.Evolvion;

/// <summary>
/// Adaptive species culling system.
/// Removes underperforming, stagnant species with low diversity.
/// </summary>
public static class SpeciesCuller
{
    /// <summary>
    /// Cull stagnant species from the population and replace with diversified offspring.
    /// Species are culled if grace period has passed AND ANY other criterion is met (NEAT-style OR logic):
    /// 1. Age > grace period (REQUIRED)
    /// 2. No improvement for stagnation threshold generations (OR)
    /// 3. Best individual fitness < relativePerformanceThreshold * overall best fitness (OR)
    /// 4. Fitness variance < diversity threshold (OR)
    /// NEVER culls the species containing the best individual.
    /// </summary>
    /// <param name="population">Population to cull from.</param>
    /// <param name="config">Evolution configuration.</param>
    /// <param name="random">Random number generator.</param>
    public static void CullStagnantSpecies(
        Population population,
        EvolutionConfig config,
        Random random)
    {
        // Don't cull if at minimum species count
        if (population.AllSpecies.Count <= config.MinSpeciesCount)
            return;

        // Find the species containing the best individual - it must never be culled
        Species? bestSpecies = null;
        float bestFitness = float.MinValue;
        foreach (var species in population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                if (individual.Fitness > bestFitness)
                {
                    bestFitness = individual.Fitness;
                    bestSpecies = species;
                }
            }
        }

        // Find eligible species
        var eligible = FindEligibleForCulling(population, config);

        // Remove best species from eligible list
        if (bestSpecies != null)
        {
            eligible.Remove(bestSpecies);
        }

        // Only cull if at least 2 species are eligible after removing best
        if (eligible.Count < 2)
            return;

        // Cull the worst performer among eligible species (by best individual fitness)
        var worstSpecies = eligible
            .OrderBy(s => s.Stats.BestFitnessEver)
            .First();

        population.AllSpecies.Remove(worstSpecies);

        // Replace with diversified offspring from top-2 species
        var newSpecies = SpeciesDiversification.CreateDiversifiedSpecies(
            population,
            config,
            random);

        population.AllSpecies.Add(newSpecies);
        population.TotalSpeciesCreated++;
    }

    /// <summary>
    /// Find all species eligible for culling based on NEAT-style OR logic.
    /// Grace period is required, but then ANY other condition triggers culling.
    /// </summary>
    /// <param name="population">Population to evaluate.</param>
    /// <param name="config">Evolution configuration.</param>
    /// <returns>List of eligible species.</returns>
    public static List<Species> FindEligibleForCulling(
        Population population,
        EvolutionConfig config)
    {
        if (population.AllSpecies.Count == 0)
            return new List<Species>();

        // Find best fitness across all species (by best individual in each species)
        float bestFitnessEver = population.AllSpecies
            .Max(s => s.Stats.BestFitnessEver);

        var eligible = new List<Species>();

        foreach (var species in population.AllSpecies)
        {
            // Grace period is REQUIRED (must be past it)
            if (!StagnationTracker.IsPastGracePeriod(species, config.GraceGenerations))
                continue;

            // NEAT-style OR logic: ANY of these conditions triggers culling
            bool isStagnant = StagnationTracker.IsStagnant(species, config.StagnationThreshold);

            // Calculate relative performance (works for both positive and negative fitness)
            // For negative fitness (e.g., MSE loss), normalize the gap between species and best
            float relativePerf;
            if (bestFitnessEver >= 0)
            {
                // Positive fitness: standard ratio
                relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);
            }
            else
            {
                // Negative fitness: normalize gap (species is worse if more negative)
                // relativePerf = 1.0 means equal, < 1.0 means worse
                float gap = Math.Abs(bestFitnessEver - species.Stats.BestFitnessEver);
                relativePerf = 1.0f - gap / (Math.Abs(bestFitnessEver) + 1e-9f);
            }
            bool belowPerformanceThreshold = relativePerf < config.RelativePerformanceThreshold;

            bool hasLowDiversity = StagnationTracker.HasLowDiversity(species, config.SpeciesDiversityThreshold);

            // If ANY condition is true, species is eligible for culling
            bool shouldCull = isStagnant || belowPerformanceThreshold || hasLowDiversity;

            if (shouldCull)
            {
                eligible.Add(species);
            }
        }

        return eligible;
    }

    /// <summary>
    /// Check if a specific species is eligible for culling (NEAT-style OR logic).
    /// </summary>
    /// <param name="species">Species to check.</param>
    /// <param name="population">Population context (for relative performance).</param>
    /// <param name="config">Evolution configuration.</param>
    /// <returns>True if eligible.</returns>
    public static bool IsEligibleForCulling(
        Species species,
        Population population,
        EvolutionConfig config)
    {
        // Grace period check (REQUIRED)
        if (!StagnationTracker.IsPastGracePeriod(species, config.GraceGenerations))
            return false;

        // NEAT-style OR logic: ANY of these conditions triggers culling
        bool isStagnant = StagnationTracker.IsStagnant(species, config.StagnationThreshold);

        float bestFitnessEver = population.AllSpecies
            .Max(s => s.Stats.BestFitnessEver);

        // Calculate relative performance (works for both positive and negative fitness)
        float relativePerf;
        if (bestFitnessEver >= 0)
        {
            relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);
        }
        else
        {
            float gap = Math.Abs(bestFitnessEver - species.Stats.BestFitnessEver);
            relativePerf = 1.0f - gap / (Math.Abs(bestFitnessEver) + 1e-9f);
        }
        bool belowPerformanceThreshold = relativePerf < config.RelativePerformanceThreshold;

        bool hasLowDiversity = StagnationTracker.HasLowDiversity(species, config.SpeciesDiversityThreshold);

        // If ANY condition is true, species is eligible for culling
        return isStagnant || belowPerformanceThreshold || hasLowDiversity;
    }

    /// <summary>
    /// Get culling status report for all species.
    /// Useful for debugging and visualization.
    /// </summary>
    /// <param name="population">Population to analyze.</param>
    /// <param name="config">Evolution configuration.</param>
    /// <returns>Dictionary mapping species to eligibility reasons.</returns>
    public static Dictionary<Species, CullingStatus> GetCullingReport(
        Population population,
        EvolutionConfig config)
    {
        var report = new Dictionary<Species, CullingStatus>();
        float bestFitnessEver = population.AllSpecies.Max(s => s.Stats.BestFitnessEver);

        foreach (var species in population.AllSpecies)
        {
            // Calculate relative performance (works for both positive and negative fitness)
            float relativePerf;
            if (bestFitnessEver >= 0)
            {
                // Positive fitness: standard ratio
                relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);
            }
            else
            {
                // Negative fitness: normalize gap (species is worse if more negative)
                float gap = Math.Abs(bestFitnessEver - species.Stats.BestFitnessEver);
                relativePerf = 1.0f - gap / (Math.Abs(bestFitnessEver) + 1e-9f);
            }

            var status = new CullingStatus
            {
                PastGracePeriod = StagnationTracker.IsPastGracePeriod(
                    species,
                    config.GraceGenerations),

                IsStagnant = StagnationTracker.IsStagnant(
                    species,
                    config.StagnationThreshold),

                RelativePerformance = relativePerf,

                BelowPerformanceThreshold = relativePerf < config.RelativePerformanceThreshold,

                HasLowDiversity = StagnationTracker.HasLowDiversity(
                    species,
                    config.SpeciesDiversityThreshold),

                FitnessVariance = species.Stats.FitnessVariance
            };

            // NEAT-style OR logic: grace period required, then ANY other condition
            status.IsEligible =
                status.PastGracePeriod &&
                (status.IsStagnant || status.BelowPerformanceThreshold || status.HasLowDiversity);

            report[species] = status;
        }

        return report;
    }
}

/// <summary>
/// Culling eligibility status for a species.
/// </summary>
public struct CullingStatus
{
    public bool PastGracePeriod;
    public bool IsStagnant;
    public float RelativePerformance;
    public bool BelowPerformanceThreshold;
    public bool HasLowDiversity;
    public float FitnessVariance;
    public bool IsEligible;
}
