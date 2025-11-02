namespace Evolvatron.Evolvion;

/// <summary>
/// Adaptive species culling system.
/// Removes underperforming, stagnant species with low diversity.
/// </summary>
public static class SpeciesCuller
{
    /// <summary>
    /// Cull stagnant species from the population and replace with diversified offspring.
    /// Species are culled only if ALL criteria are met:
    /// 1. Age > grace period
    /// 2. No improvement for stagnation threshold generations
    /// 3. Best individual fitness < relativePerformanceThreshold * overall best fitness
    /// 4. Fitness variance < diversity threshold
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
    /// Find all species eligible for culling based on 4 criteria.
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
            // Criterion 1: Past grace period
            if (!StagnationTracker.IsPastGracePeriod(species, config.GraceGenerations))
                continue;

            // Criterion 2: Stagnant (no improvement for threshold generations)
            if (!StagnationTracker.IsStagnant(species, config.StagnationThreshold))
                continue;

            // Criterion 3: Relative performance below threshold (using best individual)
            float relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);

            if (relativePerf >= config.RelativePerformanceThreshold)
                continue;

            // Criterion 4: Low diversity
            if (!StagnationTracker.HasLowDiversity(species, config.SpeciesDiversityThreshold))
                continue;

            // All criteria met - eligible for culling
            eligible.Add(species);
        }

        return eligible;
    }

    /// <summary>
    /// Check if a specific species is eligible for culling.
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
        // Grace period check
        if (!StagnationTracker.IsPastGracePeriod(species, config.GraceGenerations))
            return false;

        // Stagnation check
        if (!StagnationTracker.IsStagnant(species, config.StagnationThreshold))
            return false;

        // Relative performance check (using best individual fitness)
        float bestFitnessEver = population.AllSpecies
            .Max(s => s.Stats.BestFitnessEver);

        float relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);

        if (relativePerf >= config.RelativePerformanceThreshold)
            return false;

        // Diversity check
        if (!StagnationTracker.HasLowDiversity(species, config.SpeciesDiversityThreshold))
            return false;

        return true;
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
            float relativePerf = species.Stats.BestFitnessEver / (bestFitnessEver + 1e-9f);

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

            status.IsEligible =
                status.PastGracePeriod &&
                status.IsStagnant &&
                status.BelowPerformanceThreshold &&
                status.HasLowDiversity;

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
