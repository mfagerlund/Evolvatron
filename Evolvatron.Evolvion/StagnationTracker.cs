namespace Evolvatron.Evolvion;

/// <summary>
/// Tracks species performance and stagnation metrics over generations.
/// Used to determine eligibility for adaptive culling.
/// </summary>
public static class StagnationTracker
{
    /// <summary>
    /// Update species statistics after fitness evaluation.
    /// Computes median, variance, and stagnation counters.
    /// </summary>
    /// <param name="species">Species to update.</param>
    public static void UpdateSpeciesStats(Species species)
    {
        if (species.Individuals.Count == 0)
            return;

        // Step 1: Compute median fitness
        var fitnesses = species.Individuals
            .Select(i => i.Fitness)
            .OrderBy(f => f)
            .ToArray();

        species.Stats.MedianFitness = fitnesses[fitnesses.Length / 2];

        // Step 2: Compute fitness variance (diversity metric)
        float mean = fitnesses.Average();
        float variance = fitnesses.Sum(f => (f - mean) * (f - mean)) / fitnesses.Length;
        species.Stats.FitnessVariance = variance;

        // Step 3: Update best fitness ever and stagnation counter
        float currentBest = fitnesses.Max();
        if (currentBest > species.Stats.BestFitnessEver)
        {
            species.Stats.BestFitnessEver = currentBest;
            species.Stats.GenerationsSinceImprovement = 0;
        }
        else
        {
            species.Stats.GenerationsSinceImprovement++;
        }

        // Step 4: Update fitness history (rolling window of last 10 generations)
        UpdateFitnessHistory(ref species.Stats, species.Stats.MedianFitness);
    }

    /// <summary>
    /// Update the rolling fitness history window.
    /// Shifts old values and adds new median fitness.
    /// </summary>
    private static void UpdateFitnessHistory(ref SpeciesStats stats, float newMedianFitness)
    {
        if (stats.FitnessHistory == null || stats.FitnessHistory.Length == 0)
        {
            stats.FitnessHistory = new float[10];
        }

        // Shift all values left and add new value at the end
        for (int i = 0; i < stats.FitnessHistory.Length - 1; i++)
        {
            stats.FitnessHistory[i] = stats.FitnessHistory[i + 1];
        }
        stats.FitnessHistory[^1] = newMedianFitness;
    }

    /// <summary>
    /// Check if a species is stagnant (no improvement for threshold generations).
    /// </summary>
    /// <param name="species">Species to check.</param>
    /// <param name="threshold">Stagnation threshold in generations.</param>
    /// <returns>True if stagnant.</returns>
    public static bool IsStagnant(Species species, int threshold)
    {
        return species.Stats.GenerationsSinceImprovement >= threshold;
    }

    /// <summary>
    /// Check if a species has low diversity (fitness variance below threshold).
    /// </summary>
    /// <param name="species">Species to check.</param>
    /// <param name="threshold">Minimum variance threshold.</param>
    /// <returns>True if low diversity.</returns>
    public static bool HasLowDiversity(Species species, float threshold)
    {
        return species.Stats.FitnessVariance < threshold;
    }

    /// <summary>
    /// Check if a species is past the grace period.
    /// </summary>
    /// <param name="species">Species to check.</param>
    /// <param name="graceGenerations">Grace period in generations.</param>
    /// <returns>True if past grace period.</returns>
    public static bool IsPastGracePeriod(Species species, int graceGenerations)
    {
        return species.Age > graceGenerations;
    }

    /// <summary>
    /// Compute relative performance compared to best species.
    /// </summary>
    /// <param name="species">Species to evaluate.</param>
    /// <param name="bestMedianFitness">Median fitness of best-performing species.</param>
    /// <returns>Ratio of species median to best median (0.0 to 1.0+).</returns>
    public static float ComputeRelativePerformance(Species species, float bestMedianFitness)
    {
        if (bestMedianFitness <= 0f)
            return 1f; // Avoid division by zero

        return species.Stats.MedianFitness / bestMedianFitness;
    }

    /// <summary>
    /// Get the fitness trend over the last N generations.
    /// </summary>
    /// <param name="species">Species to analyze.</param>
    /// <returns>Slope of linear regression over fitness history (positive = improving).</returns>
    public static float GetFitnessTrend(Species species)
    {
        if (species.Stats.FitnessHistory == null || species.Stats.FitnessHistory.Length < 2)
            return 0f;

        // Simple linear regression: slope = cov(x,y) / var(x)
        int n = species.Stats.FitnessHistory.Length;
        float meanX = (n - 1) / 2f;
        float meanY = species.Stats.FitnessHistory.Average();

        float covariance = 0f;
        float variance = 0f;

        for (int i = 0; i < n; i++)
        {
            float dx = i - meanX;
            float dy = species.Stats.FitnessHistory[i] - meanY;
            covariance += dx * dy;
            variance += dx * dx;
        }

        return variance > 0f ? covariance / variance : 0f;
    }

    /// <summary>
    /// Update statistics for all species in a population.
    /// </summary>
    /// <param name="population">Population to update.</param>
    public static void UpdateAllSpecies(Population population)
    {
        foreach (var species in population.AllSpecies)
        {
            UpdateSpeciesStats(species);
        }
    }
}
