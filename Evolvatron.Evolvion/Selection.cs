namespace Evolvatron.Evolvion;

/// <summary>
/// Selection operators for evolutionary algorithm.
/// Implements tournament selection and fitness ranking.
/// </summary>
public static class Selection
{
    /// <summary>
    /// Tournament selection: randomly select K individuals and return the best.
    /// Provides selection pressure proportional to tournament size.
    /// </summary>
    /// <param name="individuals">Pool of individuals to select from.</param>
    /// <param name="tournamentSize">Number of competitors per tournament.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>The selected individual (winner of tournament).</returns>
    public static Individual TournamentSelect(
        List<Individual> individuals,
        int tournamentSize,
        Random random)
    {
        if (individuals.Count == 0)
            throw new ArgumentException("Cannot select from empty population.");

        if (tournamentSize <= 0)
            throw new ArgumentException("Tournament size must be positive.");

        // Handle edge case: tournament larger than population
        int actualTournamentSize = Math.Min(tournamentSize, individuals.Count);

        // Select random competitors
        Individual best = individuals[random.Next(individuals.Count)];
        float bestFitness = best.Fitness;

        for (int i = 1; i < actualTournamentSize; i++)
        {
            var competitor = individuals[random.Next(individuals.Count)];
            if (competitor.Fitness > bestFitness)
            {
                best = competitor;
                bestFitness = competitor.Fitness;
            }
        }

        return best;
    }

    /// <summary>
    /// Rank individuals by fitness in descending order (best first).
    /// </summary>
    /// <param name="individuals">Individuals to rank.</param>
    /// <returns>List of individuals sorted by fitness (highest first).</returns>
    public static List<Individual> RankByFitness(List<Individual> individuals)
    {
        return individuals
            .OrderByDescending(i => i.Fitness)
            .ToList();
    }

    /// <summary>
    /// Generate offspring using tournament selection.
    /// </summary>
    /// <param name="individuals">Parent pool.</param>
    /// <param name="count">Number of offspring to generate.</param>
    /// <param name="tournamentSize">Tournament size for selection pressure.</param>
    /// <param name="random">Random number generator.</param>
    /// <param name="parentPoolPercentage">Top percentage eligible as parents (0.0 to 1.0).</param>
    /// <returns>List of selected parents (clones, to be mutated later).</returns>
    public static List<Individual> GenerateOffspring(
        List<Individual> individuals,
        int count,
        int tournamentSize,
        Random random,
        float parentPoolPercentage = 1.0f)
    {
        // Filter to top X% by fitness
        List<Individual> parentPool = individuals;
        if (parentPoolPercentage < 1.0f)
        {
            int poolSize = Math.Max(1, (int)(individuals.Count * parentPoolPercentage));
            parentPool = RankByFitness(individuals).Take(poolSize).ToList();
        }

        var offspring = new List<Individual>(count);

        for (int i = 0; i < count; i++)
        {
            var parent = TournamentSelect(parentPool, tournamentSize, random);
            var child = new Individual(parent); // Deep copy to avoid shared arrays
            offspring.Add(child);
        }

        return offspring;
    }

    /// <summary>
    /// Compute rank-based selection probabilities.
    /// Used for fitness-scale-independent selection.
    /// </summary>
    /// <param name="individuals">Individuals to compute probabilities for.</param>
    /// <returns>Array of selection probabilities (sums to 1.0).</returns>
    public static float[] ComputeRankProbabilities(List<Individual> individuals)
    {
        if (individuals.Count == 0)
            return Array.Empty<float>();

        int n = individuals.Count;
        var probabilities = new float[n];

        // Rank-based: P(i) = (n - rank(i)) / sum(1..n)
        // Higher fitness = lower rank number = higher probability
        var ranked = RankByFitness(individuals);
        float sum = n * (n + 1) / 2f;

        for (int i = 0; i < n; i++)
        {
            probabilities[i] = (n - i) / sum;
        }

        return probabilities;
    }
}
