namespace Evolvatron.Evolvion;

/// <summary>
/// Elitism operators preserve top-performing individuals across generations.
/// Elite individuals are copied unchanged (no mutation).
/// </summary>
public static class Elitism
{
    /// <summary>
    /// Extract elite individuals (top N by fitness) from the population.
    /// Elites are sorted by fitness (best first).
    /// </summary>
    /// <param name="individuals">Population to select from.</param>
    /// <param name="eliteCount">Number of elites to preserve.</param>
    /// <returns>List of elite individuals.</returns>
    public static List<Individual> PreserveElites(
        List<Individual> individuals,
        int eliteCount)
    {
        if (eliteCount <= 0)
            return new List<Individual>();

        if (eliteCount >= individuals.Count)
            return Selection.RankByFitness(individuals);

        return individuals
            .OrderByDescending(i => i.Fitness)
            .Take(eliteCount)
            .ToList();
    }

    /// <summary>
    /// Copy elite individuals from source to destination population.
    /// Elites are exact clones (no mutation applied).
    /// </summary>
    /// <param name="source">Source population containing elites.</param>
    /// <param name="destination">Destination list to populate with elites.</param>
    /// <param name="eliteCount">Number of elites to copy.</param>
    public static void CopyElites(
        List<Individual> source,
        List<Individual> destination,
        int eliteCount)
    {
        var elites = PreserveElites(source, eliteCount);
        destination.AddRange(elites);
    }

    /// <summary>
    /// Create a new generation combining elites and offspring.
    /// Elites are preserved unchanged; offspring are selected via tournament and mutated.
    /// </summary>
    /// <param name="currentGen">Current generation individuals.</param>
    /// <param name="eliteCount">Number of elites to preserve.</param>
    /// <param name="populationSize">Target population size.</param>
    /// <param name="tournamentSize">Tournament size for offspring selection.</param>
    /// <param name="random">Random number generator.</param>
    /// <returns>New generation (elites + selected offspring, unmutated).</returns>
    public static List<Individual> CreateNextGeneration(
        List<Individual> currentGen,
        int eliteCount,
        int populationSize,
        int tournamentSize,
        Random random)
    {
        var nextGen = new List<Individual>(populationSize);

        // Step 1: Preserve elites
        var elites = PreserveElites(currentGen, eliteCount);
        nextGen.AddRange(elites);

        // Step 2: Generate offspring via tournament selection
        int offspringCount = populationSize - elites.Count;
        if (offspringCount > 0)
        {
            var offspring = Selection.GenerateOffspring(
                currentGen,
                offspringCount,
                tournamentSize,
                random);
            nextGen.AddRange(offspring);
        }

        return nextGen;
    }

    /// <summary>
    /// Verify that elites are exact copies (no mutation occurred).
    /// Used for debugging and validation.
    /// </summary>
    /// <param name="elites">Elite individuals to verify.</param>
    /// <param name="nextGen">Next generation containing the elites.</param>
    /// <returns>True if all elites are present and unchanged.</returns>
    public static bool VerifyElitesPreserved(
        List<Individual> elites,
        List<Individual> nextGen)
    {
        if (elites.Count > nextGen.Count)
            return false;

        // Check that top N individuals in nextGen match elites
        for (int i = 0; i < elites.Count; i++)
        {
            if (!AreIndividualsIdentical(elites[i], nextGen[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Check if two individuals are identical (same weights, params, activations).
    /// </summary>
    private static bool AreIndividualsIdentical(Individual a, Individual b)
    {
        if (a.Weights.Length != b.Weights.Length)
            return false;
        if (a.NodeParams.Length != b.NodeParams.Length)
            return false;
        if (a.Activations.Length != b.Activations.Length)
            return false;

        for (int i = 0; i < a.Weights.Length; i++)
        {
            if (Math.Abs(a.Weights[i] - b.Weights[i]) > 1e-9f)
                return false;
        }

        for (int i = 0; i < a.NodeParams.Length; i++)
        {
            if (Math.Abs(a.NodeParams[i] - b.NodeParams[i]) > 1e-9f)
                return false;
        }

        for (int i = 0; i < a.Activations.Length; i++)
        {
            if (a.Activations[i] != b.Activations[i])
                return false;
        }

        return true;
    }
}
