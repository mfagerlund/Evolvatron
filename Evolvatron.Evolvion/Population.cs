namespace Evolvatron.Evolvion;

/// <summary>
/// Represents the entire population across all species.
/// Manages evolutionary progression through generations.
/// </summary>
public class Population
{
    /// <summary>
    /// All species in the population.
    /// </summary>
    public List<Species> AllSpecies { get; set; }

    /// <summary>
    /// Current generation number.
    /// </summary>
    public int Generation { get; set; }

    /// <summary>
    /// Total number of species created throughout evolution (including extinct species).
    /// Incremented each time a new species is born via SpeciesDiversification.
    /// </summary>
    public int TotalSpeciesCreated { get; set; }

    /// <summary>
    /// Configuration parameters for evolution.
    /// </summary>
    public EvolutionConfig Config { get; set; }

    /// <summary>
    /// Create a new population with the given configuration.
    /// </summary>
    public Population(EvolutionConfig config)
    {
        AllSpecies = new List<Species>();
        Generation = 0;
        Config = config;
    }

    /// <summary>
    /// Total number of individuals across all species.
    /// </summary>
    public int TotalIndividuals => AllSpecies.Sum(s => s.Individuals.Count);

    /// <summary>
    /// Get the best individual across all species.
    /// </summary>
    public (Individual individual, Species species)? GetBestIndividual()
    {
        Individual? best = null;
        Species? bestSpecies = null;
        float bestFitness = float.MinValue;

        foreach (var species in AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                if (individual.Fitness > bestFitness)
                {
                    bestFitness = individual.Fitness;
                    best = individual;
                    bestSpecies = species;
                }
            }
        }

        return best != null && bestSpecies != null ? (best.Value, bestSpecies) : null;
    }

    /// <summary>
    /// Get statistics summary across all species.
    /// </summary>
    public PopulationStats GetStatistics()
    {
        var allFitness = AllSpecies
            .SelectMany(s => s.Individuals)
            .Select(i => i.Fitness)
            .ToArray();

        if (allFitness.Length == 0)
        {
            return new PopulationStats
            {
                BestFitness = float.MinValue,
                MeanFitness = 0f,
                MedianFitness = 0f,
                WorstFitness = float.MaxValue
            };
        }

        Array.Sort(allFitness);

        return new PopulationStats
        {
            BestFitness = allFitness[^1],
            MeanFitness = allFitness.Average(),
            MedianFitness = allFitness[allFitness.Length / 2],
            WorstFitness = allFitness[0]
        };
    }
}

/// <summary>
/// Population-wide statistics.
/// </summary>
public struct PopulationStats
{
    public float BestFitness;
    public float MeanFitness;
    public float MedianFitness;
    public float WorstFitness;
}
