namespace Evolvatron.Evolvion;

/// <summary>
/// Tracks stagnation and performance metrics for a species.
/// Used to determine eligibility for adaptive culling.
/// </summary>
public struct SpeciesStats
{
    /// <summary>
    /// Historical peak fitness achieved by this species.
    /// </summary>
    public float BestFitnessEver;

    /// <summary>
    /// Number of generations since last improvement to BestFitnessEver.
    /// Used for stagnation detection.
    /// </summary>
    public int GenerationsSinceImprovement;

    /// <summary>
    /// Rolling window of median fitness values (last 10 generations).
    /// Used for trend analysis.
    /// </summary>
    public float[] FitnessHistory;

    /// <summary>
    /// Current generation's median fitness.
    /// </summary>
    public float MedianFitness;

    /// <summary>
    /// Current generation's fitness variance across individuals.
    /// Measures diversity within the species.
    /// </summary>
    public float FitnessVariance;

    /// <summary>
    /// Initialize a new SpeciesStats instance.
    /// </summary>
    public SpeciesStats()
    {
        BestFitnessEver = float.MinValue;
        GenerationsSinceImprovement = 0;
        FitnessHistory = new float[10];
        MedianFitness = 0f;
        FitnessVariance = 0f;
    }
}
