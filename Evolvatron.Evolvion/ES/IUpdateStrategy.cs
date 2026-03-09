namespace Evolvatron.Evolvion.ES;

/// <summary>
/// Shared interface for distribution-based update strategies.
/// Both CEM and ES maintain per-island (μ, σ) and update them from fitness results.
/// </summary>
public interface IUpdateStrategy
{
    void GenerateSamples(Island island, Span<float> paramVectors, int popSize, Random rng);

    void Update(Island island, ReadOnlySpan<float> fitnesses,
                ReadOnlySpan<float> paramVectors, int popSize);
}
