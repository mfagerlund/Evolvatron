namespace Evolvatron.Evolvion;

/// <summary>
/// Represents a species - a group of individuals with identical topology
/// but differing weights and node parameters.
/// </summary>
public class Species
{
    /// <summary>
    /// The topology shared by all individuals in this species.
    /// </summary>
    public SpeciesSpec Topology { get; set; }

    /// <summary>
    /// All individuals belonging to this species.
    /// </summary>
    public List<Individual> Individuals { get; set; }

    /// <summary>
    /// Performance and stagnation tracking metrics.
    /// Note: Using a field instead of property to allow direct struct modification.
    /// </summary>
    public SpeciesStats Stats;

    /// <summary>
    /// Number of generations this species has existed.
    /// Used for grace period protection.
    /// </summary>
    public int Age { get; set; }

    /// <summary>
    /// Create a new species with the given topology.
    /// </summary>
    public Species(SpeciesSpec topology)
    {
        Topology = topology;
        Individuals = new List<Individual>();
        Stats = new SpeciesStats();
        Age = 0;
    }
}
