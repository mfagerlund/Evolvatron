namespace Evolvatron.Evolvion;

/// <summary>
/// Main evolutionary algorithm orchestrator.
/// Manages generation stepping, fitness evaluation, selection, mutation, and species culling.
/// </summary>
public class Evolver
{
    private readonly Random _random;

    public Evolver(int seed = 42)
    {
        _random = new Random(seed);
    }

    /// <summary>
    /// Step the population forward one generation.
    /// Process:
    /// 1. Evaluate all individuals (fitness must be set externally before calling)
    /// 2. Update species statistics
    /// 3. Adaptive species culling
    /// 4. Within-species selection and reproduction
    /// 5. Apply mutations
    /// 6. Increment generation counter
    /// </summary>
    /// <param name="population">Population to evolve.</param>
    /// <returns>Evolved population (same instance, modified in place).</returns>
    public Population StepGeneration(Population population)
    {
        var config = population.Config;

        // Step 1: Evaluate all individuals (assumed done externally)
        // Note: Fitness values must be set on all individuals before calling this method

        // Step 2: Update species statistics
        StagnationTracker.UpdateAllSpecies(population);

        // Step 3: Adaptive species culling
        if (population.AllSpecies.Count > config.MinSpeciesCount)
        {
            SpeciesCuller.CullStagnantSpecies(population, config, _random);
        }

        // Step 4: Within-species selection and reproduction
        foreach (var species in population.AllSpecies)
        {
            EvolveSpecies(species, config);
        }

        // Step 5: Increment generation counter and species ages
        population.Generation++;
        foreach (var species in population.AllSpecies)
        {
            species.Age++;
        }

        return population;
    }

    /// <summary>
    /// Evolve a single species through selection, reproduction, and mutation.
    /// </summary>
    /// <param name="species">Species to evolve.</param>
    /// <param name="config">Evolution configuration.</param>
    private void EvolveSpecies(Species species, EvolutionConfig config)
    {
        if (species.Individuals.Count == 0)
            return;

        var topology = species.Topology;
        int targetPopSize = config.IndividualsPerSpecies;

        // Step 1: Preserve elites
        var elites = Elitism.PreserveElites(species.Individuals, config.Elites);

        // Step 2: Generate offspring via tournament selection
        int offspringCount = targetPopSize - elites.Count;
        var offspring = new List<Individual>(offspringCount);

        if (offspringCount > 0)
        {
            offspring = Selection.GenerateOffspring(
                species.Individuals,
                offspringCount,
                config.TournamentSize,
                _random,
                config.ParentPoolPercentage);
        }

        // Step 3: Apply mutations to offspring (not elites)
        foreach (var individual in offspring)
        {
            ApplyMutations(individual, topology, config);
        }

        // Step 4: Combine elites + offspring
        species.Individuals = elites.Concat(offspring).ToList();
    }

    /// <summary>
    /// Apply all mutation operators to an individual.
    /// Includes weight mutations, activation swaps, and edge topology mutations.
    /// </summary>
    /// <param name="individual">Individual to mutate.</param>
    /// <param name="topology">Species topology (may be modified by edge mutations).</param>
    /// <param name="config">Evolution configuration.</param>
    private void ApplyMutations(
        Individual individual,
        SpeciesSpec topology,
        EvolutionConfig config)
    {
        // Weight-level mutations (apply to both weights and biases)
        if (_random.NextSingle() < config.MutationRates.WeightJitter)
        {
            MutationOperators.ApplyWeightJitter(
                individual,
                config.MutationRates.WeightJitterStdDev,
                _random);
            MutationOperators.ApplyBiasJitter(
                individual,
                config.MutationRates.WeightJitterStdDev,
                _random);
        }

        if (_random.NextSingle() < config.MutationRates.WeightReset)
        {
            MutationOperators.ApplyWeightReset(individual, _random);
            MutationOperators.ApplyBiasReset(individual, _random);
        }

        if (_random.NextSingle() < config.MutationRates.WeightL1Shrink)
        {
            MutationOperators.ApplyWeightL1Shrink(
                individual,
                config.MutationRates.L1ShrinkFactor);
            MutationOperators.ApplyBiasL1Shrink(
                individual,
                config.MutationRates.L1ShrinkFactor);
        }

        // Activation mutations
        if (_random.NextSingle() < config.MutationRates.ActivationSwap)
        {
            MutationOperators.ApplyActivationSwap(individual, topology, _random);
        }

        // Node parameter mutations
        if (_random.NextSingle() < config.MutationRates.NodeParamMutate)
        {
            MutationOperators.ApplyNodeParamMutate(individual, _random);
        }

        // Edge topology mutations (probabilistic)
        ApplyEdgeMutations(individual, topology, config.EdgeMutations);
    }

    /// <summary>
    /// Apply edge topology mutations (add, delete, split, etc.).
    /// </summary>
    private void ApplyEdgeMutations(
        Individual individual,
        SpeciesSpec topology,
        EdgeMutationConfig edgeConfig)
    {
        // TODO: Integrate edge topology mutations once EdgeTopologyMutations API is finalized
        // For now, these are disabled during regular evolution to avoid complexity

        // EdgeAdd
        // if (_random.NextSingle() < edgeConfig.EdgeAdd)
        // {
        //     EdgeTopologyMutations.AddEdge(topology, individual, edgeConfig, _random);
        // }

        // EdgeDelete
        // if (_random.NextSingle() < edgeConfig.EdgeDeleteRandom)
        // {
        //     EdgeTopologyMutations.DeleteEdge(topology, individual, edgeConfig, _random);
        // }

        // ... other edge mutations ...
    }

    /// <summary>
    /// Initialize a new population with random species.
    /// </summary>
    /// <param name="config">Evolution configuration.</param>
    /// <param name="defaultTopology">Default topology for all species.</param>
    /// <returns>Initialized population.</returns>
    public Population InitializePopulation(
        EvolutionConfig config,
        SpeciesSpec defaultTopology)
    {
        var population = new Population(config);

        for (int i = 0; i < config.SpeciesCount; i++)
        {
            // Clone topology for each species
            var topology = SpeciesDiversification.CloneTopology(defaultTopology);

            // Apply some initial diversification
            if (i > 0) // Keep first species as baseline
            {
                SpeciesDiversification.CloneTopology(topology);
            }

            // Create species
            var species = new Species(topology);

            // Initialize individuals
            species.Individuals = SpeciesDiversification.InitializePopulation(
                topology,
                config.IndividualsPerSpecies,
                _random);

            population.AllSpecies.Add(species);
        }

        return population;
    }

    /// <summary>
    /// Get summary statistics for the current generation.
    /// </summary>
    public string GetGenerationSummary(Population population)
    {
        var stats = population.GetStatistics();
        var best = population.GetBestIndividual();

        var summary = $"Generation {population.Generation}\n";
        summary += $"  Species: {population.AllSpecies.Count}\n";
        summary += $"  Best Fitness: {stats.BestFitness:F4}\n";
        summary += $"  Mean Fitness: {stats.MeanFitness:F4}\n";
        summary += $"  Median Fitness: {stats.MedianFitness:F4}\n";
        summary += $"  Worst Fitness: {stats.WorstFitness:F4}\n";

        if (best.HasValue)
        {
            var (individual, species) = best.Value;
            int speciesIdx = population.AllSpecies.IndexOf(species);
            summary += $"  Best Individual: Species {speciesIdx}, Age {individual.Age}\n";
        }

        return summary;
    }
}
