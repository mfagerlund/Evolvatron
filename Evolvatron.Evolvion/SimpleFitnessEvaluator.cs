namespace Evolvatron.Evolvion;

/// <summary>
/// Simple fitness evaluator for single-seed evaluation.
/// Connects CPUEvaluator with IEnvironment to compute fitness.
/// </summary>
public class SimpleFitnessEvaluator
{
    /// <summary>
    /// Evaluate a single individual on an environment.
    /// Returns cumulative reward over the episode.
    /// </summary>
    public float Evaluate(
        Individual individual,
        SpeciesSpec topology,
        IEnvironment environment,
        int seed = 0)
    {
        // Create evaluator for this topology
        var evaluator = new CPUEvaluator(topology);

        // Reset environment
        environment.Reset(seed);

        float totalReward = 0f;
        int step = 0;

        var observations = new float[environment.InputCount];

        while (!environment.IsTerminal() && step < environment.MaxSteps)
        {
            // Get observations from environment
            environment.GetObservations(observations);

            // Run neural network forward pass
            var outputs = evaluator.Evaluate(individual, observations);

            // Check for NaN in outputs
            if (ContainsNaN(outputs))
            {
                return -1000f; // Severe penalty for NaN
            }

            // Step environment and accumulate reward
            float reward = environment.Step(outputs);
            totalReward += reward;

            step++;
        }

        // Return final fitness from environment (for goal-based tasks)
        // Falls back to cumulative reward if GetFinalFitness returns 0
        float finalFitness = environment.GetFinalFitness();
        return finalFitness != 0f ? finalFitness : totalReward;
    }

    /// <summary>
    /// Evaluate all individuals in a population.
    /// </summary>
    public void EvaluatePopulation(
        Population population,
        IEnvironment environment,
        int seed = 0)
    {
        foreach (var species in population.AllSpecies)
        {
            for (int i = 0; i < species.Individuals.Count; i++)
            {
                var individual = species.Individuals[i];
                individual.Fitness = Evaluate(individual, species.Topology, environment, seed);
                species.Individuals[i] = individual;
            }
        }
    }

    /// <summary>
    /// Check if any value in the span is NaN.
    /// </summary>
    private static bool ContainsNaN(ReadOnlySpan<float> values)
    {
        foreach (float v in values)
        {
            if (float.IsNaN(v))
                return true;
        }
        return false;
    }
}
