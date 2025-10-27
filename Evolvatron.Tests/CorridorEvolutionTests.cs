using Evolvatron.Demo;
using Xunit;

namespace Evolvatron.Tests;

public class CorridorEvolutionTests
{
    [Fact]
    public void CorridorEvolution_WithDefaultParameters_ShouldSolve()
    {
        var config = new CorridorEvaluationRunner.RunConfig
        {
            MaxTimeoutMs = 15 * 60 * 1000,
            SolvedThreshold = 0.9f
        };

        var runner = new CorridorEvaluationRunner(
            config: config,
            progressCallback: update =>
            {
                if (update.Generation % 100 == 0)
                {
                    Console.WriteLine($"Gen {update.Generation}: Best={update.BestFitness:F3} ({update.BestFitness * 100:F1}%)");
                }
            }
        );

        var result = runner.Run();

        Console.WriteLine($"\n=== TEST SUMMARY ===");
        Console.WriteLine($"Generations: {result.generation}");
        Console.WriteLine($"Final fitness: {result.bestFitness:F3} ({result.bestFitness * 100:F1}%)");
        Console.WriteLine($"Status: {(result.solved ? "SOLVED!" : "FAILED")}");
        Console.WriteLine($"Total time: {result.elapsedMs / 1000.0:F1}s");

        Assert.True(result.solved, $"Evolution should solve within {config.MaxTimeoutMs / 1000}s. Final fitness: {result.bestFitness:F3}");
    }
}
