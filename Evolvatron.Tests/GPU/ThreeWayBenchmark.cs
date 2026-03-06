using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using Evolvatron.Evolvion.TrajectoryOptimization;
using System.Diagnostics;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.GPU;

public class ThreeWayBenchmark
{
    private readonly ITestOutputHelper _output;

    public ThreeWayBenchmark(ITestOutputHelper output) => _output = output;

    // Same obstacles used across all three methods
    private static readonly OBBCollider[] ObstacleOBBs =
    {
        OBBCollider.FromAngle(-8f, 5f, 5f, 0.3f, 30f * MathF.PI / 180f),
        OBBCollider.FromAngle(8f, 5f, 5f, 0.3f, -30f * MathF.PI / 180f),
        OBBCollider.AxisAligned(-6f, 0f, 2f, 0.2f),
        OBBCollider.AxisAligned(6f, 0f, 2f, 0.2f),
    };

    private static readonly GPUOBBCollider[] GpuObstacles =
    {
        new GPUOBBCollider
        {
            CX = -8f, CY = 5f,
            UX = MathF.Cos(30f * MathF.PI / 180f), UY = MathF.Sin(30f * MathF.PI / 180f),
            HalfExtentX = 5f, HalfExtentY = 0.3f
        },
        new GPUOBBCollider
        {
            CX = 8f, CY = 5f,
            UX = MathF.Cos(-30f * MathF.PI / 180f), UY = MathF.Sin(-30f * MathF.PI / 180f),
            HalfExtentX = 5f, HalfExtentY = 0.3f
        },
        new GPUOBBCollider { CX = -6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
        new GPUOBBCollider { CX = 6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
    };

    [Fact]
    public void TwoWayRace_ObstacleScene()
    {
        const int timeBudgetSeconds = 60;
        const int gpuPopSize = 10000;
        const int maxSteps = 900;
        const int evalRounds = 3;

        _output.WriteLine("=== 2-Way Benchmark: GPU Mega Evolution vs LS Trajectory Optimization ===");
        _output.WriteLine($"Time budget: {timeBudgetSeconds}s each, Obstacles: {ObstacleOBBs.Length}");
        _output.WriteLine($"GPU pop: {gpuPopSize}, MaxSteps: {maxSteps}\n");

        // --- GPU Mega Evolution ---
        _output.WriteLine("--- GPU Mega Evolution (12 inputs, 4 sensors) ---");
        var gpuTrace = RunMegaGPUEvolution(gpuPopSize, maxSteps, timeBudgetSeconds, evalRounds);
        _output.WriteLine($"  {gpuTrace.Count} gens, {gpuTrace.Last().TotalEvals} evals, " +
                          $"best={gpuTrace.Last().BestFitness:F1}, landings={gpuTrace.Last().Landings}\n");

        // --- LS Trajectory Optimization ---
        _output.WriteLine("--- LS Trajectory Optimization (Levenberg-Marquardt) ---");
        var lsTrace = RunTrajectoryOptimization(timeBudgetSeconds);
        _output.WriteLine($"  {lsTrace.Attempts} attempts, {lsTrace.Landings} landings, " +
                          $"best cost={lsTrace.BestCost:F2}\n");

        // --- Summary table ---
        _output.WriteLine("=== SUMMARY ===");
        _output.WriteLine($"{"Method",-25} | {"Gens/Attempts",14} | {"Landings",10} | {"Best Score",12} | {"Total Evals",12}");
        _output.WriteLine(new string('-', 82));
        _output.WriteLine($"{"GPU Mega (10K, sensors)",-25} | {gpuTrace.Count,14} | {gpuTrace.Last().Landings,10} | {gpuTrace.Last().BestFitness,12:F1} | {gpuTrace.Last().TotalEvals,12}");
        _output.WriteLine($"{"LS Trajectory Opt",-25} | {lsTrace.Attempts,14} | {lsTrace.Landings,10} | {lsTrace.BestCost,12:F2} | {lsTrace.Attempts,12}");

        // --- Time series table ---
        _output.WriteLine($"\n--- GPU Mega over time ---");
        _output.WriteLine($"{"Time(s)",8} | {"Best",10} {"Landings",10} {"Land%",8} {"Evals",10}");
        _output.WriteLine(new string('-', 55));
        int gi = 0;
        for (int t = 0; t <= timeBudgetSeconds; t += 5)
        {
            while (gi < gpuTrace.Count - 1 && gpuTrace[gi + 1].ElapsedSeconds <= t) gi++;
            var g = gi < gpuTrace.Count ? gpuTrace[gi] : gpuTrace.Last();
            float landRate = 100f * g.Landings / (gpuPopSize * evalRounds);
            _output.WriteLine($"{t,8} | {g.BestFitness,10:F1} {g.Landings,10} {landRate,7:F1}% {g.TotalEvals,10}");
        }

        // --- HTML chart ---
        var htmlPath = Path.Combine(
            Path.GetDirectoryName(typeof(ThreeWayBenchmark).Assembly.Location)!,
            "..", "..", "..", "..", "docs", "artifacts");
        Directory.CreateDirectory(htmlPath);
        var chartFile = Path.Combine(htmlPath, "gpu_vs_ls_race.html");
        GenerateChart(chartFile, gpuTrace, lsTrace, gpuPopSize, timeBudgetSeconds, evalRounds);
        _output.WriteLine($"\nChart written to: {chartFile}");
    }

    private List<TracePoint> RunMegaGPUEvolution(int popSize, int maxSteps, int timeBudgetSeconds, int evalRounds)
    {
        var topology = new SpeciesBuilder()
            .AddInputRow(12) // 8 base + 4 sensors
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var trace = new List<TracePoint>();
        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = popSize };
        var population = evolver.InitializePopulation(config, topology);

        using var megaEval = new GPURocketLandingMegaEvaluator(maxIndividuals: popSize + 100);
        megaEval.SpawnXMin = 8f;
        megaEval.SpawnXRange = 15f;
        megaEval.SpawnAngleRange = 0.44f;
        megaEval.InitialVelXRange = 4f;
        megaEval.InitialVelYMax = 4f;
        megaEval.MaxThrust = 130f;
        megaEval.MaxLandingAngle = 8f * MathF.PI / 180f;
        megaEval.Obstacles = new List<GPUOBBCollider>(GpuObstacles);
        megaEval.SensorCount = 4;

        // Warmup
        var allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
        megaEval.EvaluatePopulation(topology, allIndividuals, seed: 99, maxSteps: maxSteps);

        var overallSw = Stopwatch.StartNew();
        int generation = 0;
        long totalEvals = 0;

        while (overallSw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
            int n = allIndividuals.Count;
            var avgFitness = new float[n];
            int totalLandings = 0;

            for (int k = 0; k < evalRounds; k++)
            {
                var (fitness, landings, _) = megaEval.EvaluatePopulation(
                    topology, allIndividuals, seed: generation * evalRounds + k, maxSteps: maxSteps);
                for (int i = 0; i < n; i++)
                    avgFitness[i] += fitness[i];
                totalLandings += landings;
            }
            for (int i = 0; i < n; i++)
                avgFitness[i] /= evalRounds;

            int idx = 0;
            foreach (var species in population.AllSpecies)
            {
                for (int i = 0; i < species.Individuals.Count; i++)
                {
                    var ind = species.Individuals[i];
                    ind.Fitness = avgFitness[idx++];
                    species.Individuals[i] = ind;
                }
            }

            totalEvals += n * evalRounds;

            trace.Add(new TracePoint
            {
                ElapsedSeconds = (float)overallSw.Elapsed.TotalSeconds,
                Generation = generation,
                BestFitness = avgFitness.Max(),
                MeanFitness = avgFitness.Average(),
                Landings = totalLandings,
                TotalEvals = totalEvals
            });

            if (generation % 10 == 0)
                _output.WriteLine($"  GPU gen {generation}: best={avgFitness.Max():F1}, land={totalLandings}");

            evolver.StepGeneration(population);
            generation++;
        }

        return trace;
    }

    private LSTrace RunTrajectoryOptimization(int timeBudgetSeconds)
    {
        var rng = new Random(42);
        var result = new LSTrace();
        var overallSw = Stopwatch.StartNew();

        // Spawn conditions matching evolution difficulty
        float spawnXMin = 8f;
        float spawnXRange = 15f;
        float spawnAngleRange = 0.44f;
        float initialVelXRange = 4f;
        float initialVelYMax = 4f;

        while (overallSw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            // Random spawn
            float side = rng.NextDouble() < 0.5 ? -1f : 1f;
            float startX = side * (spawnXMin + (float)(rng.NextDouble() * (spawnXRange - spawnXMin)));
            float startY = 15f + (float)(rng.NextDouble() * 3f);
            float startAngle = MathF.PI / 2f + (float)(rng.NextDouble() * spawnAngleRange * 2 - spawnAngleRange);
            float startVelX = (float)(rng.NextDouble() * initialVelXRange * 2 - initialVelXRange);
            float startVelY = (float)(rng.NextDouble() * -initialVelYMax);

            var opts = new TrajectoryOptimizerOptions
            {
                MaxThrust = 130f,
                MaxGimbalTorque = 50f,
                MaxIterations = 80,
                ControlSteps = 60,
                PhysicsStepsPerControl = 15,
                Obstacles = new List<OBBCollider>(ObstacleOBBs)
            };

            var optimizer = new TrajectoryOptimizer(opts);
            var trajectory = optimizer.Optimize(startX, startY, startVelX, startVelY, startAngle);

            result.Attempts++;

            // Check if landed: final state near pad, low velocity, upright
            var finalState = trajectory.States[^1];
            float errX = MathF.Abs(finalState.X - opts.PadX);
            float errY = MathF.Abs(finalState.Y - opts.PadY);
            float speed = MathF.Sqrt(finalState.VelX * finalState.VelX + finalState.VelY * finalState.VelY);
            float angleErr = MathF.Abs(finalState.Angle - MathF.PI / 2f);

            bool landed = errX < 2f && errY < 2f && speed < 2f && angleErr < 8f * MathF.PI / 180f;
            if (landed) result.Landings++;

            if (trajectory.FinalCost < result.BestCost)
                result.BestCost = trajectory.FinalCost;

            result.TracePoints.Add(new LSTracePoint
            {
                ElapsedSeconds = (float)overallSw.Elapsed.TotalSeconds,
                Attempt = result.Attempts,
                Cost = trajectory.FinalCost,
                BestCost = result.BestCost,
                Landings = result.Landings,
                Landed = landed
            });

            if (result.Attempts % 5 == 0)
                _output.WriteLine($"  LS attempt {result.Attempts}: cost={trajectory.FinalCost:F2}, " +
                                  $"landed={landed}, total_landings={result.Landings}");
        }

        return result;
    }

    private struct TracePoint
    {
        public float ElapsedSeconds;
        public int Generation;
        public float BestFitness;
        public float MeanFitness;
        public int Landings;
        public long TotalEvals;
    }

    private struct LSTracePoint
    {
        public float ElapsedSeconds;
        public int Attempt;
        public double Cost;
        public double BestCost;
        public int Landings;
        public bool Landed;
    }

    private class LSTrace
    {
        public int Attempts;
        public int Landings;
        public double BestCost = double.MaxValue;
        public List<LSTracePoint> TracePoints = new();
    }

    private void GenerateChart(string path,
        List<TracePoint> gpuTrace, LSTrace lsTrace,
        int gpuPop, int timeBudget, int evalRounds)
    {
        var ic = System.Globalization.CultureInfo.InvariantCulture;

        var gpuTimes = string.Join(",", gpuTrace.Select(t => t.ElapsedSeconds.ToString("F2", ic)));
        var gpuBest = string.Join(",", gpuTrace.Select(t => t.BestFitness.ToString("F1", ic)));
        var gpuLandings = string.Join(",", gpuTrace.Select(t => t.Landings));
        var gpuLandRate = string.Join(",", gpuTrace.Select(t =>
            (100.0 * t.Landings / (gpuPop * evalRounds)).ToString("F1", ic)));

        var lsTimes = string.Join(",", lsTrace.TracePoints.Select(t => t.ElapsedSeconds.ToString("F2", ic)));
        var lsCosts = string.Join(",", lsTrace.TracePoints.Select(t => t.BestCost.ToString("F2", ic)));
        var lsLandings = string.Join(",", lsTrace.TracePoints.Select(t => t.Landings));
        var lsLandRate = string.Join(",", lsTrace.TracePoints.Select(t =>
            (100.0 * t.Landings / Math.Max(1, t.Attempt)).ToString("F1", ic)));

        var gpuLastGen = gpuTrace.Last().Generation;

        var html = $@"<!DOCTYPE html>
<html><head>
<meta charset=""utf-8"">
<title>GPU Evolution vs LS Trajectory Optimization</title>
<script src=""https://cdn.jsdelivr.net/npm/chart.js""></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #fff; text-align: center; margin-bottom: 5px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }}
  .chart-box {{ background: #16213e; border-radius: 8px; padding: 15px; }}
  .chart-box canvas {{ max-height: 350px; }}
  .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px auto; max-width: 1200px; flex-wrap: wrap; }}
  .stat-card {{ background: #16213e; border-radius: 8px; padding: 15px 20px; text-align: center; min-width: 120px; }}
  .stat-card .label {{ color: #888; font-size: 11px; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
  .gpu {{ color: #00d4aa; }}
  .ls {{ color: #ffd700; }}
</style>
</head><body>
<h1>GPU Evolution vs Trajectory Optimization</h1>
<p class=""subtitle"">Obstacle Scene (4 OBBs, funnel) &middot; {timeBudget}s wall time each &middot; GPU {gpuPop:N0} + LS single-trajectory</p>

<div class=""stats"">
  <div class=""stat-card"">
    <div class=""label"">GPU Gens</div>
    <div class=""value gpu"">{gpuLastGen}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">GPU Landings</div>
    <div class=""value gpu"">{gpuTrace.Last().Landings}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">LS Attempts</div>
    <div class=""value ls"">{lsTrace.Attempts}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">LS Landings</div>
    <div class=""value ls"">{lsTrace.Landings}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">LS Best Cost</div>
    <div class=""value ls"">{lsTrace.BestCost:F1}</div>
  </div>
</div>

<div class=""charts"">
  <div class=""chart-box""><canvas id=""landingRate""></canvas></div>
  <div class=""chart-box""><canvas id=""cumulativeLandings""></canvas></div>
  <div class=""chart-box""><canvas id=""gpuFitness""></canvas></div>
  <div class=""chart-box""><canvas id=""lsCost""></canvas></div>
</div>

<script>
const gpuColor = '#00d4aa';
const lsColor = '#ffd700';
const gridColor = 'rgba(255,255,255,0.08)';

const defaultOpts = {{
  responsive: true,
  plugins: {{ legend: {{ labels: {{ color: '#ccc' }} }} }},
  scales: {{
    x: {{ title: {{ display: true, text: 'Wall Time (s)', color: '#888' }}, ticks: {{ color: '#888' }}, grid: {{ color: gridColor }} }},
    y: {{ ticks: {{ color: '#888' }}, grid: {{ color: gridColor }} }}
  }}
}};

function makeDataset(label, times, values, color) {{
  return {{
    label, borderColor: color, backgroundColor: color + '20',
    data: times.map((t,i) => ({{ x: t, y: values[i] }})),
    showLine: true, pointRadius: 0, borderWidth: 2, tension: 0.3
  }};
}}

const gpuTimes = [{gpuTimes}];
const lsTimes = [{lsTimes}];
const gpuBest = [{gpuBest}];
const gpuLandings = [{gpuLandings}];
const lsLandings = [{lsLandings}];
const gpuLandRate = [{gpuLandRate}];
const lsLandRate = [{lsLandRate}];
const lsCosts = [{lsCosts}];

new Chart('landingRate', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('GPU Landing %', gpuTimes, gpuLandRate, gpuColor),
    makeDataset('LS Landing %', lsTimes, lsLandRate, lsColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Landing Rate % vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Landing Rate %', color: '#888' }} }} }}
  }}
}});

new Chart('cumulativeLandings', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('GPU Landings', gpuTimes, gpuLandings, gpuColor),
    makeDataset('LS Landings', lsTimes, lsLandings, lsColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Landings per Generation/Attempt', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Landings', color: '#888' }} }} }}
  }}
}});

new Chart('gpuFitness', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('GPU Best Fitness', gpuTimes, gpuBest, gpuColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'GPU Best Fitness vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Best Fitness (3-seed avg)', color: '#888' }} }} }}
  }}
}});

new Chart('lsCost', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('LS Best Cost', lsTimes, lsCosts, lsColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'LS Trajectory Optimization: Best Cost vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Best Cost (lower = better)', color: '#888' }} }} }}
  }}
}});
</script>
</body></html>";

        File.WriteAllText(path, html);
    }
}
