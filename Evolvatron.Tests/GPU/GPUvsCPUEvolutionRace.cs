using Evolvatron.Core.GPU;
using Evolvatron.Evolvion;
using Evolvatron.Evolvion.GPU;
using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.GPU;

public class GPUvsCPUEvolutionRace
{
    private readonly ITestOutputHelper _output;

    public GPUvsCPUEvolutionRace(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Races the fused mega-kernel GPU evaluator against the original multi-dispatch GPU evaluator.
    /// Same population size, same time budget. Measures generation throughput improvement.
    /// </summary>
    [Fact]
    public void RaceMegaVsOriginalGPU()
    {
        const int timeBudgetSeconds = 60;
        const int popSize = 10000;
        const int maxSteps = 900;
        const int evalRounds = 3;

        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        _output.WriteLine($"Time budget: {timeBudgetSeconds}s, Pop: {popSize}");
        _output.WriteLine($"MaxSteps: {maxSteps}, EvalRounds: {evalRounds}\n");

        _output.WriteLine("=== Original GPU (35 dispatches/step) ===");
        var origTrace = RunGPUEvolution(topology, popSize, maxSteps, timeBudgetSeconds, evalRounds);
        _output.WriteLine($"  {origTrace.Count} gens, {origTrace.Last().TotalEvals} evals, best={origTrace.Last().BestFitness:F1}, land={origTrace.Last().Landings}\n");

        _output.WriteLine("=== Mega GPU (1 dispatch/step) ===");
        var megaTrace = RunMegaGPUEvolution(topology, popSize, maxSteps, timeBudgetSeconds, evalRounds);
        _output.WriteLine($"  {megaTrace.Count} gens, {megaTrace.Last().TotalEvals} evals, best={megaTrace.Last().BestFitness:F1}, land={megaTrace.Last().Landings}\n");

        float speedup = (float)megaTrace.Count / origTrace.Count;
        _output.WriteLine($"Generation throughput: {speedup:F2}x ({megaTrace.Count} vs {origTrace.Count} gens in {timeBudgetSeconds}s)");

        _output.WriteLine($"\n{"Time(s)",8} | {"Orig Best",10} {"Orig Land",10} {"Orig Gen",10} | {"Mega Best",10} {"Mega Land",10} {"Mega Gen",10}");
        _output.WriteLine(new string('-', 85));

        int oi = 0, mi = 0;
        for (int t = 0; t <= timeBudgetSeconds; t += 5)
        {
            while (oi < origTrace.Count - 1 && origTrace[oi + 1].ElapsedSeconds <= t) oi++;
            while (mi < megaTrace.Count - 1 && megaTrace[mi + 1].ElapsedSeconds <= t) mi++;

            var o = oi < origTrace.Count ? origTrace[oi] : origTrace.Last();
            var m = mi < megaTrace.Count ? megaTrace[mi] : megaTrace.Last();

            _output.WriteLine($"{t,8} | {o.BestFitness,10:F1} {o.Landings,10} {o.Generation,10} | {m.BestFitness,10:F1} {m.Landings,10} {m.Generation,10}");
        }

        // Generate chart
        var htmlPath = Path.Combine(
            Path.GetDirectoryName(typeof(GPUvsCPUEvolutionRace).Assembly.Location)!,
            "..", "..", "..", "..", "docs", "artifacts");
        Directory.CreateDirectory(htmlPath);
        var chartFile = Path.Combine(htmlPath, "mega_vs_original_gpu_race.html");
        GenerateMegaVsOrigChart(chartFile, origTrace, megaTrace, popSize, timeBudgetSeconds);
        _output.WriteLine($"\nChart written to: {chartFile}");
    }

    [Theory]
    [InlineData(6)]
    [InlineData(12)]
    [InlineData(16)]
    public void BenchmarkGPUSolverIterations(int solverIterations)
    {
        const int timeBudgetSeconds = 60;
        const int gpuPopSize = 10000;
        const int maxSteps = 900;
        const int evalRounds = 3;

        var topology = new SpeciesBuilder()
            .AddInputRow(8)
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        _output.WriteLine($"GPU Solver Iterations: {solverIterations}");
        _output.WriteLine($"Time budget: {timeBudgetSeconds}s, Pop: {gpuPopSize}\n");

        var trace = RunGPUEvolution(topology, gpuPopSize, maxSteps, timeBudgetSeconds, evalRounds,
            solverIterations: solverIterations);

        _output.WriteLine($"  {trace.Count} generations, {trace.Last().TotalEvals} total evals");
        _output.WriteLine($"  Best fitness: {trace.Last().BestFitness:F1}");
        _output.WriteLine($"  Landings: {trace.Last().Landings}\n");

        _output.WriteLine($"{"Time(s)",8} | {"Best",10} {"Landings",10} {"Land%",8} {"Evals",10}");
        _output.WriteLine(new string('-', 55));

        int gi = 0;
        for (int t = 0; t <= timeBudgetSeconds; t += 5)
        {
            while (gi < trace.Count - 1 && trace[gi + 1].ElapsedSeconds <= t) gi++;
            var g = gi < trace.Count ? trace[gi] : trace.Last();
            float landRate = 100f * g.Landings / (gpuPopSize * evalRounds);
            _output.WriteLine($"{t,8} | {g.BestFitness,10:F1} {g.Landings,10} {landRate,7:F1}% {g.TotalEvals,10}");
        }
    }

    /// <summary>
    /// Evolution race with obstacles and distance sensors.
    /// Uses 12-input NN (8 base + 4 sensors), funnel obstacles, harder spawns.
    /// </summary>
    [Fact]
    public void RaceWithObstacles()
    {
        const int timeBudgetSeconds = 60;
        const int popSize = 10000;
        const int maxSteps = 900;
        const int evalRounds = 3;

        var topology = new SpeciesBuilder()
            .AddInputRow(12)  // 8 base + 4 distance sensors
            .AddHiddenRow(16, ActivationType.Tanh)
            .AddHiddenRow(12, ActivationType.Tanh)
            .AddOutputRow(2, ActivationType.Tanh)
            .InitializeDense(new Random(42))
            .Build();

        var obstacles = new List<GPUOBBCollider>
        {
            // Left angled wall
            new GPUOBBCollider
            {
                CX = -8f, CY = 5f,
                UX = MathF.Cos(30f * MathF.PI / 180f), UY = MathF.Sin(30f * MathF.PI / 180f),
                HalfExtentX = 5f, HalfExtentY = 0.3f
            },
            // Right angled wall
            new GPUOBBCollider
            {
                CX = 8f, CY = 5f,
                UX = MathF.Cos(-30f * MathF.PI / 180f), UY = MathF.Sin(-30f * MathF.PI / 180f),
                HalfExtentX = 5f, HalfExtentY = 0.3f
            },
            // Left platform
            new GPUOBBCollider { CX = -6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
            // Right platform
            new GPUOBBCollider { CX = 6f, CY = 0f, UX = 1f, UY = 0f, HalfExtentX = 2f, HalfExtentY = 0.2f },
        };

        _output.WriteLine($"Time budget: {timeBudgetSeconds}s, Pop: {popSize}");
        _output.WriteLine($"Obstacles: {obstacles.Count}, Sensors: 4");
        _output.WriteLine($"MaxSteps: {maxSteps}, EvalRounds: {evalRounds}\n");

        var trace = RunMegaGPUEvolutionWithObstacles(
            topology, popSize, maxSteps, timeBudgetSeconds, evalRounds, obstacles, sensorCount: 4);

        _output.WriteLine($"  {trace.Count} gens, {trace.Last().TotalEvals} evals");
        _output.WriteLine($"  Best fitness: {trace.Last().BestFitness:F1}");
        _output.WriteLine($"  Landings: {trace.Last().Landings}\n");

        _output.WriteLine($"{"Time(s)",8} | {"Best",10} {"Landings",10} {"Land%",8} {"Evals",10}");
        _output.WriteLine(new string('-', 55));

        int gi = 0;
        for (int t = 0; t <= timeBudgetSeconds; t += 5)
        {
            while (gi < trace.Count - 1 && trace[gi + 1].ElapsedSeconds <= t) gi++;
            var g = gi < trace.Count ? trace[gi] : trace.Last();
            float landRate = 100f * g.Landings / (popSize * evalRounds);
            _output.WriteLine($"{t,8} | {g.BestFitness,10:F1} {g.Landings,10} {landRate,7:F1}% {g.TotalEvals,10}");
        }
    }

    private List<TracePoint> RunMegaGPUEvolutionWithObstacles(
        SpeciesSpec topology, int popSize, int maxSteps, int timeBudgetSeconds, int evalRounds,
        List<GPUOBBCollider> obstacles, int sensorCount)
    {
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
        megaEval.Obstacles = obstacles;
        megaEval.SensorCount = sensorCount;

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

            evolver.StepGeneration(population);
            generation++;
        }

        return trace;
    }

    private List<TracePoint> RunGPUEvolution(
        SpeciesSpec topology, int popSize, int maxSteps, int timeBudgetSeconds, int evalRounds,
        int solverIterations = 8)
    {
        var trace = new List<TracePoint>();
        var evolver = new Evolver(seed: 42);
        var config = new EvolutionConfig { SpeciesCount = 1, IndividualsPerSpecies = popSize };
        var population = evolver.InitializePopulation(config, topology);

        using var gpuEval = new GPURocketLandingEvaluator(maxIndividuals: popSize + 100);
        gpuEval.SolverIterations = solverIterations;
        gpuEval.SpawnXMin = 8f;
        gpuEval.SpawnXRange = 15f;
        gpuEval.SpawnAngleRange = 0.44f;
        gpuEval.InitialVelXRange = 4f;
        gpuEval.InitialVelYMax = 4f;
        gpuEval.MaxThrust = 130f;
        gpuEval.MaxLandingAngle = 8f * MathF.PI / 180f;

        // Warmup
        var allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
        gpuEval.EvaluatePopulation(topology, allIndividuals, seed: 99, maxSteps: maxSteps);

        var overallSw = Stopwatch.StartNew();
        int generation = 0;
        long totalEvals = 0;

        while (overallSw.Elapsed.TotalSeconds < timeBudgetSeconds)
        {
            allIndividuals = population.AllSpecies.SelectMany(s => s.Individuals).ToList();
            int n = allIndividuals.Count;
            var avgFitness = new float[n];
            int totalLandings = 0;

            // Multi-seed evaluation: average fitness across K spawn conditions
            for (int k = 0; k < evalRounds; k++)
            {
                var (fitness, landings) = gpuEval.EvaluatePopulation(
                    topology, allIndividuals, seed: generation * evalRounds + k, maxSteps: maxSteps);
                for (int i = 0; i < n; i++)
                    avgFitness[i] += fitness[i];
                totalLandings += landings;
            }
            for (int i = 0; i < n; i++)
                avgFitness[i] /= evalRounds;

            // Write averaged fitness back
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

            evolver.StepGeneration(population);
            generation++;
        }

        return trace;
    }

    private List<TracePoint> RunMegaGPUEvolution(
        SpeciesSpec topology, int popSize, int maxSteps, int timeBudgetSeconds, int evalRounds)
    {
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

            evolver.StepGeneration(population);
            generation++;
        }

        return trace;
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

    private void GenerateMegaVsOrigChart(string path, List<TracePoint> origTrace, List<TracePoint> megaTrace,
        int popSize, int timeBudget)
    {
        var ic = System.Globalization.CultureInfo.InvariantCulture;
        var origTimes = string.Join(",", origTrace.Select(t => t.ElapsedSeconds.ToString("F2", ic)));
        var origBest = string.Join(",", origTrace.Select(t => t.BestFitness.ToString("F1", ic)));
        var origMean = string.Join(",", origTrace.Select(t => t.MeanFitness.ToString("F1", ic)));
        var origLandings = string.Join(",", origTrace.Select(t => t.Landings));

        var megaTimes = string.Join(",", megaTrace.Select(t => t.ElapsedSeconds.ToString("F2", ic)));
        var megaBest = string.Join(",", megaTrace.Select(t => t.BestFitness.ToString("F1", ic)));
        var megaMean = string.Join(",", megaTrace.Select(t => t.MeanFitness.ToString("F1", ic)));
        var megaLandings = string.Join(",", megaTrace.Select(t => t.Landings));

        var origGenTimes = string.Join(",", origTrace.Select(t => t.ElapsedSeconds.ToString("F2", ic)));
        var megaGenTimes = string.Join(",", megaTrace.Select(t => t.ElapsedSeconds.ToString("F2", ic)));

        int evalRounds = 3;
        var origLandRate = string.Join(",", origTrace.Select(t =>
            (100.0 * t.Landings / (popSize * evalRounds)).ToString("F1", ic)));
        var megaLandRate = string.Join(",", megaTrace.Select(t =>
            (100.0 * t.Landings / (popSize * evalRounds)).ToString("F1", ic)));

        float speedup = (float)megaTrace.Count / origTrace.Count;

        var html = $@"<!DOCTYPE html>
<html><head>
<meta charset=""utf-8"">
<title>Mega vs Original GPU Evaluator — Rocket Landing</title>
<script src=""https://cdn.jsdelivr.net/npm/chart.js""></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1 {{ color: #fff; text-align: center; margin-bottom: 5px; }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }}
  .chart-box {{ background: #16213e; border-radius: 8px; padding: 15px; }}
  .chart-box canvas {{ max-height: 350px; }}
  .stats {{ display: flex; justify-content: center; gap: 40px; margin: 20px auto; max-width: 1000px; flex-wrap: wrap; }}
  .stat-card {{ background: #16213e; border-radius: 8px; padding: 15px 25px; text-align: center; }}
  .stat-card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 24px; font-weight: bold; margin-top: 4px; }}
  .mega {{ color: #00d4aa; }}
  .orig {{ color: #ff6b6b; }}
  .speedup {{ color: #ffd700; }}
</style>
</head><body>
<h1>Mega Kernel vs Original GPU Evaluator</h1>
<p class=""subtitle"">Fused kernel + terminal skip vs 35 dispatches/step &middot; {popSize:N0} pop &middot; {timeBudget}s wall time &middot; 3-seed avg</p>

<div class=""stats"">
  <div class=""stat-card"">
    <div class=""label"">Mega Generations</div>
    <div class=""value mega"">{megaTrace.Last().Generation}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">Original Generations</div>
    <div class=""value orig"">{origTrace.Last().Generation}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">Gen Throughput</div>
    <div class=""value speedup"">{speedup:F1}x</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">Mega Total Evals</div>
    <div class=""value mega"">{megaTrace.Last().TotalEvals:N0}</div>
  </div>
  <div class=""stat-card"">
    <div class=""label"">Original Total Evals</div>
    <div class=""value orig"">{origTrace.Last().TotalEvals:N0}</div>
  </div>
</div>

<div class=""charts"">
  <div class=""chart-box""><canvas id=""bestFitness""></canvas></div>
  <div class=""chart-box""><canvas id=""landingRate""></canvas></div>
  <div class=""chart-box""><canvas id=""meanFitness""></canvas></div>
  <div class=""chart-box""><canvas id=""landings""></canvas></div>
</div>

<script>
const megaColor = '#00d4aa';
const origColor = '#ff6b6b';
const gridColor = 'rgba(255,255,255,0.08)';

const megaGenTimesArr = [{megaGenTimes}];
const origGenTimesArr = [{origGenTimes}];

const genStripesPlugin = {{
  id: 'genStripes',
  beforeDraw(chart) {{
    const ctx = chart.ctx;
    const xScale = chart.scales.x;
    const yTop = chart.chartArea.top;
    const yBot = chart.chartArea.bottom;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0, 212, 170, 0.35)';
    megaGenTimesArr.forEach((t, i) => {{
      if (i % 5 !== 0) return;
      const x = xScale.getPixelForValue(t);
      if (x >= chart.chartArea.left && x <= chart.chartArea.right) {{
        ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
      }}
    }});
    ctx.strokeStyle = 'rgba(255, 107, 107, 0.35)';
    origGenTimesArr.forEach((t, i) => {{
      if (i % 5 !== 0) return;
      const x = xScale.getPixelForValue(t);
      if (x >= chart.chartArea.left && x <= chart.chartArea.right) {{
        ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
      }}
    }});
    ctx.restore();
  }}
}};

Chart.register(genStripesPlugin);

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

const megaTimes = [{megaTimes}];
const origTimes = [{origTimes}];
const megaBest = [{megaBest}];
const origBest = [{origBest}];
const megaMean = [{megaMean}];
const origMean = [{origMean}];
const megaLandings = [{megaLandings}];
const origLandings = [{origLandings}];
const megaLandRate = [{megaLandRate}];
const origLandRate = [{origLandRate}];

new Chart('bestFitness', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('Mega Best', megaTimes, megaBest, megaColor),
    makeDataset('Original Best', origTimes, origBest, origColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Best Avg Fitness vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Best Fitness (3-seed avg)', color: '#888' }} }} }}
  }}
}});

new Chart('landingRate', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('Mega Landing %', megaTimes, megaLandRate, megaColor),
    makeDataset('Original Landing %', origTimes, origLandRate, origColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Landing Rate % vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Landing Rate %', color: '#888' }} }} }}
  }}
}});

new Chart('meanFitness', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('Mega Mean', megaTimes, megaMean, megaColor),
    makeDataset('Original Mean', origTimes, origMean, origColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Mean Fitness vs Wall Time', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Mean Fitness', color: '#888' }} }} }}
  }}
}});

new Chart('landings', {{
  type: 'scatter', data: {{ datasets: [
    makeDataset('Mega Landings', megaTimes, megaLandings, megaColor),
    makeDataset('Original Landings', origTimes, origLandings, origColor)
  ]}},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins, title: {{ display: true, text: 'Landings per Generation (raw count)', color: '#fff' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y, title: {{ display: true, text: 'Landings', color: '#888' }} }} }}
  }}
}});
</script>
</body></html>";

        File.WriteAllText(path, html);
    }
}
