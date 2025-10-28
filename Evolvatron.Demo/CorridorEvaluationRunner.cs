using Evolvatron.Evolvion;
using Evolvatron.Evolvion.Environments;
using System.Diagnostics;
using Colonel.Tests.HagridTests.FollowTheCorridor;

namespace Evolvatron.Demo;

public class CorridorEvaluationRunner
{
    public class RunConfig
    {
        public int SpeciesCount { get; set; } = 40;
        public int IndividualsPerSpecies { get; set; } = 40;
        public int Elites { get; set; } = 1;
        public int TournamentSize { get; set; } = 4;
        public float ParentPoolPercentage { get; set; } = 0.2f;
        public int MinSpeciesCount { get; set; } = 4;
        public int EvolutionSeed { get; set; } = 42;
        public int MaxGenerations { get; set; } = 1000;
        public float SolvedThreshold { get; set; } = 0.9f;
        public int MaxTimeoutMs { get; set; } = int.MaxValue;
        public int MaxStepsPerEpisode { get; set; } = 320;
    }

    public class GenerationSnapshot
    {
        public int Generation { get; set; }
        public float BestFitness { get; set; }
        public int CurrentStep { get; set; }
        public int ActiveCount { get; set; }
        public long ElapsedMs { get; set; }
        public bool IsSimulating { get; set; }
        public List<EnvironmentSnapshot> Environments { get; set; } = new();
        public Dictionary<Evolvion.Environments.DeathCause, int> DeathCounts { get; set; } = new();
    }

    public class EnvironmentSnapshot
    {
        public Godot.Vector2 Position { get; set; }
        public float Heading { get; set; }
        public bool IsTerminal { get; set; }
        public Evolvion.Environments.DeathCause DeathCause { get; set; }
    }

    public class ProgressUpdate
    {
        public int Generation { get; set; }
        public float BestFitness { get; set; }
        public int CurrentStep { get; set; }
        public long ElapsedMs { get; set; }
    }

    private readonly RunConfig _config;
    private readonly Action<ProgressUpdate>? _progressCallback;
    private readonly Func<GenerationSnapshot>? _snapshotProvider;

    private Population _population;
    private Evolver _evolver;
    private List<FollowTheCorridorEnvironment> _environments;
    private List<CPUEvaluator> _evaluators;
    private List<Individual> _individuals;
    private float[] _totalRewards;

    private int _generation;
    private int _currentStep;
    private float _bestFitness;
    private bool _solved;
    private Stopwatch _stopwatch;

    public CorridorEvaluationRunner(
        RunConfig? config = null,
        Action<ProgressUpdate>? progressCallback = null,
        Func<GenerationSnapshot>? snapshotProvider = null)
    {
        _config = config ?? new RunConfig();
        _progressCallback = progressCallback;
        _snapshotProvider = snapshotProvider;

        _population = null!;
        _evolver = null!;
        _environments = null!;
        _evaluators = null!;
        _individuals = null!;
        _totalRewards = null!;
        _stopwatch = null!;
    }

    public (float bestFitness, int generation, bool solved, long elapsedMs) Run()
    {
        Initialize();

        while (_stopwatch.ElapsedMilliseconds < _config.MaxTimeoutMs &&
               _generation < _config.MaxGenerations &&
               !_solved)
        {
            RebuildEvaluatorsIfNeeded();
            RunGeneration();

            if (!_solved)
            {
                _evolver.StepGeneration(_population);
                _generation++;
            }
        }

        _stopwatch.Stop();
        return (_bestFitness, _generation, _solved, _stopwatch.ElapsedMilliseconds);
    }

    public GenerationSnapshot CreateSnapshot()
    {
        var snapshot = new GenerationSnapshot
        {
            Generation = _generation,
            BestFitness = _bestFitness,
            CurrentStep = _currentStep,
            ElapsedMs = _stopwatch?.ElapsedMilliseconds ?? 0,
            IsSimulating = false
        };

        if (_environments != null)
        {
            try
            {
                var envCount = _environments.Count;
                snapshot.ActiveCount = 0;

                for (int i = 0; i < envCount; i++)
                {
                    var env = _environments[i];
                    bool isTerminal = env.IsTerminal();

                    if (!isTerminal)
                        snapshot.ActiveCount++;
                    else
                    {
                        // Count death causes
                        var cause = env.CauseOfDeath;
                        if (!snapshot.DeathCounts.ContainsKey(cause))
                            snapshot.DeathCounts[cause] = 0;
                        snapshot.DeathCounts[cause]++;
                    }

                    snapshot.Environments.Add(new EnvironmentSnapshot
                    {
                        Position = env.GetCarPosition(),
                        Heading = env.GetCarHeading(),
                        IsTerminal = isTerminal,
                        DeathCause = env.CauseOfDeath
                    });
                }
            }
            catch
            {
                snapshot.ActiveCount = 0;
            }
        }
        else
        {
            snapshot.ActiveCount = 0;
        }

        return snapshot;
    }

    private void Initialize()
    {
        var random = new Random(_config.EvolutionSeed);
        var topology = CreateCorridorTopology(random);

        var evolutionConfig = new EvolutionConfig
        {
            SpeciesCount = _config.SpeciesCount,
            IndividualsPerSpecies = _config.IndividualsPerSpecies,
            Elites = _config.Elites,
            TournamentSize = _config.TournamentSize,
            ParentPoolPercentage = _config.ParentPoolPercentage,
            MinSpeciesCount = _config.MinSpeciesCount
        };

        _evolver = new Evolver(seed: _config.EvolutionSeed);
        _population = _evolver.InitializePopulation(evolutionConfig, topology);

        _environments = new List<FollowTheCorridorEnvironment>();
        _evaluators = new List<CPUEvaluator>();
        _individuals = new List<Individual>();

        foreach (var species in _population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                _environments.Add(new FollowTheCorridorEnvironment(maxSteps: _config.MaxStepsPerEpisode));
                _evaluators.Add(new CPUEvaluator(species.Topology));
                _individuals.Add(individual);
            }
        }

        _totalRewards = new float[_environments.Count];
        _generation = 0;
        _currentStep = 0;
        _bestFitness = float.MinValue;
        _solved = false;
        _stopwatch = Stopwatch.StartNew();
    }

    private void RunGeneration()
    {
        for (int i = 0; i < _environments.Count; i++)
        {
            _environments[i].Reset(seed: _generation);
            _totalRewards[i] = 0f;
        }

        _currentStep = 0;
        var observations = new float[_environments[0].InputCount];

        for (int step = 0; step < _config.MaxStepsPerEpisode; step++)
        {
            _currentStep = step;

            for (int i = 0; i < _environments.Count; i++)
            {
                if (!_environments[i].IsTerminal())
                {
                    _environments[i].GetObservations(observations);
                    var actions = _evaluators[i].Evaluate(_individuals[i], observations);
                    float reward = _environments[i].Step(actions);
                    _totalRewards[i] += reward;
                }
            }

            if (_snapshotProvider != null && step % 10 == 0)
            {
                _snapshotProvider();
            }
        }

        int envIdx = 0;
        foreach (var species in _population.AllSpecies)
        {
            for (int indIdx = 0; indIdx < species.Individuals.Count; indIdx++)
            {
                var ind = species.Individuals[indIdx];
                ind.Fitness = _totalRewards[envIdx];
                species.Individuals[indIdx] = ind;
                _individuals[envIdx] = ind;
                envIdx++;
            }
        }

        var best = _population.GetBestIndividual();
        _bestFitness = best.HasValue ? best.Value.individual.Fitness : 0f;

        _progressCallback?.Invoke(new ProgressUpdate
        {
            Generation = _generation,
            BestFitness = _bestFitness,
            CurrentStep = _currentStep,
            ElapsedMs = _stopwatch.ElapsedMilliseconds
        });

        if (_bestFitness >= _config.SolvedThreshold)
        {
            _solved = true;
        }
    }

    private void RebuildEvaluatorsIfNeeded()
    {
        int newPopSize = _population.AllSpecies.Sum(s => s.Individuals.Count);

        while (_environments.Count < newPopSize)
            _environments.Add(new FollowTheCorridorEnvironment(maxSteps: _config.MaxStepsPerEpisode));
        while (_environments.Count > newPopSize)
            _environments.RemoveAt(_environments.Count - 1);

        if (_totalRewards.Length != newPopSize)
            _totalRewards = new float[newPopSize];

        _evaluators.Clear();
        _individuals.Clear();

        foreach (var species in _population.AllSpecies)
        {
            foreach (var individual in species.Individuals)
            {
                _evaluators.Add(new CPUEvaluator(species.Topology));
                _individuals.Add(individual);
            }
        }
    }

    public static SpeciesSpec CreateCorridorTopology(Random random)
    {
        return new SpeciesBuilder()
            .AddInputRow(9)
            .AddHiddenRow(12, ActivationType.Linear, ActivationType.Tanh, ActivationType.ReLU, ActivationType.Sigmoid, ActivationType.LeakyReLU, ActivationType.ELU, ActivationType.Softsign, ActivationType.Softplus, ActivationType.Sin, ActivationType.Gaussian, ActivationType.GELU)
            .AddOutputRow(2, ActivationType.Tanh)
            .WithMaxInDegree(12)
            .InitializeSparse(random)
            .Build();
    }
}
