using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator that integrates batched physics simulation
/// with neural network evaluation for evolutionary rocket control.
///
/// This class orchestrates the full evaluation loop:
/// 1. Upload rocket template and colliders to GPUBatchedEnvironment
/// 2. Upload neural network weights from individuals
/// 3. For each timestep until all terminal:
///    - Get observations from physics state
///    - Run neural network forward pass (GPU kernels)
///    - Copy outputs to actions buffer
///    - Apply actions to physics
///    - Step physics simulation
///    - Check terminal conditions, compute rewards
/// 4. Download fitness values
///
/// Each "world" in the batched physics corresponds to one individual being evaluated.
/// </summary>
public class GPUBatchedFitnessEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _maxIndividuals;
    private bool _disposed;

    // Neural network state
    private GPUEvolvionState? _neuralState;
    private SpeciesSpec? _currentSpec;

    // Neural network kernels
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>
        _setInputsForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPUEdge>, ArrayView<float>,
        ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int, int>
        _evaluateRowForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<GPURowPlan>, int, int, int, int>
        _getOutputsForEpisodesKernel;

    // Physics environment (created per evaluation)
    private GPUBatchedEnvironment? _physicsEnv;

    /// <summary>
    /// Creates a new GPU batched fitness evaluator.
    /// </summary>
    /// <param name="maxIndividuals">Maximum number of individuals that can be evaluated in one batch.</param>
    public GPUBatchedFitnessEvaluator(int maxIndividuals = 1000)
    {
        _maxIndividuals = maxIndividuals;

        // Initialize ILGPU context and accelerator
        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        if (cudaDevice != null)
        {
            _accelerator = cudaDevice.CreateAccelerator(_context);
        }
        else
        {
            Console.WriteLine("WARNING: No CUDA device found, falling back to preferred device");
            _accelerator = _context.GetPreferredDevice(preferCPU: false)
                .CreateAccelerator(_context);
        }

        Console.WriteLine($"GPUBatchedFitnessEvaluator initialized on: {_accelerator.Name}");

        // Load neural network kernels
        _setInputsForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            GPUEvolvionKernels.SetInputsForEpisodesKernel);

        _evaluateRowForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<GPUEdge>, ArrayView<float>,
            ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<GPURowPlan>,
            int, int, int, int, int>(
            GPUEvolvionKernels.EvaluateRowForEpisodesKernel);

        _getOutputsForEpisodesKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<GPURowPlan>,
            int, int, int, int>(
            GPUEvolvionKernels.GetOutputsForEpisodesKernel);
    }

    /// <summary>
    /// Evaluates all individuals in a population using GPU-accelerated batched physics.
    /// Each individual runs in its own parallel physics world.
    /// </summary>
    /// <param name="spec">The species specification (neural network topology).</param>
    /// <param name="individuals">List of individuals to evaluate.</param>
    /// <param name="rocketTemplate">Template for rocket rigid bodies.</param>
    /// <param name="geomTemplate">Template for rocket collision geometries.</param>
    /// <param name="jointTemplate">Template for rocket joints.</param>
    /// <param name="colliders">Static arena colliders (shared across all worlds).</param>
    /// <param name="simConfig">Physics simulation configuration.</param>
    /// <param name="envConfig">Environment configuration (rewards, bounds, etc.).</param>
    /// <param name="seed">Random seed for target placement.</param>
    /// <param name="maxThrust">Maximum thrust force for rockets.</param>
    /// <param name="maxGimbalTorque">Maximum gimbal torque for rockets.</param>
    /// <returns>Array of fitness values, one per individual.</returns>
    public float[] EvaluatePopulation(
        SpeciesSpec spec,
        List<Individual> individuals,
        GPURigidBody[] rocketTemplate,
        GPURigidBodyGeom[] geomTemplate,
        GPURevoluteJoint[] jointTemplate,
        GPUOBBCollider[] colliders,
        SimulationConfig simConfig,
        GPUBatchedEnvironmentConfig envConfig,
        int seed,
        float maxThrust = 100f,
        float maxGimbalTorque = 20f)
    {
        int worldCount = individuals.Count;
        if (worldCount == 0)
            return Array.Empty<float>();

        if (worldCount > _maxIndividuals)
            throw new ArgumentException($"Population size ({worldCount}) exceeds maximum ({_maxIndividuals})");

        // Update environment config world count
        envConfig.WorldCount = worldCount;

        // Create physics world config matching the rocket template
        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = rocketTemplate.Length,
            GeomsPerWorld = geomTemplate.Length,
            JointsPerWorld = jointTemplate.Length,
            SharedColliderCount = colliders.Length,
            TargetsPerWorld = 10, // Default targets
            MaxContactsPerWorld = 32
        };

        // Initialize neural network state if needed
        InitializeNeuralState(spec, individuals);

        // Create physics environment
        _physicsEnv?.Dispose();
        _physicsEnv = new GPUBatchedEnvironment(_accelerator, worldConfig, envConfig);

        // Upload rocket template and colliders
        _physicsEnv.UploadRocketTemplate(rocketTemplate, geomTemplate, jointTemplate);
        _physicsEnv.UploadSharedColliders(colliders);

        // Reset all environments
        _physicsEnv.Reset(seed);

        // Get buffer sizes
        int observationsPerWorld = envConfig.ObservationsPerWorld;
        int actionsPerWorld = envConfig.ActionsPerWorld;
        int totalEpisodes = worldCount;
        int episodesPerIndividual = 1; // 1 episode per individual (1 world = 1 individual)

        // Main simulation loop
        int maxSteps = envConfig.MaxStepsPerEpisode;
        for (int step = 0; step < maxSteps; step++)
        {
            // Check if all worlds are terminal (early exit)
            if (_physicsEnv.AllTerminal())
                break;

            // 1. Get observations from physics state
            var observationsView = _physicsEnv.GetObservationsView();
            _accelerator.Synchronize();

            // 2. Run neural network forward pass
            RunNeuralNetworkForwardPass(
                observationsView,
                _physicsEnv.ActionsView,
                spec,
                totalEpisodes,
                episodesPerIndividual,
                observationsPerWorld,
                actionsPerWorld);

            // 3. Apply actions and step physics
            _physicsEnv.Step(simConfig, maxThrust, maxGimbalTorque);
        }

        // Compute final fitness values
        _physicsEnv.ComputeFitness(targetBonusMultiplier: 200f);

        // Download and return fitness values
        return _physicsEnv.GetFitness();
    }

    /// <summary>
    /// Simplified evaluation method using default configurations.
    /// Creates simple spherical rocket chase scenario with X/Y thrust.
    /// </summary>
    public float[] EvaluatePopulation(
        SpeciesSpec spec,
        List<Individual> individuals,
        int seed)
    {
        // Single spherical rocket body - starts at center
        var rocketTemplate = new GPURigidBody[]
        {
            new GPURigidBody
            {
                X = 0f,
                Y = 0f,
                Angle = 0f,
                VelX = 0f,
                VelY = 0f,
                AngularVel = 0f,
                PrevX = 0f,
                PrevY = 0f,
                PrevAngle = 0f,
                InvMass = 1f,       // mass = 1
                InvInertia = 0f,    // No rotation
                GeomStartIndex = 0,
                GeomCount = 1
            }
        };

        // Single sphere collision geom
        var geomTemplate = new GPURigidBodyGeom[]
        {
            new GPURigidBodyGeom { LocalX = 0f, LocalY = 0f, Radius = 0.5f, BodyIndex = 0 }
        };

        // No joints
        var jointTemplate = Array.Empty<GPURevoluteJoint>();

        // Arena walls (symmetric box: -10 to 10 in X and Y)
        var colliders = new GPUOBBCollider[]
        {
            // Bottom wall
            new GPUOBBCollider { CX = 0f, CY = -10.5f, UX = 1f, UY = 0f, HalfExtentX = 11f, HalfExtentY = 0.5f },
            // Top wall
            new GPUOBBCollider { CX = 0f, CY = 10.5f, UX = 1f, UY = 0f, HalfExtentX = 11f, HalfExtentY = 0.5f },
            // Left wall
            new GPUOBBCollider { CX = -10.5f, CY = 0f, UX = 0f, UY = 1f, HalfExtentX = 0.5f, HalfExtentY = 11f },
            // Right wall
            new GPUOBBCollider { CX = 10.5f, CY = 0f, UX = 0f, UY = 1f, HalfExtentX = 0.5f, HalfExtentY = 11f }
        };

        // Simple physics config - no gravity, some damping
        var simConfig = new SimulationConfig
        {
            Dt = 1f / 60f,
            XpbdIterations = 4,
            Substeps = 1,
            GravityX = 0f,
            GravityY = 0f,            // No gravity - pure pursuit task
            FrictionMu = 0.5f,
            Restitution = 0.5f,
            GlobalDamping = 0.05f,    // Some velocity damping
            AngularDamping = 0f,
            VelocityStabilizationBeta = 1f,
            MaxVelocity = 15f
        };

        // Default environment config
        var envConfig = GPUBatchedEnvironmentConfig.ForTargetChase(individuals.Count);

        return EvaluatePopulation(
            spec,
            individuals,
            rocketTemplate,
            geomTemplate,
            jointTemplate,
            colliders,
            simConfig,
            envConfig,
            seed,
            maxThrust: 30f,          // Moderate thrust
            maxGimbalTorque: 0f);    // No rotation
    }

    /// <summary>
    /// Initializes neural network GPU state for the given species and individuals.
    /// </summary>
    private void InitializeNeuralState(SpeciesSpec spec, List<Individual> individuals)
    {
        _currentSpec = spec;

        // Create or resize neural state if needed
        if (_neuralState == null ||
            individuals.Count > _neuralState.MaxIndividuals ||
            spec.TotalNodes > _neuralState.MaxNodes ||
            spec.TotalEdges > _neuralState.MaxEdges)
        {
            _neuralState?.Dispose();
            _neuralState = new GPUEvolvionState(
                _accelerator,
                maxIndividuals: Math.Max(_maxIndividuals, individuals.Count),
                maxNodes: Math.Max(100, spec.TotalNodes),
                maxEdges: Math.Max(500, spec.TotalEdges));
        }

        // Upload topology and individual weights
        _neuralState.UploadTopology(spec);
        _neuralState.UploadIndividuals(individuals, spec);
    }

    /// <summary>
    /// Runs neural network forward pass on GPU.
    /// Takes observations from physics, produces actions for physics.
    /// </summary>
    private void RunNeuralNetworkForwardPass(
        ArrayView<float> observations,
        ArrayView<float> actions,
        SpeciesSpec spec,
        int totalEpisodes,
        int episodesPerIndividual,
        int observationSize,
        int actionSize)
    {
        if (_neuralState == null || _currentSpec == null)
            throw new InvalidOperationException("Neural state not initialized");

        // 1. Copy observations to neural network input layer
        _setInputsForEpisodesKernel(
            totalEpisodes,
            _neuralState.NodeValues.View,
            observations,
            totalEpisodes,
            episodesPerIndividual,
            spec.TotalNodes,
            observationSize);
        _accelerator.Synchronize();

        // 2. Evaluate hidden layers row by row
        for (int rowIdx = 1; rowIdx < spec.RowPlans.Length; rowIdx++)
        {
            _evaluateRowForEpisodesKernel(
                totalEpisodes,
                _neuralState.NodeValues.View,
                _neuralState.Edges.View,
                _neuralState.Individuals.Weights.View,
                _neuralState.Individuals.Biases.View,
                _neuralState.Individuals.Activations.View,
                _neuralState.Individuals.NodeParams.View,
                _neuralState.RowPlans.View,
                rowIdx,
                totalEpisodes,
                episodesPerIndividual,
                spec.TotalNodes,
                spec.TotalEdges);
            _accelerator.Synchronize();
        }

        // 3. Copy output layer values to actions buffer
        int outputRowIdx = spec.RowPlans.Length - 1;
        _getOutputsForEpisodesKernel(
            totalEpisodes,
            _neuralState.NodeValues.View,
            actions,
            _neuralState.RowPlans.View,
            outputRowIdx,
            totalEpisodes,
            spec.TotalNodes,
            actionSize);
        _accelerator.Synchronize();
    }

    /// <summary>
    /// Gets the underlying accelerator for advanced usage.
    /// </summary>
    public Accelerator Accelerator => _accelerator;

    /// <summary>
    /// Gets the current physics environment (null if not in evaluation).
    /// </summary>
    public GPUBatchedEnvironment? PhysicsEnvironment => _physicsEnv;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _physicsEnv?.Dispose();
        _neuralState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
