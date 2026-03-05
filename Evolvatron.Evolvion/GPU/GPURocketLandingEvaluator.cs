using ILGPU;
using ILGPU.Runtime;
using Evolvatron.Core;
using Evolvatron.Core.GPU;
using Evolvatron.Core.GPU.Batched;

namespace Evolvatron.Evolvion.GPU;

/// <summary>
/// GPU-accelerated fitness evaluator for rocket landing tasks.
/// Uses full 3-body rigid body rocket (body + 2 legs with revolute joints)
/// matching CPU RocketEnvironment exactly for physics parity.
/// 8D observations, 2D actions (throttle + gimbal).
///
/// The evaluator orchestrates:
/// 1. Upload 3-body rocket template + joints + ground collider
/// 2. Upload neural network weights
/// 3. Simulation loop: observations → NN forward pass → actions → physics → terminal check
/// 4. Compute terminal-state fitness, download results
/// </summary>
public class GPURocketLandingEvaluator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private readonly int _maxIndividuals;
    private bool _disposed;

    // Neural network state (reusable across evaluations)
    private GPUEvolvionState? _neuralState;
    private SpeciesSpec? _currentSpec;

    // Physics state (recreated when world count changes)
    private GPUBatchedWorldState? _worldState;
    private GPUBatchedStepper? _stepper;
    private int _lastWorldCount;

    // Landing-specific GPU buffers
    private MemoryBuffer1D<float, Stride1D.Dense>? _observations;
    private MemoryBuffer1D<float, Stride1D.Dense>? _actions;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentThrottle;
    private MemoryBuffer1D<float, Stride1D.Dense>? _currentGimbal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _isTerminal;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _hasLanded;
    private MemoryBuffer1D<int, Stride1D.Dense>? _stepCounters;
    private MemoryBuffer1D<float, Stride1D.Dense>? _fitnessValues;

    // Landing kernels
    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPURigidBody>,
        ArrayView<float>, ArrayView<float>, ArrayView<byte>,
        int, float, float> _getLandingObservationsKernel;

    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<float>,
        ArrayView<float>, ArrayView<float>, ArrayView<byte>,
        int, float, float, float> _applyRocketActionsKernel;

    private readonly Action<Index1D, ArrayView<GPURigidBody>, ArrayView<int>,
        ArrayView<byte>, ArrayView<byte>,
        int, int, float, float, float, float, float, float, float> _checkLandingTerminalKernel;

    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPURigidBody>,
        ArrayView<int>, ArrayView<byte>,
        int, int, float, float, float> _computeLandingFitnessKernel;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>> _resetLandingStateKernel;

    // Neural network kernels
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int, int>
        _setInputsForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<GPUEdge>, ArrayView<float>,
        ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<GPURowPlan>,
        int, int, int, int, int>
        _evaluateRowForEpisodesKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<GPURowPlan>,
        int, int, int, int>
        _getOutputsForEpisodesKernel;

    // Host caches
    private byte[]? _terminalCache;
    private float[]? _fitnessCache;
    private byte[]? _landedCache;

    // Landing parameters (matching RocketEnvironment defaults)
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public float GroundY { get; set; } = -5f;
    public float SpawnHeight { get; set; } = 15f;
    public float PadX { get; set; } = 0f;
    public float PadY => GroundY + 0.5f;
    public float PadHalfWidth { get; set; } = 2f;
    public float MaxLandingVel { get; set; } = 2f;
    public float MaxLandingAngle { get; set; } = 15f * MathF.PI / 180f;
    public float LandingBonus { get; set; } = 200f;

    // Difficulty knobs (defaults match original easy settings)
    public float SpawnXRange { get; set; } = 2f;         // max distance from pad center
    public float SpawnXMin { get; set; } = 0f;           // min distance from pad (0 = any)
    public float SpawnAngleRange { get; set; } = 0f;     // no initial tilt (radians)
    public float InitialVelXRange { get; set; } = 1f;    // ±1 m/s horizontal
    public float InitialVelYMax { get; set; } = 2f;      // 0 to -2 m/s (downward)

    // Solver config (GPU Jacobi solver may need more iterations than CPU Gauss-Seidel)
    public int SolverIterations { get; set; } = 8;

    public Accelerator Accelerator => _accelerator;

    public GPURocketLandingEvaluator(int maxIndividuals = 1500)
    {
        _maxIndividuals = maxIndividuals;

        _context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = _context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        _accelerator = cudaDevice != null
            ? cudaDevice.CreateAccelerator(_context)
            : _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        Console.WriteLine($"GPURocketLandingEvaluator on: {_accelerator.Name}");

        // Load landing kernels
        _getLandingObservationsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<GPURigidBody>,
            ArrayView<float>, ArrayView<float>, ArrayView<byte>,
            int, float, float>(
            GPUBatchedRocketLandingKernels.GetLandingObservationsKernel);

        _applyRocketActionsKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, ArrayView<byte>,
            int, float, float, float>(
            GPUBatchedRocketLandingKernels.ApplyRocketActionsKernel);

        _checkLandingTerminalKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<GPURigidBody>, ArrayView<int>,
            ArrayView<byte>, ArrayView<byte>,
            int, int, float, float, float, float, float, float, float>(
            GPUBatchedRocketLandingKernels.CheckLandingTerminalKernel);

        _computeLandingFitnessKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<GPURigidBody>,
            ArrayView<int>, ArrayView<byte>,
            int, int, float, float, float>(
            GPUBatchedRocketLandingKernels.ComputeLandingFitnessKernel);

        _resetLandingStateKernel = _accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>>(
            GPUBatchedRocketLandingKernels.ResetLandingStateKernel);

        // Load NN kernels
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
    /// Evaluates all individuals using GPU-accelerated batched rocket landing simulation.
    /// </summary>
    /// <returns>Tuple of (fitness array, landing count).</returns>
    public (float[] fitness, int landings) EvaluatePopulation(
        SpeciesSpec spec,
        List<Individual> individuals,
        int seed,
        int maxSteps = 600)
    {
        int worldCount = individuals.Count;
        if (worldCount == 0)
            return (Array.Empty<float>(), 0);

        if (worldCount > _maxIndividuals)
            throw new ArgumentException($"Population size ({worldCount}) exceeds maximum ({_maxIndividuals})");

        // Initialize neural network
        InitializeNeuralState(spec, individuals);

        // Create/resize physics state and landing buffers
        EnsureBuffers(worldCount);

        // Create and upload rocket template with per-world spawn randomization
        CreateAndUploadRocketTemplate(worldCount, seed);

        // Upload ground collider
        _worldState!.UploadSharedColliders(new GPUOBBCollider[]
        {
            new GPUOBBCollider { CX = 0f, CY = GroundY, UX = 1f, UY = 0f, HalfExtentX = 30f, HalfExtentY = 0.5f }
        });

        // Reset landing state
        _resetLandingStateKernel(
            worldCount,
            _stepCounters!.View,
            _isTerminal!.View,
            _hasLanded!.View,
            _currentThrottle!.View,
            _currentGimbal!.View,
            _fitnessValues!.View);
        _worldState.ClearContactCounts();
        _worldState.ClearContactCache();
        _accelerator.Synchronize();

        // Physics config matching RocketEnvironment
        var simConfig = new SimulationConfig
        {
            Dt = 1f / 120f,
            XpbdIterations = SolverIterations,
            Substeps = 1,
            GravityX = 0f,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f,
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f,
            VelocityStabilizationBeta = 1f,
            MaxVelocity = 10f
        };

        int bodiesPerWorld = 3;

        // Main simulation loop — all kernels enqueued without sync.
        // ILGPU's default stream guarantees execution order.
        // Only sync every 60 steps for early-exit check + one final sync before download.
        for (int step = 0; step < maxSteps; step++)
        {
            // Early exit check every 60 steps (requires sync to read GPU memory)
            if (step > 0 && step % 60 == 0)
            {
                _accelerator.Synchronize();
                _isTerminal!.CopyToCPU(_terminalCache!);
                bool allDone = true;
                for (int i = 0; i < worldCount; i++)
                {
                    if (_terminalCache![i] == 0) { allDone = false; break; }
                }
                if (allDone) break;
            }

            // 1. Get observations (no sync — stream ordering)
            _getLandingObservationsKernel(
                worldCount,
                _observations!.View,
                _worldState.RigidBodies.View,
                _currentThrottle!.View,
                _currentGimbal!.View,
                _isTerminal!.View,
                bodiesPerWorld,
                PadX, PadY);

            // 2. Neural network forward pass (no sync — stream ordering)
            RunNeuralNetworkForwardPassNoSync(
                _observations!.View,
                _actions!.View,
                spec,
                worldCount, 1, 8, 2);

            // 3. Apply actions (no sync — stream ordering)
            _applyRocketActionsKernel(
                worldCount,
                _worldState.RigidBodies.View,
                _actions!.View,
                _currentThrottle!.View,
                _currentGimbal!.View,
                _isTerminal!.View,
                bodiesPerWorld,
                MaxThrust, MaxGimbalTorque, simConfig.Dt);

            // 4. Physics step (no sync — stream ordering)
            _stepper!.StepNoSync(_worldState, simConfig);

            // 5. Check terminal conditions (no sync — stream ordering)
            _checkLandingTerminalKernel(
                worldCount,
                _worldState.RigidBodies.View,
                _stepCounters!.View,
                _isTerminal!.View,
                _hasLanded!.View,
                bodiesPerWorld,
                maxSteps,
                PadX, PadY, PadHalfWidth,
                MaxLandingVel, MaxLandingAngle,
                GroundY, SpawnHeight);
        }

        // Compute terminal-state fitness
        _computeLandingFitnessKernel(
            worldCount,
            _fitnessValues!.View,
            _worldState.RigidBodies.View,
            _stepCounters!.View,
            _hasLanded!.View,
            bodiesPerWorld,
            maxSteps,
            PadX, PadY, LandingBonus);

        // Single final sync before downloading results to CPU
        _accelerator.Synchronize();
        _fitnessValues!.CopyToCPU(_fitnessCache!);
        _hasLanded!.CopyToCPU(_landedCache!);

        int landings = 0;
        for (int i = 0; i < worldCount; i++)
            if (_landedCache![i] != 0) landings++;

        var result = new float[worldCount];
        Array.Copy(_fitnessCache!, result, worldCount);
        return (result, landings);
    }

    private void EnsureBuffers(int worldCount)
    {
        if (_lastWorldCount == worldCount && _worldState != null)
            return;

        // Dispose old buffers
        _worldState?.Dispose();
        _stepper?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _currentThrottle?.Dispose();
        _currentGimbal?.Dispose();
        _isTerminal?.Dispose();
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();

        // 3-body rocket: body(5 geoms) + leftLeg(7 geoms) + rightLeg(7 geoms) = 19 geoms, 2 joints
        var worldConfig = new GPUBatchedWorldConfig
        {
            WorldCount = worldCount,
            RigidBodiesPerWorld = 3,
            GeomsPerWorld = 19,
            JointsPerWorld = 2,
            SharedColliderCount = 1,
            TargetsPerWorld = 0,
            MaxContactsPerWorld = 48
        };

        _worldState = new GPUBatchedWorldState(_accelerator, worldConfig);
        _stepper = new GPUBatchedStepper(_accelerator);

        _observations = _accelerator.Allocate1D<float>(worldCount * 8);
        _actions = _accelerator.Allocate1D<float>(worldCount * 2);
        _currentThrottle = _accelerator.Allocate1D<float>(worldCount);
        _currentGimbal = _accelerator.Allocate1D<float>(worldCount);
        _isTerminal = _accelerator.Allocate1D<byte>(worldCount);
        _hasLanded = _accelerator.Allocate1D<byte>(worldCount);
        _stepCounters = _accelerator.Allocate1D<int>(worldCount);
        _fitnessValues = _accelerator.Allocate1D<float>(worldCount);

        _terminalCache = new byte[worldCount];
        _fitnessCache = new float[worldCount];
        _landedCache = new byte[worldCount];
        _lastWorldCount = worldCount;
    }

    private void CreateAndUploadRocketTemplate(int worldCount, int seed)
    {
        // 3-body rocket matching RocketEnvironment + RigidBodyRocketTemplate exactly:
        // Body capsule (8 kg) + 2 leg capsules (1.5 kg each)
        const float bodyHeight = 1.5f;
        const float bodyRadius = 0.2f;
        const float bodyMass = 8f;
        const float bodyHalfLength = bodyHeight * 0.5f; // 0.75
        float bodyInertia = bodyMass * (bodyRadius * bodyRadius * 0.25f + bodyHeight * bodyHeight / 12f);

        const float legLength = 1.0f;
        const float legRadius = 0.1f;
        const float legMass = 1.5f;
        const float legHalfLength = legLength * 0.5f; // 0.5
        float legInertia = legMass * (legRadius * legRadius * 0.25f + legLength * legLength / 12f);

        // Body angle = PI/2 (vertical), leg angles from RigidBodyRocketTemplate
        const float bodyAngle = MathF.PI / 2f;
        float leftLegAngle = 225f * MathF.PI / 180f;
        float rightLegAngle = 315f * MathF.PI / 180f;

        // Leg offset from center (attachment point = bottom of body = centerY)
        float leftOffX = MathF.Cos(leftLegAngle) * legHalfLength;
        float leftOffY = MathF.Sin(leftLegAngle) * legHalfLength;
        float rightOffX = MathF.Cos(rightLegAngle) * legHalfLength;
        float rightOffY = MathF.Sin(rightLegAngle) * legHalfLength;

        // Reference angles for joints (leg angle - body angle)
        float leftRefAngle = leftLegAngle - bodyAngle;
        float rightRefAngle = rightLegAngle - bodyAngle;

        // Geom counts: body=5, each leg=7 (matching RigidBodyFactory.CreateCapsule formula)
        int bodyGeomCount = Math.Clamp((int)(bodyHalfLength / bodyRadius) + 2, 3, 7); // 5
        int legGeomCount = Math.Clamp((int)(legHalfLength / legRadius) + 2, 3, 7);    // 7

        // Build geom templates (local coords, same for all worlds)
        // Body geoms: 5 circles along capsule
        var bodyGeoms = new GPURigidBodyGeom[bodyGeomCount];
        for (int i = 0; i < bodyGeomCount; i++)
        {
            float t = (float)i / (bodyGeomCount - 1);
            bodyGeoms[i] = new GPURigidBodyGeom
            {
                LocalX = -bodyHalfLength + t * bodyHeight,
                LocalY = 0f,
                Radius = bodyRadius,
                BodyIndex = 0
            };
        }

        // Leg geoms: 7 circles each
        var legGeoms = new GPURigidBodyGeom[legGeomCount];
        for (int i = 0; i < legGeomCount; i++)
        {
            float t = (float)i / (legGeomCount - 1);
            legGeoms[i] = new GPURigidBodyGeom
            {
                LocalX = -legHalfLength + t * legLength,
                LocalY = 0f,
                Radius = legRadius,
                BodyIndex = 0 // Will be set per-body below
            };
        }

        var rng = new Random(seed);
        var config = _worldState!.Config;

        var allBodies = new GPURigidBody[config.TotalRigidBodies];
        var allGeoms = new GPURigidBodyGeom[config.TotalGeoms];
        var allJoints = new GPURevoluteJoint[config.TotalJoints];

        for (int w = 0; w < worldCount; w++)
        {
            // Randomized spawn matching RocketEnvironment.Reset()
            float spawnX;
            if (SpawnXMin > 0f)
            {
                float side = rng.NextDouble() < 0.5 ? -1f : 1f;
                spawnX = side * (SpawnXMin + (float)(rng.NextDouble() * (SpawnXRange - SpawnXMin)));
            }
            else
            {
                spawnX = (float)(rng.NextDouble() * SpawnXRange * 2 - SpawnXRange);
            }
            float spawnY = SpawnHeight + (float)(rng.NextDouble() * 3.0);
            float spawnTilt = (float)(rng.NextDouble() * SpawnAngleRange * 2 - SpawnAngleRange);
            float velX = (float)(rng.NextDouble() * InitialVelXRange * 2 - InitialVelXRange);
            float velY = (float)(rng.NextDouble() * -InitialVelYMax);

            // centerY = spawnY (base of rocket body, bottom attachment point)
            float centerY = spawnY;

            // Rotate all body positions around (spawnX, centerY) by spawnTilt
            float cosT = MathF.Cos(spawnTilt);
            float sinT = MathF.Sin(spawnTilt);

            // Body center (0, bodyHalfLength) rotated
            float bodyX = spawnX + 0f * cosT - bodyHalfLength * sinT;
            float bodyY = centerY + 0f * sinT + bodyHalfLength * cosT;
            float bodyAngleFinal = bodyAngle + spawnTilt;

            // Body (local index 0)
            allBodies[config.GetRigidBodyIndex(w, 0)] = new GPURigidBody
            {
                X = bodyX, Y = bodyY,
                Angle = bodyAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = bodyX, PrevY = bodyY, PrevAngle = bodyAngleFinal,
                InvMass = 1f / bodyMass,
                InvInertia = 1f / bodyInertia,
                GeomStartIndex = 0,
                GeomCount = bodyGeomCount
            };

            // Left leg: rotate (leftOffX, leftOffY) around origin
            float leftLegX = spawnX + leftOffX * cosT - leftOffY * sinT;
            float leftLegY = centerY + leftOffX * sinT + leftOffY * cosT;
            float leftAngleFinal = leftLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 1)] = new GPURigidBody
            {
                X = leftLegX, Y = leftLegY,
                Angle = leftAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = leftLegX, PrevY = leftLegY, PrevAngle = leftAngleFinal,
                InvMass = 1f / legMass,
                InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount,
                GeomCount = legGeomCount
            };

            // Right leg: rotate (rightOffX, rightOffY) around origin
            float rightLegX = spawnX + rightOffX * cosT - rightOffY * sinT;
            float rightLegY = centerY + rightOffX * sinT + rightOffY * cosT;
            float rightAngleFinal = rightLegAngle + spawnTilt;
            allBodies[config.GetRigidBodyIndex(w, 2)] = new GPURigidBody
            {
                X = rightLegX, Y = rightLegY,
                Angle = rightAngleFinal,
                VelX = velX, VelY = velY, AngularVel = 0f,
                PrevX = rightLegX, PrevY = rightLegY, PrevAngle = rightAngleFinal,
                InvMass = 1f / legMass,
                InvInertia = 1f / legInertia,
                GeomStartIndex = bodyGeomCount + legGeomCount,
                GeomCount = legGeomCount
            };

            // Body geoms (local body index 0)
            for (int i = 0; i < bodyGeomCount; i++)
            {
                var g = bodyGeoms[i];
                g.BodyIndex = 0;
                allGeoms[config.GetGeomIndex(w, i)] = g;
            }

            // Left leg geoms (local body index 1)
            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i];
                g.BodyIndex = 1;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + i)] = g;
            }

            // Right leg geoms (local body index 2)
            for (int i = 0; i < legGeomCount; i++)
            {
                var g = legGeoms[i];
                g.BodyIndex = 2;
                allGeoms[config.GetGeomIndex(w, bodyGeomCount + legGeomCount + i)] = g;
            }

            // Joints use GLOBAL body indices (the joint kernels index directly into the flat body array)
            int globalBody = config.GetRigidBodyIndex(w, 0);
            int globalLeftLeg = config.GetRigidBodyIndex(w, 1);
            int globalRightLeg = config.GetRigidBodyIndex(w, 2);

            // Left leg joint
            allJoints[config.GetJointIndex(w, 0)] = new GPURevoluteJoint
            {
                BodyA = globalBody,
                BodyB = globalLeftLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = leftRefAngle,
                EnableLimits = 0,
                LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1,
                MotorSpeed = 0f,
                MaxMotorTorque = 1000f
            };

            // Right leg joint
            allJoints[config.GetJointIndex(w, 1)] = new GPURevoluteJoint
            {
                BodyA = globalBody,
                BodyB = globalRightLeg,
                LocalAnchorAX = -bodyHalfLength, LocalAnchorAY = 0f,
                LocalAnchorBX = -legHalfLength, LocalAnchorBY = 0f,
                ReferenceAngle = rightRefAngle,
                EnableLimits = 0,
                LowerAngle = 0f, UpperAngle = 0f,
                EnableMotor = 1,
                MotorSpeed = 0f,
                MaxMotorTorque = 1000f
            };
        }

        _worldState.RigidBodies.CopyFromCPU(allBodies);
        _worldState.Geoms.CopyFromCPU(allGeoms);
        _worldState.Joints.CopyFromCPU(allJoints);
    }

    private void InitializeNeuralState(SpeciesSpec spec, List<Individual> individuals)
    {
        _currentSpec = spec;

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

        _neuralState.UploadTopology(spec);
        _neuralState.UploadIndividuals(individuals, spec);
    }

    private void RunNeuralNetworkForwardPass(
        ArrayView<float> observations,
        ArrayView<float> actions,
        SpeciesSpec spec,
        int totalEpisodes,
        int episodesPerIndividual,
        int observationSize,
        int actionSize)
    {
        RunNeuralNetworkForwardPassNoSync(observations, actions, spec,
            totalEpisodes, episodesPerIndividual, observationSize, actionSize);
        _accelerator.Synchronize();
    }

    private void RunNeuralNetworkForwardPassNoSync(
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

        _setInputsForEpisodesKernel(
            totalEpisodes,
            _neuralState.NodeValues.View,
            observations,
            totalEpisodes,
            episodesPerIndividual,
            spec.TotalNodes,
            observationSize);

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
        }

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
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _worldState?.Dispose();
        _stepper?.Dispose();
        _observations?.Dispose();
        _actions?.Dispose();
        _currentThrottle?.Dispose();
        _currentGimbal?.Dispose();
        _isTerminal?.Dispose();
        _hasLanded?.Dispose();
        _stepCounters?.Dispose();
        _fitnessValues?.Dispose();
        _neuralState?.Dispose();
        _accelerator?.Dispose();
        _context?.Dispose();
    }
}
