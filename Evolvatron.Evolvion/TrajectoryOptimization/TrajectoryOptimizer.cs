using Evolvatron.Core;
using Evolvatron.Core.Templates;
using Colonel.Core.Optimization;

namespace Evolvatron.Evolvion.TrajectoryOptimization;

/// <summary>
/// Optimizes a rocket landing trajectory using Levenberg-Marquardt least squares.
/// Finite-difference Jacobian computed through the actual Rigidon physics simulation.
/// Exploits causal sparsity: perturbing step s only affects residuals at steps >= s.
/// </summary>
public sealed class TrajectoryOptimizer
{
    // Problem sizing
    private readonly int _controlSteps;
    private readonly int _physicsStepsPerControl;
    private readonly int _residualsPerStep;
    private readonly int _totalParams;
    private readonly int _totalResiduals;

    // Residual weights
    private readonly double _wPos;
    private readonly double _wVel;
    private readonly double _wAngle;
    private readonly double _wSmooth;
    private readonly double _wFuel;
    private readonly double _wTerminal;
    private const int TerminalResidualCount = 5;

    // Physics config
    private readonly float _maxThrust;
    private readonly float _maxGimbalTorque;
    private readonly float _padX;
    private readonly float _padY;
    private readonly float _groundY;
    private readonly float _spawnHeight;

    // Solver config
    private readonly int _maxIterations;
    private readonly double _initialDamping;
    private readonly double _epsilon;
    private readonly Action<string>? _logCallback;
    private readonly Action<IterationSnapshot>? _onIterationComplete;

    // Reusable state
    private readonly WorldState _world;
    private readonly SimulationConfig _config;

    /// <summary>The physics config used by the optimizer. Playback MUST use the same config.</summary>
    public SimulationConfig Config => _config;
    private readonly CPUStepper _stepper;
    private readonly WorldStateSnapshot[] _snapshots;
    private readonly WorldStateSnapshot _tempSnapshot;

    // Start position (set per optimization call for glide path targets)
    private float _startX, _startY;

    // Iteration recording
    private readonly List<IterationSnapshot> _iterationSnapshots = new();
    private int _evalCount;

    public TrajectoryOptimizer(TrajectoryOptimizerOptions? options = null)
    {
        var opt = options ?? new TrajectoryOptimizerOptions();

        _controlSteps = opt.ControlSteps;
        _physicsStepsPerControl = opt.PhysicsStepsPerControl;
        _residualsPerStep = 8;
        _totalParams = _controlSteps * 2;
        _totalResiduals = _controlSteps * _residualsPerStep + TerminalResidualCount;

        _wPos = opt.WeightPosition;
        _wVel = opt.WeightVelocity;
        _wAngle = opt.WeightAngle;
        _wSmooth = opt.WeightSmooth;
        _wFuel = opt.WeightFuel;
        _wTerminal = opt.WeightTerminal;

        _maxThrust = opt.MaxThrust;
        _maxGimbalTorque = opt.MaxGimbalTorque;
        _padX = opt.PadX;
        _padY = opt.PadY;
        _groundY = opt.GroundY;
        _spawnHeight = opt.SpawnHeight;

        _maxIterations = opt.MaxIterations;
        _initialDamping = opt.InitialDamping;
        _epsilon = opt.Epsilon;
        _logCallback = opt.LogCallback;
        _onIterationComplete = opt.OnIterationComplete;

        _world = new WorldState(64);
        _config = new SimulationConfig
        {
            Dt = 1f / 120f,
            XpbdIterations = 8,
            Substeps = 1,
            GravityY = -9.81f,
            FrictionMu = 0.8f,
            Restitution = 0.0f,
            GlobalDamping = 0.02f,
            AngularDamping = 0.1f
        };
        _stepper = new CPUStepper();

        _snapshots = new WorldStateSnapshot[_controlSteps + 1];
        for (int i = 0; i <= _controlSteps; i++)
            _snapshots[i] = new WorldStateSnapshot();
        _tempSnapshot = new WorldStateSnapshot();
    }

    /// <summary>
    /// Optimizes a landing trajectory from the given initial conditions.
    /// </summary>
    public TrajectoryResult Optimize(
        float startX, float startY,
        float startVelX = 0f, float startVelY = 0f,
        float startAngle = MathF.PI / 2f)
    {
        _startX = startX;
        _startY = startY;

        // Build initial guess: hover throttle, zero gimbal
        var rocketIndices = SetupWorld(startX, startY, startVelX, startVelY, startAngle);
        float totalMass = GetTotalMass(rocketIndices);
        float hoverThrottle = totalMass * 9.81f / _maxThrust;

        double[] parameters = new double[_totalParams];
        for (int s = 0; s < _controlSteps; s++)
        {
            parameters[2 * s] = hoverThrottle;
            parameters[2 * s + 1] = 0.0;
        }

        // Capture initial snapshot for re-use in residual evaluations
        var initialSnapshot = WorldStateSnapshot.Capture(_world);

        _iterationSnapshots.Clear();
        _evalCount = 0;

        var solverOptions = new LeastSquaresOptions
        {
            MaxIterations = _maxIterations,
            AdaptiveDamping = true,
            InitialDamping = _initialDamping,
            CostTolerance = 1e-4,
            ParamTolerance = 1e-6,
            GradientTolerance = 1e-6,
            Verbose = _logCallback != null,
            LogCallback = _logCallback
        };

        var result = NonlinearLeastSquaresSolver.Solve(
            parameters,
            p => EvaluateWithJacobian(p, initialSnapshot, rocketIndices),
            solverOptions);

        // Build final trajectory by rolling out optimized controls
        initialSnapshot.Restore(_world);
        var trajectory = RolloutTrajectory(parameters, rocketIndices);

        trajectory.Controls = (double[])parameters.Clone();
        trajectory.IterationSnapshots = new List<IterationSnapshot>(_iterationSnapshots);
        trajectory.Success = result.Success;
        trajectory.FinalCost = result.FinalCost;
        trajectory.Iterations = result.Iterations;
        trajectory.ComputationTimeMs = result.ComputationTimeMs;
        trajectory.ConvergenceReason = result.ConvergenceReason;

        return trajectory;
    }

    private ResidualEvaluation EvaluateWithJacobian(
        double[] parameters,
        WorldStateSnapshot initialSnapshot,
        int[] rocketIndices)
    {
        // 1. Baseline rollout: simulate full trajectory, save snapshots at each control boundary
        initialSnapshot.Restore(_world);
        double[] baselineResiduals = new double[_totalResiduals];
        var baselineStates = new TrajectoryState[_controlSteps + 1];

        baselineStates[0] = ExtractState(rocketIndices, 0f, 0f);
        _snapshots[0].CaptureFrom(_world);

        for (int s = 0; s < _controlSteps; s++)
        {
            float throttle = ClampThrottle(parameters[2 * s]);
            float gimbal = ClampGimbal(parameters[2 * s + 1]);

            ApplyControlAndStep(rocketIndices, throttle, gimbal);

            baselineStates[s + 1] = ExtractState(rocketIndices, throttle, gimbal);
            _snapshots[s + 1].CaptureFrom(_world);

            float prevThrottle = s > 0 ? ClampThrottle(parameters[2 * (s - 1)]) : 0f;
            float prevGimbal = s > 0 ? ClampGimbal(parameters[2 * (s - 1) + 1]) : 0f;
            ComputeStepResiduals(baselineResiduals, s, baselineStates[s + 1], throttle, gimbal, prevThrottle, prevGimbal);
        }

        // Terminal residuals: heavily penalize final state errors
        ComputeTerminalResiduals(baselineResiduals, baselineStates[_controlSteps]);

        // Record this iteration's trajectory
        double cost = 0;
        for (int i = 0; i < _totalResiduals; i++) cost += baselineResiduals[i] * baselineResiduals[i];
        var snapshot = new IterationSnapshot
        {
            Iteration = _evalCount++,
            Cost = cost,
            States = (TrajectoryState[])baselineStates.Clone()
        };
        _iterationSnapshots.Add(snapshot);
        _onIterationComplete?.Invoke(snapshot);

        // 2. Finite-difference Jacobian with causal sparsity
        double[,] jacobian = new double[_totalResiduals, _totalParams];
        double[] perturbedResiduals = new double[_totalResiduals];

        for (int col = 0; col < _totalParams; col++)
        {
            int affectedStep = col / 2;

            // Restore snapshot at the affected step
            _snapshots[affectedStep].Restore(_world);

            // Perturb parameter
            double originalValue = parameters[col];
            parameters[col] = originalValue + _epsilon;

            // Re-simulate from affected step to end
            for (int s = affectedStep; s < _controlSteps; s++)
            {
                float throttle = ClampThrottle(parameters[2 * s]);
                float gimbal = ClampGimbal(parameters[2 * s + 1]);

                ApplyControlAndStep(rocketIndices, throttle, gimbal);

                var state = ExtractState(rocketIndices, throttle, gimbal);
                float prevThrottle = s > 0 ? ClampThrottle(parameters[2 * (s - 1)]) : 0f;
                float prevGimbal = s > 0 ? ClampGimbal(parameters[2 * (s - 1) + 1]) : 0f;
                ComputeStepResiduals(perturbedResiduals, s, state, throttle, gimbal, prevThrottle, prevGimbal);

                // Capture final state for terminal residuals
                if (s == _controlSteps - 1)
                    ComputeTerminalResiduals(perturbedResiduals, state);
            }

            // Compute Jacobian column (only rows >= affectedStep * residualsPerStep)
            int startRow = affectedStep * _residualsPerStep;
            for (int row = startRow; row < _totalResiduals; row++)
            {
                jacobian[row, col] = (perturbedResiduals[row] - baselineResiduals[row]) / _epsilon;
            }

            // Restore parameter
            parameters[col] = originalValue;
        }

        return new ResidualEvaluation(baselineResiduals, jacobian);
    }

    private void ComputeStepResiduals(
        double[] residuals, int step,
        TrajectoryState state,
        float throttle, float gimbal,
        float prevThrottle, float prevGimbal)
    {
        double t = (step + 1.0) / _controlSteps;
        double wt = Math.Pow(t, 1.5); // Time weight: weak early, strong at landing

        // Glide path: target position interpolates from start to pad
        // This guides the optimizer to descend rather than hover
        double targetX = _startX + (_padX - _startX) * t;
        double targetY = _startY + (_padY - _startY) * t;

        int baseIdx = step * _residualsPerStep;
        residuals[baseIdx + 0] = wt * _wPos * (state.X - targetX);
        residuals[baseIdx + 1] = wt * _wPos * (state.Y - targetY);
        residuals[baseIdx + 2] = wt * _wVel * state.VelX;
        residuals[baseIdx + 3] = wt * _wVel * state.VelY;
        residuals[baseIdx + 4] = wt * _wAngle * Math.Sin(state.Angle - MathF.PI / 2f);
        residuals[baseIdx + 5] = _wSmooth * (throttle - prevThrottle);
        residuals[baseIdx + 6] = _wSmooth * (gimbal - prevGimbal);
        residuals[baseIdx + 7] = _wFuel * throttle;
    }

    private void ComputeTerminalResiduals(double[] residuals, TrajectoryState finalState)
    {
        int baseIdx = _controlSteps * _residualsPerStep;
        residuals[baseIdx + 0] = _wTerminal * (finalState.X - _padX);
        residuals[baseIdx + 1] = _wTerminal * (finalState.Y - _padY);
        residuals[baseIdx + 2] = _wTerminal * finalState.VelX;
        residuals[baseIdx + 3] = _wTerminal * finalState.VelY;
        residuals[baseIdx + 4] = _wTerminal * Math.Sin(finalState.Angle - MathF.PI / 2f);
    }

    private TrajectoryResult RolloutTrajectory(double[] parameters, int[] rocketIndices)
    {
        var result = new TrajectoryResult();
        result.Throttles = new float[_controlSteps];
        result.Gimbals = new float[_controlSteps];
        result.States = new TrajectoryState[_controlSteps + 1];

        result.States[0] = ExtractState(rocketIndices, 0f, 0f);

        for (int s = 0; s < _controlSteps; s++)
        {
            float throttle = ClampThrottle(parameters[2 * s]);
            float gimbal = ClampGimbal(parameters[2 * s + 1]);

            result.Throttles[s] = throttle;
            result.Gimbals[s] = gimbal;

            ApplyControlAndStep(rocketIndices, throttle, gimbal);
            result.States[s + 1] = ExtractState(rocketIndices, throttle, gimbal);
        }

        return result;
    }

    private void ApplyControlAndStep(int[] rocketIndices, float throttle, float gimbal)
    {
        for (int p = 0; p < _physicsStepsPerControl; p++)
        {
            RigidBodyRocketTemplate.ApplyThrust(_world, rocketIndices, throttle, _maxThrust, _config.Dt);
            RigidBodyRocketTemplate.ApplyGimbal(_world, rocketIndices, gimbal * _maxGimbalTorque, _config.Dt);
            _stepper.Step(_world, _config);
        }
    }

    private TrajectoryState ExtractState(int[] rocketIndices, float throttle, float gimbal)
    {
        RigidBodyRocketTemplate.GetCenterOfMass(_world, rocketIndices, out float comX, out float comY);
        RigidBodyRocketTemplate.GetVelocity(_world, rocketIndices, out float velX, out float velY);
        var body = _world.RigidBodies[rocketIndices[0]];

        return new TrajectoryState
        {
            X = comX, Y = comY,
            VelX = velX, VelY = velY,
            Angle = body.Angle, AngularVel = body.AngularVel,
            Throttle = throttle, Gimbal = gimbal
        };
    }

    private int[] SetupWorld(float startX, float startY, float startVelX, float startVelY, float startAngle)
    {
        _world.Clear();

        // Ground
        _world.Obbs.Add(OBBCollider.AxisAligned(0f, _groundY, 30f, 0.5f));

        // Create rocket
        var rocketIndices = RigidBodyRocketTemplate.CreateRocket(
            _world,
            centerX: startX,
            centerY: startY,
            bodyHeight: 1.5f,
            bodyRadius: 0.2f,
            legLength: 1.0f,
            legRadius: 0.1f,
            bodyMass: 8f,
            legMass: 1.5f);

        // Apply initial velocity and angle offset
        float angleDelta = startAngle - MathF.PI / 2f;
        for (int i = 0; i < rocketIndices.Length; i++)
        {
            var rb = _world.RigidBodies[rocketIndices[i]];
            rb.VelX = startVelX;
            rb.VelY = startVelY;
            rb.Angle += angleDelta;
            _world.RigidBodies[rocketIndices[i]] = rb;
        }

        return rocketIndices;
    }

    private float GetTotalMass(int[] rocketIndices)
    {
        float total = 0f;
        foreach (int idx in rocketIndices)
        {
            var rb = _world.RigidBodies[idx];
            if (rb.InvMass > 0f) total += 1f / rb.InvMass;
        }
        return total;
    }

    private static float ClampThrottle(double v) => (float)Math.Clamp(v, 0.0, 1.0);
    private static float ClampGimbal(double v) => (float)Math.Clamp(v, -1.0, 1.0);
}

/// <summary>
/// Configuration for the trajectory optimizer.
/// </summary>
public sealed class TrajectoryOptimizerOptions
{
    // Problem sizing
    public int ControlSteps { get; set; } = 40;
    public int PhysicsStepsPerControl { get; set; } = 15;

    // Residual weights
    public double WeightPosition { get; set; } = 1.0;
    public double WeightVelocity { get; set; } = 1.5;
    public double WeightAngle { get; set; } = 2.0;
    public double WeightSmooth { get; set; } = 0.3;
    public double WeightFuel { get; set; } = 0.1;
    public double WeightTerminal { get; set; } = 4.0;

    // Physics
    public float MaxThrust { get; set; } = 200f;
    public float MaxGimbalTorque { get; set; } = 50f;
    public float PadX { get; set; } = 0f;
    public float PadY { get; set; } = -4.5f;
    public float GroundY { get; set; } = -5f;
    public float SpawnHeight { get; set; } = 15f;

    // Solver
    public int MaxIterations { get; set; } = 50;
    public double InitialDamping { get; set; } = 1e-2;
    public double Epsilon { get; set; } = 1e-4;
    public Action<string>? LogCallback { get; set; }

    /// <summary>Called from the optimizer thread each time a baseline rollout completes.</summary>
    public Action<IterationSnapshot>? OnIterationComplete { get; set; }
}
