namespace Evolvatron.Demo;

/// <summary>
/// Evolvatron graphical demo launcher.
/// Shows real-time XPBD particle and rigid body physics, or evolutionary ML demos.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "corridor")
        {
            // Launch corridor evolution demo
            FollowTheCorridorDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "sweep")
        {
            // Launch hyperparameter sweep
            CorridorHyperparameterSweep.Run();
        }
        else if (args.Length > 0 && args[0] == "timing")
        {
            // Launch simple timing test
            SimpleTimingTest.Run();
        }
        else if (args.Length > 0 && args[0] == "angles")
        {
            // Launch angle constraint demo
            AngleConstraintDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "mutations")
        {
            // Launch mutation progression visualization demo
            MutationProgressionDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "initial")
        {
            // Launch initial topology grid demo
            InitialTopologyGridDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "spiral")
        {
            // Launch spiral classification demo
            SpiralClassificationDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "landscape")
        {
            // Launch landscape benchmark demo
            LandscapeBenchmarkDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "chase")
        {
            // Launch AI target chase demo
            TargetChaseDemo.Run();
        }
        else if (args.Length > 0 && args[0] == "batched")
        {
            // Launch GPU batched evolution demo
            BatchedTargetChaseDemo.Run();
        }
        else
        {
            // Launch physics demo by default
            GraphicalDemo.Run();
        }
    }
}
