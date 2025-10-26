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
        else if (args.Length > 0 && args[0] == "angles")
        {
            // Launch angle constraint demo
            AngleConstraintDemo.Run();
        }
        else
        {
            // Launch physics demo by default
            GraphicalDemo.Run();
        }
    }
}
