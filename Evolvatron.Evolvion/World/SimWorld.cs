namespace Evolvatron.Evolvion.World;

/// <summary>
/// C# mirror of the editor's SimWorld JSON export.
/// All angles are in radians after loading (converted by SimWorldLoader).
/// These are plain data objects — no IDs, no lookups.
/// </summary>
public class SimWorld
{
    public float GroundY { get; set; }
    public SimLandingPad LandingPad { get; set; } = null!;
    public SimSpawn Spawn { get; set; } = null!;
    public SimObstacle[] Obstacles { get; set; } = [];
    public SimCheckpoint[] Checkpoints { get; set; } = [];
    public SimSpeedZone[] SpeedZones { get; set; } = [];
    public SimDangerZone[] DangerZones { get; set; } = [];
    public SimAttractor[] Attractors { get; set; } = [];
    public SimSimulationConfig SimulationConfig { get; set; } = null!;
    public SimRewardWeights RewardWeights { get; set; } = null!;
}

public class SimLandingPad
{
    public float PadX { get; set; }
    public float PadY { get; set; }
    public float PadHalfWidth { get; set; }
    public float PadHalfHeight { get; set; }
    public float LandingBonus { get; set; }
    public float MaxLandingVelocity { get; set; }
    public float MaxLandingAngle { get; set; }
    public float AttractionMagnitude { get; set; }
    public float AttractionRadius { get; set; }
}

public class SimSpawn
{
    public float X { get; set; }
    public float Y { get; set; }
    public float XRange { get; set; }
    public float HeightRange { get; set; }
    public float AngleRange { get; set; }
    public float VelXRange { get; set; }
    public float VelYMax { get; set; }
    public int SpawnCount { get; set; }
    public int SpawnSeed { get; set; }
}

public class SimObstacle
{
    public float CX { get; set; }
    public float CY { get; set; }
    public float UX { get; set; }
    public float UY { get; set; }
    public float HalfExtentX { get; set; }
    public float HalfExtentY { get; set; }
    public bool IsLethal { get; set; }
    public float PenaltyPerStep { get; set; }
    public float InfluenceRadius { get; set; }
}

public class SimCheckpoint
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Radius { get; set; }
    public int Order { get; set; }
    public float RewardBonus { get; set; }
    public float InfluenceRadius { get; set; }
}

public class SimSpeedZone
{
    public float X { get; set; }
    public float Y { get; set; }
    public float HalfExtentX { get; set; }
    public float HalfExtentY { get; set; }
    public float MaxSpeed { get; set; }
    public float RewardPerStep { get; set; }
}

public class SimDangerZone
{
    public float X { get; set; }
    public float Y { get; set; }
    public float HalfExtentX { get; set; }
    public float HalfExtentY { get; set; }
    public float PenaltyPerStep { get; set; }
    public bool IsLethal { get; set; }
    public float InfluenceRadius { get; set; }
}

public class SimAttractor
{
    public float X { get; set; }
    public float Y { get; set; }
    public float HalfExtentX { get; set; }
    public float HalfExtentY { get; set; }
    public float Magnitude { get; set; }
    public float InfluenceRadius { get; set; }
    public float ContactBonus { get; set; }
}

public class SimSimulationConfig
{
    public float Dt { get; set; }
    public float GravityY { get; set; }
    public float FrictionMu { get; set; }
    public float Restitution { get; set; }
    public float GlobalDamping { get; set; }
    public float AngularDamping { get; set; }
    public int SolverIterations { get; set; }
    public float MaxThrust { get; set; }
    public float MaxGimbalAngle { get; set; }
    public int SensorCount { get; set; }
    public int MaxSteps { get; set; }
    public float HasteBonus { get; set; }
}

public class SimRewardWeights
{
    public float PositionWeight { get; set; }
    public float VelocityWeight { get; set; }
    public float AngleWeight { get; set; }
    public float AngularVelocityWeight { get; set; }
    public float ControlEffortWeight { get; set; }
}
