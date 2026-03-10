import type { World, ObstacleModule } from '../model/types';

export interface SimObstacle {
  CX: number;
  CY: number;
  UX: number;
  UY: number;
  HalfExtentX: number;
  HalfExtentY: number;
  IsLethal: boolean;
  PenaltyPerStep: number;
  InfluenceRadius: number;
}

export interface SimCheckpoint {
  X: number;
  Y: number;
  Radius: number;
  Order: number;
  RewardBonus: number;
  InfluenceRadius: number;
}

export interface SimSpeedZone {
  X: number;
  Y: number;
  HalfExtentX: number;
  HalfExtentY: number;
  MaxSpeed: number;
  RewardPerStep: number;
}

export interface SimDangerZone {
  X: number;
  Y: number;
  HalfExtentX: number;
  HalfExtentY: number;
  PenaltyPerStep: number;
  IsLethal: boolean;
  InfluenceRadius: number;
}

export interface SimAttractor {
  X: number;
  Y: number;
  HalfExtentX: number;
  HalfExtentY: number;
  Magnitude: number;
  InfluenceRadius: number;
  ContactBonus: number;
}

export interface SimWorld {
  GroundY: number;
  LandingPad: {
    PadX: number;
    PadY: number;
    PadHalfWidth: number;
    PadHalfHeight: number;
    LandingBonus: number;
    MaxLandingVelocity: number;
    MaxLandingAngle: number;
    AttractionMagnitude: number;
    AttractionRadius: number;
  };
  Spawn: {
    X: number;
    Y: number;
    XRange: number;
    HeightRange: number;
    AngleRange: number;
    VelXRange: number;
    VelYMax: number;
    SpawnCount: number;
    SpawnSeed: number;
  };
  Obstacles: SimObstacle[];
  Checkpoints: SimCheckpoint[];
  SpeedZones: SimSpeedZone[];
  DangerZones: SimDangerZone[];
  Attractors: SimAttractor[];
  SimulationConfig: {
    Dt: number;
    GravityY: number;
    FrictionMu: number;
    Restitution: number;
    GlobalDamping: number;
    AngularDamping: number;
    SolverIterations: number;
    MaxThrust: number;
    MaxGimbalAngle: number;
    SensorCount: number;
    MaxSteps: number;
    HasteBonus: number;
  };
  RewardWeights: {
    PositionWeight: number;
    VelocityWeight: number;
    AngleWeight: number;
    AngularVelocityWeight: number;
    ControlEffortWeight: number;
  };
}

export function obstacleToSim(obs: ObstacleModule): SimObstacle {
  const rad = (obs.rotation * Math.PI) / 180;
  return {
    CX: obs.position.x,
    CY: obs.position.y,
    UX: Math.cos(rad),
    UY: Math.sin(rad),
    HalfExtentX: obs.halfExtentX,
    HalfExtentY: obs.halfExtentY,
    IsLethal: obs.isLethal,
    PenaltyPerStep: obs.penaltyPerStep,
    InfluenceRadius: obs.influenceRadius,
  };
}

export function exportSimWorld(world: World): SimWorld {
  const obstacles: SimObstacle[] = [];
  const checkpoints: SimCheckpoint[] = [];
  const speedZones: SimSpeedZone[] = [];
  const dangerZones: SimDangerZone[] = [];
  const attractors: SimAttractor[] = [];

  for (const mod of world.modules) {
    switch (mod.kind) {
      case 'obstacle':
        obstacles.push(obstacleToSim(mod));
        break;
      case 'checkpoint':
        checkpoints.push({
          X: mod.position.x,
          Y: mod.position.y,
          Radius: mod.radius,
          Order: mod.order,
          RewardBonus: mod.rewardBonus,
          InfluenceRadius: mod.influenceRadius,
        });
        break;
      case 'speedZone':
        speedZones.push({
          X: mod.position.x,
          Y: mod.position.y,
          HalfExtentX: mod.halfExtentX,
          HalfExtentY: mod.halfExtentY,
          MaxSpeed: mod.maxSpeed,
          RewardPerStep: mod.rewardPerStep,
        });
        break;
      case 'dangerZone':
        dangerZones.push({
          X: mod.position.x,
          Y: mod.position.y,
          HalfExtentX: mod.halfExtentX,
          HalfExtentY: mod.halfExtentY,
          PenaltyPerStep: mod.penaltyPerStep,
          IsLethal: mod.isLethal,
          InfluenceRadius: mod.influenceRadius,
        });
        break;
      case 'attractor':
        attractors.push({
          X: mod.position.x,
          Y: mod.position.y,
          HalfExtentX: mod.halfExtentX,
          HalfExtentY: mod.halfExtentY,
          Magnitude: mod.magnitude,
          InfluenceRadius: mod.influenceRadius,
          ContactBonus: mod.contactBonus,
        });
        break;
    }
  }

  const pad = world.landingPad;
  const spawn = world.spawnArea;
  const cfg = world.simulationConfig;
  const rw = world.rewardWeights;

  return {
    GroundY: world.groundY,
    LandingPad: {
      PadX: pad.position.x,
      PadY: pad.position.y,
      PadHalfWidth: pad.halfWidth,
      PadHalfHeight: pad.halfHeight,
      LandingBonus: pad.landingBonus,
      MaxLandingVelocity: pad.maxLandingVelocity,
      MaxLandingAngle: pad.maxLandingAngle,
      AttractionMagnitude: pad.attractionMagnitude,
      AttractionRadius: pad.attractionRadius,
    },
    Spawn: {
      X: spawn.position.x,
      Y: spawn.position.y,
      XRange: spawn.xRange,
      HeightRange: spawn.heightRange,
      AngleRange: spawn.angleRange,
      VelXRange: spawn.velXRange,
      VelYMax: spawn.velYMax,
      SpawnCount: spawn.spawnCount,
      SpawnSeed: spawn.spawnSeed,
    },
    Obstacles: obstacles,
    Checkpoints: checkpoints,
    SpeedZones: speedZones,
    DangerZones: dangerZones,
    Attractors: attractors,
    SimulationConfig: {
      Dt: cfg.dt,
      GravityY: cfg.gravityY,
      FrictionMu: cfg.frictionMu,
      Restitution: cfg.restitution,
      GlobalDamping: cfg.globalDamping,
      AngularDamping: cfg.angularDamping,
      SolverIterations: cfg.solverIterations,
      MaxThrust: cfg.maxThrust,
      MaxGimbalAngle: cfg.maxGimbalAngle,
      SensorCount: cfg.sensorCount,
      MaxSteps: cfg.maxSteps,
      HasteBonus: cfg.hasteBonus,
    },
    RewardWeights: {
      PositionWeight: rw.positionWeight,
      VelocityWeight: rw.velocityWeight,
      AngleWeight: rw.angleWeight,
      AngularVelocityWeight: rw.angularVelocityWeight,
      ControlEffortWeight: rw.controlEffortWeight,
    },
  };
}

export function exportSim(world: World): string {
  return JSON.stringify(exportSimWorld(world), null, 2);
}
