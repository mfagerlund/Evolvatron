import type {
  World, LandingPad, SpawnArea, SimulationConfig, RewardWeights,
  ObstacleModule, CheckpointModule, SpeedZoneModule, DangerZoneModule, AttractorModule,
} from './types';

let nextId = 1;
export function generateId(): string {
  return `mod_${nextId++}_${Date.now().toString(36)}`;
}

export function resetIdCounter(): void {
  nextId = 1;
}

export function createLandingPad(): LandingPad {
  return {
    position: { x: 0, y: -4.5 },
    halfWidth: 2,
    halfHeight: 0.25,
    landingBonus: 100,
    maxLandingVelocity: 2.0,
    maxLandingAngle: 15,
    attractionMagnitude: 10,
    attractionRadius: 10,
  };
}

export function createSpawnArea(): SpawnArea {
  return {
    position: { x: 0, y: 15 },
    xRange: 8,
    heightRange: 2,
    angleRange: 30,
    velXRange: 2,
    velYMax: 3,
    spawnCount: 10,
    spawnSeed: 0,
  };
}

export function createSimulationConfig(): SimulationConfig {
  return {
    dt: 1 / 120,
    gravityY: -9.81,
    frictionMu: 0.8,
    restitution: 0.0,
    globalDamping: 0.02,
    angularDamping: 0.1,
    solverIterations: 6,
    maxThrust: 200,
    maxGimbalAngle: 15,
    sensorCount: 4,
    maxSteps: 600,
    hasteBonus: 1.0,
  };
}

export function createRewardWeights(): RewardWeights {
  return {
    positionWeight: 1.0,
    velocityWeight: 0.5,
    angleWeight: 0.3,
    angularVelocityWeight: 0.1,
    controlEffortWeight: 0.05,
  };
}

export function createDefaultWorld(): World {
  return {
    landingPad: createLandingPad(),
    spawnArea: createSpawnArea(),
    modules: createDemoModules(),
    groundY: -5,
    simulationConfig: createSimulationConfig(),
    rewardWeights: createRewardWeights(),
  };
}

function createDemoModules(): World['modules'] {
  const obs = createObstacle(-9.517, 9.467);
  obs.halfExtentX = 2.583;
  obs.halfExtentY = 2.267;

  const cp0 = createCheckpoint(-1.933, 9.583, 0);
  cp0.radius = 1.792;

  const cp1 = createCheckpoint(3.7, 10.6, 1);
  cp1.radius = 1.447;

  const sz = createSpeedZone(6.383, 7.783);
  sz.halfExtentX = 1.25;
  sz.halfExtentY = 1.783;

  const dz = createDangerZone(-4.617, 3.517);
  dz.halfExtentX = 0.75;
  dz.halfExtentY = 1.083;

  const att = createAttractor(2.683, 1.15);
  att.halfExtentX = 1.883;
  att.halfExtentY = 1.95;

  return [obs, cp0, cp1, sz, dz, att];
}

export function createObstacle(x = 0, y = 0): ObstacleModule {
  return {
    id: generateId(),
    kind: 'obstacle',
    position: { x, y },
    halfExtentX: 2,
    halfExtentY: 2,
    rotation: 0,
    isLethal: false,
    penaltyPerStep: 10,
    influenceRadius: 3,
  };
}

export function createCheckpoint(x = 0, y = 0, order = 0): CheckpointModule {
  return {
    id: generateId(),
    kind: 'checkpoint',
    position: { x, y },
    radius: 1.5,
    order,
    rewardBonus: 20,
    influenceRadius: 2,
  };
}

export function createSpeedZone(x = 0, y = 0): SpeedZoneModule {
  return {
    id: generateId(),
    kind: 'speedZone',
    position: { x, y },
    halfExtentX: 2,
    halfExtentY: 2,
    maxSpeed: 5,
    rewardPerStep: 0.1,
  };
}

export function createDangerZone(x = 0, y = 0): DangerZoneModule {
  return {
    id: generateId(),
    kind: 'dangerZone',
    position: { x, y },
    halfExtentX: 2,
    halfExtentY: 2,
    penaltyPerStep: 10,
    isLethal: false,
    influenceRadius: 3,
  };
}

export function createAttractor(x = 0, y = 0): AttractorModule {
  return {
    id: generateId(),
    kind: 'attractor',
    position: { x, y },
    halfExtentX: 1.5,
    halfExtentY: 1.5,
    magnitude: 10,
    influenceRadius: 3,
    contactBonus: 50,
  };
}
