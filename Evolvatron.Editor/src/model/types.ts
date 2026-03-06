export interface Vec2 {
  x: number;
  y: number;
}

export type SelectableId = string | 'landingPad' | 'spawnArea';

export interface LandingPad {
  position: Vec2;
  halfWidth: number;
  halfHeight: number;
  landingBonus: number;
  maxLandingVelocity: number;
  maxLandingAngle: number; // degrees
}

export interface SpawnArea {
  position: Vec2;
  xRange: number;
  heightRange: number;
  angleRange: number; // degrees
  velXRange: number;
  velYMax: number;
}

export type ModuleKind = 'obstacle' | 'checkpoint' | 'speedZone' | 'dangerZone' | 'attractor';

export interface ObstacleModule {
  id: string;
  kind: 'obstacle';
  position: Vec2;
  halfExtentX: number;
  halfExtentY: number;
  rotation: number; // degrees
  isLethal: boolean;
}

export interface CheckpointModule {
  id: string;
  kind: 'checkpoint';
  position: Vec2;
  radius: number;
  order: number;
  rewardBonus: number;
  /** Multiplier on radius for influence falloff region (1 = no falloff beyond core) */
  influenceFactor: number;
}

export interface SpeedZoneModule {
  id: string;
  kind: 'speedZone';
  position: Vec2;
  halfExtentX: number;
  halfExtentY: number;
  maxSpeed: number;
  rewardPerStep: number;
}

export interface DangerZoneModule {
  id: string;
  kind: 'dangerZone';
  position: Vec2;
  halfExtentX: number;
  halfExtentY: number;
  penaltyPerStep: number;
  isLethal: boolean;
  /** Multiplier on size for influence falloff region (1 = no falloff beyond core) */
  influenceFactor: number;
}

export interface AttractorModule {
  id: string;
  kind: 'attractor';
  position: Vec2;
  halfExtentX: number;
  halfExtentY: number;
  /** Positive = reward (attractor), negative = penalty (repulsor) */
  magnitude: number;
  /** Multiplier on size for influence falloff region */
  influenceFactor: number;
  /** One-time bonus when entering the core zone (0 to disable) */
  contactBonus: number;
}

export type Module = ObstacleModule | CheckpointModule | SpeedZoneModule | DangerZoneModule | AttractorModule;

export interface SimulationConfig {
  dt: number;
  gravityY: number;
  frictionMu: number;
  restitution: number;
  globalDamping: number;
  angularDamping: number;
  solverIterations: number;
  maxThrust: number;
  maxGimbalAngle: number; // degrees
  sensorCount: number;
  maxSteps: number;
}

export interface RewardWeights {
  positionWeight: number;
  velocityWeight: number;
  angleWeight: number;
  angularVelocityWeight: number;
  controlEffortWeight: number;
}

export interface World {
  landingPad: LandingPad;
  spawnArea: SpawnArea;
  modules: Module[];
  groundY: number;
  simulationConfig: SimulationConfig;
  rewardWeights: RewardWeights;
}
