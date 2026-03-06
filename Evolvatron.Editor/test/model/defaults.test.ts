import { describe, it, expect, beforeEach } from 'vitest';
import {
  createDefaultWorld,
  createObstacle,
  createCheckpoint,
  createSpeedZone,
  createDangerZone,
  createLandingPad,
  createSpawnArea,
  createSimulationConfig,
  createRewardWeights,
  resetIdCounter,
  generateId,
} from '../../src/model/defaults';

beforeEach(() => {
  resetIdCounter();
});

describe('createDefaultWorld', () => {
  it('returns a world with all required fields', () => {
    const w = createDefaultWorld();
    expect(w.landingPad).toBeDefined();
    expect(w.spawnArea).toBeDefined();
    expect(w.modules).toEqual([]);
    expect(typeof w.groundY).toBe('number');
    expect(w.simulationConfig).toBeDefined();
    expect(w.rewardWeights).toBeDefined();
  });

  it('has spawn above landing pad above ground', () => {
    const w = createDefaultWorld();
    expect(w.spawnArea.position.y).toBeGreaterThan(w.landingPad.position.y);
    expect(w.landingPad.position.y).toBeGreaterThan(w.groundY);
  });
});

describe('createLandingPad', () => {
  it('returns valid landing pad with positive dimensions', () => {
    const pad = createLandingPad();
    expect(pad.halfWidth).toBeGreaterThan(0);
    expect(pad.halfHeight).toBeGreaterThan(0);
    expect(pad.landingBonus).toBeGreaterThan(0);
    expect(pad.maxLandingVelocity).toBeGreaterThan(0);
    expect(pad.maxLandingAngle).toBeGreaterThan(0);
  });
});

describe('createSpawnArea', () => {
  it('returns valid spawn area with positive ranges', () => {
    const spawn = createSpawnArea();
    expect(spawn.xRange).toBeGreaterThan(0);
    expect(spawn.heightRange).toBeGreaterThan(0);
    expect(spawn.angleRange).toBeGreaterThan(0);
  });
});

describe('createSimulationConfig', () => {
  it('returns config with sensible physics values', () => {
    const cfg = createSimulationConfig();
    expect(cfg.dt).toBeGreaterThan(0);
    expect(cfg.gravityY).toBeLessThan(0);
    expect(cfg.solverIterations).toBeGreaterThanOrEqual(1);
    expect(cfg.maxSteps).toBeGreaterThan(0);
    expect(cfg.sensorCount).toBeGreaterThanOrEqual(0);
  });
});

describe('createRewardWeights', () => {
  it('returns weights that are all non-negative', () => {
    const rw = createRewardWeights();
    expect(rw.positionWeight).toBeGreaterThanOrEqual(0);
    expect(rw.velocityWeight).toBeGreaterThanOrEqual(0);
    expect(rw.angleWeight).toBeGreaterThanOrEqual(0);
    expect(rw.angularVelocityWeight).toBeGreaterThanOrEqual(0);
    expect(rw.controlEffortWeight).toBeGreaterThanOrEqual(0);
  });
});

describe('module factories', () => {
  it('createObstacle returns correct kind and position', () => {
    const m = createObstacle(3, 7);
    expect(m.kind).toBe('obstacle');
    expect(m.position).toEqual({ x: 3, y: 7 });
    expect(m.halfExtentX).toBeGreaterThan(0);
    expect(m.halfExtentY).toBeGreaterThan(0);
    expect(m.rotation).toBe(0);
    expect(typeof m.isLethal).toBe('boolean');
  });

  it('createCheckpoint returns correct kind with order', () => {
    const m = createCheckpoint(1, 2, 5);
    expect(m.kind).toBe('checkpoint');
    expect(m.position).toEqual({ x: 1, y: 2 });
    expect(m.order).toBe(5);
    expect(m.radius).toBeGreaterThan(0);
    expect(m.rewardBonus).toBeGreaterThan(0);
  });

  it('createSpeedZone returns correct kind', () => {
    const m = createSpeedZone(-1, 4);
    expect(m.kind).toBe('speedZone');
    expect(m.position).toEqual({ x: -1, y: 4 });
    expect(m.maxSpeed).toBeGreaterThan(0);
  });

  it('createDangerZone returns correct kind', () => {
    const m = createDangerZone(0, 0);
    expect(m.kind).toBe('dangerZone');
    expect(m.position).toEqual({ x: 0, y: 0 });
    expect(m.penaltyPerStep).toBeGreaterThan(0);
  });
});

describe('generateId / resetIdCounter', () => {
  it('produces unique IDs', () => {
    const a = generateId();
    const b = generateId();
    const c = generateId();
    expect(a).not.toBe(b);
    expect(b).not.toBe(c);
  });

  it('resetIdCounter makes the counter restart', () => {
    const first = generateId(); // mod_1_...
    resetIdCounter();
    const afterReset = generateId(); // mod_1_... again (same prefix)
    expect(first.startsWith('mod_1_')).toBe(true);
    expect(afterReset.startsWith('mod_1_')).toBe(true);
  });

  it('factory IDs are unique across different factories', () => {
    const ids = new Set([
      createObstacle().id,
      createCheckpoint().id,
      createSpeedZone().id,
      createDangerZone().id,
    ]);
    expect(ids.size).toBe(4);
  });
});
