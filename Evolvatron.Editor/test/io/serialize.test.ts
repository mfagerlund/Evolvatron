import { describe, it, expect, beforeEach } from 'vitest';
import { serialize, deserialize } from '../../src/io/serialize';
import { createDefaultWorld, createObstacle, createCheckpoint, resetIdCounter } from '../../src/model/defaults';

describe('serialize/deserialize', () => {
  beforeEach(() => resetIdCounter());

  it('roundtrips default world', () => {
    const world = createDefaultWorld();
    const json = serialize(world);
    const loaded = deserialize(json);
    expect(loaded.groundY).toBe(world.groundY);
    expect(loaded.landingPad.position.x).toBe(world.landingPad.position.x);
    expect(loaded.spawnArea.xRange).toBe(world.spawnArea.xRange);
    expect(loaded.modules).toHaveLength(0);
  });

  it('roundtrips world with modules', () => {
    const world = createDefaultWorld();
    world.modules.push(createObstacle(3, 5));
    world.modules.push(createCheckpoint(-2, 8, 0));
    const json = serialize(world);
    const loaded = deserialize(json);
    expect(loaded.modules).toHaveLength(2);
    expect(loaded.modules[0].kind).toBe('obstacle');
    expect(loaded.modules[0].position.x).toBe(3);
    expect(loaded.modules[1].kind).toBe('checkpoint');
  });

  it('preserves simulation config', () => {
    const world = createDefaultWorld();
    world.simulationConfig.maxThrust = 999;
    const loaded = deserialize(serialize(world));
    expect(loaded.simulationConfig.maxThrust).toBe(999);
  });

  it('preserves reward weights', () => {
    const world = createDefaultWorld();
    world.rewardWeights.positionWeight = 42;
    const loaded = deserialize(serialize(world));
    expect(loaded.rewardWeights.positionWeight).toBe(42);
  });

  it('rejects invalid version', () => {
    const json = JSON.stringify({ version: 999, world: {} });
    expect(() => deserialize(json)).toThrow('Unsupported format version');
  });

  it('rejects missing fields', () => {
    const json = JSON.stringify({ version: 1, world: { landingPad: {} } });
    expect(() => deserialize(json)).toThrow('missing required fields');
  });
});
