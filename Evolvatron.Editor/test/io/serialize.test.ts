import { describe, it, expect, beforeEach } from 'vitest';
import { serialize, deserialize } from '../../src/io/serialize';
import { createDefaultWorld, createObstacle, createCheckpoint, resetIdCounter } from '../../src/model/defaults';

describe('serialize/deserialize', () => {
  beforeEach(() => resetIdCounter());

  it('roundtrips default world', () => {
    const world = createDefaultWorld();
    const json = serialize(world);
    const { world: loaded } = deserialize(json);
    expect(loaded.groundY).toBe(world.groundY);
    expect(loaded.landingPad.position.x).toBe(world.landingPad.position.x);
    expect(loaded.spawnArea.xRange).toBe(world.spawnArea.xRange);
    expect(loaded.modules).toHaveLength(world.modules.length);
  });

  it('roundtrips world with added modules', () => {
    const world = createDefaultWorld();
    const baseCount = world.modules.length;
    world.modules.push(createObstacle(3, 5));
    world.modules.push(createCheckpoint(-2, 8, 0));
    const json = serialize(world);
    const { world: loaded } = deserialize(json);
    expect(loaded.modules).toHaveLength(baseCount + 2);
    expect(loaded.modules[baseCount].kind).toBe('obstacle');
    expect(loaded.modules[baseCount].position.x).toBe(3);
    expect(loaded.modules[baseCount + 1].kind).toBe('checkpoint');
  });

  it('preserves simulation config', () => {
    const world = createDefaultWorld();
    world.simulationConfig.maxThrust = 999;
    const { world: loaded } = deserialize(serialize(world));
    expect(loaded.simulationConfig.maxThrust).toBe(999);
  });

  it('preserves reward weights', () => {
    const world = createDefaultWorld();
    world.rewardWeights.positionWeight = 42;
    const { world: loaded } = deserialize(serialize(world));
    expect(loaded.rewardWeights.positionWeight).toBe(42);
  });

  it('roundtrips camera state', () => {
    const world = createDefaultWorld();
    const json = serialize(world, { centerX: 5, centerY: 10, zoom: 50 });
    const { camera } = deserialize(json);
    expect(camera).toEqual({ centerX: 5, centerY: 10, zoom: 50 });
  });

  it('handles missing camera gracefully', () => {
    const world = createDefaultWorld();
    const json = serialize(world);
    const { camera } = deserialize(json);
    expect(camera).toBeUndefined();
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
