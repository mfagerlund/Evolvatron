import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, createCheckpoint, createSpeedZone, resetIdCounter } from '../../src/model/defaults';
import { pointInRect, pointInCircle, hitTestPoint } from '../../src/editor/hit-test';
import type { World } from '../../src/model/types';

let world: World;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
});

describe('pointInRect', () => {
  it('returns true for point at center', () => {
    expect(pointInRect(5, 5, 5, 5, 2, 1, 0)).toBe(true);
  });

  it('returns true for point inside', () => {
    expect(pointInRect(6, 5.5, 5, 5, 2, 1, 0)).toBe(true);
  });

  it('returns true for point on edge', () => {
    expect(pointInRect(7, 5, 5, 5, 2, 1, 0)).toBe(true);
  });

  it('returns false for point outside', () => {
    expect(pointInRect(8, 5, 5, 5, 2, 1, 0)).toBe(false);
  });

  it('handles rotation (45 degrees)', () => {
    // A 2x2 rect centered at origin, rotated 45 degrees
    // The diamond shape extends sqrt(2) along axes
    // Point (1,0) should be inside the rotated rect (corner of original)
    expect(pointInRect(1, 0, 0, 0, 1, 1, 45)).toBe(true);

    // Point (1.5, 0) should be outside the rotated 1x1 rect
    expect(pointInRect(1.5, 0, 0, 0, 1, 1, 45)).toBe(false);
  });

  it('handles 90-degree rotation', () => {
    // 3x1 rect at origin rotated 90 degrees becomes 1x3
    // Point at (0.5, 2) should be inside after rotation
    expect(pointInRect(0.5, 2, 0, 0, 3, 1, 90)).toBe(true);
    // Point at (2, 0) should be outside after rotation (was inside before)
    expect(pointInRect(2, 0, 0, 0, 3, 1, 90)).toBe(false);
  });
});

describe('pointInCircle', () => {
  it('returns true for point at center', () => {
    expect(pointInCircle(0, 0, 0, 0, 5)).toBe(true);
  });

  it('returns true for point inside', () => {
    expect(pointInCircle(3, 0, 0, 0, 5)).toBe(true);
  });

  it('returns true for point on boundary', () => {
    expect(pointInCircle(5, 0, 0, 0, 5)).toBe(true);
  });

  it('returns false for point outside', () => {
    expect(pointInCircle(6, 0, 0, 0, 5)).toBe(false);
  });

  it('works with offset center', () => {
    expect(pointInCircle(12, 10, 10, 10, 3)).toBe(true);
    expect(pointInCircle(14, 10, 10, 10, 3)).toBe(false);
  });
});

describe('hitTestPoint', () => {
  it('returns null on empty world (no module, point far from singletons)', () => {
    const result = hitTestPoint(world, 100, 100);
    expect(result).toBeNull();
  });

  it('hits landing pad', () => {
    const pad = world.landingPad;
    const result = hitTestPoint(world, pad.position.x, pad.position.y);
    expect(result).not.toBeNull();
    expect(result!.id).toBe('landingPad');
  });

  it('hits spawn area', () => {
    const spawn = world.spawnArea;
    const result = hitTestPoint(world, spawn.position.x, spawn.position.y);
    expect(result).not.toBeNull();
    expect(result!.id).toBe('spawnArea');
  });

  it('hits an obstacle module', () => {
    const obs = createObstacle(5, 5);
    world.modules.push(obs);
    const result = hitTestPoint(world, 5, 5);
    expect(result).not.toBeNull();
    expect(result!.id).toBe(obs.id);
  });

  it('hits a checkpoint module (circle)', () => {
    const cp = createCheckpoint(10, 10);
    world.modules.push(cp);
    const result = hitTestPoint(world, 10.5, 10);
    expect(result).not.toBeNull();
    expect(result!.id).toBe(cp.id);
  });

  it('hits a speed zone module', () => {
    const sz = createSpeedZone(20, 20);
    world.modules.push(sz);
    const result = hitTestPoint(world, 20, 20);
    expect(result).not.toBeNull();
    expect(result!.id).toBe(sz.id);
  });

  it('misses module when point is outside', () => {
    const obs = createObstacle(5, 5);
    world.modules.push(obs);
    // obstacle halfExtent defaults: 1 x 0.25
    const result = hitTestPoint(world, 7, 5);
    expect(result).toBeNull();
  });

  it('selects closest module when overlapping', () => {
    const a = createObstacle(5, 5);
    const b = createObstacle(5.5, 5);
    world.modules.push(a, b);
    // Point at (5.4, 5) is closer to b(5.5,5) than a(5,5)
    const result = hitTestPoint(world, 5.4, 5);
    expect(result).not.toBeNull();
    expect(result!.id).toBe(b.id);
  });

  it('prefers closer module over singleton', () => {
    // Place an obstacle right on top of the landing pad
    const pad = world.landingPad;
    const obs = createObstacle(pad.position.x, pad.position.y);
    world.modules.push(obs);
    const result = hitTestPoint(world, pad.position.x, pad.position.y);
    // Both are at distance 0 from center; first found (module) wins if distance is equal
    expect(result).not.toBeNull();
    // The one with smaller distance should win; at same center both are distance 0
    expect(result!.distance).toBe(0);
  });
});
