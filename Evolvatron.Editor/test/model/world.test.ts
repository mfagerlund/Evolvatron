import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, createCheckpoint, resetIdCounter } from '../../src/model/defaults';
import { findModule, findModuleIndex, allSelectableObjects, validate } from '../../src/model/world';
import type { World } from '../../src/model/types';

let world: World;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
});

describe('findModule', () => {
  it('returns undefined for empty modules list', () => {
    expect(findModule(world, 'nonexistent')).toBeUndefined();
  });

  it('finds an existing module by id', () => {
    const obs = createObstacle(1, 2);
    world.modules.push(obs);
    expect(findModule(world, obs.id)).toBe(obs);
  });

  it('returns undefined for wrong id', () => {
    world.modules.push(createObstacle());
    expect(findModule(world, 'wrong_id')).toBeUndefined();
  });
});

describe('findModuleIndex', () => {
  it('returns -1 for missing module', () => {
    expect(findModuleIndex(world, 'missing')).toBe(-1);
  });

  it('returns correct index', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    world.modules.push(a, b);
    expect(findModuleIndex(world, a.id)).toBe(0);
    expect(findModuleIndex(world, b.id)).toBe(1);
  });
});

describe('allSelectableObjects', () => {
  it('always includes landingPad and spawnArea', () => {
    const objs = allSelectableObjects(world);
    const ids = objs.map(o => o.id);
    expect(ids).toContain('landingPad');
    expect(ids).toContain('spawnArea');
    expect(objs.length).toBe(2);
  });

  it('includes modules', () => {
    const obs = createObstacle(5, 5);
    world.modules.push(obs);
    const objs = allSelectableObjects(world);
    expect(objs.length).toBe(3);
    expect(objs[2].id).toBe(obs.id);
    expect(objs[2].position).toEqual({ x: 5, y: 5 });
  });

  it('positions match actual world positions', () => {
    const objs = allSelectableObjects(world);
    const pad = objs.find(o => o.id === 'landingPad')!;
    expect(pad.position).toBe(world.landingPad.position);
    const spawn = objs.find(o => o.id === 'spawnArea')!;
    expect(spawn.position).toBe(world.spawnArea.position);
  });
});

describe('validate', () => {
  it('returns no errors for default world', () => {
    expect(validate(world)).toEqual([]);
  });

  it('reports error when landing pad is below ground', () => {
    world.landingPad.position.y = -10;
    world.groundY = -5;
    const errors = validate(world);
    expect(errors.some(e => e.includes('Landing pad'))).toBe(true);
  });

  it('reports error when spawn area is below landing pad', () => {
    world.spawnArea.position.y = world.landingPad.position.y - 1;
    const errors = validate(world);
    expect(errors.some(e => e.includes('Spawn area'))).toBe(true);
  });

  it('reports error for duplicate checkpoint orders', () => {
    world.modules.push(createCheckpoint(0, 5, 1));
    world.modules.push(createCheckpoint(2, 5, 1)); // duplicate order
    const errors = validate(world);
    expect(errors.some(e => e.includes('Checkpoint orders'))).toBe(true);
  });

  it('no error for unique checkpoint orders', () => {
    world.modules.push(createCheckpoint(0, 5, 0));
    world.modules.push(createCheckpoint(2, 5, 1));
    expect(validate(world)).toEqual([]);
  });
});
