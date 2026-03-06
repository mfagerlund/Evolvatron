import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, createCheckpoint, resetIdCounter } from '../../src/model/defaults';
import { AddModuleCommand } from '../../src/commands/add-module';
import { findModule } from '../../src/model/world';
import type { World } from '../../src/model/types';

let world: World;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
});

describe('AddModuleCommand', () => {
  it('execute adds module to world', () => {
    const obs = createObstacle(3, 4);
    const cmd = new AddModuleCommand(obs);
    cmd.execute(world);
    expect(world.modules.length).toBe(1);
    expect(findModule(world, obs.id)).toBe(obs);
  });

  it('undo removes the added module', () => {
    const obs = createObstacle();
    const cmd = new AddModuleCommand(obs);
    cmd.execute(world);
    expect(world.modules.length).toBe(1);

    cmd.undo(world);
    expect(world.modules.length).toBe(0);
    expect(findModule(world, obs.id)).toBeUndefined();
  });

  it('undo only removes the specific module', () => {
    const a = createObstacle(1, 1);
    const b = createCheckpoint(2, 2);
    world.modules.push(a);
    const cmd = new AddModuleCommand(b);
    cmd.execute(world);
    expect(world.modules.length).toBe(2);

    cmd.undo(world);
    expect(world.modules.length).toBe(1);
    expect(world.modules[0]).toBe(a);
  });

  it('has a meaningful description', () => {
    const obs = createObstacle();
    const cmd = new AddModuleCommand(obs);
    expect(cmd.description).toContain('obstacle');
    expect(cmd.description).toContain(obs.id);
  });

  it('execute then undo is a no-op', () => {
    const obs = createObstacle();
    const cmd = new AddModuleCommand(obs);
    const before = [...world.modules];
    cmd.execute(world);
    cmd.undo(world);
    expect(world.modules).toEqual(before);
  });
});
