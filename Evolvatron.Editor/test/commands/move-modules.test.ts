import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, resetIdCounter } from '../../src/model/defaults';
import { MoveModulesCommand } from '../../src/commands/move-modules';
import type { World, SelectableId } from '../../src/model/types';

let world: World;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
});

describe('MoveModulesCommand', () => {
  it('moves a module by delta', () => {
    const obs = createObstacle(5, 10);
    world.modules.push(obs);
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      [obs.id, { x: 3, y: -2 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    expect(obs.position).toEqual({ x: 8, y: 8 });
  });

  it('undo restores original position', () => {
    const obs = createObstacle(5, 10);
    world.modules.push(obs);
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      [obs.id, { x: 3, y: -2 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    cmd.undo(world);
    expect(obs.position).toEqual({ x: 5, y: 10 });
  });

  it('moves landing pad singleton', () => {
    const origX = world.landingPad.position.x;
    const origY = world.landingPad.position.y;
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      ['landingPad', { x: 1, y: 2 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    expect(world.landingPad.position.x).toBe(origX + 1);
    expect(world.landingPad.position.y).toBe(origY + 2);
  });

  it('moves spawn area singleton', () => {
    const origX = world.spawnArea.position.x;
    const origY = world.spawnArea.position.y;
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      ['spawnArea', { x: -5, y: 0 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    expect(world.spawnArea.position.x).toBe(origX - 5);
    expect(world.spawnArea.position.y).toBe(origY);
  });

  it('moves multiple objects simultaneously', () => {
    const obs = createObstacle(0, 0);
    world.modules.push(obs);
    const padOrig = { ...world.landingPad.position };
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      [obs.id, { x: 10, y: 10 }],
      ['landingPad', { x: -1, y: -1 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    expect(obs.position).toEqual({ x: 10, y: 10 });
    expect(world.landingPad.position).toEqual({ x: padOrig.x - 1, y: padOrig.y - 1 });
  });

  it('undo after moving multiple objects restores all', () => {
    const obs = createObstacle(2, 3);
    world.modules.push(obs);
    const padOrig = { ...world.landingPad.position };
    const deltas = new Map<SelectableId, { x: number; y: number }>([
      [obs.id, { x: 10, y: 10 }],
      ['landingPad', { x: -1, y: -1 }],
    ]);
    const cmd = new MoveModulesCommand(deltas);
    cmd.execute(world);
    cmd.undo(world);
    expect(obs.position).toEqual({ x: 2, y: 3 });
    expect(world.landingPad.position).toEqual(padOrig);
  });

  it('description reflects object count', () => {
    const deltas1 = new Map<SelectableId, { x: number; y: number }>([
      ['landingPad', { x: 1, y: 1 }],
    ]);
    expect(new MoveModulesCommand(deltas1).description).toContain('landingPad');

    const deltas2 = new Map<SelectableId, { x: number; y: number }>([
      ['landingPad', { x: 1, y: 1 }],
      ['spawnArea', { x: 2, y: 2 }],
    ]);
    expect(new MoveModulesCommand(deltas2).description).toContain('2');
  });
});
