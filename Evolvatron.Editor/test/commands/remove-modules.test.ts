import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, createCheckpoint, createSpeedZone, resetIdCounter } from '../../src/model/defaults';
import { RemoveModulesCommand } from '../../src/commands/remove-modules';
import type { World } from '../../src/model/types';

let world: World;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
});

describe('RemoveModulesCommand', () => {
  it('removes a single module', () => {
    const obs = createObstacle();
    world.modules.push(obs);
    const cmd = new RemoveModulesCommand([obs.id]);
    cmd.execute(world);
    expect(world.modules.length).toBe(0);
  });

  it('removes multiple modules', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    const c = createSpeedZone();
    world.modules.push(a, b, c);
    const cmd = new RemoveModulesCommand([a.id, c.id]);
    cmd.execute(world);
    expect(world.modules.length).toBe(1);
    expect(world.modules[0]).toBe(b);
  });

  it('undo restores modules at correct indices', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    const c = createSpeedZone();
    world.modules.push(a, b, c);

    const cmd = new RemoveModulesCommand([a.id, c.id]);
    cmd.execute(world);
    expect(world.modules.length).toBe(1);

    cmd.undo(world);
    expect(world.modules.length).toBe(3);
    expect(world.modules[0]).toBe(a);
    expect(world.modules[1]).toBe(b);
    expect(world.modules[2]).toBe(c);
  });

  it('undo restores middle element at correct position', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    const c = createSpeedZone();
    world.modules.push(a, b, c);

    const cmd = new RemoveModulesCommand([b.id]);
    cmd.execute(world);
    expect(world.modules.map(m => m.id)).toEqual([a.id, c.id]);

    cmd.undo(world);
    expect(world.modules.map(m => m.id)).toEqual([a.id, b.id, c.id]);
  });

  it('does nothing if ids do not match', () => {
    const obs = createObstacle();
    world.modules.push(obs);
    const cmd = new RemoveModulesCommand(['nonexistent']);
    cmd.execute(world);
    expect(world.modules.length).toBe(1);
  });

  it('description reflects count', () => {
    const cmd1 = new RemoveModulesCommand(['a']);
    expect(cmd1.description).toContain('"a"');

    const cmd2 = new RemoveModulesCommand(['a', 'b', 'c']);
    expect(cmd2.description).toContain('3');
  });
});
