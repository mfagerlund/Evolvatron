import { describe, it, expect, beforeEach } from 'vitest';
import { createDefaultWorld, createObstacle, createCheckpoint, resetIdCounter } from '../../src/model/defaults';
import { CommandHistory } from '../../src/commands/command';
import { AddModuleCommand } from '../../src/commands/add-module';
import { RemoveModulesCommand } from '../../src/commands/remove-modules';
import type { World } from '../../src/model/types';

let world: World;
let history: CommandHistory;

beforeEach(() => {
  resetIdCounter();
  world = createDefaultWorld();
  history = new CommandHistory();
});

describe('CommandHistory', () => {
  it('starts with no undo/redo', () => {
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
    expect(history.undoDescription).toBeUndefined();
    expect(history.redoDescription).toBeUndefined();
  });

  it('execute enables undo', () => {
    const obs = createObstacle();
    history.execute(new AddModuleCommand(obs), world);
    expect(history.canUndo).toBe(true);
    expect(history.canRedo).toBe(false);
    expect(world.modules.length).toBe(1);
  });

  it('undo reverses the command', () => {
    const obs = createObstacle();
    history.execute(new AddModuleCommand(obs), world);
    const result = history.undo(world);
    expect(result).toBe(true);
    expect(world.modules.length).toBe(0);
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(true);
  });

  it('redo re-applies the command', () => {
    const obs = createObstacle();
    history.execute(new AddModuleCommand(obs), world);
    history.undo(world);
    const result = history.redo(world);
    expect(result).toBe(true);
    expect(world.modules.length).toBe(1);
    expect(history.canUndo).toBe(true);
    expect(history.canRedo).toBe(false);
  });

  it('undo returns false when stack is empty', () => {
    expect(history.undo(world)).toBe(false);
  });

  it('redo returns false when stack is empty', () => {
    expect(history.redo(world)).toBe(false);
  });

  it('new command clears redo stack', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    history.execute(new AddModuleCommand(a), world);
    history.undo(world);
    expect(history.canRedo).toBe(true);

    history.execute(new AddModuleCommand(b), world);
    expect(history.canRedo).toBe(false);
  });

  it('multiple undo/redo chain', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    history.execute(new AddModuleCommand(a), world);
    history.execute(new AddModuleCommand(b), world);
    expect(world.modules.length).toBe(2);

    history.undo(world);
    expect(world.modules.length).toBe(1);
    expect(world.modules[0].id).toBe(a.id);

    history.undo(world);
    expect(world.modules.length).toBe(0);

    history.redo(world);
    expect(world.modules.length).toBe(1);

    history.redo(world);
    expect(world.modules.length).toBe(2);
  });

  it('undoDescription and redoDescription reflect top of stack', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    history.execute(new AddModuleCommand(a), world);
    history.execute(new AddModuleCommand(b), world);

    expect(history.undoDescription).toContain(b.id);
    history.undo(world);
    expect(history.undoDescription).toContain(a.id);
    expect(history.redoDescription).toContain(b.id);
  });

  it('clear empties both stacks', () => {
    const obs = createObstacle();
    history.execute(new AddModuleCommand(obs), world);
    history.undo(world);
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(true);

    history.clear();
    expect(history.canUndo).toBe(false);
    expect(history.canRedo).toBe(false);
  });

  it('complex sequence: add, add, remove, undo remove, undo add', () => {
    const a = createObstacle();
    const b = createCheckpoint();
    history.execute(new AddModuleCommand(a), world);
    history.execute(new AddModuleCommand(b), world);
    expect(world.modules.length).toBe(2);

    history.execute(new RemoveModulesCommand([a.id]), world);
    expect(world.modules.length).toBe(1);
    expect(world.modules[0].id).toBe(b.id);

    history.undo(world); // undo remove
    expect(world.modules.length).toBe(2);

    history.undo(world); // undo add b
    expect(world.modules.length).toBe(1);
    expect(world.modules[0].id).toBe(a.id);
  });
});
