import type { World } from '../model/types';

export interface Command {
  execute(world: World): void;
  undo(world: World): void;
  description: string;
}

export class CommandHistory {
  private undoStack: Command[] = [];
  private redoStack: Command[] = [];

  execute(command: Command, world: World): void {
    command.execute(world);
    this.undoStack.push(command);
    this.redoStack.length = 0;
  }

  undo(world: World): boolean {
    const cmd = this.undoStack.pop();
    if (!cmd) return false;
    cmd.undo(world);
    this.redoStack.push(cmd);
    return true;
  }

  redo(world: World): boolean {
    const cmd = this.redoStack.pop();
    if (!cmd) return false;
    cmd.execute(world);
    this.undoStack.push(cmd);
    return true;
  }

  get canUndo(): boolean {
    return this.undoStack.length > 0;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  get undoDescription(): string | undefined {
    return this.undoStack.at(-1)?.description;
  }

  get redoDescription(): string | undefined {
    return this.redoStack.at(-1)?.description;
  }

  clear(): void {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
  }
}
