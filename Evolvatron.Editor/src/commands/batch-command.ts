import type { World } from '../model/types';
import type { Command } from './command';

export class BatchCommand implements Command {
  description: string;

  constructor(
    private commands: Command[],
    description?: string,
  ) {
    this.description = description ?? `Batch (${commands.length} operations)`;
  }

  execute(world: World): void {
    for (const cmd of this.commands) {
      cmd.execute(world);
    }
  }

  undo(world: World): void {
    for (let i = this.commands.length - 1; i >= 0; i--) {
      this.commands[i].undo(world);
    }
  }
}
