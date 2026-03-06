import type { World, Module } from '../model/types';
import type { Command } from './command';

interface RemovedEntry {
  module: Module;
  index: number;
}

export class RemoveModulesCommand implements Command {
  description: string;
  private removed: RemovedEntry[] = [];

  constructor(private ids: string[]) {
    this.description = ids.length === 1
      ? `Remove module "${ids[0]}"`
      : `Remove ${ids.length} modules`;
  }

  execute(world: World): void {
    this.removed = [];
    const idSet = new Set(this.ids);
    for (let i = world.modules.length - 1; i >= 0; i--) {
      if (idSet.has(world.modules[i].id)) {
        this.removed.push({ module: world.modules[i], index: i });
        world.modules.splice(i, 1);
      }
    }
    this.removed.reverse();
  }

  undo(world: World): void {
    for (const entry of this.removed) {
      world.modules.splice(entry.index, 0, entry.module);
    }
  }
}
