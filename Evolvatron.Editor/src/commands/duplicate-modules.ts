import type { World, Module } from '../model/types';
import type { Command } from './command';
import { findModuleIndex } from '../model/world';

export class DuplicateModulesCommand implements Command {
  description: string;
  private cloneIds: string[];

  constructor(
    sourceIds: string[],
    private clones: Map<string, Module>,
  ) {
    this.cloneIds = [...clones.keys()];
    this.description = sourceIds.length === 1
      ? `Duplicate "${sourceIds[0]}"`
      : `Duplicate ${sourceIds.length} modules`;
  }

  execute(world: World): void {
    for (const clone of this.clones.values()) {
      world.modules.push(clone);
    }
  }

  undo(world: World): void {
    for (let i = this.cloneIds.length - 1; i >= 0; i--) {
      const idx = findModuleIndex(world, this.cloneIds[i]);
      if (idx !== -1) {
        world.modules.splice(idx, 1);
      }
    }
  }
}
