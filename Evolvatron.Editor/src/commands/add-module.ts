import type { World, Module } from '../model/types';
import type { Command } from './command';
import { findModuleIndex } from '../model/world';

export class AddModuleCommand implements Command {
  description: string;

  constructor(private module: Module) {
    this.description = `Add ${module.kind} "${module.id}"`;
  }

  execute(world: World): void {
    world.modules.push(this.module);
  }

  undo(world: World): void {
    const idx = findModuleIndex(world, this.module.id);
    if (idx !== -1) {
      world.modules.splice(idx, 1);
    }
  }
}
