import type { World } from '../model/types';
import type { Command } from './command';
import { findModule } from '../model/world';

export interface ResizeSnapshot {
  [key: string]: number;
}

export class ResizeModuleCommand implements Command {
  description: string;

  constructor(
    private moduleId: string,
    private oldValues: ResizeSnapshot,
    private newValues: ResizeSnapshot,
  ) {
    this.description = `Resize "${moduleId}"`;
  }

  execute(world: World): void {
    this.apply(world, this.newValues);
  }

  undo(world: World): void {
    this.apply(world, this.oldValues);
  }

  private apply(world: World, values: ResizeSnapshot): void {
    const mod = findModule(world, this.moduleId);
    if (!mod) return;
    if ('posX' in values) mod.position.x = values.posX;
    if ('posY' in values) mod.position.y = values.posY;
    for (const [key, val] of Object.entries(values)) {
      if (key !== 'posX' && key !== 'posY') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (mod as any)[key] = val;
      }
    }
  }
}

export class ResizeSingletonCommand implements Command {
  description: string;

  constructor(
    private target: 'landingPad' | 'spawnArea',
    private oldValues: ResizeSnapshot,
    private newValues: ResizeSnapshot,
  ) {
    this.description = `Resize ${target}`;
  }

  execute(world: World): void {
    this.apply(world, this.newValues);
  }

  undo(world: World): void {
    this.apply(world, this.oldValues);
  }

  private apply(world: World, values: ResizeSnapshot): void {
    const obj = world[this.target];
    if ('posX' in values) obj.position.x = values.posX;
    if ('posY' in values) obj.position.y = values.posY;
    for (const [key, val] of Object.entries(values)) {
      if (key !== 'posX' && key !== 'posY') {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (obj as any)[key] = val;
      }
    }
  }
}
