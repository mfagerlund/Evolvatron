import type { World } from '../model/types';
import type { Command } from './command';
import { findModule } from '../model/world';

export class UpdatePropertyCommand implements Command {
  description: string;

  constructor(
    private moduleId: string,
    private propertyPath: string,
    private oldValue: unknown,
    private newValue: unknown,
  ) {
    this.description = `Set ${propertyPath} on "${moduleId}"`;
  }

  execute(world: World): void {
    this.setValue(world, this.newValue);
  }

  undo(world: World): void {
    this.setValue(world, this.oldValue);
  }

  private setValue(world: World, value: unknown): void {
    const module = findModule(world, this.moduleId);
    if (!module) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (module as any)[this.propertyPath] = value;
  }
}
