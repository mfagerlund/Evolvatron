import type { World } from '../model/types';
import type { Command } from './command';

type SingletonTarget = 'landingPad' | 'spawnArea';

export class UpdateSingletonCommand implements Command {
  description: string;

  constructor(
    private target: SingletonTarget,
    private propertyPath: string,
    private oldValue: unknown,
    private newValue: unknown,
  ) {
    this.description = `Set ${propertyPath} on ${target}`;
  }

  execute(world: World): void {
    this.setValue(world, this.newValue);
  }

  undo(world: World): void {
    this.setValue(world, this.oldValue);
  }

  private setValue(world: World, value: unknown): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (world[this.target] as any)[this.propertyPath] = value;
  }
}
