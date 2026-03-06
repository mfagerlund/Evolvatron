import type { World, Vec2, SelectableId } from '../model/types';
import type { Command } from './command';
import { findModule } from '../model/world';

export class MoveModulesCommand implements Command {
  description: string;

  constructor(private deltas: Map<SelectableId, Vec2>) {
    this.description = deltas.size === 1
      ? `Move "${[...deltas.keys()][0]}"`
      : `Move ${deltas.size} objects`;
  }

  execute(world: World): void {
    this.applyDeltas(world, 1);
  }

  undo(world: World): void {
    this.applyDeltas(world, -1);
  }

  private applyDeltas(world: World, sign: number): void {
    for (const [id, delta] of this.deltas) {
      const pos = this.getPosition(world, id);
      if (pos) {
        pos.x += sign * delta.x;
        pos.y += sign * delta.y;
      }
    }
  }

  private getPosition(world: World, id: SelectableId): Vec2 | undefined {
    if (id === 'landingPad') return world.landingPad.position;
    if (id === 'spawnArea') return world.spawnArea.position;
    return findModule(world, id)?.position;
  }
}
