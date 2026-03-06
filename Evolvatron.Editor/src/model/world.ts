import type { World, Module, SelectableId } from './types';

export interface SelectableObject {
  id: SelectableId;
  position: { x: number; y: number };
}

export function findModule(world: World, id: string): Module | undefined {
  return world.modules.find(m => m.id === id);
}

export function findModuleIndex(world: World, id: string): number {
  return world.modules.findIndex(m => m.id === id);
}

export function allSelectableObjects(world: World): SelectableObject[] {
  const objects: SelectableObject[] = [
    { id: 'landingPad', position: world.landingPad.position },
    { id: 'spawnArea', position: world.spawnArea.position },
  ];
  for (const m of world.modules) {
    objects.push({ id: m.id, position: m.position });
  }
  return objects;
}

export function getSelectablePosition(world: World, id: SelectableId): { x: number; y: number } | undefined {
  if (id === 'landingPad') return world.landingPad.position;
  if (id === 'spawnArea') return world.spawnArea.position;
  const mod = findModule(world, id);
  return mod?.position;
}

export function removeModules(world: World, ids: Set<string>): Module[] {
  const removed: Module[] = [];
  world.modules = world.modules.filter(m => {
    if (ids.has(m.id)) {
      removed.push(m);
      return false;
    }
    return true;
  });
  return removed;
}

export function validate(world: World): string[] {
  const errors: string[] = [];
  if (world.groundY >= world.landingPad.position.y) {
    errors.push('Landing pad must be above ground');
  }
  if (world.spawnArea.position.y <= world.landingPad.position.y) {
    errors.push('Spawn area should be above landing pad');
  }
  const checkpoints = world.modules.filter(m => m.kind === 'checkpoint');
  const orders = checkpoints.map(c => c.order);
  const uniqueOrders = new Set(orders);
  if (uniqueOrders.size !== orders.length) {
    errors.push('Checkpoint orders must be unique');
  }
  return errors;
}
