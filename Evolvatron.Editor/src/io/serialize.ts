import type { World } from '../model/types';

const FORMAT_VERSION = 1;

interface SerializedWorld {
  version: number;
  world: World;
}

export function serialize(world: World): string {
  const data: SerializedWorld = {
    version: FORMAT_VERSION,
    world,
  };
  return JSON.stringify(data, null, 2);
}

export function deserialize(json: string): World {
  const data = JSON.parse(json) as SerializedWorld;
  if (!data.version || data.version > FORMAT_VERSION) {
    throw new Error(`Unsupported format version: ${data.version}`);
  }
  const w = data.world;
  if (!w.landingPad || !w.spawnArea || !Array.isArray(w.modules)) {
    throw new Error('Invalid world data: missing required fields');
  }
  return w;
}
