import type { World } from '../model/types';

const FORMAT_VERSION = 1;

export interface CameraState {
  centerX: number;
  centerY: number;
  zoom: number;
}

interface SerializedWorld {
  version: number;
  world: World;
  camera?: CameraState;
}

export function serialize(world: World, camera?: CameraState): string {
  const data: SerializedWorld = {
    version: FORMAT_VERSION,
    world,
    camera,
  };
  return JSON.stringify(data, null, 2);
}

export interface DeserializeResult {
  world: World;
  camera?: CameraState;
}

export function deserialize(json: string): DeserializeResult {
  const data = JSON.parse(json) as SerializedWorld;
  if (!data.version || data.version > FORMAT_VERSION) {
    throw new Error(`Unsupported format version: ${data.version}`);
  }
  const w = data.world;
  if (!w.landingPad || !w.spawnArea || !Array.isArray(w.modules)) {
    throw new Error('Invalid world data: missing required fields');
  }
  // Migrate: fill defaults for fields added after initial format
  w.spawnArea.spawnCount ??= 10;
  w.spawnArea.spawnSeed ??= 0;
  w.landingPad.attractionMagnitude ??= 10;
  w.landingPad.attractionRadius ??= 10;
  if (w.simulationConfig) {
    w.simulationConfig.hasteBonus ??= 1.0;
  }
  for (const mod of w.modules) {
    if (mod.kind === 'obstacle') {
      (mod as any).penaltyPerStep ??= 10;
      (mod as any).influenceRadius ??= 3;
    }
  }
  return { world: w, camera: data.camera };
}
