import type { World } from '../model/types';
import type { Camera } from '../editor/camera';

interface RewardSource {
  cx: number; cy: number;
  hx: number; hy: number;
  magnitude: number;
  influenceFactor: number;
}

/**
 * Compute the reward value at a world-space point from all reward sources.
 * Returns positive for attractors, negative for repulsors/danger zones.
 */
export function sampleReward(world: World, wx: number, wy: number): number {
  let total = 0;
  for (const mod of world.modules) {
    if (mod.kind === 'attractor') {
      total += rewardAt(mod.position.x, mod.position.y, mod.halfExtentX, mod.halfExtentY, mod.magnitude, mod.influenceFactor, wx, wy);
    } else if (mod.kind === 'dangerZone') {
      total += rewardAt(mod.position.x, mod.position.y, mod.halfExtentX, mod.halfExtentY, -mod.penaltyPerStep, mod.influenceFactor, wx, wy);
    }
  }
  return total;
}

function rewardAt(
  cx: number, cy: number, hx: number, hy: number,
  magnitude: number, influenceFactor: number,
  wx: number, wy: number,
): number {
  const dx = Math.abs(wx - cx);
  const dy = Math.abs(wy - cy);

  const ihx = hx * influenceFactor;
  const ihy = hy * influenceFactor;

  // Outside influence region entirely
  if (dx > ihx || dy > ihy) return 0;

  // Inside core zone
  if (dx <= hx && dy <= hy) return magnitude;

  // Normalized distance: 0 at core edge, 1 at influence edge
  const fx = (ihx > hx) ? Math.max(0, (dx - hx) / (ihx - hx)) : 0;
  const fy = (ihy > hy) ? Math.max(0, (dy - hy) / (ihy - hy)) : 0;
  const t = Math.max(fx, fy);

  // Smooth falloff from core to influence edge
  const falloff = 1 - t * t; // quadratic falloff
  return magnitude * falloff;
}

function collectRewardSources(world: World): RewardSource[] {
  const sources: RewardSource[] = [];
  for (const mod of world.modules) {
    if (mod.kind === 'attractor') {
      sources.push({ cx: mod.position.x, cy: mod.position.y, hx: mod.halfExtentX, hy: mod.halfExtentY, magnitude: mod.magnitude, influenceFactor: mod.influenceFactor });
    } else if (mod.kind === 'dangerZone') {
      sources.push({ cx: mod.position.x, cy: mod.position.y, hx: mod.halfExtentX, hy: mod.halfExtentY, magnitude: -mod.penaltyPerStep, influenceFactor: mod.influenceFactor });
    }
  }
  return sources;
}

/**
 * Render the reward landscape as a colored overlay.
 * Green = positive reward, Red = negative reward.
 */
export function drawRewardOverlay(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
): void {
  const sources = collectRewardSources(world);
  if (sources.length === 0) return;

  // Determine the bounding box of all influence regions in world space
  let minWx = Infinity, maxWx = -Infinity;
  let minWy = Infinity, maxWy = -Infinity;
  for (const src of sources) {
    const ihx = src.hx * src.influenceFactor;
    const ihy = src.hy * src.influenceFactor;
    minWx = Math.min(minWx, src.cx - ihx);
    maxWx = Math.max(maxWx, src.cx + ihx);
    minWy = Math.min(minWy, src.cy - ihy);
    maxWy = Math.max(maxWy, src.cy + ihy);
  }

  // Convert to screen space and clip to canvas
  const topLeft = camera.worldToScreen(minWx, maxWy); // maxWy because Y is inverted
  const bottomRight = camera.worldToScreen(maxWx, minWy);

  const sx0 = Math.max(0, Math.floor(topLeft.x));
  const sy0 = Math.max(0, Math.floor(topLeft.y));
  const sx1 = Math.min(camera.canvasWidth, Math.ceil(bottomRight.x));
  const sy1 = Math.min(camera.canvasHeight, Math.ceil(bottomRight.y));

  if (sx1 <= sx0 || sy1 <= sy0) return;

  const width = sx1 - sx0;
  const height = sy1 - sy0;

  // Sample at reduced resolution for performance (every Nth pixel)
  const step = Math.max(1, Math.floor(4 / (camera.zoom / 40))); // adaptive step
  const gridW = Math.ceil(width / step);
  const gridH = Math.ceil(height / step);

  const imageData = ctx.createImageData(gridW, gridH);
  const data = imageData.data;

  // Find max magnitude for normalization
  let maxMag = 0;
  for (const src of sources) {
    maxMag = Math.max(maxMag, Math.abs(src.magnitude));
  }
  if (maxMag === 0) return;

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const screenX = sx0 + gx * step;
      const screenY = sy0 + gy * step;
      const w = camera.screenToWorld(screenX, screenY);
      const reward = sampleReward(world, w.x, w.y);

      const idx = (gy * gridW + gx) * 4;
      if (Math.abs(reward) < 0.001) {
        data[idx + 3] = 0; // transparent
        continue;
      }

      const normalized = Math.min(1, Math.abs(reward) / maxMag);
      const alpha = Math.floor(normalized * 100); // max alpha ~100/255

      if (reward > 0) {
        // Green for positive reward
        data[idx] = 0;
        data[idx + 1] = 220;
        data[idx + 2] = 100;
      } else {
        // Red for negative reward
        data[idx] = 255;
        data[idx + 1] = 50;
        data[idx + 2] = 50;
      }
      data[idx + 3] = alpha;
    }
  }

  // Draw the low-res image scaled up
  const offscreen = new OffscreenCanvas(gridW, gridH);
  const offCtx = offscreen.getContext('2d')!;
  offCtx.putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offscreen, sx0, sy0, width, height);
  ctx.imageSmoothingEnabled = false;
}
