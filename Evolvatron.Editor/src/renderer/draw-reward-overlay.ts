import type { World } from '../model/types';
import type { Camera } from '../editor/camera';

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
      total += rewardAt(mod.position.x, mod.position.y, mod.halfExtentX, mod.halfExtentY, -mod.penaltyPerStep, mod.influenceFactor ?? 1, wx, wy);
    } else if (mod.kind === 'checkpoint') {
      total += rewardAtCircle(mod.position.x, mod.position.y, mod.radius, mod.rewardBonus, mod.influenceFactor ?? 1, wx, wy);
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

function rewardAtCircle(
  cx: number, cy: number, radius: number,
  magnitude: number, influenceFactor: number,
  wx: number, wy: number,
): number {
  const dx = wx - cx;
  const dy = wy - cy;
  const dist = Math.sqrt(dx * dx + dy * dy);

  const influenceRadius = radius * influenceFactor;

  if (dist > influenceRadius) return 0;
  if (dist <= radius) return magnitude;

  const t = (dist - radius) / (influenceRadius - radius);
  const falloff = 1 - t * t;
  return magnitude * falloff;
}

/**
 * Render the reward landscape as a colored overlay across the entire visible screen.
 * Samples on a low-res grid, then scales up with bilinear filtering.
 * Green = positive reward, Red = negative reward.
 */
export function drawRewardOverlay(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
): void {
  const hasAnySources = world.modules.some(
    m => m.kind === 'attractor' || m.kind === 'dangerZone' || m.kind === 'checkpoint',
  );
  if (!hasAnySources) return;

  const canvasW = camera.canvasWidth;
  const canvasH = camera.canvasHeight;

  // Sample every Nth pixel (adaptive to zoom)
  const step = Math.max(2, Math.min(8, Math.floor(6 / (camera.zoom / 40))));
  const gridW = Math.ceil(canvasW / step);
  const gridH = Math.ceil(canvasH / step);

  const imageData = ctx.createImageData(gridW, gridH);
  const data = imageData.data;

  // Find max magnitude for normalization (single scale for both positive and negative)
  let maxMag = 0;
  for (const mod of world.modules) {
    if (mod.kind === 'attractor') maxMag = Math.max(maxMag, Math.abs(mod.magnitude));
    else if (mod.kind === 'dangerZone') maxMag = Math.max(maxMag, mod.penaltyPerStep);
    else if (mod.kind === 'checkpoint') maxMag = Math.max(maxMag, mod.rewardBonus);
  }
  if (maxMag === 0) return;

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const screenX = gx * step;
      const screenY = gy * step;
      const w = camera.screenToWorld(screenX, screenY);
      const reward = sampleReward(world, w.x, w.y);

      const idx = (gy * gridW + gx) * 4;
      if (Math.abs(reward) < 0.001) {
        data[idx + 3] = 0;
        continue;
      }

      const normalized = Math.min(1, Math.abs(reward) / maxMag);
      const alpha = Math.floor(30 + normalized * 90);

      if (reward > 0) {
        data[idx] = 0;
        data[idx + 1] = 220;
        data[idx + 2] = 100;
      } else {
        data[idx] = 255;
        data[idx + 1] = 50;
        data[idx + 2] = 50;
      }
      data[idx + 3] = alpha;
    }
  }

  const offscreen = new OffscreenCanvas(gridW, gridH);
  const offCtx = offscreen.getContext('2d')!;
  offCtx.putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);
  ctx.imageSmoothingEnabled = false;
}
