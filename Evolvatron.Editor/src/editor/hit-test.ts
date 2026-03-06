import type { World } from '../model/types';

export function pointInRect(
  px: number, py: number,
  cx: number, cy: number,
  hx: number, hy: number,
  rotationDeg: number,
): boolean {
  const rad = (rotationDeg * Math.PI) / 180;
  const cos = Math.cos(-rad);
  const sin = Math.sin(-rad);
  const dx = px - cx;
  const dy = py - cy;
  const lx = dx * cos - dy * sin;
  const ly = dx * sin + dy * cos;
  return Math.abs(lx) <= hx && Math.abs(ly) <= hy;
}

export function pointInCircle(
  px: number, py: number,
  cx: number, cy: number,
  radius: number,
): boolean {
  const dx = px - cx;
  const dy = py - cy;
  return dx * dx + dy * dy <= radius * radius;
}

export function rectIntersectsRect(
  ax: number, ay: number, ahx: number, ahy: number,
  bx: number, by: number, bhx: number, bhy: number,
): boolean {
  return (
    Math.abs(ax - bx) <= ahx + bhx &&
    Math.abs(ay - by) <= ahy + bhy
  );
}

export function circleIntersectsRect(
  cx: number, cy: number, radius: number,
  rx: number, ry: number, rhx: number, rhy: number,
): boolean {
  const closestX = Math.max(rx - rhx, Math.min(cx, rx + rhx));
  const closestY = Math.max(ry - rhy, Math.min(cy, ry + rhy));
  const dx = cx - closestX;
  const dy = cy - closestY;
  return dx * dx + dy * dy <= radius * radius;
}

export interface HitResult {
  id: string | 'landingPad' | 'spawnArea';
  distance: number;
}

export function hitTestPoint(world: World, wx: number, wy: number): HitResult | null {
  let best: HitResult | null = null;

  for (const mod of world.modules) {
    let hit = false;
    if (mod.kind === 'checkpoint') {
      hit = pointInCircle(wx, wy, mod.position.x, mod.position.y, mod.radius);
    } else if (mod.kind === 'obstacle') {
      hit = pointInRect(
        wx, wy,
        mod.position.x, mod.position.y,
        mod.halfExtentX, mod.halfExtentY,
        mod.rotation,
      );
    } else {
      hit = pointInRect(
        wx, wy,
        mod.position.x, mod.position.y,
        mod.halfExtentX, mod.halfExtentY,
        0,
      );
    }
    if (hit) {
      const dx = wx - mod.position.x;
      const dy = wy - mod.position.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (!best || dist < best.distance) {
        best = { id: mod.id, distance: dist };
      }
    }
  }

  const pad = world.landingPad;
  if (pointInRect(wx, wy, pad.position.x, pad.position.y, pad.halfWidth, pad.halfHeight, 0)) {
    const dx = wx - pad.position.x;
    const dy = wy - pad.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (!best || dist < best.distance) {
      best = { id: 'landingPad', distance: dist };
    }
  }

  const spawn = world.spawnArea;
  const spawnHx = spawn.xRange / 2;
  const spawnHy = spawn.heightRange / 2;
  if (pointInRect(wx, wy, spawn.position.x, spawn.position.y, spawnHx, spawnHy, 0)) {
    const dx = wx - spawn.position.x;
    const dy = wy - spawn.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (!best || dist < best.distance) {
      best = { id: 'spawnArea', distance: dist };
    }
  }

  return best;
}

export function boxSelectModules(
  world: World,
  minX: number, minY: number,
  maxX: number, maxY: number,
): HitResult[] {
  const results: HitResult[] = [];
  const bx = (minX + maxX) / 2;
  const by = (minY + maxY) / 2;
  const bhx = (maxX - minX) / 2;
  const bhy = (maxY - minY) / 2;

  for (const mod of world.modules) {
    let hit = false;
    if (mod.kind === 'checkpoint') {
      hit = circleIntersectsRect(
        mod.position.x, mod.position.y, mod.radius,
        bx, by, bhx, bhy,
      );
    } else if (mod.kind === 'obstacle') {
      // Approximate: use AABB of rotated rect
      const rad = Math.abs(mod.rotation * Math.PI / 180);
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      const aabbHx = mod.halfExtentX * cos + mod.halfExtentY * sin;
      const aabbHy = mod.halfExtentX * sin + mod.halfExtentY * cos;
      hit = rectIntersectsRect(mod.position.x, mod.position.y, aabbHx, aabbHy, bx, by, bhx, bhy);
    } else {
      hit = rectIntersectsRect(
        mod.position.x, mod.position.y, mod.halfExtentX, mod.halfExtentY,
        bx, by, bhx, bhy,
      );
    }
    if (hit) {
      results.push({ id: mod.id, distance: 0 });
    }
  }

  const pad = world.landingPad;
  if (rectIntersectsRect(pad.position.x, pad.position.y, pad.halfWidth, pad.halfHeight, bx, by, bhx, bhy)) {
    results.push({ id: 'landingPad', distance: 0 });
  }

  const spawn = world.spawnArea;
  if (rectIntersectsRect(spawn.position.x, spawn.position.y, spawn.xRange / 2, spawn.heightRange / 2, bx, by, bhx, bhy)) {
    results.push({ id: 'spawnArea', distance: 0 });
  }

  return results;
}
