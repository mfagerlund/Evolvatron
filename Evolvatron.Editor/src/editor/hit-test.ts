import type { World, SelectableId } from '../model/types';
import type { Camera } from './camera';

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

/** Which handle is being hit. Cardinals resize one axis, corners resize both. */
export type HandleDirection =
  | 'n' | 's' | 'e' | 'w'
  | 'ne' | 'nw' | 'se' | 'sw';

export type HandleKind = 'size' | 'influence';

export interface HandleHitResult {
  id: SelectableId;
  direction: HandleDirection;
  kind: HandleKind;
}

const HANDLE_RADIUS_PX = 7;

/**
 * Test if screen coords (sx, sy) hit a resize handle on any selected object.
 * Returns the handle info or null.
 */
export function hitTestHandle(
  world: World,
  camera: Camera,
  selectedIds: ReadonlySet<SelectableId>,
  sx: number,
  sy: number,
): HandleHitResult | null {
  // Test influence handles first (they're further out, so test them before size handles)
  for (const id of selectedIds) {
    const infHandles = getInfluenceHandleScreenPositions(world, camera, id);
    if (!infHandles) continue;
    for (const h of infHandles) {
      const dx = sx - h.sx;
      const dy = sy - h.sy;
      if (dx * dx + dy * dy <= HANDLE_RADIUS_PX * HANDLE_RADIUS_PX) {
        return { id, direction: h.dir, kind: 'influence' };
      }
    }
  }
  for (const id of selectedIds) {
    const handles = getHandleScreenPositions(world, camera, id);
    if (!handles) continue;
    for (const h of handles) {
      const dx = sx - h.sx;
      const dy = sy - h.sy;
      if (dx * dx + dy * dy <= HANDLE_RADIUS_PX * HANDLE_RADIUS_PX) {
        return { id, direction: h.dir, kind: 'size' };
      }
    }
  }
  return null;
}

export interface HandlePos {
  sx: number;
  sy: number;
  dir: HandleDirection;
}

const DIRS: { dir: HandleDirection; dx: number; dy: number }[] = [
  { dir: 'n',  dx:  0, dy: -1 },
  { dir: 's',  dx:  0, dy:  1 },
  { dir: 'w',  dx: -1, dy:  0 },
  { dir: 'e',  dx:  1, dy:  0 },
  { dir: 'nw', dx: -1, dy: -1 },
  { dir: 'ne', dx:  1, dy: -1 },
  { dir: 'sw', dx: -1, dy:  1 },
  { dir: 'se', dx:  1, dy:  1 },
];

const CARDINAL_DIRS = DIRS.filter(d => d.dir === 'n' || d.dir === 's' || d.dir === 'e' || d.dir === 'w');

export function getHandleScreenPositions(
  world: World,
  camera: Camera,
  id: SelectableId,
): HandlePos[] | null {
  if (id === 'landingPad') {
    const pad = world.landingPad;
    const s = camera.worldToScreen(pad.position.x, pad.position.y);
    const hw = (camera.worldToScreenScale(pad.halfWidth * 2) + 6) / 2;
    const hh = (camera.worldToScreenScale(pad.halfHeight * 2) + 6) / 2;
    return DIRS.map(d => ({ sx: s.x + d.dx * hw, sy: s.y + d.dy * hh, dir: d.dir }));
  }

  if (id === 'spawnArea') {
    const spawn = world.spawnArea;
    const s = camera.worldToScreen(spawn.position.x, spawn.position.y);
    const hw = (camera.worldToScreenScale(spawn.xRange) + 6) / 2;
    const hh = (camera.worldToScreenScale(spawn.heightRange) + 6) / 2;
    return DIRS.map(d => ({ sx: s.x + d.dx * hw, sy: s.y + d.dy * hh, dir: d.dir }));
  }

  const mod = world.modules.find(m => m.id === id);
  if (!mod) return null;

  const s = camera.worldToScreen(mod.position.x, mod.position.y);

  if (mod.kind === 'checkpoint') {
    const r = camera.worldToScreenScale(mod.radius) + 4;
    return DIRS.map(d => {
      const angle = Math.atan2(d.dy, d.dx);
      return { sx: s.x + Math.cos(angle) * r, sy: s.y + Math.sin(angle) * r, dir: d.dir };
    });
  }

  if (mod.kind === 'obstacle') {
    const hw = (camera.worldToScreenScale(mod.halfExtentX * 2) + 6) / 2;
    const hh = (camera.worldToScreenScale(mod.halfExtentY * 2) + 6) / 2;
    const rad = (-mod.rotation * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    return DIRS.map(d => {
      const lx = d.dx * hw;
      const ly = d.dy * hh;
      return {
        sx: s.x + lx * cos - ly * sin,
        sy: s.y + lx * sin + ly * cos,
        dir: d.dir,
      };
    });
  }

  // speedZone, dangerZone
  const hw = (camera.worldToScreenScale(mod.halfExtentX * 2) + 6) / 2;
  const hh = (camera.worldToScreenScale(mod.halfExtentY * 2) + 6) / 2;
  return DIRS.map(d => ({ sx: s.x + d.dx * hw, sy: s.y + d.dy * hh, dir: d.dir }));
}

/**
 * Get screen positions for influence-factor handles.
 * Only returns handles for modules with influenceRadius > 0.
 * Uses 4 cardinal handles on the influence region boundary.
 */
export function getInfluenceHandleScreenPositions(
  world: World,
  camera: Camera,
  id: SelectableId,
): HandlePos[] | null {
  if (id === 'landingPad' || id === 'spawnArea') return null;

  const mod = world.modules.find(m => m.id === id);
  if (!mod) return null;

  const s = camera.worldToScreen(mod.position.x, mod.position.y);

  if (mod.kind === 'checkpoint') {
    if (mod.influenceRadius <= 0) return null;
    const ir = camera.worldToScreenScale(mod.radius + mod.influenceRadius);
    return CARDINAL_DIRS.map(d => {
      const angle = Math.atan2(d.dy, d.dx);
      return { sx: s.x + Math.cos(angle) * ir, sy: s.y + Math.sin(angle) * ir, dir: d.dir };
    });
  }

  if (mod.kind === 'attractor') {
    if (mod.influenceRadius <= 0) return null;
    const ihw = camera.worldToScreenScale(mod.halfExtentX + mod.influenceRadius);
    const ihh = camera.worldToScreenScale(mod.halfExtentY + mod.influenceRadius);
    return CARDINAL_DIRS.map(d => ({ sx: s.x + d.dx * ihw, sy: s.y + d.dy * ihh, dir: d.dir }));
  }

  if (mod.kind === 'dangerZone') {
    if (mod.influenceRadius <= 0) return null;
    const ihw = camera.worldToScreenScale(mod.halfExtentX + mod.influenceRadius);
    const ihh = camera.worldToScreenScale(mod.halfExtentY + mod.influenceRadius);
    return CARDINAL_DIRS.map(d => ({ sx: s.x + d.dx * ihw, sy: s.y + d.dy * ihh, dir: d.dir }));
  }

  return null;
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
