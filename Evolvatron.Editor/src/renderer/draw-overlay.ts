import type { World, Module, SelectableId } from '../model/types';
import type { Camera } from '../editor/camera';
import type { Selection } from '../editor/selection';
import type { HoverState } from '../editor/hover-state';
import { getInfluenceHandleScreenPositions } from '../editor/hit-test';
import { findModule } from '../model/world';
import { COLORS } from './colors';

export function drawSelectionHighlights(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
  selection: Selection,
): void {
  for (const id of selection.ids) {
    drawSelectionForId(ctx, camera, world, id);
  }
}

function drawSelectionForId(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
  id: SelectableId,
): void {
  ctx.strokeStyle = COLORS.selection;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);

  if (id === 'landingPad') {
    const pad = world.landingPad;
    const s = camera.worldToScreen(pad.position.x, pad.position.y);
    const w = camera.worldToScreenScale(pad.halfWidth * 2) + 6;
    const h = camera.worldToScreenScale(pad.halfHeight * 2) + 6;
    ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
    drawHandles(ctx, s.x, s.y, w, h);
  } else if (id === 'spawnArea') {
    const spawn = world.spawnArea;
    const s = camera.worldToScreen(spawn.position.x, spawn.position.y);
    const w = camera.worldToScreenScale(spawn.xRange) + 6;
    const h = camera.worldToScreenScale(spawn.heightRange) + 6;
    ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
    drawHandles(ctx, s.x, s.y, w, h);
  } else {
    const mod = findModule(world, id);
    if (!mod) return;
    drawModuleSelection(ctx, camera, mod);
  }

  ctx.setLineDash([]);

  // Draw influence handles if applicable
  const infHandles = getInfluenceHandleScreenPositions(world, camera, id);
  if (infHandles) {
    drawInfluenceHandles(ctx, infHandles);
  }
}

function drawModuleSelection(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: Module,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);

  if (mod.kind === 'checkpoint') {
    const r = camera.worldToScreenScale(mod.radius) + 4;
    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
    ctx.stroke();
    drawCircleHandles(ctx, s.x, s.y, r);
  } else if (mod.kind === 'obstacle') {
    const w = camera.worldToScreenScale(mod.halfExtentX * 2) + 6;
    const h = camera.worldToScreenScale(mod.halfExtentY * 2) + 6;
    ctx.save();
    ctx.translate(s.x, s.y);
    ctx.rotate((-mod.rotation * Math.PI) / 180);
    ctx.strokeRect(-w / 2, -h / 2, w, h);
    drawHandles(ctx, 0, 0, w, h);
    ctx.restore();
  } else {
    const w = camera.worldToScreenScale(mod.halfExtentX * 2) + 6;
    const h = camera.worldToScreenScale(mod.halfExtentY * 2) + 6;
    ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
    drawHandles(ctx, s.x, s.y, w, h);
  }
}

function drawHandles(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  w: number, h: number,
): void {
  const size = 5;
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.handle;
  const points = [
    // Corners
    [cx - w / 2, cy - h / 2],
    [cx + w / 2, cy - h / 2],
    [cx - w / 2, cy + h / 2],
    [cx + w / 2, cy + h / 2],
    // Cardinals
    [cx,         cy - h / 2],
    [cx,         cy + h / 2],
    [cx - w / 2, cy        ],
    [cx + w / 2, cy        ],
  ];
  for (const [x, y] of points) {
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
  }
  ctx.setLineDash([4, 4]);
}

function drawCircleHandles(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  r: number,
): void {
  const size = 5;
  ctx.setLineDash([]);
  ctx.fillStyle = COLORS.handle;
  // 8 handles evenly spaced at 0, 45, 90, ... 315 degrees
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 4;
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
  }
  ctx.setLineDash([4, 4]);
}

function drawInfluenceHandles(
  ctx: CanvasRenderingContext2D,
  handles: import('../editor/hit-test').HandlePos[],
): void {
  const r = 4;
  ctx.strokeStyle = COLORS.influenceHandle;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  for (const h of handles) {
    ctx.beginPath();
    ctx.arc(h.sx, h.sy, r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.setLineDash([4, 4]);
}

export function drawBoxSelection(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number,
  x1: number, y1: number,
): void {
  const x = Math.min(x0, x1);
  const y = Math.min(y0, y1);
  const w = Math.abs(x1 - x0);
  const h = Math.abs(y1 - y0);

  ctx.fillStyle = COLORS.boxSelect;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = COLORS.boxSelectBorder;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
}

export function drawGhostModule(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  kind: string,
  wx: number, wy: number,
  anchorWx?: number, anchorWy?: number,
): void {
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 2;

  if (anchorWx !== undefined && anchorWy !== undefined) {
    // Two-point preview: show actual shape between anchor and cursor
    if (kind === 'checkpoint') {
      const dx = wx - anchorWx;
      const dy = wy - anchorWy;
      const r = camera.worldToScreenScale(Math.sqrt(dx * dx + dy * dy) / 2);
      const center = camera.worldToScreen((anchorWx + wx) / 2, (anchorWy + wy) / 2);
      ctx.beginPath();
      ctx.arc(center.x, center.y, Math.max(1, r), 0, Math.PI * 2);
      ctx.strokeStyle = COLORS.checkpoint;
      ctx.stroke();
    } else {
      const s0 = camera.worldToScreen(anchorWx, anchorWy);
      const s1 = camera.worldToScreen(wx, wy);
      const x = Math.min(s0.x, s1.x);
      const y = Math.min(s0.y, s1.y);
      const w = Math.abs(s1.x - s0.x);
      const h = Math.abs(s1.y - s0.y);
      const color = kind === 'obstacle' ? COLORS.obstacle
        : kind === 'speedZone' ? COLORS.speedZone
        : kind === 'attractor' ? COLORS.attractor
        : COLORS.dangerZone;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);
    }
  } else {
    // Single-point preview: show crosshair at cursor position
    const s = camera.worldToScreen(wx, wy);
    ctx.strokeStyle = COLORS.selection;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(s.x - 10, s.y);
    ctx.lineTo(s.x + 10, s.y);
    ctx.moveTo(s.x, s.y - 10);
    ctx.lineTo(s.x, s.y + 10);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.globalAlpha = 1.0;
}

// --- Hover highlights ---

export function drawHoverHighlights(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
  hoverState: HoverState,
): void {
  for (const [id, intensity] of hoverState.entries()) {
    if (intensity <= 0) continue;
    drawHoverForId(ctx, camera, world, id, intensity);
  }
}

function drawHoverForId(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  world: World,
  id: SelectableId,
  intensity: number,
): void {
  if (id === 'landingPad') {
    const pad = world.landingPad;
    const s = camera.worldToScreen(pad.position.x, pad.position.y);
    const w = camera.worldToScreenScale(pad.halfWidth * 2);
    const h = camera.worldToScreenScale(pad.halfHeight * 2);
    drawRectHover(ctx, s.x, s.y, w, h, 0, COLORS.landingPad, intensity);
    return;
  }
  if (id === 'spawnArea') {
    const spawn = world.spawnArea;
    const s = camera.worldToScreen(spawn.position.x, spawn.position.y);
    const w = camera.worldToScreenScale(spawn.xRange);
    const h = camera.worldToScreenScale(spawn.heightRange);
    drawRectHover(ctx, s.x, s.y, w, h, 0, COLORS.spawnArea, intensity);
    return;
  }
  const mod = findModule(world, id);
  if (!mod) return;
  const s = camera.worldToScreen(mod.position.x, mod.position.y);

  switch (mod.kind) {
    case 'obstacle': {
      const w = camera.worldToScreenScale(mod.halfExtentX * 2);
      const h = camera.worldToScreenScale(mod.halfExtentY * 2);
      const color = mod.isLethal ? COLORS.obstaclLethal : COLORS.obstacle;
      drawRectHover(ctx, s.x, s.y, w, h, mod.rotation, color, intensity);
      break;
    }
    case 'checkpoint': {
      const r = camera.worldToScreenScale(mod.radius);
      drawCircleHover(ctx, s.x, s.y, r, COLORS.checkpoint, intensity);
      break;
    }
    case 'speedZone': {
      const w = camera.worldToScreenScale(mod.halfExtentX * 2);
      const h = camera.worldToScreenScale(mod.halfExtentY * 2);
      drawRectHover(ctx, s.x, s.y, w, h, 0, COLORS.speedZone, intensity);
      break;
    }
    case 'dangerZone': {
      const w = camera.worldToScreenScale(mod.halfExtentX * 2);
      const h = camera.worldToScreenScale(mod.halfExtentY * 2);
      const color = mod.isLethal ? COLORS.dangerZoneLethal : COLORS.dangerZone;
      drawRectHover(ctx, s.x, s.y, w, h, 0, color, intensity);
      break;
    }
    case 'attractor': {
      const w = camera.worldToScreenScale(mod.halfExtentX * 2);
      const h = camera.worldToScreenScale(mod.halfExtentY * 2);
      const color = mod.magnitude < 0 ? COLORS.repulsor : COLORS.attractor;
      drawRectHover(ctx, s.x, s.y, w, h, 0, color, intensity);
      break;
    }
  }
}

function drawRectHover(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  w: number, h: number,
  rotationDeg: number,
  color: string,
  intensity: number,
): void {
  ctx.save();
  ctx.translate(cx, cy);
  if (rotationDeg !== 0) {
    ctx.rotate((-rotationDeg * Math.PI) / 180);
  }

  ctx.shadowColor = color;
  ctx.shadowBlur = 14 * intensity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.5 * intensity;
  ctx.strokeRect(-w / 2, -h / 2, w, h);

  ctx.shadowBlur = 0;
  ctx.globalAlpha = 0.08 * intensity;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(-w / 2, -h / 2, w, h);

  ctx.restore();
}

function drawCircleHover(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  r: number,
  color: string,
  intensity: number,
): void {
  ctx.save();

  ctx.shadowColor = color;
  ctx.shadowBlur = 14 * intensity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.5 * intensity;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.globalAlpha = 0.08 * intensity;
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

// --- Paste ghosts ---

export function drawPasteGhosts(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  modules: Module[],
): void {
  ctx.globalAlpha = 0.5;

  for (const mod of modules) {
    const s = camera.worldToScreen(mod.position.x, mod.position.y);

    switch (mod.kind) {
      case 'obstacle': {
        const w = camera.worldToScreenScale(mod.halfExtentX * 2);
        const h = camera.worldToScreenScale(mod.halfExtentY * 2);
        ctx.save();
        ctx.translate(s.x, s.y);
        ctx.rotate((-mod.rotation * Math.PI) / 180);
        ctx.strokeStyle = mod.isLethal ? COLORS.obstaclLethal : COLORS.obstacle;
        ctx.lineWidth = 2;
        ctx.strokeRect(-w / 2, -h / 2, w, h);
        ctx.restore();
        break;
      }
      case 'checkpoint': {
        const r = camera.worldToScreenScale(mod.radius);
        ctx.strokeStyle = COLORS.checkpoint;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.stroke();
        break;
      }
      case 'speedZone': {
        const w = camera.worldToScreenScale(mod.halfExtentX * 2);
        const h = camera.worldToScreenScale(mod.halfExtentY * 2);
        ctx.strokeStyle = COLORS.speedZone;
        ctx.lineWidth = 2;
        ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
        break;
      }
      case 'dangerZone': {
        const w = camera.worldToScreenScale(mod.halfExtentX * 2);
        const h = camera.worldToScreenScale(mod.halfExtentY * 2);
        ctx.strokeStyle = mod.isLethal ? COLORS.dangerZoneLethal : COLORS.dangerZone;
        ctx.lineWidth = 2;
        ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
        break;
      }
      case 'attractor': {
        const w = camera.worldToScreenScale(mod.halfExtentX * 2);
        const h = camera.worldToScreenScale(mod.halfExtentY * 2);
        ctx.strokeStyle = mod.magnitude < 0 ? COLORS.repulsor : COLORS.attractor;
        ctx.lineWidth = 2;
        ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
        break;
      }
    }
  }

  ctx.globalAlpha = 1.0;
}
