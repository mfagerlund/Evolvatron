import type { World } from '../model/types';
import type { Camera } from '../editor/camera';
import { COLORS } from './colors';

export function drawGround(ctx: CanvasRenderingContext2D, camera: Camera, groundY: number): void {
  const left = camera.screenToWorld(0, 0);
  const right = camera.screenToWorld(camera.canvasWidth, 0);

  const groundScreen = camera.worldToScreen(0, groundY);
  ctx.fillStyle = COLORS.ground;
  ctx.fillRect(0, groundScreen.y, camera.canvasWidth, camera.canvasHeight - groundScreen.y);

  ctx.strokeStyle = COLORS.groundLine;
  ctx.lineWidth = 2;
  ctx.beginPath();
  const gl = camera.worldToScreen(left.x - 10, groundY);
  const gr = camera.worldToScreen(right.x + 10, groundY);
  ctx.moveTo(gl.x, gl.y);
  ctx.lineTo(gr.x, gr.y);
  ctx.stroke();
}

export function drawGrid(ctx: CanvasRenderingContext2D, camera: Camera): void {
  const topLeft = camera.screenToWorld(0, 0);
  const bottomRight = camera.screenToWorld(camera.canvasWidth, camera.canvasHeight);

  const minX = Math.floor(topLeft.x);
  const maxX = Math.ceil(bottomRight.x);
  const minY = Math.floor(bottomRight.y);
  const maxY = Math.ceil(topLeft.y);

  ctx.lineWidth = 0.5;

  for (let x = minX; x <= maxX; x++) {
    ctx.strokeStyle = x % 5 === 0 ? COLORS.gridMajor : COLORS.grid;
    const s = camera.worldToScreen(x, 0);
    ctx.beginPath();
    ctx.moveTo(s.x, 0);
    ctx.lineTo(s.x, camera.canvasHeight);
    ctx.stroke();
  }

  for (let y = minY; y <= maxY; y++) {
    ctx.strokeStyle = y % 5 === 0 ? COLORS.gridMajor : COLORS.grid;
    const s = camera.worldToScreen(0, y);
    ctx.beginPath();
    ctx.moveTo(0, s.y);
    ctx.lineTo(camera.canvasWidth, s.y);
    ctx.stroke();
  }
}

export function drawLandingPad(ctx: CanvasRenderingContext2D, camera: Camera, world: World): void {
  const pad = world.landingPad;
  const s = camera.worldToScreen(pad.position.x, pad.position.y);
  const w = camera.worldToScreenScale(pad.halfWidth * 2);
  const h = camera.worldToScreenScale(pad.halfHeight * 2);

  // Draw attraction influence gradient — UPWARD ONLY, increasing toward pad
  if (pad.attractionMagnitude > 0 && pad.attractionRadius > 0) {
    const inflScreen = camera.worldToScreenScale(pad.attractionRadius);
    const coreHW = camera.worldToScreenScale(pad.halfWidth);
    const padTopScreen = s.y - h / 2;
    const padBottomScreen = s.y + h / 2;
    drawUpwardInfluenceGradient(ctx, s.x, padTopScreen, padBottomScreen, coreHW, inflScreen, '0, 200, 100');

    // Dashed outline at outermost boundary (upward-only shape)
    const hw = coreHW + inflScreen;
    const topY = padTopScreen - inflScreen;
    ctx.strokeStyle = 'rgba(0, 200, 100, 0.4)';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    topRoundedRect(ctx, s.x - hw, topY, hw * 2, padBottomScreen - topY, inflScreen);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = COLORS.landingPadFill;
  ctx.fillRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.strokeStyle = COLORS.landingPad;
  ctx.lineWidth = 2;
  ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);

  ctx.fillStyle = COLORS.landingPad;
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('LANDING PAD', s.x, s.y + 4);
}

export function drawSpawnArea(ctx: CanvasRenderingContext2D, camera: Camera, world: World): void {
  const spawn = world.spawnArea;
  const s = camera.worldToScreen(spawn.position.x, spawn.position.y);
  const w = camera.worldToScreenScale(spawn.xRange);
  const h = camera.worldToScreenScale(spawn.heightRange);

  ctx.fillStyle = COLORS.spawnAreaFill;
  ctx.fillRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.strokeStyle = COLORS.spawnArea;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.setLineDash([]);

  drawSpawnDots(ctx, camera, spawn);

  ctx.fillStyle = COLORS.spawnArea;
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('SPAWN', s.x, s.y + 4);
}

/** Mulberry32 seeded PRNG — returns values in [0, 1). */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function drawSpawnDots(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  spawn: import('../model/types').SpawnArea,
): void {
  const count = spawn.spawnCount || 10;
  const seed = spawn.spawnSeed || 0;
  ctx.fillStyle = COLORS.spawnDot;

  // Spawn area rect is centered on position, half-extents = xRange/2 and heightRange/2
  const hx = spawn.xRange / 2;
  const hy = spawn.heightRange / 2;

  for (let i = 0; i < count; i++) {
    const rng = mulberry32(seed * 1000 + i);
    const spawnX = spawn.position.x + (rng() * 2 - 1) * hx;
    const spawnY = spawn.position.y + (rng() * 2 - 1) * hy;

    const dot = camera.worldToScreen(spawnX, spawnY);
    ctx.beginPath();
    ctx.arc(dot.x, dot.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawModules(ctx: CanvasRenderingContext2D, camera: Camera, world: World): void {
  for (const mod of world.modules) {
    switch (mod.kind) {
      case 'obstacle':
        drawObstacle(ctx, camera, mod);
        break;
      case 'checkpoint':
        drawCheckpoint(ctx, camera, mod);
        break;
      case 'speedZone':
        drawSpeedZone(ctx, camera, mod);
        break;
      case 'dangerZone':
        drawDangerZone(ctx, camera, mod);
        break;
      case 'attractor':
        drawAttractor(ctx, camera, mod);
        break;
    }
  }
}

function drawObstacle(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: import('../model/types').ObstacleModule,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);
  const w = camera.worldToScreenScale(mod.halfExtentX * 2);
  const h = camera.worldToScreenScale(mod.halfExtentY * 2);

  const stroke = mod.isLethal ? COLORS.obstaclLethal : COLORS.obstacle;

  // Draw influence gradient (axis-aligned, increasing toward obstacle)
  if (mod.influenceRadius > 0) {
    const inflScreen = camera.worldToScreenScale(mod.influenceRadius);
    const coreHW = camera.worldToScreenScale(mod.halfExtentX);
    const coreHH = camera.worldToScreenScale(mod.halfExtentY);
    drawInfluenceGradient(ctx, s.x, s.y, coreHW, coreHH, inflScreen, '255, 51, 102');

    // Dashed outline at outermost boundary
    const iw = (coreHW + inflScreen) * 2;
    const ih = (coreHH + inflScreen) * 2;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    roundedRect(ctx, s.x - iw / 2, s.y - ih / 2, iw, ih, inflScreen);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.save();
  ctx.translate(s.x, s.y);
  ctx.rotate((-mod.rotation * Math.PI) / 180);

  const fill = mod.isLethal ? 'rgba(255, 34, 0, 0.5)' : COLORS.obstacleFill;

  ctx.fillStyle = fill;
  ctx.fillRect(-w / 2, -h / 2, w, h);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.strokeRect(-w / 2, -h / 2, w, h);

  ctx.fillStyle = stroke;
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(mod.isLethal ? 'LETHAL' : 'OBS', 0, 4);

  ctx.restore();
}

function drawCheckpoint(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: import('../model/types').CheckpointModule,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);
  const r = camera.worldToScreenScale(mod.radius);

  // Draw influence gradient (increasing toward checkpoint)
  if (mod.influenceRadius > 0) {
    const coreScreen = camera.worldToScreenScale(mod.radius);
    const inflScreen = camera.worldToScreenScale(mod.influenceRadius);
    drawCircularInfluenceGradient(ctx, s.x, s.y, coreScreen, inflScreen, '255, 204, 0');

    // Dashed outline at outermost boundary
    const ir = camera.worldToScreenScale(mod.radius + mod.influenceRadius);
    ctx.strokeStyle = COLORS.checkpoint;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.arc(s.x, s.y, ir, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.checkpointFill;
  ctx.fill();
  ctx.strokeStyle = COLORS.checkpoint;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = COLORS.checkpoint;
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`CP ${mod.order}`, s.x, s.y + 4);
}

function drawSpeedZone(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: import('../model/types').SpeedZoneModule,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);
  const w = camera.worldToScreenScale(mod.halfExtentX * 2);
  const h = camera.worldToScreenScale(mod.halfExtentY * 2);

  ctx.fillStyle = COLORS.speedZoneFill;
  ctx.fillRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.strokeStyle = COLORS.speedZone;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.setLineDash([]);

  ctx.fillStyle = COLORS.speedZone;
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(`SPD <${mod.maxSpeed}`, s.x, s.y + 4);
}

function drawDangerZone(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: import('../model/types').DangerZoneModule,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);
  const w = camera.worldToScreenScale(mod.halfExtentX * 2);
  const h = camera.worldToScreenScale(mod.halfExtentY * 2);

  const fill = mod.isLethal ? 'rgba(255, 0, 51, 0.3)' : COLORS.dangerZoneFill;
  const stroke = mod.isLethal ? COLORS.dangerZoneLethal : COLORS.dangerZone;

  // Draw influence gradient (increasing toward danger zone)
  if (mod.influenceRadius > 0) {
    const inflScreen = camera.worldToScreenScale(mod.influenceRadius);
    const coreHW = camera.worldToScreenScale(mod.halfExtentX);
    const coreHH = camera.worldToScreenScale(mod.halfExtentY);
    drawInfluenceGradient(ctx, s.x, s.y, coreHW, coreHH, inflScreen, '255, 51, 102');

    // Dashed outline at outermost boundary
    const iw = (coreHW + inflScreen) * 2;
    const ih = (coreHH + inflScreen) * 2;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 0.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    roundedRect(ctx, s.x - iw / 2, s.y - ih / 2, iw, ih, inflScreen);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = fill;
  ctx.fillRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);

  ctx.fillStyle = stroke;
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(mod.isLethal ? 'LETHAL' : 'DANGER', s.x, s.y + 4);
}

function drawAttractor(
  ctx: CanvasRenderingContext2D,
  camera: Camera,
  mod: import('../model/types').AttractorModule,
): void {
  const s = camera.worldToScreen(mod.position.x, mod.position.y);
  const w = camera.worldToScreenScale(mod.halfExtentX * 2);
  const h = camera.worldToScreenScale(mod.halfExtentY * 2);
  const isRepulsor = mod.magnitude < 0;

  if (mod.influenceRadius <= 0) {
    drawAttractorCore(ctx, s, w, h, mod);
    return;
  }

  // Draw influence gradient (increasing toward attractor/repulsor core)
  const inflScreen = camera.worldToScreenScale(mod.influenceRadius);
  const coreHW = camera.worldToScreenScale(mod.halfExtentX);
  const coreHH = camera.worldToScreenScale(mod.halfExtentY);
  const rgb = isRepulsor ? '255, 102, 68' : '68, 255, 170';
  drawInfluenceGradient(ctx, s.x, s.y, coreHW, coreHH, inflScreen, rgb);

  // Dashed outline at outermost boundary
  const iw = (coreHW + inflScreen) * 2;
  const ih = (coreHH + inflScreen) * 2;
  ctx.strokeStyle = isRepulsor ? COLORS.repulsor : COLORS.attractor;
  ctx.lineWidth = 0.5;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  roundedRect(ctx, s.x - iw / 2, s.y - ih / 2, iw, ih, inflScreen);
  ctx.stroke();
  ctx.setLineDash([]);

  drawAttractorCore(ctx, s, w, h, mod);
}

function drawAttractorCore(
  ctx: CanvasRenderingContext2D,
  s: { x: number; y: number },
  w: number, h: number,
  mod: import('../model/types').AttractorModule,
): void {
  const isRepulsor = mod.magnitude < 0;

  ctx.fillStyle = isRepulsor ? COLORS.repulsorFill : COLORS.attractorFill;
  ctx.fillRect(s.x - w / 2, s.y - h / 2, w, h);
  ctx.strokeStyle = isRepulsor ? COLORS.repulsor : COLORS.attractor;
  ctx.lineWidth = 2;
  ctx.strokeRect(s.x - w / 2, s.y - h / 2, w, h);

  ctx.fillStyle = isRepulsor ? COLORS.repulsor : COLORS.attractor;
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  const label = isRepulsor ? `REP ${mod.magnitude}` : `ATT +${mod.magnitude}`;
  ctx.fillText(label, s.x, s.y + 4);
}

// ── Influence gradient helpers ──────────────────────────────────────────

const GRADIENT_STEPS = 6;
const ALPHA_PER_STEP = 0.025;

/**
 * Draw concentric rounded-rect bands from outside→inside.
 * Each band adds ALPHA_PER_STEP opacity, so layers accumulate:
 *   outer edge  = 1 layer  = faint
 *   core edge   = N layers = strongest
 * This makes the field visually INCREASE toward the target.
 */
function drawInfluenceGradient(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  coreHalfW: number, coreHalfH: number,
  influenceScreen: number,
  rgb: string,
): void {
  for (let i = 0; i < GRADIENT_STEPS; i++) {
    const frac = (GRADIENT_STEPS - i) / GRADIENT_STEPS; // 1.0 outermost → small innermost
    const r = influenceScreen * frac;
    const hw = coreHalfW + r;
    const hh = coreHalfH + r;
    ctx.fillStyle = `rgba(${rgb}, ${ALPHA_PER_STEP})`;
    ctx.beginPath();
    roundedRect(ctx, cx - hw, cy - hh, hw * 2, hh * 2, r);
    ctx.fill();
  }
}

/**
 * Circular version for checkpoints — same accumulating-alpha approach.
 */
function drawCircularInfluenceGradient(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  coreR: number,
  influenceScreen: number,
  rgb: string,
): void {
  for (let i = 0; i < GRADIENT_STEPS; i++) {
    const frac = (GRADIENT_STEPS - i) / GRADIENT_STEPS;
    const r = coreR + influenceScreen * frac;
    ctx.fillStyle = `rgba(${rgb}, ${ALPHA_PER_STEP})`;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Landing pad influence — extends UPWARD and to the sides, NOT below the pad.
 * Shape: rounded top corners, flat bottom at pad's bottom edge.
 * Same accumulating-alpha gradient as other zones.
 */
function drawUpwardInfluenceGradient(
  ctx: CanvasRenderingContext2D,
  cx: number,
  padTopScreen: number,
  padBottomScreen: number,
  coreHalfW: number,
  influenceScreen: number,
  rgb: string,
): void {
  for (let i = 0; i < GRADIENT_STEPS; i++) {
    const frac = (GRADIENT_STEPS - i) / GRADIENT_STEPS;
    const r = influenceScreen * frac;
    const hw = coreHalfW + r;
    const topY = padTopScreen - r;
    ctx.fillStyle = `rgba(${rgb}, ${ALPHA_PER_STEP})`;
    ctx.beginPath();
    topRoundedRect(ctx, cx - hw, topY, hw * 2, padBottomScreen - topY, r);
    ctx.fill();
  }
}

// ── Shape helpers ───────────────────────────────────────────────────────

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  r = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

/**
 * Rounded rect with only the TOP corners rounded, flat bottom edge.
 * Used for landing pad upward-only influence.
 */
function topRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  r = Math.min(r, w / 2, h);
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
}
