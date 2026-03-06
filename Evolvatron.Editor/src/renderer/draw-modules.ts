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

  ctx.fillStyle = COLORS.spawnArea;
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('SPAWN', s.x, s.y + 4);
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

  ctx.save();
  ctx.translate(s.x, s.y);
  ctx.rotate((-mod.rotation * Math.PI) / 180);

  const fill = mod.isLethal ? 'rgba(255, 34, 0, 0.5)' : COLORS.obstacleFill;
  const stroke = mod.isLethal ? COLORS.obstaclLethal : COLORS.obstacle;

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
