import type { Vec2 } from '../model/types';

export class Camera {
  centerX = 0;
  centerY = 5;
  zoom = 30; // pixels per meter
  canvasWidth = 800;
  canvasHeight = 600;

  worldToScreen(wx: number, wy: number): Vec2 {
    return {
      x: (wx - this.centerX) * this.zoom + this.canvasWidth / 2,
      y: -(wy - this.centerY) * this.zoom + this.canvasHeight / 2,
    };
  }

  screenToWorld(sx: number, sy: number): Vec2 {
    return {
      x: (sx - this.canvasWidth / 2) / this.zoom + this.centerX,
      y: -((sy - this.canvasHeight / 2) / this.zoom) + this.centerY,
    };
  }

  worldToScreenScale(meters: number): number {
    return meters * this.zoom;
  }

  screenToWorldScale(pixels: number): number {
    return pixels / this.zoom;
  }

  pan(dxPixels: number, dyPixels: number): void {
    this.centerX -= dxPixels / this.zoom;
    this.centerY += dyPixels / this.zoom;
  }

  zoomAt(screenX: number, screenY: number, factor: number): void {
    const worldBefore = this.screenToWorld(screenX, screenY);
    this.zoom *= factor;
    this.zoom = Math.max(5, Math.min(200, this.zoom));
    const worldAfter = this.screenToWorld(screenX, screenY);
    this.centerX += worldBefore.x - worldAfter.x;
    this.centerY += worldBefore.y - worldAfter.y;
  }

  resize(width: number, height: number): void {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }
}
