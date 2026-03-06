import { describe, it, expect, beforeEach } from 'vitest';
import { Camera } from '../../src/editor/camera';

let cam: Camera;

beforeEach(() => {
  cam = new Camera();
});

describe('Camera', () => {
  it('has sensible defaults', () => {
    expect(cam.zoom).toBeGreaterThan(0);
    expect(cam.canvasWidth).toBeGreaterThan(0);
    expect(cam.canvasHeight).toBeGreaterThan(0);
  });

  describe('worldToScreen / screenToWorld roundtrip', () => {
    it('roundtrips at default settings', () => {
      const wx = 3, wy = 7;
      const s = cam.worldToScreen(wx, wy);
      const w = cam.screenToWorld(s.x, s.y);
      expect(w.x).toBeCloseTo(wx, 10);
      expect(w.y).toBeCloseTo(wy, 10);
    });

    it('roundtrips at non-default center and zoom', () => {
      cam.centerX = -10;
      cam.centerY = 20;
      cam.zoom = 50;
      const wx = -5, wy = 25;
      const s = cam.worldToScreen(wx, wy);
      const w = cam.screenToWorld(s.x, s.y);
      expect(w.x).toBeCloseTo(wx, 10);
      expect(w.y).toBeCloseTo(wy, 10);
    });

    it('center of camera maps to center of canvas', () => {
      cam.centerX = 5;
      cam.centerY = 10;
      const s = cam.worldToScreen(5, 10);
      expect(s.x).toBeCloseTo(cam.canvasWidth / 2);
      expect(s.y).toBeCloseTo(cam.canvasHeight / 2);
    });
  });

  describe('worldToScreenScale / screenToWorldScale', () => {
    it('converts meters to pixels', () => {
      cam.zoom = 40;
      expect(cam.worldToScreenScale(2)).toBe(80);
    });

    it('converts pixels to meters', () => {
      cam.zoom = 40;
      expect(cam.screenToWorldScale(80)).toBe(2);
    });

    it('are inverses', () => {
      const meters = 3.7;
      const pixels = cam.worldToScreenScale(meters);
      expect(cam.screenToWorldScale(pixels)).toBeCloseTo(meters);
    });
  });

  describe('pan', () => {
    it('shifts center in world space', () => {
      const origX = cam.centerX;
      const origY = cam.centerY;
      cam.pan(cam.zoom, 0); // pan 1 meter right in screen = left in world
      expect(cam.centerX).toBeCloseTo(origX - 1);
      expect(cam.centerY).toBeCloseTo(origY);
    });

    it('positive dy moves center up in world (screen y is inverted)', () => {
      const origY = cam.centerY;
      cam.pan(0, cam.zoom);
      expect(cam.centerY).toBeCloseTo(origY + 1);
    });

    it('pan then inverse pan restores center', () => {
      const origX = cam.centerX;
      const origY = cam.centerY;
      cam.pan(100, -50);
      cam.pan(-100, 50);
      expect(cam.centerX).toBeCloseTo(origX, 10);
      expect(cam.centerY).toBeCloseTo(origY, 10);
    });
  });

  describe('zoomAt', () => {
    it('changes zoom level', () => {
      const origZoom = cam.zoom;
      cam.zoomAt(400, 300, 1.5);
      expect(cam.zoom).toBeCloseTo(origZoom * 1.5);
    });

    it('zoom at canvas center does not shift center', () => {
      const origX = cam.centerX;
      const origY = cam.centerY;
      cam.zoomAt(cam.canvasWidth / 2, cam.canvasHeight / 2, 2);
      expect(cam.centerX).toBeCloseTo(origX, 5);
      expect(cam.centerY).toBeCloseTo(origY, 5);
    });

    it('clamps zoom to minimum', () => {
      cam.zoom = 10;
      cam.zoomAt(400, 300, 0.01);
      expect(cam.zoom).toBeGreaterThanOrEqual(5);
    });

    it('clamps zoom to maximum', () => {
      cam.zoom = 150;
      cam.zoomAt(400, 300, 10);
      expect(cam.zoom).toBeLessThanOrEqual(200);
    });

    it('zoom preserves the world point under cursor', () => {
      const sx = 200, sy = 150;
      const worldBefore = cam.screenToWorld(sx, sy);
      cam.zoomAt(sx, sy, 1.5);
      const worldAfter = cam.screenToWorld(sx, sy);
      expect(worldAfter.x).toBeCloseTo(worldBefore.x, 5);
      expect(worldAfter.y).toBeCloseTo(worldBefore.y, 5);
    });
  });

  describe('resize', () => {
    it('updates canvas dimensions', () => {
      cam.resize(1920, 1080);
      expect(cam.canvasWidth).toBe(1920);
      expect(cam.canvasHeight).toBe(1080);
    });
  });
});
