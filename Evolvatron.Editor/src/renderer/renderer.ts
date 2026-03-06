import type { World } from '../model/types';
import type { Camera } from '../editor/camera';
import type { Selection } from '../editor/selection';
import type { EditorState } from '../editor/editor';
import { COLORS } from './colors';
import { drawGrid, drawGround, drawLandingPad, drawSpawnArea, drawModules } from './draw-modules';
import { drawSelectionHighlights, drawBoxSelection, drawGhostModule } from './draw-overlay';
import { drawRewardOverlay } from './draw-reward-overlay';

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  showRewardOverlay = false;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  render(world: World, camera: Camera, selection: Selection, editorState: EditorState): void {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, w, h);

    drawGrid(ctx, camera);

    if (this.showRewardOverlay) {
      drawRewardOverlay(ctx, camera, world);
    }

    drawGround(ctx, camera, world.groundY);
    drawSpawnArea(ctx, camera, world);
    drawModules(ctx, camera, world);
    drawLandingPad(ctx, camera, world);

    if (selection.size > 0) {
      drawSelectionHighlights(ctx, camera, world, selection);
    }

    if (editorState.mode === 'boxSelecting') {
      drawBoxSelection(
        ctx,
        editorState.boxStart!.x, editorState.boxStart!.y,
        editorState.boxEnd!.x, editorState.boxEnd!.y,
      );
    }

    if ((editorState.mode === 'placing' || editorState.mode === 'placingDrag') && editorState.ghostPosition) {
      drawGhostModule(
        ctx, camera,
        editorState.placingKind!,
        editorState.ghostPosition.x, editorState.ghostPosition.y,
        editorState.placingAnchor?.x, editorState.placingAnchor?.y,
      );
    }

    this.drawStatusBar(ctx, camera, editorState, selection, w, h);
  }

  private drawStatusBar(
    ctx: CanvasRenderingContext2D,
    camera: Camera,
    editorState: EditorState,
    selection: Selection,
    w: number, h: number,
  ): void {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, h - 24, w, 24);

    ctx.fillStyle = COLORS.textSecondary;
    ctx.font = '11px monospace';
    ctx.textAlign = 'left';

    const modeText = editorState.mode === 'placing'
      ? `Place: ${editorState.placingKind} (click first point)`
      : editorState.mode === 'placingDrag'
      ? `Place: ${editorState.placingKind} (drag to second point)`
      : editorState.mode;

    const selText = selection.size > 0 ? `  |  ${selection.size} selected` : '';
    const zoomText = `  |  zoom: ${camera.zoom.toFixed(0)}px/m`;

    ctx.fillText(`Mode: ${modeText}${selText}${zoomText}`, 8, h - 7);
  }
}
