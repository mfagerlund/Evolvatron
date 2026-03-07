import type { World, Module } from '../model/types';
import type { Camera } from '../editor/camera';
import type { Selection } from '../editor/selection';
import type { HoverState } from '../editor/hover-state';
import type { EditorState } from '../editor/editor';
import { COLORS } from './colors';
import { drawGrid, drawGround, drawLandingPad, drawSpawnArea, drawModules } from './draw-modules';
import { drawSelectionHighlights, drawBoxSelection, drawGhostModule, drawHoverHighlights, drawPasteGhosts } from './draw-overlay';
import { drawRewardOverlay, sampleReward } from './draw-reward-overlay';

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  showRewardOverlay = true;

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
  }

  render(world: World, camera: Camera, selection: Selection, editorState: EditorState, hoverState: HoverState, pasteGhosts?: Module[], mouseWorldX?: number, mouseWorldY?: number, statusMessage?: string): void {
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
    drawHoverHighlights(ctx, camera, world, hoverState);

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

    if (pasteGhosts && pasteGhosts.length > 0) {
      drawPasteGhosts(ctx, camera, pasteGhosts);
    }

    this.drawStatusBar(ctx, camera, editorState, selection, w, h, world, mouseWorldX, mouseWorldY, statusMessage);
  }

  private drawStatusBar(
    ctx: CanvasRenderingContext2D,
    camera: Camera,
    editorState: EditorState,
    selection: Selection,
    w: number, h: number,
    world: World,
    mouseWorldX?: number,
    mouseWorldY?: number,
    statusMessage?: string,
  ): void {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, h - 24, w, 24);

    ctx.font = '11px monospace';
    ctx.textAlign = 'left';

    // Transient status message takes priority (right-aligned, highlighted)
    if (statusMessage) {
      ctx.fillStyle = '#88ddbb';
      ctx.textAlign = 'right';
      ctx.fillText(statusMessage, w - 8, h - 7);
      ctx.textAlign = 'left';
    }

    ctx.fillStyle = COLORS.textSecondary;

    const modeText = editorState.mode === 'placing'
      ? `Place: ${editorState.placingKind} (click first point)`
      : editorState.mode === 'placingDrag'
      ? `Place: ${editorState.placingKind} (drag to second point)`
      : editorState.mode === 'pasting'
      ? 'Paste (click to place, Esc to cancel)'
      : editorState.mode;

    const selText = selection.size > 0 ? `  |  ${selection.size} selected` : '';
    const zoomText = `  |  zoom: ${camera.zoom.toFixed(0)}px/m`;

    let rewardText = '';
    if (mouseWorldX !== undefined && mouseWorldY !== undefined) {
      const reward = sampleReward(world, mouseWorldX, mouseWorldY);
      if (Math.abs(reward) > 0.001) {
        rewardText = `  |  reward: ${reward >= 0 ? '+' : ''}${reward.toFixed(2)}`;
      }
    }

    ctx.fillText(`Mode: ${modeText}${selText}${zoomText}${rewardText}`, 8, h - 7);
  }
}
