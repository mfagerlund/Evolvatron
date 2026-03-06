import type { World, Vec2, Module, ModuleKind, SelectableId } from '../model/types';
import { CommandHistory } from '../commands/command';
import { AddModuleCommand } from '../commands/add-module';
import { RemoveModulesCommand } from '../commands/remove-modules';
import { MoveModulesCommand } from '../commands/move-modules';
import { DuplicateModulesCommand } from '../commands/duplicate-modules';
import { Selection } from './selection';
import { Camera } from './camera';
import { hitTestPoint, hitTestHandle, boxSelectModules } from './hit-test';
import type { HandleDirection } from './hit-test';
import {
  createObstacle, createCheckpoint, createSpeedZone, createDangerZone,
} from '../model/defaults';
import { findModule } from '../model/world';
import {
  ResizeModuleCommand, ResizeSingletonCommand,
  type ResizeSnapshot,
} from '../commands/resize-module';

export type EditorMode = 'idle' | 'dragging' | 'boxSelecting' | 'placing' | 'panning' | 'resizing';

export interface EditorState {
  mode: EditorMode;
  placingKind?: ModuleKind;
  ghostPosition?: Vec2;
  boxStart?: Vec2;
  boxEnd?: Vec2;
}

/** Map from handle direction to CSS cursor name */
const HANDLE_CURSORS: Record<HandleDirection, string> = {
  n: 'ns-resize',    s: 'ns-resize',
  e: 'ew-resize',    w: 'ew-resize',
  ne: 'nesw-resize', sw: 'nesw-resize',
  nw: 'nwse-resize', se: 'nwse-resize',
};

export class Editor {
  world: World;
  selection = new Selection();
  camera = new Camera();
  history = new CommandHistory();
  state: EditorState = { mode: 'idle' };

  private dragStart: Vec2 | null = null;
  private dragWorldStart: Vec2 | null = null;
  private panStart: Vec2 | null = null;
  private isDirty = true;
  private onChange: (() => void) | null = null;

  // Resize state
  private resizeTargetId: SelectableId | null = null;
  private resizeDirection: HandleDirection | null = null;
  private resizeStartValues: ResizeSnapshot | null = null;

  /** Current cursor to display. Updated on mouse move. */
  cursor = 'default';

  constructor(world: World) {
    this.world = world;
  }

  setOnChange(fn: () => void): void {
    this.onChange = fn;
  }

  markDirty(): void {
    this.isDirty = true;
    this.onChange?.();
  }

  consumeDirty(): boolean {
    const d = this.isDirty;
    this.isDirty = false;
    return d;
  }

  // --- Tool switching ---

  startPlacing(kind: ModuleKind): void {
    this.selection.clear();
    this.state = { mode: 'placing', placingKind: kind };
    this.markDirty();
  }

  cancelPlacing(): void {
    this.state = { mode: 'idle' };
    this.markDirty();
  }

  // --- Mouse events (screen coords) ---

  onMouseDown(sx: number, sy: number, shift: boolean, middle: boolean): void {
    if (middle || this.state.mode === 'panning') {
      this.panStart = { x: sx, y: sy };
      this.state = { ...this.state, mode: 'panning' };
      return;
    }

    if (this.state.mode === 'placing') {
      const wPos = this.camera.screenToWorld(sx, sy);
      this.placeModule(wPos);
      return;
    }

    // Check if clicking a resize handle on an already-selected object
    const handleHit = hitTestHandle(this.world, this.camera, this.selection.ids, sx, sy);
    if (handleHit) {
      this.startResize(handleHit.id, handleHit.direction, sx, sy);
      return;
    }

    const wPos = this.camera.screenToWorld(sx, sy);
    const hit = hitTestPoint(this.world, wPos.x, wPos.y);

    if (hit) {
      if (shift) {
        this.selection.toggle(hit.id);
      } else if (!this.selection.has(hit.id)) {
        this.selection.set(hit.id);
      }
      this.dragStart = { x: sx, y: sy };
      this.dragWorldStart = wPos;
      this.state = { mode: 'dragging' };
    } else {
      if (!shift) {
        this.selection.clear();
      }
      this.dragStart = { x: sx, y: sy };
      this.state = {
        mode: 'boxSelecting',
        boxStart: { x: sx, y: sy },
        boxEnd: { x: sx, y: sy },
      };
    }

    this.markDirty();
  }

  onMouseMove(sx: number, sy: number): void {
    if (this.state.mode === 'panning' && this.panStart) {
      const dx = sx - this.panStart.x;
      const dy = sy - this.panStart.y;
      this.camera.pan(dx, dy);
      this.panStart = { x: sx, y: sy };
      this.markDirty();
      return;
    }

    if (this.state.mode === 'placing') {
      const wPos = this.camera.screenToWorld(sx, sy);
      this.state.ghostPosition = wPos;
      this.markDirty();
      return;
    }

    if (this.state.mode === 'resizing' && this.resizeTargetId !== null) {
      this.applyResizeDelta(sx, sy);
      this.markDirty();
      return;
    }

    if (this.state.mode === 'dragging' && this.dragStart && this.selection.size > 0) {
      const wCurrent = this.camera.screenToWorld(sx, sy);
      const dx = wCurrent.x - this.dragWorldStart!.x;
      const dy = wCurrent.y - this.dragWorldStart!.y;

      if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) {
        this.applyDragDelta(dx, dy);
        this.dragWorldStart = wCurrent;
        this.markDirty();
      }
      return;
    }

    if (this.state.mode === 'boxSelecting') {
      this.state.boxEnd = { x: sx, y: sy };
      this.markDirty();
      return;
    }

    // Idle — update cursor based on handle hover
    this.updateCursor(sx, sy);
  }

  onMouseUp(_sx: number, _sy: number, shift: boolean): void {
    if (this.state.mode === 'panning') {
      this.state = { mode: 'idle' };
      this.panStart = null;
      this.markDirty();
      return;
    }

    if (this.state.mode === 'resizing') {
      this.finalizeResize();
      this.state = { mode: 'idle' };
      this.markDirty();
      return;
    }

    if (this.state.mode === 'dragging') {
      this.finalizeDrag();
      this.state = { mode: 'idle' };
      this.dragStart = null;
      this.dragWorldStart = null;
      this.markDirty();
      return;
    }

    if (this.state.mode === 'boxSelecting' && this.state.boxStart && this.state.boxEnd) {
      const w0 = this.camera.screenToWorld(this.state.boxStart.x, this.state.boxStart.y);
      const w1 = this.camera.screenToWorld(this.state.boxEnd!.x, this.state.boxEnd!.y);
      const minX = Math.min(w0.x, w1.x);
      const minY = Math.min(w0.y, w1.y);
      const maxX = Math.max(w0.x, w1.x);
      const maxY = Math.max(w0.y, w1.y);

      const hits = boxSelectModules(this.world, minX, minY, maxX, maxY);
      if (!shift) this.selection.clear();
      this.selection.addMany(hits.map(h => h.id));

      this.state = { mode: 'idle' };
      this.dragStart = null;
      this.markDirty();
      return;
    }
  }

  onWheel(sx: number, sy: number, deltaY: number): void {
    const factor = deltaY > 0 ? 0.9 : 1.1;
    this.camera.zoomAt(sx, sy, factor);
    this.markDirty();
  }

  onKeyDown(key: string, ctrl: boolean, _shift: boolean): boolean {
    if (key === 'Escape') {
      return this.handleEscape();
    }
    if (key === 'Delete' || key === 'Backspace') {
      return this.deleteSelected();
    }
    if (ctrl && key === 'z') {
      this.undo();
      return true;
    }
    if (ctrl && key === 'y') {
      this.redo();
      return true;
    }
    if (ctrl && key === 'a') {
      this.selectAll();
      return true;
    }
    if (ctrl && key === 'd') {
      this.duplicateSelected();
      return true;
    }
    return false;
  }

  // --- Actions ---

  private handleEscape(): boolean {
    if (this.state.mode === 'placing') {
      this.cancelPlacing();
      return true;
    }
    if (this.selection.size > 0) {
      this.selection.clear();
      this.markDirty();
      return true;
    }
    return false;
  }

  private placeModule(pos: Vec2): void {
    const kind = this.state.placingKind!;
    let mod: Module;
    switch (kind) {
      case 'obstacle': mod = createObstacle(pos.x, pos.y); break;
      case 'checkpoint': {
        const existing = this.world.modules.filter(m => m.kind === 'checkpoint');
        mod = createCheckpoint(pos.x, pos.y, existing.length);
        break;
      }
      case 'speedZone': mod = createSpeedZone(pos.x, pos.y); break;
      case 'dangerZone': mod = createDangerZone(pos.x, pos.y); break;
    }
    this.history.execute(new AddModuleCommand(mod), this.world);
    this.selection.set(mod.id);
    this.markDirty();
  }

  deleteSelected(): boolean {
    const moduleIds = this.selection.moduleIds;
    if (moduleIds.length === 0) return false;
    this.history.execute(new RemoveModulesCommand(moduleIds), this.world);
    this.selection.clear();
    this.markDirty();
    return true;
  }

  selectAll(): void {
    this.selection.clear();
    this.selection.add('landingPad');
    this.selection.add('spawnArea');
    for (const mod of this.world.modules) {
      this.selection.add(mod.id);
    }
    this.markDirty();
  }

  duplicateSelected(): boolean {
    const moduleIds = this.selection.moduleIds;
    if (moduleIds.length === 0) return false;

    const clones = new Map<string, Module>();
    const offset = 0.5;
    for (const id of moduleIds) {
      const original = findModule(this.world, id);
      if (!original) continue;
      const clone = JSON.parse(JSON.stringify(original)) as Module;
      clone.id = `dup_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
      clone.position.x += offset;
      clone.position.y += offset;
      clones.set(clone.id, clone);
    }

    if (clones.size === 0) return false;
    this.history.execute(new DuplicateModulesCommand(moduleIds, clones), this.world);

    this.selection.clear();
    for (const id of clones.keys()) {
      this.selection.add(id);
    }
    this.markDirty();
    return true;
  }

  undo(): void {
    this.history.undo(this.world);
    this.selection.clear();
    this.markDirty();
  }

  redo(): void {
    this.history.redo(this.world);
    this.selection.clear();
    this.markDirty();
  }

  // --- Cursor ---

  private updateCursor(sx: number, sy: number): void {
    if (this.selection.size === 0) {
      this.cursor = 'default';
      return;
    }
    const handleHit = hitTestHandle(this.world, this.camera, this.selection.ids, sx, sy);
    if (handleHit) {
      this.cursor = HANDLE_CURSORS[handleHit.direction];
    } else {
      this.cursor = 'default';
    }
  }

  // --- Resize helpers ---

  private startResize(id: SelectableId, direction: HandleDirection, sx: number, sy: number): void {
    this.resizeTargetId = id;
    this.resizeDirection = direction;
    this.dragStart = { x: sx, y: sy };
    this.resizeStartValues = this.captureSize(id);
    this.state = { mode: 'resizing' };
    this.markDirty();
  }

  private captureSize(id: SelectableId): ResizeSnapshot {
    if (id === 'landingPad') {
      const p = this.world.landingPad;
      return { posX: p.position.x, posY: p.position.y, halfWidth: p.halfWidth, halfHeight: p.halfHeight };
    }
    if (id === 'spawnArea') {
      const s = this.world.spawnArea;
      return { posX: s.position.x, posY: s.position.y, xRange: s.xRange, heightRange: s.heightRange };
    }
    const mod = findModule(this.world, id);
    if (!mod) return {};
    if (mod.kind === 'checkpoint') {
      return { posX: mod.position.x, posY: mod.position.y, radius: mod.radius };
    }
    return { posX: mod.position.x, posY: mod.position.y, halfExtentX: mod.halfExtentX, halfExtentY: mod.halfExtentY };
  }

  private applyResizeDelta(sx: number, sy: number): void {
    if (!this.dragStart || !this.resizeTargetId || !this.resizeDirection || !this.resizeStartValues) return;

    // World-space delta from drag start
    const wStart = this.camera.screenToWorld(this.dragStart.x, this.dragStart.y);
    const wNow = this.camera.screenToWorld(sx, sy);
    let dwx = wNow.x - wStart.x;
    let dwy = wNow.y - wStart.y;

    const id = this.resizeTargetId;
    const dir = this.resizeDirection;
    const start = this.resizeStartValues;

    // For rotated obstacles, transform delta into local space
    let isRotated = false;
    let rotRad = 0;
    if (typeof id === 'string' && id !== 'landingPad' && id !== 'spawnArea') {
      const mod = findModule(this.world, id);
      if (mod && mod.kind === 'obstacle' && mod.rotation !== 0) {
        rotRad = (-mod.rotation * Math.PI) / 180;
        const cos = Math.cos(rotRad);
        const sin = Math.sin(rotRad);
        const lx = dwx * cos + dwy * sin;
        const ly = -dwx * sin + dwy * cos;
        dwx = lx;
        dwy = ly;
        isRotated = true;
      }
    }

    // Determine axis multipliers from direction
    // e.g. dragging the east handle: xSign=+1, dragging west: xSign=-1
    const xSign = dir.includes('e') ? 1 : dir.includes('w') ? -1 : 0;
    const ySign = dir.includes('n') ? 1 : dir.includes('s') ? -1 : 0;

    const MIN_SIZE = 0.1;

    // The key idea: the opposite edge stays fixed.
    // When dragging east by dwx, halfExtent grows by dwx/2 and center shifts right by dwx/2.
    // When dragging west by dwx (dwx is negative since moving left), xSign=-1:
    //   sizeChange = xSign * dwx (positive when dragging outward), center shifts by xSign * dwx / 2

    // Position shifts toward the dragged handle (xSign/ySign direction),
    // by half the world-space delta. The size grows by the same amount.
    // posShiftX = dwx / 2 (always toward cursor, regardless of which side)
    // sizeDelta = xSign * dwx / 2 (positive when dragging outward)

    if (id === 'landingPad') {
      const p = this.world.landingPad;
      if (xSign !== 0) {
        const sizeDelta = xSign * dwx / 2;
        p.halfWidth = Math.max(MIN_SIZE, start.halfWidth + sizeDelta);
        p.position.x = start.posX + dwx / 2;
      }
      if (ySign !== 0) {
        const sizeDelta = ySign * dwy / 2;
        p.halfHeight = Math.max(MIN_SIZE, start.halfHeight + sizeDelta);
        p.position.y = start.posY + dwy / 2;
      }
      return;
    }

    if (id === 'spawnArea') {
      const s = this.world.spawnArea;
      if (xSign !== 0) {
        const sizeDelta = xSign * dwx;
        s.xRange = Math.max(MIN_SIZE, start.xRange + sizeDelta);
        s.position.x = start.posX + dwx / 2;
      }
      if (ySign !== 0) {
        const sizeDelta = ySign * dwy;
        s.heightRange = Math.max(MIN_SIZE, start.heightRange + sizeDelta);
        s.position.y = start.posY + dwy / 2;
      }
      return;
    }

    const mod = findModule(this.world, id);
    if (!mod) return;

    if (mod.kind === 'checkpoint') {
      // Circle: anchor the opposite point, grow radius and shift center
      const radialDelta = Math.sqrt(dwx * dwx + dwy * dwy) *
        Math.sign(xSign * dwx + ySign * dwy || dwx + dwy);
      const sizeDelta = radialDelta / 2;
      mod.radius = Math.max(MIN_SIZE, start.radius + sizeDelta);
      // Shift center by half the world-space delta (toward cursor)
      mod.position.x = start.posX + dwx / 2;
      mod.position.y = start.posY + dwy / 2;
      return;
    }

    // Rectangles: compute local-space size and position deltas
    let localDx = 0;
    let localDy = 0;
    if (xSign !== 0) {
      const sizeDelta = xSign * dwx / 2;
      mod.halfExtentX = Math.max(MIN_SIZE, start.halfExtentX + sizeDelta);
      localDx = dwx / 2;
    }
    if (ySign !== 0) {
      const sizeDelta = ySign * dwy / 2;
      mod.halfExtentY = Math.max(MIN_SIZE, start.halfExtentY + sizeDelta);
      localDy = dwy / 2;
    }

    // Convert position shift back to world space for rotated obstacles
    if (isRotated) {
      const cos = Math.cos(-rotRad);
      const sin = Math.sin(-rotRad);
      mod.position.x = start.posX + localDx * cos - localDy * sin;
      mod.position.y = start.posY + localDx * sin + localDy * cos;
    } else {
      mod.position.x = start.posX + localDx;
      mod.position.y = start.posY + localDy;
    }
  }

  private finalizeResize(): void {
    if (!this.resizeTargetId || !this.resizeStartValues) {
      this.clearResizeState();
      return;
    }

    const id = this.resizeTargetId;
    const oldValues = this.resizeStartValues;
    const newValues = this.captureSize(id);

    // Check if anything actually changed
    const changed = Object.keys(oldValues).some(k => oldValues[k] !== newValues[k]);
    if (changed) {
      // Undo the live edit first
      this.applySize(id, oldValues);

      // Execute as command for undo support
      if (id === 'landingPad' || id === 'spawnArea') {
        this.history.execute(new ResizeSingletonCommand(id, oldValues, newValues), this.world);
      } else {
        this.history.execute(new ResizeModuleCommand(id, oldValues, newValues), this.world);
      }
    }

    this.clearResizeState();
  }

  private applySize(id: SelectableId, values: ResizeSnapshot): void {
    let target: Record<string, unknown> | null = null;
    let pos: Vec2 | null = null;

    if (id === 'landingPad') { target = this.world.landingPad as unknown as Record<string, unknown>; pos = this.world.landingPad.position; }
    else if (id === 'spawnArea') { target = this.world.spawnArea as unknown as Record<string, unknown>; pos = this.world.spawnArea.position; }
    else {
      const mod = findModule(this.world, id);
      if (mod) { target = mod as unknown as Record<string, unknown>; pos = mod.position; }
    }
    if (!target || !pos) return;

    if ('posX' in values) pos.x = values.posX;
    if ('posY' in values) pos.y = values.posY;
    for (const [key, val] of Object.entries(values)) {
      if (key !== 'posX' && key !== 'posY') target[key] = val;
    }
  }

  private clearResizeState(): void {
    this.resizeTargetId = null;
    this.resizeDirection = null;
    this.resizeStartValues = null;
    this.dragStart = null;
  }

  // --- Drag helpers ---

  private dragAccumulated = new Map<SelectableId, Vec2>();

  private applyDragDelta(dx: number, dy: number): void {
    for (const id of this.selection.ids) {
      let pos: Vec2 | undefined;
      if (id === 'landingPad') pos = this.world.landingPad.position;
      else if (id === 'spawnArea') pos = this.world.spawnArea.position;
      else pos = findModule(this.world, id)?.position;
      if (!pos) continue;

      pos.x += dx;
      pos.y += dy;

      const acc = this.dragAccumulated.get(id) ?? { x: 0, y: 0 };
      acc.x += dx;
      acc.y += dy;
      this.dragAccumulated.set(id, acc);
    }
  }

  private finalizeDrag(): void {
    if (this.dragAccumulated.size === 0) return;

    // Undo the visual drag (positions already moved)
    for (const [id, acc] of this.dragAccumulated) {
      let pos: Vec2 | undefined;
      if (id === 'landingPad') pos = this.world.landingPad.position;
      else if (id === 'spawnArea') pos = this.world.spawnArea.position;
      else pos = findModule(this.world, id)?.position;
      if (!pos) continue;
      pos.x -= acc.x;
      pos.y -= acc.y;
    }

    // Execute as a command for undo support
    this.history.execute(
      new MoveModulesCommand(new Map(this.dragAccumulated)),
      this.world,
    );
    this.dragAccumulated.clear();
  }
}
