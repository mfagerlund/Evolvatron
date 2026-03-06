import type { World, Vec2, Module, ModuleKind, SelectableId } from '../model/types';
import { CommandHistory } from '../commands/command';
import { AddModuleCommand } from '../commands/add-module';
import { RemoveModulesCommand } from '../commands/remove-modules';
import { MoveModulesCommand } from '../commands/move-modules';
import { DuplicateModulesCommand } from '../commands/duplicate-modules';
import { Selection } from './selection';
import { Camera } from './camera';
import { hitTestPoint, boxSelectModules } from './hit-test';
import {
  createObstacle, createCheckpoint, createSpeedZone, createDangerZone,
} from '../model/defaults';
import { findModule } from '../model/world';

export type EditorMode = 'idle' | 'dragging' | 'boxSelecting' | 'placing' | 'panning';

export interface EditorState {
  mode: EditorMode;
  placingKind?: ModuleKind;
  ghostPosition?: Vec2;
  boxStart?: Vec2;
  boxEnd?: Vec2;
}

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
  }

  onMouseUp(_sx: number, _sy: number, shift: boolean): void {
    if (this.state.mode === 'panning') {
      this.state = { mode: 'idle' };
      this.panStart = null;
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
