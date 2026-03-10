import { createDefaultWorld } from './model/defaults';
import { Editor } from './editor/editor';
import { Renderer } from './renderer/renderer';
import { setupToolbar } from './ui/toolbar';
import { setupPropertiesPanel } from './ui/properties-panel';
import { serialize, deserialize } from './io/serialize';
import { exportSim } from './io/export-sim';
import { autosave, loadAutosave } from './io/autosave';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const container = document.getElementById('canvas-container') as HTMLDivElement;

const world = createDefaultWorld();
const editor = new Editor(world);
const renderer = new Renderer(canvas);

// Restore from IndexedDB autosave if available
loadAutosave().then(saved => {
  if (saved) {
    Object.assign(editor.world, saved.world);
    if (saved.camera) {
      editor.camera.centerX = saved.camera.centerX;
      editor.camera.centerY = saved.camera.centerY;
      editor.camera.zoom = saved.camera.zoom;
    }
    editor.history.clear();
    editor.selection.clear();
    editor.markDirty();
  }
});

function resizeCanvas(): void {
  const rect = container.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  editor.camera.resize(canvas.width, canvas.height);
  editor.markDirty();
}

window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// --- Canvas events ---

canvas.addEventListener('mousedown', (e) => {
  // Blur any focused input so keyboard shortcuts work on the canvas
  if (document.activeElement instanceof HTMLElement && document.activeElement !== canvas) {
    document.activeElement.blur();
  }
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  editor.onMouseDown(sx, sy, e.shiftKey, e.button === 1);
});

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  editor.onMouseMove(sx, sy);
  canvas.style.cursor = editor.cursor;
  // Always re-render so status bar shows current reward at cursor
  editor.markDirty();
});

canvas.addEventListener('mouseup', (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  editor.onMouseUp(sx, sy, e.shiftKey);
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  editor.onWheel(sx, sy, e.deltaY);
}, { passive: false });

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// --- Keyboard ---

window.addEventListener('keydown', (e) => {
  const ctrl = e.ctrlKey || e.metaKey;

  // Ctrl+S always saves, regardless of focus
  if (ctrl && e.key === 's') {
    e.preventDefault();
    editor.onSave?.();
    return;
  }

  const tag = (e.target as HTMLElement)?.tagName;
  const inInput = tag === 'INPUT' || tag === 'TEXTAREA';

  if (inInput) {
    if (e.key === 'Escape') {
      (e.target as HTMLElement).blur();
      e.preventDefault();
    }
    return;
  }

  // Not in input — full editor key handling
  const wasPlacing = editor.state.mode === 'placing' || editor.state.mode === 'placingDrag';
  if (editor.onKeyDown(e.key, ctrl, e.shiftKey)) {
    e.preventDefault();
    if (wasPlacing && editor.state.mode !== 'placing' && editor.state.mode !== 'placingDrag') {
      updateToolbarActive('select');
    }
  }

  // Tool shortcuts
  if (!ctrl && !e.shiftKey) {
    switch (e.key.toLowerCase()) {
      case 'v': setActiveTool('select'); break;
      case '1': setActiveTool('obstacle'); break;
      case '2': setActiveTool('checkpoint'); break;
      case '3': setActiveTool('speedZone'); break;
      case '4': setActiveTool('dangerZone'); break;
      case '5': setActiveTool('attractor'); break;
      case 'r': toggleRewardOverlay(); break;
    }
  }
});

// --- Tool switching ---

function setActiveTool(tool: string): void {
  if (tool === 'select') {
    editor.cancelPlacing();
  } else {
    editor.startPlacing(tool as any);
  }
  updateToolbarActive(tool);
}

function updateToolbarActive(tool: string): void {
  document.querySelectorAll('#toolbar button[data-tool]').forEach(btn => {
    btn.classList.toggle('active', (btn as HTMLElement).dataset.tool === tool);
  });
}

// --- File operations ---

function handleNew(): void {
  const newWorld = createDefaultWorld();
  Object.assign(editor.world, newWorld);
  editor.history.clear();
  editor.selection.clear();
  editor.markDirty();
  autosave(editor.world);
}

let saveFileHandle: FileSystemFileHandle | null = null;

async function handleSave(): Promise<void> {
  const json = serialize(editor.world, {
    centerX: editor.camera.centerX,
    centerY: editor.camera.centerY,
    zoom: editor.camera.zoom,
  });
  try {
    if (!saveFileHandle) {
      saveFileHandle = await window.showSaveFilePicker({
        suggestedName: 'world.evo.json',
        types: [{ description: 'Editor World', accept: { 'application/json': ['.evo.json'] } }],
      });
    }
    const writable = await saveFileHandle.createWritable();
    await writable.write(json);
    await writable.close();

    // Auto-export .sim.json alongside save
    if (!exportFileHandle) {
      const simName = saveFileHandle.name.replace(/\.evo\.json$/i, '.sim.json');
      try {
        exportFileHandle = await window.showSaveFilePicker({
          suggestedName: simName,
          types: [{ description: 'Simulation World', accept: { 'application/json': ['.sim.json', '.json'] } }],
        });
      } catch (pickErr: any) {
        if (pickErr?.name === 'AbortError') {
          editor.showStatus(`Saved ${saveFileHandle.name} (export skipped)`);
          return;
        }
      }
    }
    if (exportFileHandle) {
      const simJson = exportSim(editor.world);
      const ew = await exportFileHandle.createWritable();
      await ew.write(simJson);
      await ew.close();
      editor.showStatus(`Saved ${saveFileHandle.name} + ${exportFileHandle.name}`);
    } else {
      editor.showStatus(`Saved ${saveFileHandle.name}`);
    }
  } catch (err: any) {
    if (err?.name === 'AbortError') return;
    // Fallback for unsupported browsers
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'world.evo.json';
    a.click();
    URL.revokeObjectURL(url);
    editor.showStatus('Saved world.evo.json');
  }
}

function handleLoad(): void {
  const input = document.getElementById('file-input') as HTMLInputElement;
  input.onchange = () => {
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const { world: loaded, camera } = deserialize(reader.result as string);
        Object.assign(editor.world, loaded);
        if (camera) {
          editor.camera.centerX = camera.centerX;
          editor.camera.centerY = camera.centerY;
          editor.camera.zoom = camera.zoom;
        }
        editor.history.clear();
        editor.selection.clear();
        editor.markDirty();
      } catch (err) {
        alert(`Failed to load: ${err}`);
      }
    };
    reader.readAsText(file);
    input.value = '';
  };
  input.click();
}

function toggleRewardOverlay(): void {
  renderer.showRewardOverlay = !renderer.showRewardOverlay;
  const btn = document.getElementById('btn-reward-overlay');
  btn?.classList.toggle('active', renderer.showRewardOverlay);
  editor.markDirty();
}

let exportFileHandle: FileSystemFileHandle | null = null;

async function handleExport(): Promise<void> {
  const json = exportSim(editor.world);
  try {
    if (!exportFileHandle) {
      exportFileHandle = await window.showSaveFilePicker({
        suggestedName: 'world.sim.json',
        types: [{ description: 'Simulation World', accept: { 'application/json': ['.sim.json', '.json'] } }],
      });
    }
    const writable = await exportFileHandle.createWritable();
    await writable.write(json);
    await writable.close();
    editor.showStatus(`Exported ${exportFileHandle.name}`);
  } catch (err: any) {
    if (err?.name === 'AbortError') return;
    // Fallback for unsupported browsers
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'world.sim.json';
    a.click();
    URL.revokeObjectURL(url);
    editor.showStatus('Exported world.sim.json');
  }
}

// --- Setup UI ---

setupToolbar({
  onToolSelect: setActiveTool,
  onNew: handleNew,
  onLoad: handleLoad,
  onSave: handleSave,
  onExport: handleExport,
  onToggleRewardOverlay: toggleRewardOverlay,
});

editor.onSave = handleSave;

// Sync reward overlay button with default state
document.getElementById('btn-reward-overlay')?.classList.toggle('active', renderer.showRewardOverlay);

const updateProps = setupPropertiesPanel(editor);

// --- Autosave (debounced) ---

let autosaveTimer: ReturnType<typeof setTimeout> | null = null;

function scheduleAutosave(): void {
  if (autosaveTimer !== null) clearTimeout(autosaveTimer);
  autosaveTimer = setTimeout(() => {
    autosave(editor.world, { centerX: editor.camera.centerX, centerY: editor.camera.centerY, zoom: editor.camera.zoom });
    autosaveTimer = null;
  }, 500);
}

// --- Render loop ---

editor.setOnChange(() => {
  updateProps();
  scheduleAutosave();
});

let lastFrameTime = 0;

function renderLoop(now: number): void {
  const dt = lastFrameTime > 0 ? Math.min((now - lastFrameTime) / 1000, 0.05) : 0;
  lastFrameTime = now;

  const hoverAnimating = editor.hoverState.update(dt);

  if (editor.consumeDirty() || hoverAnimating) {
    const ghosts = editor.state.mode === 'pasting' && editor.state.ghostPosition
      ? editor.getPasteGhosts(editor.state.ghostPosition)
      : undefined;
    renderer.render(editor.world, editor.camera, editor.selection, editor.state, editor.hoverState, ghosts, editor.mouseWorld.x, editor.mouseWorld.y, editor.statusMessage);
  }
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
