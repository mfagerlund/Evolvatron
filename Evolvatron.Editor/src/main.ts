import { createDefaultWorld } from './model/defaults';
import { Editor } from './editor/editor';
import { Renderer } from './renderer/renderer';
import { setupToolbar } from './ui/toolbar';
import { setupPropertiesPanel } from './ui/properties-panel';
import { serialize, deserialize } from './io/serialize';
import { exportSim } from './io/export-sim';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const container = document.getElementById('canvas-container') as HTMLDivElement;

const world = createDefaultWorld();
const editor = new Editor(world);
const renderer = new Renderer(canvas);

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
  if (editor.onKeyDown(e.key, ctrl, e.shiftKey)) {
    e.preventDefault();
  }
  // Tool shortcuts
  if (!ctrl && !e.shiftKey) {
    switch (e.key.toLowerCase()) {
      case 'v': setActiveTool('select'); break;
      case '1': setActiveTool('obstacle'); break;
      case '2': setActiveTool('checkpoint'); break;
      case '3': setActiveTool('speedZone'); break;
      case '4': setActiveTool('dangerZone'); break;
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
}

function handleSave(): void {
  const json = serialize(editor.world);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'world.evo.json';
  a.click();
  URL.revokeObjectURL(url);
}

function handleLoad(): void {
  const input = document.getElementById('file-input') as HTMLInputElement;
  input.onchange = () => {
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const loaded = deserialize(reader.result as string);
        Object.assign(editor.world, loaded);
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

function handleExport(): void {
  const json = exportSim(editor.world);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'world.sim.json';
  a.click();
  URL.revokeObjectURL(url);
}

// --- Setup UI ---

setupToolbar({
  onToolSelect: setActiveTool,
  onNew: handleNew,
  onLoad: handleLoad,
  onSave: handleSave,
  onExport: handleExport,
});

const updateProps = setupPropertiesPanel(editor);

// --- Render loop ---

editor.setOnChange(() => {
  updateProps();
});

function renderLoop(): void {
  if (editor.consumeDirty()) {
    renderer.render(editor.world, editor.camera, editor.selection, editor.state);
  }
  requestAnimationFrame(renderLoop);
}

requestAnimationFrame(renderLoop);
