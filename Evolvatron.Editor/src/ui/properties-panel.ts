import type { Editor } from '../editor/editor';

import { UpdatePropertyCommand } from '../commands/update-property';
import { UpdateSingletonCommand } from '../commands/update-singleton';
import { findModule } from '../model/world';

export function setupPropertiesPanel(editor: Editor): () => void {
  const noSelection = document.getElementById('no-selection')!;
  const propsContent = document.getElementById('props-content')!;

  function update(): void {
    const ids = editor.selection.toArray();
    if (ids.length === 0) {
      noSelection.style.display = '';
      propsContent.style.display = 'none';
      return;
    }

    noSelection.style.display = 'none';
    propsContent.style.display = '';

    if (ids.length > 1) {
      propsContent.innerHTML = `<h3>Multiple Selected</h3><p style="color:#666688">${ids.length} objects selected</p>`;
      return;
    }

    const id = ids[0];
    if (id === 'landingPad') {
      renderLandingPadProps(editor, propsContent);
    } else if (id === 'spawnArea') {
      renderSpawnAreaProps(editor, propsContent);
    } else {
      const mod = findModule(editor.world, id);
      if (mod) renderModuleProps(editor, propsContent, id);
    }
  }

  return update;
}

function renderLandingPadProps(editor: Editor, container: HTMLElement): void {
  const pad = editor.world.landingPad;
  container.innerHTML = '<h3>Landing Pad</h3>';
  appendNumberField(container, 'X', pad.position.x, v => {
    commitSingleton(editor, 'landingPad', 'position', pad.position, { ...pad.position, x: v });
  });
  appendNumberField(container, 'Y', pad.position.y, v => {
    commitSingleton(editor, 'landingPad', 'position', pad.position, { ...pad.position, y: v });
  });
  appendNumberField(container, 'Half Width', pad.halfWidth, v => {
    commitSingleton(editor, 'landingPad', 'halfWidth', pad.halfWidth, v);
  });
  appendNumberField(container, 'Half Height', pad.halfHeight, v => {
    commitSingleton(editor, 'landingPad', 'halfHeight', pad.halfHeight, v);
  });
  appendNumberField(container, 'Landing Bonus', pad.landingBonus, v => {
    commitSingleton(editor, 'landingPad', 'landingBonus', pad.landingBonus, v);
  });
  appendNumberField(container, 'Max Land Vel', pad.maxLandingVelocity, v => {
    commitSingleton(editor, 'landingPad', 'maxLandingVelocity', pad.maxLandingVelocity, v);
  });
  appendNumberField(container, 'Max Land Angle', pad.maxLandingAngle, v => {
    commitSingleton(editor, 'landingPad', 'maxLandingAngle', pad.maxLandingAngle, v);
  });
}

function renderSpawnAreaProps(editor: Editor, container: HTMLElement): void {
  const spawn = editor.world.spawnArea;
  container.innerHTML = '<h3>Spawn Area</h3>';
  appendNumberField(container, 'X', spawn.position.x, v => {
    commitSingleton(editor, 'spawnArea', 'position', spawn.position, { ...spawn.position, x: v });
  });
  appendNumberField(container, 'Y', spawn.position.y, v => {
    commitSingleton(editor, 'spawnArea', 'position', spawn.position, { ...spawn.position, y: v });
  });
  appendNumberField(container, 'X Range', spawn.xRange, v => {
    commitSingleton(editor, 'spawnArea', 'xRange', spawn.xRange, v);
  });
  appendNumberField(container, 'Height Range', spawn.heightRange, v => {
    commitSingleton(editor, 'spawnArea', 'heightRange', spawn.heightRange, v);
  });
  appendNumberField(container, 'Angle Range', spawn.angleRange, v => {
    commitSingleton(editor, 'spawnArea', 'angleRange', spawn.angleRange, v);
  });
  appendNumberField(container, 'Vel X Range', spawn.velXRange, v => {
    commitSingleton(editor, 'spawnArea', 'velXRange', spawn.velXRange, v);
  });
  appendNumberField(container, 'Vel Y Max', spawn.velYMax, v => {
    commitSingleton(editor, 'spawnArea', 'velYMax', spawn.velYMax, v);
  });
}

function renderModuleProps(editor: Editor, container: HTMLElement, id: string): void {
  const mod = findModule(editor.world, id);
  if (!mod) return;

  const kindLabel = mod.kind.charAt(0).toUpperCase() + mod.kind.slice(1).replace(/([A-Z])/g, ' $1');
  container.innerHTML = `<h3>${kindLabel}</h3>`;

  appendNumberField(container, 'X', mod.position.x, v => {
    commitProp(editor, id, 'position', mod.position, { ...mod.position, x: v });
  });
  appendNumberField(container, 'Y', mod.position.y, v => {
    commitProp(editor, id, 'position', mod.position, { ...mod.position, y: v });
  });

  switch (mod.kind) {
    case 'obstacle':
      appendNumberField(container, 'Half Extent X', mod.halfExtentX, v => commitProp(editor, id, 'halfExtentX', mod.halfExtentX, v));
      appendNumberField(container, 'Half Extent Y', mod.halfExtentY, v => commitProp(editor, id, 'halfExtentY', mod.halfExtentY, v));
      appendNumberField(container, 'Rotation', mod.rotation, v => commitProp(editor, id, 'rotation', mod.rotation, v));
      appendCheckbox(container, 'Lethal', mod.isLethal, v => commitProp(editor, id, 'isLethal', mod.isLethal, v));
      break;
    case 'checkpoint':
      appendNumberField(container, 'Radius', mod.radius, v => commitProp(editor, id, 'radius', mod.radius, v));
      appendNumberField(container, 'Order', mod.order, v => commitProp(editor, id, 'order', mod.order, v));
      appendNumberField(container, 'Reward Bonus', mod.rewardBonus, v => commitProp(editor, id, 'rewardBonus', mod.rewardBonus, v));
      appendNumberField(container, 'Influence Factor', mod.influenceFactor, v => commitProp(editor, id, 'influenceFactor', mod.influenceFactor, v));
      break;
    case 'speedZone':
      appendNumberField(container, 'Half Extent X', mod.halfExtentX, v => commitProp(editor, id, 'halfExtentX', mod.halfExtentX, v));
      appendNumberField(container, 'Half Extent Y', mod.halfExtentY, v => commitProp(editor, id, 'halfExtentY', mod.halfExtentY, v));
      appendNumberField(container, 'Max Speed', mod.maxSpeed, v => commitProp(editor, id, 'maxSpeed', mod.maxSpeed, v));
      appendNumberField(container, 'Reward/Step', mod.rewardPerStep, v => commitProp(editor, id, 'rewardPerStep', mod.rewardPerStep, v));
      break;
    case 'dangerZone':
      appendNumberField(container, 'Half Extent X', mod.halfExtentX, v => commitProp(editor, id, 'halfExtentX', mod.halfExtentX, v));
      appendNumberField(container, 'Half Extent Y', mod.halfExtentY, v => commitProp(editor, id, 'halfExtentY', mod.halfExtentY, v));
      appendNumberField(container, 'Penalty/Step', mod.penaltyPerStep, v => commitProp(editor, id, 'penaltyPerStep', mod.penaltyPerStep, v));
      appendNumberField(container, 'Influence Factor', mod.influenceFactor, v => commitProp(editor, id, 'influenceFactor', mod.influenceFactor, v));
      appendCheckbox(container, 'Lethal', mod.isLethal, v => commitProp(editor, id, 'isLethal', mod.isLethal, v));
      break;
    case 'attractor':
      appendNumberField(container, 'Half Extent X', mod.halfExtentX, v => commitProp(editor, id, 'halfExtentX', mod.halfExtentX, v));
      appendNumberField(container, 'Half Extent Y', mod.halfExtentY, v => commitProp(editor, id, 'halfExtentY', mod.halfExtentY, v));
      appendNumberField(container, 'Magnitude', mod.magnitude, v => commitProp(editor, id, 'magnitude', mod.magnitude, v));
      appendNumberField(container, 'Influence Factor', mod.influenceFactor, v => commitProp(editor, id, 'influenceFactor', mod.influenceFactor, v));
      appendNumberField(container, 'Contact Bonus', mod.contactBonus, v => commitProp(editor, id, 'contactBonus', mod.contactBonus, v));
      break;
  }
}

function appendNumberField(container: HTMLElement, label: string, value: number, onChange: (v: number) => void): void {
  const row = document.createElement('div');
  row.className = 'prop-row';
  const lbl = document.createElement('label');
  lbl.textContent = label;
  lbl.style.cursor = 'ew-resize';
  lbl.style.userSelect = 'none';

  const input = document.createElement('input');
  input.type = 'text';
  input.inputMode = 'decimal';
  input.value = Number.isInteger(value) ? value.toString() : value.toFixed(3);
  input.addEventListener('change', () => {
    const v = parseFloat(input.value);
    if (!isNaN(v)) onChange(v);
  });

  // Drag-on-label to adjust value
  let dragStartX = 0;
  let dragStartValue = 0;

  const onDragMove = (e: MouseEvent) => {
    const dx = e.clientX - dragStartX;
    // Scale: 1px = 0.01 for small values, 0.1 for larger
    const scale = Math.abs(dragStartValue) > 10 ? 0.1 : 0.01;
    const newVal = dragStartValue + dx * scale;
    const rounded = Math.round(newVal * 1000) / 1000;
    input.value = rounded.toString();
    onChange(rounded);
  };

  const onDragEnd = () => {
    document.removeEventListener('mousemove', onDragMove);
    document.removeEventListener('mouseup', onDragEnd);
    document.body.style.cursor = '';
  };

  lbl.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragStartX = e.clientX;
    dragStartValue = parseFloat(input.value) || 0;
    document.body.style.cursor = 'ew-resize';
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
  });

  row.appendChild(lbl);
  row.appendChild(input);
  container.appendChild(row);
}

function appendCheckbox(container: HTMLElement, label: string, value: boolean, onChange: (v: boolean) => void): void {
  const row = document.createElement('div');
  row.className = 'prop-row';
  const lbl = document.createElement('label');
  lbl.textContent = label;
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = value;
  input.addEventListener('change', () => onChange(input.checked));
  row.appendChild(lbl);
  row.appendChild(input);
  container.appendChild(row);
}

function commitProp(editor: Editor, id: string, prop: string, oldVal: unknown, newVal: unknown): void {
  editor.history.execute(new UpdatePropertyCommand(id, prop, oldVal, newVal), editor.world);
  editor.markDirty();
}

function commitSingleton(editor: Editor, target: 'landingPad' | 'spawnArea', prop: string, oldVal: unknown, newVal: unknown): void {
  editor.history.execute(new UpdateSingletonCommand(target, prop, oldVal, newVal), editor.world);
  editor.markDirty();
}
