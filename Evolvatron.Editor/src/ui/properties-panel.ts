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
      noSelection.style.display = 'none';
      propsContent.style.display = '';
      renderWorldProps(editor, propsContent);
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
  appendNumberField(container, 'Attraction Magnitude', pad.attractionMagnitude, v => {
    commitSingleton(editor, 'landingPad', 'attractionMagnitude', pad.attractionMagnitude, v);
  });
  appendNumberField(container, 'Attraction Radius', pad.attractionRadius, v => {
    commitSingleton(editor, 'landingPad', 'attractionRadius', pad.attractionRadius, v);
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
  appendIntField(container, 'Spawn Count', spawn.spawnCount, v => {
    commitSingleton(editor, 'spawnArea', 'spawnCount', spawn.spawnCount, v);
  });
  appendIntField(container, 'Spawn Seed', spawn.spawnSeed, v => {
    commitSingleton(editor, 'spawnArea', 'spawnSeed', spawn.spawnSeed, v);
  });
}

function renderWorldProps(editor: Editor, container: HTMLElement): void {
  const sim = editor.world.simulationConfig;
  const rw = editor.world.rewardWeights;
  container.innerHTML = '<h3>Simulation</h3>';

  appendIntField(container, 'Max Steps', sim.maxSteps, v => {
    commitSingleton(editor, 'simulationConfig', 'maxSteps', sim.maxSteps, v);
  });
  appendNumberField(container, 'Dt', sim.dt, v => {
    commitSingleton(editor, 'simulationConfig', 'dt', sim.dt, v);
  });
  const maxTime = sim.maxSteps * sim.dt;
  const timeRow = document.createElement('div');
  timeRow.className = 'prop-row';
  const timeLbl = document.createElement('label');
  timeLbl.textContent = 'Max Time';
  const timeVal = document.createElement('span');
  timeVal.textContent = `${maxTime.toFixed(1)}s`;
  timeVal.style.color = '#8888aa';
  timeRow.appendChild(timeLbl);
  timeRow.appendChild(timeVal);
  container.appendChild(timeRow);

  appendNumberField(container, 'Gravity Y', sim.gravityY, v => {
    commitSingleton(editor, 'simulationConfig', 'gravityY', sim.gravityY, v);
  });
  appendNumberField(container, 'Max Thrust', sim.maxThrust, v => {
    commitSingleton(editor, 'simulationConfig', 'maxThrust', sim.maxThrust, v);
  });
  appendNumberField(container, 'Max Gimbal Angle', sim.maxGimbalAngle, v => {
    commitSingleton(editor, 'simulationConfig', 'maxGimbalAngle', sim.maxGimbalAngle, v);
  });
  appendIntField(container, 'Sensor Count', sim.sensorCount, v => {
    commitSingleton(editor, 'simulationConfig', 'sensorCount', sim.sensorCount, v);
  });
  appendNumberField(container, 'Haste Bonus', sim.hasteBonus, v => {
    commitSingleton(editor, 'simulationConfig', 'hasteBonus', sim.hasteBonus, v);
  });
  appendIntField(container, 'Solver Iterations', sim.solverIterations, v => {
    commitSingleton(editor, 'simulationConfig', 'solverIterations', sim.solverIterations, v);
  });
  appendNumberField(container, 'Friction', sim.frictionMu, v => {
    commitSingleton(editor, 'simulationConfig', 'frictionMu', sim.frictionMu, v);
  });
  appendNumberField(container, 'Global Damping', sim.globalDamping, v => {
    commitSingleton(editor, 'simulationConfig', 'globalDamping', sim.globalDamping, v);
  });
  appendNumberField(container, 'Angular Damping', sim.angularDamping, v => {
    commitSingleton(editor, 'simulationConfig', 'angularDamping', sim.angularDamping, v);
  });

  const heading2 = document.createElement('h3');
  heading2.textContent = 'Reward Weights';
  heading2.style.marginTop = '12px';
  container.appendChild(heading2);

  appendNumberField(container, 'Position', rw.positionWeight, v => {
    commitSingleton(editor, 'rewardWeights', 'positionWeight', rw.positionWeight, v);
  });
  appendNumberField(container, 'Velocity', rw.velocityWeight, v => {
    commitSingleton(editor, 'rewardWeights', 'velocityWeight', rw.velocityWeight, v);
  });
  appendNumberField(container, 'Angle', rw.angleWeight, v => {
    commitSingleton(editor, 'rewardWeights', 'angleWeight', rw.angleWeight, v);
  });
  appendNumberField(container, 'Angular Vel', rw.angularVelocityWeight, v => {
    commitSingleton(editor, 'rewardWeights', 'angularVelocityWeight', rw.angularVelocityWeight, v);
  });
  appendNumberField(container, 'Control Effort', rw.controlEffortWeight, v => {
    commitSingleton(editor, 'rewardWeights', 'controlEffortWeight', rw.controlEffortWeight, v);
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
      appendNumberField(container, 'Penalty/Step', mod.penaltyPerStep, v => commitProp(editor, id, 'penaltyPerStep', mod.penaltyPerStep, v));
      appendNumberField(container, 'Influence Radius', mod.influenceRadius, v => commitProp(editor, id, 'influenceRadius', mod.influenceRadius, v));
      break;
    case 'checkpoint':
      appendNumberField(container, 'Radius', mod.radius, v => commitProp(editor, id, 'radius', mod.radius, v));
      appendIntField(container, 'Order', mod.order, v => commitProp(editor, id, 'order', mod.order, v));
      appendNumberField(container, 'Reward Bonus', mod.rewardBonus, v => commitProp(editor, id, 'rewardBonus', mod.rewardBonus, v));
      appendNumberField(container, 'Influence Radius', mod.influenceRadius, v => commitProp(editor, id, 'influenceRadius', mod.influenceRadius, v));
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
      appendNumberField(container, 'Influence Radius', mod.influenceRadius, v => commitProp(editor, id, 'influenceRadius', mod.influenceRadius, v));
      appendCheckbox(container, 'Lethal', mod.isLethal, v => commitProp(editor, id, 'isLethal', mod.isLethal, v));
      break;
    case 'attractor':
      appendNumberField(container, 'Half Extent X', mod.halfExtentX, v => commitProp(editor, id, 'halfExtentX', mod.halfExtentX, v));
      appendNumberField(container, 'Half Extent Y', mod.halfExtentY, v => commitProp(editor, id, 'halfExtentY', mod.halfExtentY, v));
      appendNumberField(container, 'Magnitude', mod.magnitude, v => commitProp(editor, id, 'magnitude', mod.magnitude, v));
      appendNumberField(container, 'Influence Radius', mod.influenceRadius, v => commitProp(editor, id, 'influenceRadius', mod.influenceRadius, v));
      appendNumberField(container, 'Contact Bonus', mod.contactBonus, v => commitProp(editor, id, 'contactBonus', mod.contactBonus, v));
      break;
  }
}

function appendIntField(container: HTMLElement, label: string, value: number, onChange: (v: number) => void): void {
  appendNumberField(container, label, value, onChange, true);
}

function appendNumberField(container: HTMLElement, label: string, value: number, onChange: (v: number) => void, integer = false): void {
  const row = document.createElement('div');
  row.className = 'prop-row';
  const lbl = document.createElement('label');
  lbl.textContent = label;
  lbl.style.cursor = 'ew-resize';
  lbl.style.userSelect = 'none';

  const input = document.createElement('input');
  input.type = 'text';
  input.inputMode = integer ? 'numeric' : 'decimal';
  input.value = integer ? Math.round(value).toString() : (Number.isInteger(value) ? value.toString() : value.toFixed(3));
  input.addEventListener('change', () => {
    const v = integer ? parseInt(input.value, 10) : parseFloat(input.value);
    if (!isNaN(v)) onChange(v);
  });

  // Drag-on-label to adjust value
  let dragStartX = 0;
  let dragStartValue = 0;

  const onDragMove = (e: MouseEvent) => {
    const dx = e.clientX - dragStartX;
    let newVal: number;
    if (integer) {
      const scale = Math.abs(dragStartValue) > 100 ? 0.5 : 0.2;
      newVal = Math.round(dragStartValue + dx * scale);
    } else {
      const scale = Math.abs(dragStartValue) > 10 ? 0.1 : 0.01;
      newVal = Math.round((dragStartValue + dx * scale) * 1000) / 1000;
    }
    input.value = newVal.toString();
    onChange(newVal);
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

function commitSingleton(editor: Editor, target: 'landingPad' | 'spawnArea' | 'simulationConfig' | 'rewardWeights', prop: string, oldVal: unknown, newVal: unknown): void {
  editor.history.execute(new UpdateSingletonCommand(target, prop, oldVal, newVal), editor.world);
  editor.markDirty();
}
