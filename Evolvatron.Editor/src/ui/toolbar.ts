export interface ToolbarCallbacks {
  onToolSelect: (tool: string) => void;
  onNew: () => void;
  onLoad: () => void;
  onSave: () => void;
  onExport: () => void;
}

export function setupToolbar(callbacks: ToolbarCallbacks): void {
  const toolButtons: Record<string, string> = {
    'btn-select': 'select',
    'btn-obstacle': 'obstacle',
    'btn-checkpoint': 'checkpoint',
    'btn-speed-zone': 'speedZone',
    'btn-danger-zone': 'dangerZone',
  };

  for (const [btnId, tool] of Object.entries(toolButtons)) {
    const btn = document.getElementById(btnId);
    if (!btn) continue;
    btn.dataset.tool = tool;
    btn.addEventListener('click', () => {
      callbacks.onToolSelect(tool);
      for (const [id] of Object.entries(toolButtons)) {
        document.getElementById(id)?.classList.remove('active');
      }
      btn.classList.add('active');
    });
  }

  document.getElementById('btn-new')?.addEventListener('click', callbacks.onNew);
  document.getElementById('btn-load')?.addEventListener('click', callbacks.onLoad);
  document.getElementById('btn-save')?.addEventListener('click', callbacks.onSave);
  document.getElementById('btn-export')?.addEventListener('click', callbacks.onExport);
}
