import type { SelectableId } from '../model/types';

const EASE_IN_SPEED = 10;
const EASE_OUT_SPEED = 4;
const EPSILON = 0.003;

export class HoverState {
  private intensities = new Map<SelectableId, number>();
  private target: SelectableId | null = null;

  setTarget(id: SelectableId | null): void {
    this.target = id;
  }

  /** Tick animation. Returns true if any animation is in progress. */
  update(dt: number): boolean {
    let animating = false;

    if (this.target !== null) {
      const current = this.intensities.get(this.target) ?? 0;
      if (current < 1) {
        const next = current + (1 - current) * (1 - Math.exp(-EASE_IN_SPEED * dt));
        this.intensities.set(this.target, next >= 1 - EPSILON ? 1 : next);
        if (next < 1 - EPSILON) animating = true;
      }
    }

    for (const [id, intensity] of this.intensities) {
      if (id === this.target) continue;
      const next = intensity * Math.exp(-EASE_OUT_SPEED * dt);
      if (next < EPSILON) {
        this.intensities.delete(id);
      } else {
        this.intensities.set(id, next);
        animating = true;
      }
    }

    return animating;
  }

  getIntensity(id: SelectableId): number {
    return this.intensities.get(id) ?? 0;
  }

  entries(): IterableIterator<[SelectableId, number]> {
    return this.intensities.entries();
  }
}
