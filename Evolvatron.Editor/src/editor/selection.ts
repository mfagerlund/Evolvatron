import type { SelectableId } from '../model/types';

export class Selection {
  private items = new Set<SelectableId>();

  get size(): number {
    return this.items.size;
  }

  get ids(): ReadonlySet<SelectableId> {
    return this.items;
  }

  has(id: SelectableId): boolean {
    return this.items.has(id);
  }

  set(id: SelectableId): void {
    this.items.clear();
    this.items.add(id);
  }

  toggle(id: SelectableId): void {
    if (this.items.has(id)) {
      this.items.delete(id);
    } else {
      this.items.add(id);
    }
  }

  add(id: SelectableId): void {
    this.items.add(id);
  }

  addMany(ids: Iterable<SelectableId>): void {
    for (const id of ids) this.items.add(id);
  }

  remove(id: SelectableId): void {
    this.items.delete(id);
  }

  clear(): void {
    this.items.clear();
  }

  toArray(): SelectableId[] {
    return [...this.items];
  }

  get moduleIds(): string[] {
    return this.toArray().filter(
      (id): id is string => id !== 'landingPad' && id !== 'spawnArea'
    );
  }

  get hasSingletons(): boolean {
    return this.items.has('landingPad') || this.items.has('spawnArea');
  }
}
