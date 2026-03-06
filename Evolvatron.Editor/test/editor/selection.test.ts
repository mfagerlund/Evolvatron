import { describe, it, expect, beforeEach } from 'vitest';
import { Selection } from '../../src/editor/selection';

let sel: Selection;

beforeEach(() => {
  sel = new Selection();
});

describe('Selection', () => {
  it('starts empty', () => {
    expect(sel.size).toBe(0);
    expect(sel.toArray()).toEqual([]);
    expect(sel.has('anything')).toBe(false);
  });

  describe('set', () => {
    it('selects a single item, clearing previous', () => {
      sel.add('a');
      sel.add('b');
      sel.set('c');
      expect(sel.size).toBe(1);
      expect(sel.has('c')).toBe(true);
      expect(sel.has('a')).toBe(false);
    });
  });

  describe('toggle', () => {
    it('adds item if not present', () => {
      sel.toggle('x');
      expect(sel.has('x')).toBe(true);
    });

    it('removes item if already present', () => {
      sel.add('x');
      sel.toggle('x');
      expect(sel.has('x')).toBe(false);
    });
  });

  describe('add / addMany', () => {
    it('add does not duplicate', () => {
      sel.add('a');
      sel.add('a');
      expect(sel.size).toBe(1);
    });

    it('addMany adds multiple items', () => {
      sel.addMany(['a', 'b', 'c']);
      expect(sel.size).toBe(3);
      expect(sel.has('b')).toBe(true);
    });
  });

  describe('remove', () => {
    it('removes an existing item', () => {
      sel.add('a');
      sel.remove('a');
      expect(sel.has('a')).toBe(false);
      expect(sel.size).toBe(0);
    });

    it('does nothing for non-existent item', () => {
      sel.remove('nonexistent');
      expect(sel.size).toBe(0);
    });
  });

  describe('clear', () => {
    it('removes all items', () => {
      sel.addMany(['a', 'b', 'landingPad']);
      sel.clear();
      expect(sel.size).toBe(0);
    });
  });

  describe('moduleIds', () => {
    it('filters out landingPad and spawnArea', () => {
      sel.addMany(['mod_1', 'landingPad', 'mod_2', 'spawnArea']);
      expect(sel.moduleIds).toEqual(['mod_1', 'mod_2']);
    });

    it('returns empty array when only singletons selected', () => {
      sel.addMany(['landingPad', 'spawnArea']);
      expect(sel.moduleIds).toEqual([]);
    });

    it('returns all ids when no singletons', () => {
      sel.addMany(['a', 'b']);
      expect(sel.moduleIds).toEqual(['a', 'b']);
    });
  });

  describe('hasSingletons', () => {
    it('returns false when no singletons', () => {
      sel.add('mod_1');
      expect(sel.hasSingletons).toBe(false);
    });

    it('returns true when landingPad selected', () => {
      sel.add('landingPad');
      expect(sel.hasSingletons).toBe(true);
    });

    it('returns true when spawnArea selected', () => {
      sel.add('spawnArea');
      expect(sel.hasSingletons).toBe(true);
    });
  });

  describe('ids (ReadonlySet)', () => {
    it('returns a set view of current selection', () => {
      sel.addMany(['a', 'b']);
      const ids = sel.ids;
      expect(ids.has('a')).toBe(true);
      expect(ids.has('b')).toBe(true);
      expect(ids.size).toBe(2);
    });
  });
});
