import type { World } from '../model/types';
import { serialize, deserialize } from './serialize';

const DB_NAME = 'evolvatron-editor';
const STORE_NAME = 'autosave';
const KEY = 'world';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function autosave(world: World): Promise<void> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  tx.objectStore(STORE_NAME).put(serialize(world), KEY);
  db.close();
}

export async function loadAutosave(): Promise<World | null> {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(KEY);
      req.onsuccess = () => {
        db.close();
        if (typeof req.result === 'string') {
          try {
            resolve(deserialize(req.result));
          } catch {
            resolve(null);
          }
        } else {
          resolve(null);
        }
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    });
  } catch {
    return null;
  }
}
