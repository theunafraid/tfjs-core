/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

interface LRUItem<T> {
  newer: LRUItem<T>;
  older: LRUItem<T>;
  key: T;
}

export class LRUCache<T extends object> {
  private map: WeakMap<T, LRUItem<T>> = new WeakMap();
  private newest: LRUItem<T> = null;
  private oldest: LRUItem<T> = null;

  touch(key: T) {
    if (!this.map.has(key)) {
      // This element was not in the cache before.
      const prevNewest = this.newest;
      this.newest = {older: prevNewest, key, newer: null};
      if (prevNewest != null) {
        prevNewest.newer = this.newest;
      } else {
        this.oldest = this.newest;
      }
      this.map.set(key, this.newest);
      return;
    }
    // This element was in the cache before.
    const item = this.map.get(key);
    if (item === this.newest) {
      // No need to do any updates. We touched the newest.
      return;
    }
    if (item.older) {
      item.older.newer = item.newer;
    }
    if (item.newer) {
      item.newer.older = item.older;
    }
    if (this.oldest === item) {
      this.oldest = item.newer;
    }
    item.older = this.newest;
    item.newer = null;
    if (this.newest !== item) {
      const prevNewest = this.newest;
      this.newest = item;
      prevNewest.newer = this.newest;
    }
  }

  pop(): T {
    if (this.oldest == null) {
      return null;
    }
    const key = this.oldest.key;
    this.remove(key);
    return key;
  }

  remove(key: T) {
    const item = this.map.get(key);
    if (item == null) {
      return;
    }
    if (this.newest === item) {
      this.newest = item.older;
    }
    if (this.oldest === item) {
      this.oldest = item.newer;
    }
    if (item.older) {
      item.older.newer = item.newer;
    }
    if (item.newer) {
      item.newer.older = item.older;
    }
    this.map.delete(key);
  }
}
