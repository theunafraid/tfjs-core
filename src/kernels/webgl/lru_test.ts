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

import {LRUCache} from './lru';

describe('LRU', () => {
  it('pop from an empty cache', () => {
    const cache = new LRUCache();
    expect(cache.pop()).toBeNull();
  });

  it('remove an element that was not there', () => {
    const cache = new LRUCache();
    expect(cache.remove({})).toBeUndefined();
  });

  it('add a single element and pop', () => {
    const cache = new LRUCache();
    const a = {};
    cache.touch(a);
    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBeNull();
  });

  it('add a single element and remove', () => {
    const cache = new LRUCache();
    const a = {};
    cache.touch(a);
    cache.remove(a);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and pop all three', () => {
    const cache = new LRUCache();
    const a = {};
    const b = {};
    const c = {};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);
    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and touch the oldest', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Touch the oldest.
    cache.touch(a);

    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and touch the newest', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Touch the newest.
    cache.touch(c);

    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and touch the middle', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Touch the middle.
    cache.touch(b);

    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and remove first', () => {
    const cache = new LRUCache();
    const a = {};
    const b = {};
    const c = {};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Remove first.
    cache.remove(a);

    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and remove last', () => {
    const cache = new LRUCache();
    const a = {};
    const b = {};
    const c = {};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Remove last.
    cache.remove(c);

    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBeNull();
  });

  it('add 3 elements and remove middle', () => {
    const cache = new LRUCache();
    const a = {};
    const b = {};
    const c = {};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);

    // Remove middle.
    cache.remove(b);

    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBeNull();
  });

  it('add 4 elements and touch in reverse', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    const d = {id: 4};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);
    cache.touch(d);

    // Touch in reverse order.
    cache.touch(d);
    cache.touch(c);
    cache.touch(b);
    cache.touch(a);

    expect(cache.pop()).toBe(d);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBeNull();
  });

  it('add 4 elements and touch in a different order', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    const d = {id: 4};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);
    cache.touch(d);

    // Touch in permuted order.
    cache.touch(b);
    cache.touch(d);

    expect(cache.pop()).toBe(a);
    expect(cache.pop()).toBe(c);
    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(d);
    expect(cache.pop()).toBeNull();
  });

  it('add 4 elements and remove in a different order', () => {
    const cache = new LRUCache();
    const a = {id: 1};
    const b = {id: 2};
    const c = {id: 3};
    const d = {id: 4};
    cache.touch(a);
    cache.touch(b);
    cache.touch(c);
    cache.touch(d);

    // Remove in a different order.
    cache.remove(c);
    cache.remove(a);

    expect(cache.pop()).toBe(b);
    expect(cache.pop()).toBe(d);
    expect(cache.pop()).toBeNull();
  });
});
