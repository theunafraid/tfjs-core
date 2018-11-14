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

import * as broadcast_util from '../../ops/broadcast_util';
import {getCoordsDataType} from './shader_compiler';
import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';

export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const MUL = 'return dot(a, b);';
export const DIV = `return a / b;`;

export const MAX = `return max(a, b);`;

export const MIN = `return min(a, b);`;

const dims = ['rc.x', 'rc.y', 'rc.z', 'rc.w'];

export class BinaryOpPackedProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[];
  userCode: string;
  usesPackedTextures = true;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.outputShape =
        broadcast_util.assertAndGetBroadcastShape(aShape, bShape);

    const rank = this.outputShape.length;

    const dtype = getCoordsDataType(rank);

    let aSample = `vec4 aSample = getA(${getSourceCoords(aShape.length)});`;
    let a = `vec4 a = aSample;`;
    if(aShape.length < rank) {
      const sourceCoordsA = dims.slice(0, rank).slice(-aShape.length).join(',');
      aSample = `vec4 aSample = getA(${sourceCoordsA});`;

      if(aShape.length < 2) {
        a = `
          vec4 a = vec4(aSample.xy, aSample.xy);
        `;
        if(aShape.length === 0) {
          aSample = `float aSample = getA();`;
          a = `vec4 a = vec4(aSample);`;
        }
      }
    }

    let bSample = `vec4 bSample = getB(${getSourceCoords(bShape.length)});`;
    let b = 'vec4 b = bSample;';
    if(bShape.length < rank) { // this means b's rank is smaller
      const sourceCoordsB = dims.slice(0, rank).slice(-bShape.length).join(',');
      bSample = `vec4 bSample = getB(${sourceCoordsB});`;

      if(bShape.length < 2) {
        b = `
          vec4 b = vec4(bSample.xy, bSample.xy);
        `;
        if(bShape.length === 0) {
          bSample = `float bSample = getB();`;
          b = `vec4 b = vec4(bSample);`;
        }
      }
    }

    this.userCode = `
      uniform float NAN;
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        ${dtype} rc = getOutputCoords();

        ${aSample}
        ${a}

        ${bSample}
        ${b}

        setOutput(binaryOperation(a, b));
      }
    `;
  }

  getCustomSetupFunc() {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.startLoc == null) {
        this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'NAN');
        if (this.startLoc == null) {
          // This means the compiler has optimized and realized it doesn't need
          // the uniform.
          return;
        }
      }
      gpgpu.gl.uniform1f(this.startLoc, NaN);
    };
  }
}

function getSourceCoords(rank: number): string {
  if(rank === 1) {
    return 'rc';
  }

  let coords = '';
  for (let i = 0; i < rank; i++) {
    coords += dims[i];
    if (i < rank - 1) {
      coords += ',';
    }
  }
  return coords;
}