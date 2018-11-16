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

import {Conv2DInfo} from '../../ops/conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class DepthwiseConv2DPackedProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;

    // const xNumRows = convInfo.inHeight;
    // const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    // const dilationHeight = convInfo.dilationHeight;
    // const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    let mainLoop = ``;

    let combinedPatchWidth = filterWidth + 1;
    let texelsAcross = Math.ceil(combinedPatchWidth / 2);

    for(let r=0; r<filterHeight; r++) {
      for(let c=0; c<texelsAcross; c++) {
        mainLoop += `
          vec4 xTexelR${r}C${c * 2} = getX(batch, xRCorner + ${r}, xCCorner + ${c * 2}, d1);
        `;
      }
    }

    /*
    for 3x3, we end up with:
    xTexelR0C0
    xTexelR0C2
    xTexelR1C0
    xTexelR1C2
    xTexelR2C0
    xTexelR2C2
     */

    for(let r=0; r<filterHeight; r++) {
      for(let c=0; c<filterWidth; c++) {
        mainLoop += `
          vec4 wTexelR${r}C${c} = getW(${r}, ${c}, d1, q);
        `;
      }
    }

    mainLoop += `vec4 xTexel = vec4(0.); vec4 wTexel = vec4(0.);`;

    for(let r=0; r<filterHeight; r++) {
      for(let c=0; c<filterWidth; c++) {
        let currentXTexel = `xTexelR${r}C${c}`;
        let xTexel = `xTexel = ${currentXTexel}`;

        if(c % 2 !== 0) {
          currentXTexel = `xTexelR${r}C${c - 1}`;
          let nextXTexel = `xTexelR${r}C${c + 1}`;
          xTexel = `xTexel = vec4(${currentXTexel}.zw, ${nextXTexel}.xy)`;
        }

        let wTexel = `wTexel = vec4(wTexelR${r}C${c}.xy, wTexelR${r}C${c}.xy)`;

        mainLoop += `
          ${xTexel};
          ${wTexel};
          result += dot(xTexel, wTexel);
        `;
      }
    }

    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 result = vec4(0.);

        ${mainLoop}

        setOutput(result);
      }
    `;
  }
}

/*
leftovers

implement out of bounds condition

AFTER MOBILENET WORKS

dilation
 */