import { GPGPUContext } from './gpgpu_context';
import { GPGPUProgram } from './gpgpu_math';
export declare class ClipPackedProgram implements GPGPUProgram {
    variableNames: string[];
    usesPackedTextures: boolean;
    userCode: string;
    outputShape: number[];
    minLoc: WebGLUniformLocation;
    maxLoc: WebGLUniformLocation;
    constructor(aShape: number[]);
    getCustomSetupFunc(min: number, max: number): (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void;
}
