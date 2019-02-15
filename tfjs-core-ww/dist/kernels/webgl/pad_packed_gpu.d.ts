import { GPGPUProgram } from './gpgpu_math';
export declare class PadPackedProgram implements GPGPUProgram {
    variableNames: string[];
    usesPackedTextures: boolean;
    outputShape: number[];
    userCode: string;
    constructor(xShape: number[], paddings: Array<[number, number]>, constantValue: number);
}