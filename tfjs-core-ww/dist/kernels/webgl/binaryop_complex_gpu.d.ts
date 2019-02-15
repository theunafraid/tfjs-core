import { GPGPUProgram } from './gpgpu_math';
export declare const COMPLEX_MULTIPLY: {
    REAL: string;
    IMAG: string;
};
export declare class BinaryOpComplexProgram implements GPGPUProgram {
    variableNames: string[];
    userCode: string;
    outputShape: number[];
    constructor(op: string, aShape: number[], bShape: number[]);
}
