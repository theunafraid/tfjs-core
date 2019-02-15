import { Conv2DInfo, Conv3DInfo } from '../../ops/conv_util';
import { GPGPUProgram } from './gpgpu_math';
export declare class Conv2DDerFilterProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(convInfo: Conv2DInfo);
}
export declare class Conv2DDerInputProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(convInfo: Conv2DInfo);
}
export declare class Conv3DDerFilterProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(convInfo: Conv3DInfo);
}
export declare class Conv3DDerInputProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(convInfo: Conv3DInfo);
}
