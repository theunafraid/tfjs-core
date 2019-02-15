import { ReduceInfo } from '../../ops/reduce_util';
import { GPGPUProgram } from './gpgpu_math';
export declare class ReduceProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(reduceInfo: ReduceInfo, reduceType: 'all' | 'any' | 'max' | 'min' | 'sum' | 'prod');
}