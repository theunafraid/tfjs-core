import { Tensor } from '../tensor';
export interface SegOpInfo {
    windowSize: number;
    batchSize: number;
    inSize: number;
    numSegments: number;
}
export declare function segOpComputeOptimalWindowSize(inSize: number, numSegments: number): number;
export declare function computeOutShape(aShape: number[], axis: number, numSegments: number): number[];
export interface GatherOpShapeInfo {
    batchSize: number;
    sliceSize: number;
    dimSize: number;
    outputShape: number[];
}
export declare function collectGatherOpShapeInfo(x: Tensor, indices: Tensor, axis: number): GatherOpShapeInfo;
