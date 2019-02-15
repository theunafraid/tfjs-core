import { Conv2DInfo, Conv3DInfo } from '../ops/conv_util';
import { Activation } from '../ops/fused_util';
import { DataId, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, Tensor5D } from '../tensor';
import { DataType, DataValues, Rank, ShapeMap } from '../types';
export interface BackendTimingInfo {
    kernelMs: number;
    getExtraProfileInfo?(): string;
}
export interface TensorStorage {
    read(dataId: DataId): Promise<DataValues>;
    readSync(dataId: DataId): DataValues;
    disposeData(dataId: DataId): void;
    write(dataId: DataId, values: DataValues): void;
    fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels: number): Tensor3D;
    register(dataId: DataId, shape: number[], dtype: DataType): void;
    memory(): {
        unreliable: boolean;
    };
}
export declare class DataStorage<T> {
    private dataMover;
    private data;
    constructor(dataMover: DataMover);
    get(dataId: DataId): T;
    set(dataId: DataId, value: T): void;
    has(dataId: DataId): boolean;
    delete(dataId: DataId): boolean;
}
export interface DataMover {
    moveData(dataId: DataId): void;
}
export interface BackendTimer {
    time(f: () => void): Promise<BackendTimingInfo>;
}
export declare class KernelBackend implements TensorStorage, BackendTimer {
    time(f: () => void): Promise<BackendTimingInfo>;
    read(dataId: object): Promise<DataValues>;
    readSync(dataId: object): DataValues;
    disposeData(dataId: object): void;
    write(dataId: object, values: DataValues): void;
    fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels: number): Tensor<Rank.R3>;
    register(dataId: object, shape: number[], dtype: DataType): void;
    memory(): {
        unreliable: boolean;
        reasons?: string[];
    };
    floatPrecision(): number;
    batchMatMul(a: Tensor3D, b: Tensor3D, transposeA: boolean, transposeB: boolean): Tensor3D;
    fusedBatchMatMul(a: Tensor3D, b: Tensor3D, transposeA: boolean, transposeB: boolean, bias?: Tensor, activation?: Activation): Tensor3D;
    slice<T extends Tensor>(x: T, begin: number[], size: number[]): T;
    stridedSlice<T extends Tensor>(x: T, begin: number[], end: number[], strides: number[], beginMask: number, endMask: number, ellipsisMask: number, newAxisMask: number, shrinkAxisMask: number): T;
    unstack(x: Tensor, axis: number): Tensor[];
    reverse<T extends Tensor>(a: T, axis: number[]): T;
    concat(tensors: Tensor[], axis: number): Tensor;
    neg<T extends Tensor>(a: T): T;
    add(a: Tensor, b: Tensor): Tensor;
    addN<T extends Tensor>(tensors: T[]): T;
    subtract(a: Tensor, b: Tensor): Tensor;
    multiply(a: Tensor, b: Tensor): Tensor;
    realDivide(a: Tensor, b: Tensor): Tensor;
    floorDiv(a: Tensor, b: Tensor): Tensor;
    sum(x: Tensor, axes: number[]): Tensor;
    prod(x: Tensor, axes: number[]): Tensor;
    unsortedSegmentSum<T extends Tensor>(x: T, segmentIds: Tensor1D, numSegments: number): Tensor;
    argMin(x: Tensor, axis: number): Tensor;
    argMax(x: Tensor, axis: number): Tensor;
    equal(a: Tensor, b: Tensor): Tensor;
    notEqual(a: Tensor, b: Tensor): Tensor;
    less(a: Tensor, b: Tensor): Tensor;
    lessEqual(a: Tensor, b: Tensor): Tensor;
    greater(a: Tensor, b: Tensor): Tensor;
    greaterEqual(a: Tensor, b: Tensor): Tensor;
    logicalNot<T extends Tensor>(a: T): T;
    logicalAnd(a: Tensor, b: Tensor): Tensor;
    logicalOr(a: Tensor, b: Tensor): Tensor;
    where(condition: Tensor): Tensor2D;
    select(condition: Tensor, a: Tensor, b: Tensor): Tensor;
    topk<T extends Tensor>(x: T, k: number, sorted: boolean): [T, T];
    min(x: Tensor, axes: number[]): Tensor;
    minimum(a: Tensor, b: Tensor): Tensor;
    mod(a: Tensor, b: Tensor): Tensor;
    max(x: Tensor, axes: number[]): Tensor;
    maximum(a: Tensor, b: Tensor): Tensor;
    all(x: Tensor, axes: number[]): Tensor;
    any(x: Tensor, axes: number[]): Tensor;
    squaredDifference(a: Tensor, b: Tensor): Tensor;
    ceil<T extends Tensor>(x: T): T;
    floor<T extends Tensor>(x: T): T;
    round<T extends Tensor>(x: T): T;
    sign<T extends Tensor>(x: T): T;
    pow<T extends Tensor>(a: T, b: Tensor): T;
    exp<T extends Tensor>(x: T): T;
    expm1<T extends Tensor>(x: T): T;
    log<T extends Tensor>(x: T): T;
    log1p<T extends Tensor>(x: T): T;
    sqrt<T extends Tensor>(x: T): T;
    rsqrt<T extends Tensor>(x: T): T;
    square<T extends Tensor>(x: T): T;
    reciprocal<T extends Tensor>(x: T): T;
    relu<T extends Tensor>(x: T): T;
    prelu<T extends Tensor>(x: T, a: T): T;
    elu<T extends Tensor>(x: T): T;
    eluDer<T extends Tensor>(dy: T, y: T): T;
    selu<T extends Tensor>(x: T): T;
    int<T extends Tensor>(x: T): T;
    clip<T extends Tensor>(x: T, min: number, max: number): T;
    abs<T extends Tensor>(x: T): T;
    complexAbs<T extends Tensor>(x: T): T;
    sigmoid<T extends Tensor>(x: T): T;
    softplus<T extends Tensor>(x: T): T;
    sin<T extends Tensor>(x: T): T;
    cos<T extends Tensor>(x: T): T;
    tan<T extends Tensor>(x: T): T;
    asin<T extends Tensor>(x: T): T;
    acos<T extends Tensor>(x: T): T;
    atan<T extends Tensor>(x: T): T;
    atan2<T extends Tensor>(a: T, b: T): T;
    sinh<T extends Tensor>(x: T): T;
    cosh<T extends Tensor>(x: T): T;
    tanh<T extends Tensor>(x: T): T;
    asinh<T extends Tensor>(x: T): T;
    acosh<T extends Tensor>(x: T): T;
    atanh<T extends Tensor>(x: T): T;
    erf<T extends Tensor>(x: T): T;
    step<T extends Tensor>(x: T, alpha: number): T;
    conv2d(x: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    conv2dDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    conv2dDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    depthwiseConv2D(input: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    depthwiseConv2DDerInput(dy: Tensor4D, filter: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    depthwiseConv2DDerFilter(x: Tensor4D, dY: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    conv3d(x: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D;
    conv3dDerInput(dy: Tensor5D, filter: Tensor5D, convInfo: Conv3DInfo): Tensor5D;
    conv3dDerFilter(x: Tensor5D, dY: Tensor5D, convInfo: Conv3DInfo): Tensor5D;
    maxPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    maxPoolBackprop(dy: Tensor4D, x: Tensor4D, y: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    avgPool(x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    avgPoolBackprop(dy: Tensor4D, x: Tensor4D, convInfo: Conv2DInfo): Tensor4D;
    reshape<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]): Tensor<R>;
    cast<T extends Tensor>(x: T, dtype: DataType): T;
    tile<T extends Tensor>(x: T, reps: number[]): T;
    pad<T extends Tensor>(x: T, paddings: Array<[number, number]>, constantValue: number): T;
    transpose<T extends Tensor>(x: T, perm: number[]): T;
    gather<T extends Tensor>(x: T, indices: Tensor1D, axis: number): T;
    gatherND(x: Tensor, indices: Tensor): Tensor;
    scatterND<R extends Rank>(indices: Tensor, updates: Tensor, shape: ShapeMap[R]): Tensor<R>;
    batchToSpaceND<T extends Tensor>(x: T, blockShape: number[], crops: number[][]): T;
    spaceToBatchND<T extends Tensor>(x: T, blockShape: number[], paddings: number[][]): T;
    resizeBilinear(x: Tensor4D, newHeight: number, newWidth: number, alignCorners: boolean): Tensor4D;
    resizeBilinearBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D;
    resizeNearestNeighbor(x: Tensor4D, newHEight: number, newWidth: number, alignCorners: boolean): Tensor4D;
    resizeNearestNeighborBackprop(dy: Tensor4D, x: Tensor4D, alignCorners: boolean): Tensor4D;
    batchNormalization(x: Tensor4D, mean: Tensor4D | Tensor1D, variance: Tensor4D | Tensor1D, varianceEpsilon: number, scale?: Tensor4D | Tensor1D, offset?: Tensor4D | Tensor1D): Tensor4D;
    localResponseNormalization4D(x: Tensor4D, radius: number, bias: number, alpha: number, beta: number): Tensor4D;
    LRNGrad(dy: Tensor4D, inputImage: Tensor4D, outputImage: Tensor4D, radius: number, bias: number, alpha: number, beta: number): Tensor4D;
    multinomial(logits: Tensor2D, normalized: boolean, numSamples: number, seed: number): Tensor2D;
    oneHot(indices: Tensor1D, depth: number, onValue: number, offValue: number): Tensor2D;
    cumsum(x: Tensor, axis: number, exclusive: boolean, reverse: boolean): Tensor;
    nonMaxSuppression(boxes: Tensor2D, scores: Tensor1D, maxOutputSize: number, iouThreshold: number, scoreThreshold?: number): Tensor1D;
    fft(x: Tensor2D): Tensor2D;
    ifft(x: Tensor2D): Tensor2D;
    complex<T extends Tensor>(real: T, imag: T): T;
    real<T extends Tensor>(input: T): T;
    imag<T extends Tensor>(input: T): T;
    cropAndResize(image: Tensor4D, boxes: Tensor2D, boxIndex: Tensor1D, cropSize: [number, number], method: 'bilinear' | 'nearest', extrapolationValue: number): Tensor4D;
    depthToSpace(x: Tensor4D, blockSize: number, dataFormat: string): Tensor4D;
    split<T extends Tensor>(value: T, sizeSplits: number[], axis: number): T[];
    sparseToDense<R extends Rank>(sparseIndices: Tensor, sparseValues: Tensor, outputShape: ShapeMap[R], defaultValue: Scalar): Tensor<R>;
    setDataMover(dataMover: DataMover): void;
    dispose(): void;
}