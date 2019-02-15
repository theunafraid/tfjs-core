import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function stridedSlice_(x: Tensor | TensorLike, begin: number[], end: number[], strides: number[], beginMask?: number, endMask?: number, ellipsisMask?: number, newAxisMask?: number, shrinkAxisMask?: number): Tensor;
export declare const stridedSlice: typeof stridedSlice_;
export {};
