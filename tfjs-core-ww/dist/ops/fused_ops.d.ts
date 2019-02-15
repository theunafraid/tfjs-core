import { Tensor } from '../tensor';
import { TensorLike } from '../types';
import { Activation } from './fused_util';
declare function matMul_<T extends Tensor>(a: T | TensorLike, b: T | TensorLike, transposeA?: boolean, transposeB?: boolean, bias?: Tensor | TensorLike, activation?: Activation): T;
export declare const matMul: typeof matMul_;
export { Activation };
