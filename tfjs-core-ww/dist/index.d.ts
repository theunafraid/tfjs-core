import './kernels/backend_webgl';
import './kernels/backend_cpu';
import { nextFrame } from './browser_util';
import * as environment from './environment';
import { deprecationWarn, disableDeprecationWarnings, enableProdMode } from './environment';
import * as io from './io/io';
import * as math from './math';
import * as browser from './ops/browser';
import * as serialization from './serialization';
import * as test_util from './test_util';
import * as util from './util';
import { version } from './version';
import * as webgl from './webgl';
export { InferenceModel, ModelPredictConfig } from './model_types';
export { AdadeltaOptimizer } from './optimizers/adadelta_optimizer';
export { AdagradOptimizer } from './optimizers/adagrad_optimizer';
export { AdamOptimizer } from './optimizers/adam_optimizer';
export { AdamaxOptimizer } from './optimizers/adamax_optimizer';
export { MomentumOptimizer } from './optimizers/momentum_optimizer';
export { Optimizer } from './optimizers/optimizer';
export { RMSPropOptimizer } from './optimizers/rmsprop_optimizer';
export { SGDOptimizer } from './optimizers/sgd_optimizer';
export { Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, TensorBuffer, variable, Variable } from './tensor';
export { NamedTensorMap } from './tensor_types';
export { DataType, DataTypeMap, DataValues, Rank, ShapeMap } from './types';
export * from './ops/ops';
export { LSTMCellFunc } from './ops/lstm';
export { Reduction } from './ops/loss_ops';
export * from './train';
export * from './globals';
export { Features } from './environment_util';
export { TimingInfo } from './engine';
export { ENV, Environment } from './environment';
export declare const setBackend: typeof environment.Environment.setBackend;
export declare const getBackend: typeof environment.Environment.getBackend;
export declare const disposeVariables: typeof environment.Environment.disposeVariables;
export declare const memory: typeof environment.Environment.memory;
export { version as version_core };
export { nextFrame, enableProdMode, disableDeprecationWarnings, deprecationWarn };
export { browser, environment, io, math, serialization, test_util, util, webgl };
export { KernelBackend, BackendTimingInfo, DataMover, DataStorage } from './kernels/backend';