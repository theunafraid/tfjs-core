import { GPGPUContext } from './gpgpu_context';
import { GPGPUProgram } from './gpgpu_math';
export declare const LINEAR = "return x;";
export declare const LOG = "\n  vec4 result = log(x);\n  vec4 isNaN = vec4(lessThan(x, vec4(0.0)));\n  result.r = isNaN.r == 1.0 ? NAN : result.r;\n  result.g = isNaN.g == 1.0 ? NAN : result.g;\n  result.b = isNaN.b == 1.0 ? NAN : result.b;\n  result.a = isNaN.a == 1.0 ? NAN : result.a;\n\n  return result;\n";
export declare const RELU = "\n  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));\n\n  result.r = isNaN(x.r) ? x.r : result.r;\n  result.g = isNaN(x.g) ? x.g : result.g;\n  result.b = isNaN(x.b) ? x.b : result.b;\n  result.a = isNaN(x.a) ? x.a : result.a;\n\n  return result;\n";
export declare class UnaryOpPackedProgram implements GPGPUProgram {
    variableNames: string[];
    userCode: string;
    outputShape: number[];
    usesPackedTextures: boolean;
    startLoc: WebGLUniformLocation;
    constructor(aShape: number[], opSnippet: string);
    getCustomSetupFunc(): (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void;
}