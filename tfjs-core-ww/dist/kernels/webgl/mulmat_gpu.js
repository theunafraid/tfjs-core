"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var MatMulProgram = (function () {
    function MatMulProgram(aShape, bShape, transposeA, transposeB, addBias, activation) {
        if (transposeA === void 0) { transposeA = false; }
        if (transposeB === void 0) { transposeB = false; }
        if (addBias === void 0) { addBias = false; }
        if (activation === void 0) { activation = null; }
        this.variableNames = ['matrixA', 'matrixB'];
        var batchSize = aShape[0];
        var outerShapeA = transposeA ? aShape[2] : aShape[1];
        var outerShapeB = transposeB ? bShape[1] : bShape[2];
        var sharedDim = transposeA ? aShape[1] : aShape[2];
        this.outputShape = [batchSize, outerShapeA, outerShapeB];
        var aSnippetFromOffset = function (vec4Offset, indexVar) {
            return transposeA ? "batch, " + indexVar + " + " + vec4Offset + ", aRow" :
                "batch, aRow, " + indexVar + " + " + vec4Offset;
        };
        var bSnippetFromOffset = function (vec4Offset, indexVar) {
            return transposeB ? "batch, bCol, " + indexVar + " + " + vec4Offset :
                "batch, " + indexVar + " + " + vec4Offset + ", bCol";
        };
        var sharedDimNearestVec4 = Math.floor(sharedDim / 4) * 4;
        var sharedDimVec4Remainder = sharedDim % 4;
        var activationSnippet = '', applyActivationSnippet = '';
        if (activation) {
            activationSnippet = "float activation(float x) {\n        " + activation + "\n      }";
            applyActivationSnippet = "result = activation(result);";
        }
        var addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
        if (addBias) {
            this.variableNames.push('bias');
        }
        this.userCode = "\n      " + activationSnippet + "\n\n      float dotARowBCol(int batch, int aRow, int bCol) {\n        float result = 0.0;\n        for (int i = 0; i < " + sharedDimNearestVec4 + "; i += 4) {\n          vec4 a = vec4(\n            getMatrixA(" + aSnippetFromOffset(0, 'i') + "),\n            getMatrixA(" + aSnippetFromOffset(1, 'i') + "),\n            getMatrixA(" + aSnippetFromOffset(2, 'i') + "),\n            getMatrixA(" + aSnippetFromOffset(3, 'i') + ")\n          );\n          vec4 b = vec4(\n            getMatrixB(" + bSnippetFromOffset(0, 'i') + "),\n            getMatrixB(" + bSnippetFromOffset(1, 'i') + "),\n            getMatrixB(" + bSnippetFromOffset(2, 'i') + "),\n            getMatrixB(" + bSnippetFromOffset(3, 'i') + ")\n          );\n\n          result += dot(a, b);\n        }\n\n        if (" + (sharedDimVec4Remainder === 1) + ") {\n          result += getMatrixA(" + aSnippetFromOffset(0, sharedDimNearestVec4) + ") *\n            getMatrixB(" + bSnippetFromOffset(0, sharedDimNearestVec4) + ");\n        } else if (" + (sharedDimVec4Remainder === 2) + ") {\n          vec2 a = vec2(\n            getMatrixA(" + aSnippetFromOffset(0, sharedDimNearestVec4) + "),\n            getMatrixA(" + aSnippetFromOffset(1, sharedDimNearestVec4) + ")\n          );\n          vec2 b = vec2(\n            getMatrixB(" + bSnippetFromOffset(0, sharedDimNearestVec4) + "),\n            getMatrixB(" + bSnippetFromOffset(1, sharedDimNearestVec4) + ")\n          );\n          result += dot(a, b);\n        } else if (" + (sharedDimVec4Remainder === 3) + ") {\n          vec3 a = vec3(\n            getMatrixA(" + aSnippetFromOffset(0, sharedDimNearestVec4) + "),\n            getMatrixA(" + aSnippetFromOffset(1, sharedDimNearestVec4) + "),\n            getMatrixA(" + aSnippetFromOffset(2, sharedDimNearestVec4) + ")\n          );\n          vec3 b = vec3(\n            getMatrixB(" + bSnippetFromOffset(0, sharedDimNearestVec4) + "),\n            getMatrixB(" + bSnippetFromOffset(1, sharedDimNearestVec4) + "),\n            getMatrixB(" + bSnippetFromOffset(2, sharedDimNearestVec4) + ")\n          );\n          result += dot(a, b);\n        }\n\n        return result;\n      }\n\n      void main() {\n        ivec3 resBRC = getOutputCoords();\n        float result = dotARowBCol(resBRC.x, resBRC.y, resBRC.z);\n\n        " + addBiasSnippet + "\n\n        " + applyActivationSnippet + "\n\n        setOutput(result);\n      }\n    ";
    }
    return MatMulProgram;
}());
exports.MatMulProgram = MatMulProgram;
//# sourceMappingURL=mulmat_gpu.js.map