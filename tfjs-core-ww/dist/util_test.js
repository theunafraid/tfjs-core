"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_util_env_1 = require("./tensor_util_env");
var util = require("./util");
var ops_1 = require("./ops/ops");
describe('Util', function () {
    it('Correctly gets size from shape', function () {
        expect(util.sizeFromShape([1, 2, 3, 4])).toEqual(24);
    });
    it('Correctly identifies scalars', function () {
        expect(util.isScalarShape([])).toBe(true);
        expect(util.isScalarShape([1, 2])).toBe(false);
        expect(util.isScalarShape([1])).toBe(false);
    });
    it('Number arrays equal', function () {
        expect(util.arraysEqual([1, 2, 3, 6], [1, 2, 3, 6])).toBe(true);
        expect(util.arraysEqual([1, 2], [1, 2, 3])).toBe(false);
        expect(util.arraysEqual([1, 2, 5], [1, 2])).toBe(false);
    });
    it('Is integer', function () {
        expect(util.isInt(0.5)).toBe(false);
        expect(util.isInt(1)).toBe(true);
    });
    it('Size to squarish shape (perfect square)', function () {
        expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
    });
    it('Size to squarish shape (prime number)', function () {
        expect(util.sizeToSquarishShape(11)).toEqual([4, 3]);
    });
    it('Size to squarish shape (almost square)', function () {
        expect(util.sizeToSquarishShape(35)).toEqual([6, 6]);
    });
    it('Size of 1 to squarish shape', function () {
        expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
    });
    it('infer shape single number', function () {
        expect(tensor_util_env_1.inferShape(4)).toEqual([]);
    });
    it('infer shape 1d array', function () {
        expect(tensor_util_env_1.inferShape([1, 2, 5])).toEqual([3]);
    });
    it('infer shape 2d array', function () {
        expect(tensor_util_env_1.inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
    });
    it('infer shape 3d array', function () {
        var a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
        expect(tensor_util_env_1.inferShape(a)).toEqual([2, 3, 2]);
    });
    it('infer shape 4d array', function () {
        var a = [
            [[[1], [2]], [[2], [3]], [[5], [6]]],
            [[[5], [6]], [[4], [5]], [[1], [2]]]
        ];
        expect(tensor_util_env_1.inferShape(a)).toEqual([2, 3, 2, 1]);
    });
    it('infer shape of typed array', function () {
        var a = new Float32Array([1, 2, 3, 4, 5]);
        expect(tensor_util_env_1.inferShape(a)).toEqual([5]);
    });
});
describe('util.flatten', function () {
    it('nested number arrays', function () {
        expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
        expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
            1, 2, 3, 4, 5, 6, 7, 8
        ]);
        expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
    });
    it('nested string arrays', function () {
        expect(util.flatten([['a', 'b'], ['c', [['d']]]])).toEqual([
            'a', 'b', 'c', 'd'
        ]);
        expect(util.flatten([['a', ['b']], ['c', [['d']], 'e']])).toEqual([
            'a', 'b', 'c', 'd', 'e'
        ]);
    });
    it('mixed TypedArray and number[]', function () {
        var data = [new Float32Array([1, 2]), 3, [4, 5, new Float32Array([6, 7])]];
        expect(util.flatten(data)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    });
});
describe('util.bytesFromStringArray', function () {
    it('count each character as 2 bytes', function () {
        expect(util.bytesFromStringArray(['a', 'bb', 'ccc'])).toBe(6 * 2);
        expect(util.bytesFromStringArray(['a', 'bb', 'cccddd'])).toBe(9 * 2);
        expect(util.bytesFromStringArray(['даниел'])).toBe(6 * 2);
    });
});
describe('util.inferDtype', function () {
    it('a single string => string', function () {
        expect(util.inferDtype('hello')).toBe('string');
    });
    it('a single boolean => bool', function () {
        expect(util.inferDtype(true)).toBe('bool');
        expect(util.inferDtype(false)).toBe('bool');
    });
    it('a single number => float32', function () {
        expect(util.inferDtype(0)).toBe('float32');
        expect(util.inferDtype(34)).toBe('float32');
    });
    it('a list of strings => string', function () {
        expect(util.inferDtype(['a', 'b', 'c'])).toBe('string');
        expect(util.inferDtype([
            [['a']], [['b']], [['c']], [['d']]
        ])).toBe('string');
    });
    it('a list of bools => float32', function () {
        expect(util.inferDtype([false, true, false])).toBe('bool');
        expect(util.inferDtype([
            [[true]], [[false]], [[true]], [[true]]
        ])).toBe('bool');
    });
    it('a list of numbers => float32', function () {
        expect(util.inferDtype([0, 1, 2])).toBe('float32');
        expect(util.inferDtype([[[0]], [[1]], [[2]], [[3]]])).toBe('float32');
    });
});
describe('util.repeatedTry', function () {
    it('resolves', function (doneFn) {
        var counter = 0;
        var checkFn = function () {
            counter++;
            if (counter === 2) {
                return true;
            }
            return false;
        };
        util.repeatedTry(checkFn).then(doneFn).catch(function () {
            throw new Error('Rejected backoff.');
        });
    });
    it('rejects', function (doneFn) {
        var checkFn = function () { return false; };
        util.repeatedTry(checkFn, function () { return 0; }, 5)
            .then(function () {
            throw new Error('Backoff resolved');
        })
            .catch(doneFn);
    });
});
describe('util.inferFromImplicitShape', function () {
    it('empty shape', function () {
        var result = util.inferFromImplicitShape([], 0);
        expect(result).toEqual([]);
    });
    it('[2, 3, 4] -> [2, 3, 4]', function () {
        var result = util.inferFromImplicitShape([2, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, 4] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([2, -1, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[-1, 3, 4] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([-1, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, 3, -1] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([2, 3, -1], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, -1] throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, -1, -1], 24); }).toThrowError();
    });
    it('[2, 3, -1] size=13 throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, 3, -1], 13); }).toThrowError();
    });
    it('[2, 3, 4] size=25 (should be 24) throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, 3, 4], 25); }).toThrowError();
    });
});
describe('util parseAxisParam', function () {
    it('axis=null returns no axes for scalar', function () {
        var axis = null;
        var shape = [];
        expect(util.parseAxisParam(axis, shape)).toEqual([]);
    });
    it('axis=null returns 0 axis for Tensor1D', function () {
        var axis = null;
        var shape = [4];
        expect(util.parseAxisParam(axis, shape)).toEqual([0]);
    });
    it('axis=null returns all axes for Tensor3D', function () {
        var axis = null;
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 1, 2]);
    });
    it('axis as a single number', function () {
        var axis = 1;
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([1]);
    });
    it('axis as single negative number', function () {
        var axis = -1;
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2]);
        var axis2 = -2;
        expect(util.parseAxisParam(axis2, shape)).toEqual([1]);
        var axis3 = -3;
        expect(util.parseAxisParam(axis3, shape)).toEqual([0]);
    });
    it('axis as list of negative numbers', function () {
        var axis = [-1, -3];
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2, 0]);
    });
    it('axis as list of positive numbers', function () {
        var axis = [0, 2];
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis as combo of positive and negative numbers', function () {
        var axis = [0, -1];
        var shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis out of range throws error', function () {
        var axis = -4;
        var shape = [3, 1, 2];
        expect(function () { return util.parseAxisParam(axis, shape); }).toThrowError();
        var axis2 = 4;
        expect(function () { return util.parseAxisParam(axis2, shape); }).toThrowError();
    });
    it('axis a list with one number out of range throws error', function () {
        var axis = [0, 4];
        var shape = [3, 1, 2];
        expect(function () { return util.parseAxisParam(axis, shape); }).toThrowError();
    });
    it('axis with decimal value throws error', function () {
        var axis = 0.5;
        var shape = [3, 1, 2];
        expect(function () { return util.parseAxisParam(axis, shape); }).toThrowError();
    });
});
describe('util.squeezeShape', function () {
    it('scalar', function () {
        var _a = util.squeezeShape([]), newShape = _a.newShape, keptDims = _a.keptDims;
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x1 reduced to scalar', function () {
        var _a = util.squeezeShape([1, 1]), newShape = _a.newShape, keptDims = _a.keptDims;
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x3x1 reduced to [3]', function () {
        var _a = util.squeezeShape([1, 3, 1]), newShape = _a.newShape, keptDims = _a.keptDims;
        expect(newShape).toEqual([3]);
        expect(keptDims).toEqual([1]);
    });
    it('1x1x4 reduced to [4]', function () {
        var _a = util.squeezeShape([1, 1, 4]), newShape = _a.newShape, keptDims = _a.keptDims;
        expect(newShape).toEqual([4]);
        expect(keptDims).toEqual([2]);
    });
    it('2x3x4 not reduction', function () {
        var _a = util.squeezeShape([2, 3, 4]), newShape = _a.newShape, keptDims = _a.keptDims;
        expect(newShape).toEqual([2, 3, 4]);
        expect(keptDims).toEqual([0, 1, 2]);
    });
    describe('with axis', function () {
        it('should only reduce dimensions specified by axis', function () {
            var _a = util.squeezeShape([1, 1, 1, 1, 4], [1, 2]), newShape = _a.newShape, keptDims = _a.keptDims;
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 3, 4]);
        });
        it('should only reduce dimensions specified by negative axis', function () {
            var _a = util.squeezeShape([1, 1, 1, 1, 4], [-2, -3]), newShape = _a.newShape, keptDims = _a.keptDims;
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 1, 4]);
        });
        it('should only reduce dimensions specified by negative axis', function () {
            var axis = [-2, -3];
            util.squeezeShape([1, 1, 1, 1, 4], axis);
            expect(axis).toEqual([-2, -3]);
        });
        it('throws error when specified axis is not squeezable', function () {
            expect(function () { return util.squeezeShape([1, 1, 2, 1, 4], [1, 2]); }).toThrowError();
        });
        it('throws error when specified negative axis is not squeezable', function () {
            expect(function () { return util.squeezeShape([1, 1, 2, 1, 4], [-1, -2]); }).toThrowError();
        });
        it('throws error when specified axis is out of range', function () {
            expect(function () { return util.squeezeShape([1, 1, 2, 1, 4], [11, 22]); }).toThrowError();
        });
        it('throws error when specified negative axis is out of range', function () {
            expect(function () { return util.squeezeShape([1, 1, 2, 1, 4], [
                -11, -22
            ]); }).toThrowError();
        });
    });
});
describe('util.checkComputationForErrors', function () {
    it('Float32Array has NaN', function () {
        expect(function () { return util.checkComputationForErrors(new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32', ''); })
            .toThrowError();
    });
    it('Float32Array has Infinity', function () {
        expect(function () { return util.checkComputationForErrors(new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32', ''); })
            .toThrowError();
    });
    it('Float32Array no NaN', function () {
        expect(function () { return util.checkComputationForErrors(new Float32Array([1, 2, 3, 4, -1, 255]), 'float32', ''); })
            .not.toThrowError();
    });
});
describe('util.checkConversionForErrors', function () {
    it('Float32Array has NaN', function () {
        expect(function () { return util.checkConversionForErrors(new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32'); })
            .toThrowError();
    });
    it('Float32Array has Infinity', function () {
        expect(function () { return util.checkConversionForErrors(new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32'); })
            .toThrowError();
    });
    it('Int32Array has NaN', function () {
        expect(function () { return util.checkConversionForErrors([1, 2, 3, 4, NaN], 'int32'); })
            .toThrowError();
    });
});
describe('util.hasEncodingLoss', function () {
    it('complex64 to any', function () {
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'bool')).toBe(true);
    });
    it('any to complex64', function () {
        expect(util.hasEncodingLoss('bool', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
    });
    it('any to float32', function () {
        expect(util.hasEncodingLoss('bool', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
    });
    it('float32 to any', function () {
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'bool')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
    });
    it('int32 to lower', function () {
        expect(util.hasEncodingLoss('int32', 'int32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'bool')).toBe(true);
    });
    it('lower to int32', function () {
        expect(util.hasEncodingLoss('bool', 'int32')).toBe(false);
    });
    it('bool to bool', function () {
        expect(util.hasEncodingLoss('bool', 'bool')).toBe(false);
    });
});
describe('util.toNestedArray', function () {
    it('2 dimensions', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6]);
        expect(util.toNestedArray([2, 3], a))
            .toEqual([[1, 2, 3], [4, 5, 6]]);
    });
    it('3 dimensions (2x2x3)', function () {
        var a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([2, 2, 3], a))
            .toEqual([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]);
    });
    it('3 dimensions (3x2x2)', function () {
        var a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([3, 2, 2], a))
            .toEqual([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]);
    });
    it('invalid dimension', function () {
        var a = new Float32Array([1, 2, 3]);
        expect(function () { return util.toNestedArray([2, 2], a); }).toThrowError();
    });
    it('tensor to nested array', function () {
        var x = ops_1.tensor2d([1, 2, 3, 4], [2, 2]);
        expect(util.toNestedArray(x.shape, x.dataSync()))
            .toEqual([[1, 2], [3, 4]]);
    });
    it('scalar to nested array', function () {
        var x = ops_1.scalar(1);
        expect(util.toNestedArray(x.shape, x.dataSync())).toEqual(1);
    });
    it('tensor with zero shape', function () {
        var a = new Float32Array([0, 1]);
        expect(util.toNestedArray([1, 0, 2], a)).toEqual([]);
    });
});
describe('util.monitorPromisesProgress', function () {
    it('Default progress from 0 to 1', function (done) {
        var expectFractions = [0.25, 0.50, 0.75, 1.00];
        var fractionList = [];
        var tasks = Array(4).fill(0).map(function () {
            return Promise.resolve();
        });
        util.monitorPromisesProgress(tasks, function (progress) {
            fractionList.push(parseFloat(progress.toFixed(2)));
        }).then(function () {
            expect(fractionList).toEqual(expectFractions);
            done();
        });
    });
    it('Progress with pre-defined range', function (done) {
        var startFraction = 0.2;
        var endFraction = 0.8;
        var expectFractions = [0.35, 0.50, 0.65, 0.80];
        var fractionList = [];
        var tasks = Array(4).fill(0).map(function () {
            return Promise.resolve();
        });
        util.monitorPromisesProgress(tasks, function (progress) {
            fractionList.push(parseFloat(progress.toFixed(2)));
        }, startFraction, endFraction).then(function () {
            expect(fractionList).toEqual(expectFractions);
            done();
        });
    });
    it('throws error when progress fraction is out of range', function () {
        expect(function () {
            var startFraction = -1;
            var endFraction = 1;
            var tasks = Array(4).fill(0).map(function () {
                return Promise.resolve();
            });
            util.monitorPromisesProgress(tasks, function (progress) { }, startFraction, endFraction);
        }).toThrowError();
    });
    it('throws error when startFraction more than endFraction', function () {
        expect(function () {
            var startFraction = 0.8;
            var endFraction = 0.2;
            var tasks = Array(4).fill(0).map(function () {
                return Promise.resolve();
            });
            util.monitorPromisesProgress(tasks, function (progress) { }, startFraction, endFraction);
        }).toThrowError();
    });
    it('throws error when promises is null', function () {
        expect(function () {
            util.monitorPromisesProgress(null, function (progress) { });
        }).toThrowError();
    });
    it('throws error when promises is empty array', function () {
        expect(function () {
            util.monitorPromisesProgress([], function (progress) { });
        }).toThrowError();
    });
});
//# sourceMappingURL=util_test.js.map