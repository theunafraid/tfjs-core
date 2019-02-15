"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('slice1d', test_util_1.ALL_ENVS, function () {
    it('slices 1x1 into 1x1 (effectively a copy)', function () {
        var a = tf.tensor1d([5]);
        var result = tf.slice1d(a, 0, 1);
        expect(result.shape).toEqual([1]);
        test_util_1.expectArraysClose(result, 5);
    });
    it('slices 5x1 into shape 2x1 starting at 3', function () {
        var a = tf.tensor1d([1, 2, 3, 4, 5]);
        var result = tf.slice1d(a, 3, 2);
        expect(result.shape).toEqual([2]);
        test_util_1.expectArraysClose(result, [4, 5]);
    });
    it('slices 5x1 into shape 3x1 starting at 1', function () {
        var a = tf.tensor1d([1, 2, 3, 4, 5]);
        var result = tf.slice1d(a, 1, 3);
        expect(result.shape).toEqual([3]);
        test_util_1.expectArraysClose(result, [2, 3, 4]);
    });
    it('grad', function () {
        var a = tf.tensor1d([1, 2, 3, 4, 5]);
        var dy = tf.tensor1d([10, 100]);
        var da = tf.grad(function (x) { return tf.slice1d(a, 1, 2); })(a, dy);
        expect(da.shape).toEqual([5]);
        test_util_1.expectArraysClose(da, [0, 10, 100, 0, 0]);
    });
    it('accepts a tensor-like object', function () {
        var a = [5];
        var result = tf.slice1d(a, 0, 1);
        expect(result.shape).toEqual([1]);
        test_util_1.expectArraysClose(result, 5);
    });
});
jasmine_util_1.describeWithFlags('slice2d', test_util_1.ALL_ENVS, function () {
    it('slicing a 1x1 from a 1x1 returns a 1x1', function () {
        var a = tf.tensor2d([0], [1, 1]);
        var b = tf.slice2d(a, [0, 0], [1, 1]);
        expect(b.shape).toEqual([1, 1]);
    });
    it('returns a tensor of slice size', function () {
        var a = tf.zeros([100, 100]);
        var b = tf.slice2d(a, [0, 0], [12, 34]);
        expect(b.shape).toEqual([12, 34]);
    });
    it('returns the upper-left submatrix when begin is [0, 0]', function () {
        var a = tf.randomUniform([10, 10], -1, 1);
        var b = tf.slice2d(a, [0, 0], [2, 2]);
        var aValues = a.dataSync();
        test_util_1.expectArraysClose(b, [aValues[0], aValues[1], aValues[10], aValues[11]]);
    });
    it('returns the rectangle specified', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
        var b = tf.slice2d(a, [1, 1], [3, 2]);
        test_util_1.expectArraysClose(b, [5, 6, 8, 9, 11, 12]);
    });
    it('throws when requesting out of bounds slice', function () {
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);
        expect(function () { return tf.slice2d(a, [1, 1], [10, 10]); }).toThrowError();
    });
    it('grad', function () {
        var a = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
        var dy = tf.tensor2d([[20], [50]]);
        var da = tf.grad(function (x) { return tf.slice2d(a, [0, 1], [2, 1]); })(a, dy);
        expect(da.shape).toEqual([2, 3]);
        test_util_1.expectArraysClose(da, [0, 20, 0, 0, 50, 0]);
    });
    it('accepts a tensor-like object', function () {
        var a = [[0]];
        var b = tf.slice2d(a, [0, 0], [1, 1]);
        expect(b.shape).toEqual([1, 1]);
    });
    it('slice an already sliced tensor, first was not continous', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [0, 1]);
        var c = tf.slice(b, [1, 1], [1, 1]);
        expect(c.shape).toEqual([1, 1]);
        test_util_1.expectArraysClose(c, [7]);
    });
    it('slice an already sliced tensor, first was continous', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [1, 0]);
        var c = tf.slice(b, [1, 0]);
        expect(c.shape).toEqual([1, 4]);
        test_util_1.expectArraysClose(c, [9, 10, 11, 12]);
    });
    it('slice an already sliced tensor and do async read', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, b, c, _a;
        return __generator(this, function (_b) {
            switch (_b.label) {
                case 0:
                    a = [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ];
                    b = tf.slice(a, [0, 1]);
                    c = tf.slice(b, [1, 1], [1, 1]);
                    expect(c.shape).toEqual([1, 1]);
                    _a = test_util_1.expectArraysClose;
                    return [4, c.data()];
                case 1:
                    _a.apply(void 0, [_b.sent(), new Float32Array([7])]);
                    return [2];
            }
        });
    }); });
    it('square a sliced texture, followed by non-sliced texture of same shape', function () {
        var input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
        var slicedInput = tf.slice(input, [0, 0], [3, 2]);
        var a = slicedInput.square();
        test_util_1.expectArraysClose(a, [1, 4, 9, 16, 25, 36]);
        var b = tf.square(input);
        test_util_1.expectArraysClose(b, [1, 4, 9, 16, 25, 36]);
    });
    it('square a non-sliced texture, followed by a sliced texture of same shape', function () {
        var input = tf.tensor([[1, 2, 3], [4, 5, 6]]).abs().as2D(3, 2);
        var slicedInput = tf.slice(input, [0, 0], [3, 2]);
        var a = input.square();
        test_util_1.expectArraysClose(a, [1, 4, 9, 16, 25, 36]);
        var b = tf.square(slicedInput);
        test_util_1.expectArraysClose(b, [1, 4, 9, 16, 25, 36]);
    });
    it('slice a tensor and do async read', function () { return __awaiter(_this, void 0, void 0, function () {
        var a, b, vals;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    a = [
                        [1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                    ];
                    b = tf.slice(a, [0, 1], [3, 2]);
                    expect(b.shape).toEqual([3, 2]);
                    return [4, b.data()];
                case 1:
                    vals = _a.sent();
                    test_util_1.expectArraysClose(vals, new Float32Array([2, 3, 6, 7, 10, 11]));
                    return [2];
            }
        });
    }); });
    it('flatten a sliced tensor that was continous in memory', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [1, 0]).flatten();
        expect(b.shape).toEqual([8]);
        test_util_1.expectArraysClose(b, [5, 6, 7, 8, 9, 10, 11, 12]);
    });
    it('slice a tensor that was not continous in memory', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [0, 1]);
        expect(b.shape).toEqual([3, 3]);
        test_util_1.expectArraysClose(b, [2, 3, 4, 6, 7, 8, 10, 11, 12]);
    });
    it('flatten a sliced tensor that was not continous in memory', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [0, 1]).flatten();
        expect(b.shape).toEqual([9]);
        test_util_1.expectArraysClose(b, [2, 3, 4, 6, 7, 8, 10, 11, 12]);
    });
    it('flatten a sliced tensor not continous in memory and run program', function () {
        var a = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ];
        var b = tf.slice(a, [0, 1]).flatten();
        var c = tf.square(b);
        test_util_1.expectArraysClose(c, [4, 9, 16, 36, 49, 64, 100, 121, 144]);
    });
    it('reshape a sliced 1d into a 2d tensor', function () {
        var a = [1, 2, 3, 4, 5];
        var b = tf.slice(a, 1).as2D(2, 2);
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [2, 3, 4, 5]);
    });
    it('reshape a sliced 1d into a 2d tensor and run program', function () {
        var a = [1, 2, 3, 4, 5];
        var b = tf.slice(a, 1).as2D(2, 2).square();
        expect(b.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(b, [4, 9, 16, 25]);
    });
    it('broadcast the original with the sliced tensor', function () {
        var a = [[1, 2], [3, 4]];
        var b = tf.slice(a, [0, 1]);
        var c = tf.add(a, b);
        expect(c.shape).toEqual([2, 2]);
        test_util_1.expectArraysClose(c, [3, 4, 7, 8]);
    });
});
jasmine_util_1.describeWithFlags('slice a packed texture', test_util_1.WEBGL_ENVS, function () {
    beforeAll(function () {
        tf.ENV.set('WEBGL_PACK', true);
    });
    it('slice after a matmul', function () {
        var a = [[1, 2], [3, 4]];
        var b = [[5, 6], [7, 8]];
        var c = tf.matMul(a, b);
        test_util_1.expectArraysClose(c.slice([0, 0]), [19, 22, 43, 50]);
        test_util_1.expectArraysClose(c.slice([0, 1]), [22, 50]);
        test_util_1.expectArraysClose(c.slice([1, 0]), [43, 50]);
        test_util_1.expectArraysClose(c.slice([1, 1]), [50]);
    });
});
jasmine_util_1.describeWithFlags('slice3d', test_util_1.ALL_ENVS, function () {
    it('slices 1x1x1 into shape 1x1x1 (effectively a copy)', function () {
        var a = tf.tensor3d([[[5]]], [1, 1, 1]);
        var result = tf.slice3d(a, [0, 0, 0], [1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
    it('slices 2x2x2 array into 1x2x2 starting at [1, 0, 0]', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = tf.slice3d(a, [1, 0, 0], [1, 2, 2]);
        expect(result.shape).toEqual([1, 2, 2]);
        test_util_1.expectArraysClose(result, [5, 6, 7, 8]);
    });
    it('slices 2x2x2 array into 2x1x1 starting at [0, 1, 1]', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = tf.slice3d(a, [0, 1, 1], [2, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1]);
        test_util_1.expectArraysClose(result, [4, 8]);
    });
    it('accepts a tensor-like object', function () {
        var a = [[[5]]];
        var result = tf.slice3d(a, [0, 0, 0], [1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
});
jasmine_util_1.describeWithFlags('slice4d', test_util_1.ALL_ENVS, function () {
    it('slices 1x1x1x1 into shape 1x1x1x1 (effectively a copy)', function () {
        var a = tf.tensor4d([[[[5]]]], [1, 1, 1, 1]);
        var result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
    it('slices 2x2x2x2 array into 1x2x2x2 starting at [1, 0, 0, 0]', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88], [2, 2, 2, 2]);
        var result = tf.slice4d(a, [1, 0, 0, 0], [1, 2, 2, 2]);
        expect(result.shape).toEqual([1, 2, 2, 2]);
        test_util_1.expectArraysClose(result, [11, 22, 33, 44, 55, 66, 77, 88]);
    });
    it('slices 2x2x2x2 array into 2x1x1x1 starting at [0, 1, 1, 1]', function () {
        var a = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8, 11, 22, 33, 44, 55, 66, 77, 88], [2, 2, 2, 2]);
        var result = tf.slice4d(a, [0, 1, 1, 1], [2, 1, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [8, 88]);
    });
    it('accepts a tensor-like object', function () {
        var a = [[[[5]]]];
        var result = tf.slice4d(a, [0, 0, 0, 0], [1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
});
jasmine_util_1.describeWithFlags('slice5d', test_util_1.ALL_ENVS, function () {
    it('slices 1x1x1x1x1 into shape 1x1x1x1x1 (effectively a copy)', function () {
        var a = tf.tensor5d([[[[[5]]]]], [1, 1, 1, 1, 1]);
        var result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
    it('slices 2x2x2x2x2 array into 1x2x2x2x2 starting at [1,0,0,0,0]', function () {
        var a = tf.tensor5d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 11, 22, 33, 44, 55, 66,
            77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ], [2, 2, 2, 2, 2]);
        var result = tf.slice(a, [1, 0, 0, 0, 0], [1, 2, 2, 2, 2]);
        expect(result.shape).toEqual([1, 2, 2, 2, 2]);
        test_util_1.expectArraysClose(result, [
            11, 22, 33, 44, 55, 66, 77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ]);
    });
    it('slices 2x2x2x2x2 array into 2x1x1x1x1 starting at [0,1,1,1,1]', function () {
        var a = tf.tensor5d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 11, 22, 33, 44, 55, 66,
            77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ], [2, 2, 2, 2, 2]);
        var result = tf.slice(a, [0, 1, 1, 1, 1], [2, 1, 1, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [16, 888]);
    });
    it('accepts a tensor-like object', function () {
        var a = [[[[[5]]]]];
        var result = tf.slice(a, [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
});
jasmine_util_1.describeWithFlags('slice6d', test_util_1.ALL_ENVS, function () {
    it('slices 1x1x1x1x1x1 into shape 1x1x1x1x1x1 (effectively a copy)', function () {
        var a = tf.tensor6d([[[[[[5]]]]]], [1, 1, 1, 1, 1, 1]);
        var result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
    it('slices 2x2x2x2x2x2 array into 1x2x2x2x2x2 starting at [1,0,0,0,0,0]', function () {
        var a = tf.tensor6d([
            31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311,
            312, 313, 314, 315, 316, 311, 322, 333, 344, 355, 366,
            377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 11, 22, 33, 44, 55, 66,
            77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ], [2, 2, 2, 2, 2, 2]);
        var result = tf.slice(a, [1, 0, 0, 0, 0, 0], [1, 2, 2, 2, 2, 2]);
        expect(result.shape).toEqual([1, 2, 2, 2, 2, 2]);
        test_util_1.expectArraysClose(result, [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 11, 22, 33, 44, 55, 66,
            77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ]);
    });
    it('slices 2x2x2x2x2x2 array into 2x1x1x1x1x1 starting at [0,1,1,1,1,1]', function () {
        var a = tf.tensor6d([
            31, 32, 33, 34, 35, 36, 37, 38, 39, 310, 311,
            312, 313, 314, 315, 316, 311, 322, 333, 344, 355, 366,
            377, 388, 3111, 3222, 3333, 3444, 3555, 3666, 3777, 3888,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 11, 22, 33, 44, 55, 66,
            77, 88, 111, 222, 333, 444, 555, 666, 777, 888
        ], [2, 2, 2, 2, 2, 2]);
        var result = tf.slice(a, [0, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [3888, 888]);
    });
    it('accepts a tensor-like object', function () {
        var a = [[[[[[5]]]]]];
        var result = tf.slice(a, [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]);
        expect(result.shape).toEqual([1, 1, 1, 1, 1, 1]);
        test_util_1.expectArraysClose(result, [5]);
    });
});
jasmine_util_1.describeWithFlags('slice ergonomics', test_util_1.ALL_ENVS, function () {
    it('slices 2x2x2 array into 2x1x1 no size', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = a.slice([0, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1]);
        test_util_1.expectArraysClose(result, [4, 8]);
    });
    it('slices 2x2x2 array into 1x2x2 with scalar begin no size', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = a.slice(1);
        expect(result.shape).toEqual([1, 2, 2]);
        test_util_1.expectArraysClose(result, [5, 6, 7, 8]);
    });
    it('slices 2x2x2 array using 2d size and 2d size', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = a.slice([0, 1]);
        expect(result.shape).toEqual([2, 1, 2]);
        test_util_1.expectArraysClose(result, [3, 4, 7, 8]);
    });
    it('slices 2x2x2 array using negative size', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = a.slice([0, 1], [-1, 1]);
        expect(result.shape).toEqual([2, 1, 2]);
        test_util_1.expectArraysClose(result, [3, 4, 7, 8]);
    });
    it('slices 2x2x2 array using 1d size', function () {
        var a = tf.tensor3d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        var result = a.slice(0, 1);
        expect(result.shape).toEqual([1, 2, 2]);
        test_util_1.expectArraysClose(result, [1, 2, 3, 4]);
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.slice({}, 0, 0); })
            .toThrowError(/Argument 'x' passed to 'slice' must be a Tensor/);
    });
    it('accepts a tensor-like object', function () {
        var a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        var result = tf.slice(a, [0, 1, 1]);
        expect(result.shape).toEqual([2, 1, 1]);
        test_util_1.expectArraysClose(result, [4, 8]);
    });
});
//# sourceMappingURL=slice_test.js.map