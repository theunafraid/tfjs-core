"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var device_util = require("./device_util");
var environment_1 = require("./environment");
var environment_util_1 = require("./environment_util");
var tf = require("./index");
var jasmine_util_1 = require("./jasmine_util");
var backend_cpu_1 = require("./kernels/backend_cpu");
var backend_webgl_1 = require("./kernels/backend_webgl");
var test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', test_util_1.WEBGL_ENVS, function () {
    it('disjoint query timer disabled', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 0 };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 1 };
        spyOn(device_util, 'isMobile').and.returnValue(true);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, not mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION': 1 };
        spyOn(device_util, 'isMobile').and.returnValue(false);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(true);
    });
});
jasmine_util_1.describeWithFlags('WEBGL_NUM_MB_BEFORE_PAGING webgl', test_util_1.WEBGL_ENVS, function () {
    it('should be set in a browser', function () {
        var expectedMBBeforePaging = window.screen.width * window.screen.height *
            window.devicePixelRatio * environment_util_1.BEFORE_PAGING_CONSTANT / 1024;
        var features = { 'IS_BROWSER': true, 'PROD': false };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_NUM_MB_BEFORE_PAGING')).toBe(expectedMBBeforePaging);
    });
    it('should be off when the environment is prod', function () {
        var features = { 'IS_BROWSER': true, 'PROD': true };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_NUM_MB_BEFORE_PAGING'))
            .toBe(Number.POSITIVE_INFINITY);
    });
    it('should not cause errors when paging is turned off', function () {
        environment_1.ENV.set('WEBGL_NUM_MB_BEFORE_PAGING', Number.POSITIVE_INFINITY);
        var a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
        var b = tf.tensor2d([0, 1, -3, 2, 2, 1], [3, 2]);
        var c = tf.matMul(a, b);
        test_util_1.expectArraysClose(c, [0, 8, -3, 20]);
    });
});
jasmine_util_1.describeWithFlags('WEBGL_NUM_MB_BEFORE_PAGING node', test_util_1.NODE_ENVS, function () {
    it('should be off when the environment is node', function () {
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_NUM_MB_BEFORE_PAGING'))
            .toBe(Number.POSITIVE_INFINITY);
    });
});
describe('Backend', function () {
    beforeAll(function () {
        spyOn(console, 'warn');
    });
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('custom cpu registration', function () {
        var backend;
        environment_1.ENV.registerBackend('custom-cpu', function () {
            backend = new backend_cpu_1.MathBackendCPU();
            return backend;
        });
        expect(environment_1.ENV.findBackend('custom-cpu')).toBe(backend);
        environment_1.Environment.setBackend('custom-cpu');
        expect(environment_1.ENV.backend).toBe(backend);
        environment_1.ENV.removeBackend('custom-cpu');
    });
    it('webgl not supported, falls back to cpu', function () {
        environment_1.ENV.setFeatures({ 'WEBGL_VERSION': 0 });
        var cpuBackend;
        environment_1.ENV.registerBackend('custom-cpu', function () {
            cpuBackend = new backend_cpu_1.MathBackendCPU();
            return cpuBackend;
        }, 103);
        var success = environment_1.ENV.registerBackend('custom-webgl', function () { return new backend_webgl_1.MathBackendWebGL(); }, 104);
        expect(success).toBe(false);
        expect(environment_1.ENV.findBackend('custom-webgl') == null).toBe(true);
        expect(environment_1.Environment.getBackend()).toBe('custom-cpu');
        expect(environment_1.ENV.backend).toBe(cpuBackend);
        environment_1.ENV.removeBackend('custom-cpu');
    });
    it('default custom background null', function () {
        expect(environment_1.ENV.findBackend('custom')).toBeNull();
    });
    it('allow custom backend', function () {
        var backend = new backend_cpu_1.MathBackendCPU();
        var success = environment_1.ENV.registerBackend('custom', function () { return backend; });
        expect(success).toBeTruthy();
        expect(environment_1.ENV.findBackend('custom')).toEqual(backend);
        environment_1.ENV.removeBackend('custom');
    });
});
describe('environment_util.getQueryParams', function () {
    it('basic', function () {
        expect(environment_util_1.getQueryParams('?a=1&b=hi&f=animal'))
            .toEqual({ 'a': '1', 'b': 'hi', 'f': 'animal' });
    });
});
describe('public api tf.*', function () {
    beforeEach(function () {
        environment_1.ENV.reset();
    });
    afterEach(function () {
        environment_1.ENV.reset();
    });
    it('tf.enableProdMode', function () {
        tf.enableProdMode();
        expect(environment_1.ENV.get('PROD')).toBe(true);
    });
});
jasmine_util_1.describeWithFlags('max texture size', test_util_1.WEBGL_ENVS, function () {
    it('should not throw exception', function () {
        expect(function () { return environment_1.ENV.get('WEBGL_MAX_TEXTURE_SIZE'); }).not.toThrow();
    });
});
jasmine_util_1.describeWithFlags('epsilon', {}, function () {
    it('Epsilon is a function of float precision', function () {
        var epsilonValue = environment_1.ENV.backend.floatPrecision() === 32 ? environment_1.EPSILON_FLOAT32 : environment_1.EPSILON_FLOAT16;
        expect(environment_1.ENV.get('EPSILON')).toBe(epsilonValue);
    });
    it('abs(epsilon) > 0', function () {
        expect(tf.abs(environment_1.ENV.get('EPSILON')).arraySync()).toBeGreaterThan(0);
    });
});
jasmine_util_1.describeWithFlags('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', test_util_1.ALL_ENVS, function () {
    it('disabled when prod is enabled', function () {
        var env = new environment_1.Environment();
        env.set('PROD', true);
        expect(env.get('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(false);
    });
    it('enabled when prod is disabled', function () {
        var env = new environment_1.Environment();
        env.set('PROD', false);
        expect(env.get('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')).toBe(true);
    });
});
jasmine_util_1.describeWithFlags('WEBGL_SIZE_UPLOAD_UNIFORM', test_util_1.WEBGL_ENVS, function () {
    it('is 0 when there is no float32 bit support', function () {
        var env = new environment_1.Environment();
        env.set('WEBGL_RENDER_FLOAT32_ENABLED', false);
        expect(env.get('WEBGL_SIZE_UPLOAD_UNIFORM')).toBe(0);
    });
    it('is > 0 when there is float32 bit support', function () {
        var env = new environment_1.Environment();
        env.set('WEBGL_RENDER_FLOAT32_ENABLED', true);
        expect(env.get('WEBGL_SIZE_UPLOAD_UNIFORM')).toBeGreaterThan(0);
    });
});
describe('deprecation warnings', function () {
    var oldWarn;
    beforeEach(function () {
        oldWarn = console.warn;
        spyOn(console, 'warn').and.callFake(function (msg) { return null; });
        environment_1.ENV.set('DEPRECATION_WARNINGS_ENABLED', true);
    });
    afterEach(function () {
        console.warn = oldWarn;
    });
    it('deprecationWarn warns', function () {
        tf.deprecationWarn('xyz is deprecated.');
        expect(console.warn).toHaveBeenCalledTimes(1);
        expect(console.warn)
            .toHaveBeenCalledWith('xyz is deprecated. You can disable deprecation warnings with ' +
            'tf.disableDeprecationWarnings().');
    });
    it('disableDeprecationWarnings called, deprecationWarn doesnt warn', function () {
        tf.disableDeprecationWarnings();
        expect(console.warn).toHaveBeenCalledTimes(1);
        expect(console.warn)
            .toHaveBeenCalledWith('TensorFlow.js deprecation warnings have been disabled.');
        tf.deprecationWarn('xyz is deprecated.');
        expect(console.warn).toHaveBeenCalledTimes(1);
    });
});
//# sourceMappingURL=environment_test.js.map