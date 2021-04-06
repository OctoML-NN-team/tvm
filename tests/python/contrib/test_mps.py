# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
import tvm.testing
from tvm import te, relay
import numpy as np
from tvm.contrib import mps


def _get_model(
    shape,
    kernel=(3, 3),
    padding=(1, 1),
    strides=(1, 1),
    dilation=(1, 1),
    groups=1,
    dtype="float32",
    channels=-1,  # -1 means same as input channels
    bias_type="none",
    activation_type="none",
):
    """Return a model and any parameters it may have"""
    if channels == -1:
        channels = shape[1]

    a = relay.var("a", shape=shape, dtype=dtype)
    weight_shape = (channels, shape[1] // groups, *kernel)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=kernel,
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}

    return out, params


def skip_runtime_test():
    """Skip test if it requires the runtime and it's not present."""
    # MPS codegen not present.
    if not tvm.get_global_func("relay.ext.mps", True):
        print("Skip because MPS codegen is not available.")
        return True
    return False


import pytest
from tvm.relay.analysis import analysis
from itertools import zip_longest, combinations
from tvm.relay.op.contrib.mps import partition_for_mps
from tvm.contrib import graph_runtime


def verify(answers, atol, rtol, verify_saturation=False, config=None):
    """Compare the array of answers. Each entry is a list of outputs."""
    if config is None:
        config = {}

    if len(answers) < 2:
        raise RuntimeError(f"No results to compare: expected at least two, found {len(answers)}")
    for answer in zip_longest(*answers):
        for outs in combinations(answer, 2):
            try:
                if verify_saturation:
                    assert (
                        np.count_nonzero(outs[0].asnumpy() == 255) < 0.25 * outs[0].asnumpy().size
                    ), "Output is saturated: {}".format(outs[0])
                    assert (
                        np.count_nonzero(outs[0].asnumpy() == 0) < 0.25 * outs[0].asnumpy().size
                    ), "Output is saturated: {}".format(outs[0])
                tvm.testing.assert_allclose(
                    outs[0].asnumpy(), outs[1].asnumpy(), rtol=rtol, atol=atol
                )
            except AssertionError as e:
                err_msg = "Results not within the acceptable tolerance.\n"
                if config:
                    err_msg += f"The test failed with the following parameters: {config}\n"
                err_msg += str(e)
                raise AssertionError(err_msg)


def build_module(mod, target, target_host, params=None, enable_bnns=True, tvm_ops=0):
    """Build module with option to build for BNNS."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        if enable_bnns:
            mod = partition_for_mps(mod)
        relay.backend.compile_engine.get().clear()
        return relay.build(mod, target=target, target_host=target_host, params=params)


def build_and_run(
    mod,
    inputs,
    outputs,
    params,
    enable_bnns=True,
    no_runs=1,
    tvm_ops=0,
    config=None,
):
    """Build and run the relay module."""
    if config is None:
        config = {}
    target = 'metal'
    target_host = "llvm -mtriple=x86_64-apple-darwin20.1.0"

    print('-' *  50)
    print(mod)
    print('-' *  50)
    try:
        #lib = build_module(mod, device.target, params, enable_bnns, tvm_ops)
        lib = build_module(mod, target, target_host, params, enable_bnns, tvm_ops)
    except Exception as e:
        err_msg = "The module could not be built.\n"
        if config:
            err_msg += f"The test failed with the following parameters: {config}\n"
        err_msg += str(e)
        raise Exception(err_msg)
    print('-' *  50)
    print("after")
    print('-' *  50)

    #lib_name = "mod.so"
    #lib.export_library(lib_name)
    #lib = update_lib(lib, device.device, device.cross_compile)
    #gen_module = graph_runtime.GraphModule(lib["default"](device.device.cpu(0)))
    ctx = tvm.metal()
    gen_module = graph_runtime.GraphModule(lib["default"](ctx))
    gen_module.set_input(**inputs)
    out = []
    for _ in range(no_runs):
        gen_module.run()
        out.append([gen_module.get_output(i) for i in range(outputs)])
    return out


def compare_inference_with_ref(func, params, atol=0.002, rtol=0.007):
    """Compare scoring results for compilation with and without BNNS.

    Provided function will be compiled two times with and without BNNS.
    The scoring results for both type of compilation will be compared
    with provided atol and rtol. The input data will be automatically
    generated based of shape and dtype info provided for var nodes.

    """
    # Generate input tensor values
    inputs = {}
    for free_param in analysis.free_vars(func):
        name = free_param.name_hint
        dtype = free_param.type_annotation.dtype
        shape = [s.value for s in free_param.type_annotation.shape]
        inputs[name] = tvm.nd.array(np.random.uniform(0, 127, shape).astype(dtype))

    # Run for both type of compilation
    outputs = []
    #for bnns in [False, True]:
    #    outputs.append(build_and_run(func, inputs, 1, params, enable_bnns=bnns)[0])
    outputs.append(build_and_run(func, inputs, 1, params, enable_bnns=True)[0])

    # Compare result tensors
    verify(outputs, atol=atol, rtol=rtol)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because MPS codegen is not available")
def test_conv2d():
    np.random.seed(0)

    #kernel_hs = [1, 2, 3, 5]
    #kernel_ws = [1, 2, 3, 5]
    #pad = [(1, 1), (2, 2), (2, 1)]
    #strides = [(1, 1), (2, 2)]
    #dilation = [(1, 1)]
    #out_channels = [1, 4, 8, 16]
    #input_shapes = [(10, 10, 14), (12, 15, 16), (20, 20, 20)]
    #batches = [1, 2]
    #groups = [1, 2]
    #bias_kind = ["none"]
    #activation_kind = ["none"]
    #trials = generate_trials(
    #    [
    #        kernel_hs,
    #        kernel_ws,
    #        pad,
    #        strides,
    #        dilation,
    #        out_channels,
    #        input_shapes,
    #        groups,
    #        batches,
    #        bias_kind,
    #        activation_kind,
    #    ],
    #    3,
    #)

    #for (
    #    kernel_h,
    #    kernel_w,
    #    pad,
    #    stride,
    #    dilation,
    #    out_channels,
    #    input_shapes,
    #    group,
    #    batch,
    #    bias,
    #    activation,
    #) in trials:
    #    if out_channels % group != 0:
    #        continue
    #    func, params = _get_model(
    #        shape=(batch, *input_shapes),
    #        kernel=(kernel_h, kernel_w),
    #        padding=pad,
    #        strides=stride,
    #        dilation=dilation,
    #        groups=group,
    #        channels=out_channels,
    #        bias_type=bias,
    #        activation_type=activation,
    #    )
    #    compare_inference_with_ref(func, params)
    func, params = _get_model(
        shape=(1, 10, 10, 14),
        kernel=(3, 3),
        padding=(1, 1),
        strides=(1, 1),
        dilation=(1, 1),
        groups=1,
        channels=2,
    )
    compare_inference_with_ref(func, params)


if __name__ == "__main__":
    test_conv2d()


#@tvm.testing.requires_metal
#def test_matmul():
#    n = 1024
#    l = 128
#    m = 256
#    A = te.placeholder((n, l), name="A")
#    B = te.placeholder((l, m), name="B")
#    C = mps.matmul(A, B)
#    D = te.compute(C.shape, lambda *i: C(*i) + 1.0)
#    s = te.create_schedule(D.op)
#    yo, xo = D.op.axis
#    block_y = te.thread_axis("blockIdx.y")
#    block_x = te.thread_axis("blockIdx.x")
#    thread_y = te.thread_axis("threadIdx.y")
#    thread_x = te.thread_axis("threadIdx.x")
#    by, ty = s[D].split(yo, factor=16)
#    bx, tx = s[D].split(xo, factor=16)
#    s[D].bind(by, block_y)
#    s[D].bind(bx, block_x)
#    s[D].bind(ty, thread_y)
#    s[D].bind(tx, thread_x)
#
#    def verify(A, B, D, s, target="metal"):
#        if not tvm.get_global_func("tvm.contrib.mps.matmul", True):
#            print("skip because extern function is not available")
#            return
#        ctx = tvm.metal(0)
#        f = tvm.build(s, [A, B, D], "metal")
#        a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
#        b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
#        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
#        f(a, b, c)
#        tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + 1, rtol=1e-5)
#
#    verify(A, B, D, s)
#
#
#@tvm.testing.requires_metal
#def test_conv2d():
#    n = 1
#    h = 14
#    w = 14
#    ci = 2
#    co = 4
#    kh = 3
#    kw = 3
#    stride = 2
#    A = te.placeholder((n, h, w, ci), name="x")
#    B = te.placeholder((co, kh, kw, ci), name="w")
#    C = mps.conv2d(A, B, "SAME", 2)
#    s1 = te.create_schedule(C.op)
#
#    def verify(A, B, C, target="llvm"):
#        if not tvm.get_global_func("tvm.contrib.mps.conv2d", True):
#            print("skip because extern function is not available")
#            return
#        ctx = tvm.metal(0)
#        f = tvm.build(s1, [A, B, C], "metal")
#        a = tvm.nd.array(np.random.uniform(size=(n, h, w, ci)).astype(A.dtype), ctx)
#        b = tvm.nd.array(np.random.uniform(size=(co, kh, kw, ci)).astype(B.dtype), ctx)
#        c = tvm.nd.array(np.zeros((n, h // stride, w // stride, co), dtype=C.dtype), ctx)
#        f(a, b, c)
#        # print(c.asnumpy())
#        # print(c.shape)
#
#    verify(A, B, C, s1)
#
#
#if __name__ == "__main__":
#    # test_matmul()
#    test_conv2d()
