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


def _get_model(a_shape, b_shape, dtype, var_names, is_a_constant=False, is_b_constant=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=a_shape, dtype=dtype)
    b = relay.var(next(var_names), shape=b_shape, dtype=dtype)
    params = {}
    if is_b_constant is True:
        b = tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype))
        params["b"] = b
        b = relay.const(b, dtype)
    if is_a_constant is True:
        a = tvm.nd.array(np.random.uniform(-128, 127, a_shape).astype(dtype))
        params["a"] = a
        a = relay.const(a, dtype)
    out = relay.nn.batch_matmul(a, b)
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
from tvm.contrib import graph_executor


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

    try:
        #lib = build_module(mod, device.target, params, enable_bnns, tvm_ops)
        lib = build_module(mod, target, target_host, params, enable_bnns, tvm_ops)
    except Exception as e:
        err_msg = "The module could not be built.\n"
        if config:
            err_msg += f"The test failed with the following parameters: {config}\n"
        err_msg += str(e)
        raise Exception(err_msg)

    #lib_name = "mod.so"
    #lib.export_library(lib_name)
    #lib = update_lib(lib, device.device, device.cross_compile)
    #gen_module = graph_runtime.GraphModule(lib["default"](device.device.cpu(0)))
    ctx = tvm.metal()
    gen_module = graph_executor.GraphModule(lib["default"](ctx))
    gen_module.set_input(**inputs)
    out = []
    for _ in range(no_runs):
        gen_module.run()
        out.append([gen_module.get_output(i) for i in range(outputs)])
    return out


def compare_inference_with_ref(func, params, inputs, atol=0.002, rtol=0.007):
    """Compare scoring results for compilation with and without BNNS.

    Provided function will be compiled two times with and without BNNS.
    The scoring results for both type of compilation will be compared
    with provided atol and rtol. The input data will be automatically
    generated based of shape and dtype info provided for var nodes.

    """
    # Run for both type of compilation
    outputs = []
    for bnns in [False, True]:
        outputs.append(build_and_run(func, inputs, 1, params, enable_bnns=bnns)[0])

    # Compare result tensors
    verify(outputs, atol=atol, rtol=rtol)


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_matmul():
    np.random.seed(0)
    dtype = "float32"
    # C[N, I, J] = A[N, I, K] * B[N, J, K]
    shapes_config = [
        # N, I, J, K
        [1, 4, 4, 3],
        [1, 16, 32, 32],
        [2, 1, 1, 3],
        [2, 16, 32, 32],
        [5, 1, 1, 3],
    ]
    data_config = [
        # A_is_constant, B_is_constant
        [False, True],
        [True, False],
        [False, False],
    ]
    for N, I, J, K in shapes_config:
        a_shape = [N, I, K]
        b_shape = [N, J, K]
        for is_a_constant, is_b_constant in data_config:
            inputs = {
                "a": tvm.nd.array(np.random.uniform(-128, 127, a_shape).astype(dtype)),
                "b": tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype)),
            }
            func, params = _get_model(
                a_shape=a_shape,
                b_shape=b_shape,
                dtype=dtype,
                var_names=iter(inputs),
                is_a_constant=is_a_constant,
                is_b_constant=is_b_constant,
            )
            compare_inference_with_ref(func, params, inputs)


if __name__ == "__main__":
    test_matmul()


