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
"""Arm Compute Library integration dense tests."""

import numpy as np
import math

import tvm
from tvm import relay
from tvm import testing
from common.infrastructure import (
    build_and_run,
    verify,
    generate_trials,
)
from .infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    build_module,
)


def _get_model(shape, weight_shape, units, dtype, var_names, has_bias=False, has_gelu=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    print("*" * 50)
    print(w.shape)
    print(w)
    print("*" * 50)
    weights = relay.const(w, dtype)
    out = relay.nn.dense(a, weights, units=units, out_dtype=dtype)
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(-128, 127, weight_shape[0]).astype(dtype))
        print("B" * 50)
        print(b.shape)
        print(b)
        print("B" * 50)
        biasc = relay.const(b, dtype)
        out = relay.op.add(out, biasc)
        params["b"] = b
    if has_gelu:
        const1 = relay.const(0.044715)
        const2 = relay.const(math.sqrt(2 / math.pi))
        bias = out
        out = relay.op.power(bias, relay.const(3.0, "float32"))
        out = relay.op.multiply(out, const1)
        out = relay.op.add(out, bias)
        out = relay.op.multiply(out, const2)
        out = relay.op.tanh(out)
        out = relay.op.add(out, relay.const(1, "float32"))
        out = relay.op.multiply(out, relay.const(0.5))
        out = relay.op.multiply(out, bias)
    return out, params


#def test_dense():
#    Device.load("test_config.json")
#
#    if skip_runtime_test():
#        return
#
#    device = Device()
#    np.random.seed(0)
#
#    dtype = ["float32"]
#    shape = [
#        ((1, 128), (16, 128), 16),
#        ((32, 32), (32, 32), 32),
#        ((1, 64), (1, 64), 1),
#        ((11, 2), (2, 2), 2),
#        ((2, 2), (1, 2), 1),
#    ]
#    composite = [False, True]
#    trials = generate_trials([dtype, shape, composite], 3)
#
#    for dtype, (shape, weight_shape, units), composite in trials:
#        outputs = []
#        inputs = {"a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}
#        func, params = _get_model(
#            shape, weight_shape, units, dtype, var_names=iter(inputs), has_bias=composite
#        )
#        for bnns in [False, True]:
#            outputs.append(
#                build_and_run(
#                    func,
#                    inputs,
#                    1,
#                    params,
#                    device,
#                    build_module,
#                    enable_framework=bnns,
#                )[0]
#            )
#
#        config = {
#            "shape": shape,
#            "weight_shape": weight_shape,
#            "units": units,
#            "dtype": dtype,
#            "composite operators (bias)": composite,
#        }
#        verify(outputs, atol=0.001, rtol=0.01, config=config)

def test_dense_bias_gelu():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    np.random.seed(0)

    dtype = ["float32"]
    shape = [
        ((1, 128), (16, 128), 16),
        ((32, 32), (32, 32), 32),
        ((1, 64), (1, 64), 1),
        ((11, 2), (2, 2), 2),
        ((2, 2), (1, 2), 1),
    ]
    composite = [False, True]
    trials = generate_trials([dtype, shape, composite], 3)

    for dtype, (shape, weight_shape, units), composite in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}
        func, params = _get_model(
            shape, weight_shape, units, dtype, var_names=iter(inputs), has_bias=True, has_gelu=True
        )
        for bnns in [True]:
            outputs.append(
                build_and_run(
                    func,
                    inputs,
                    1,
                    params,
                    device,
                    build_module,
                    enable_framework=bnns,
                )[0]
            )

        config = {
            "shape": shape,
            "weight_shape": weight_shape,
            "units": units,
            "dtype": dtype,
            "composite operators (bias)": composite,
        }
        verify(outputs, atol=0.001, rtol=0.01, config=config)


if __name__ == "__main__":
    #test_dense()
    test_dense_bias_gelu()

