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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Example code to do convolution."""

import numpy as np
import tvm
import os
import tvm.testing
import tvm.topi.testing
from tvm import te, autotvm, topi, relay
from tvm.contrib.pickle_memoize import memoize
from tvm.contrib import nvcc
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple


def verify_feature_length():
    np.random.seed(34)
    target = "opencl -device=intel_graphics"
    ctx = tvm.device(target)

    batch_size = 1

    input_shape = (1, 512, 7, 7)
    kernel_shape = (512, 512, 3, 3)

    def get_mod():
        x = relay.var("x", relay.TensorType(input_shape, "float32"))
        y = relay.var("y", relay.TensorType(kernel_shape, "float32"))
        f = relay.Function(
            [x, y], relay.nn.conv2d(x, y, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
        )
        mod = tvm.IRModule()
        mod["main"] = f
        mod = relay.transform.InferType()(mod)
        return mod, {}

    mod, params = get_mod()
    #layout_config = relay.transform.LayoutConfig()
    #desired_layouts = {"nn.conv2d": ["HCHWC", "default"]}
    #with layout_config:
    #    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
    #    with tvm.transform.PassContext(opt_level=3):
    #        mod = seq(mod)
    #mod = relay.transform.recast(mod, "int4", "int32")

    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    assert len(tasks) == 1
    task = tasks[0]

    space = task.config_space

    idx1 = np.random.randint(len(space))
    idx2 = np.random.randint(len(space))

    cfg = space.get(idx1)
    sch, arg_bufs = task.instantiate(cfg)
    fea1 = autotvm.feature.get_itervar_feature_flatten(sch, arg_bufs, take_log=True)

    cfg = space.get(idx2)
    sch, arg_bufs = task.instantiate(cfg)
    fea2 = autotvm.feature.get_itervar_feature_flatten(sch, arg_bufs, take_log=True)

    print(len(fea1))
    print(len(fea2))
    assert len(fea1) == len(fea2)


def test_conv2d_nchwc_intel_graphics():
    """Test the conv2d with tensorcore for hwnc layout"""
    verify_feature_length()


if __name__ == "__main__":
    test_conv2d_nchwc_intel_graphics()
