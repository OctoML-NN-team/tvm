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
# pylint: disable=invalid-name, unused-argument
"""Metal Performance Shaders supported operators."""
import math
import tvm.ir

from tvm.relay import transform
from tvm.relay.expr import const
from tvm.relay.build_module import bind_params_by_name

from .register import register_pattern_table, get_pattern_table
from ...dataflow_pattern import wildcard, is_op, is_expr


def partition_for_mps(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to MPS.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.DynamicToStatic(),
            transform.AlterOpLayout(),
            transform.FoldConstant(),
            transform.MergeComposite(get_pattern_table("mps")),
            transform.AnnotateTarget("mps"),
            #   If you no need in per layer performance statistic you can
            #   uncomment next line
            # transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by MPS.

    Parameters
    ----------
    op_name : Str
        The name of supported operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by MPS.
    """

    @tvm.ir.register_op_attr(op_name, "target.mps")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_matmul")


#def dtype_is_supported(dtype):
#    """Check if data type is supported by MPS backend"""
#    return dtype in ("", "float32")
#
#
#@tvm.ir.register_op_attr("nn.conv2d", "target.mps")
#def conv2d_check(expr):
#    """Check if the conv2d can be executed in MPS"""
#    attrs, args = expr.attrs, expr.args
#    data_typ = args[0].checked_type
#    print("\n\n\n\n\n 1. mps_pattert_table\n\n\n\n")
#    if len(data_typ.shape) != 4 or data_typ.dtype != "float32":
#        return False
#    if not isinstance(args[1], tvm.relay.expr.Constant):
#        return False
#    kernel_typ = args[1].checked_type
#    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
#        return False
#    if attrs.data_layout != "NCHW":
#        return False
#    if not dtype_is_supported(attrs.out_dtype):
#        return False
#    print("\n\n\n\n\n 2. mps_pattert_table\n\n\n\n")
#    return True
#
#
#def make_conv_pattern(with_bias=True, activation="none"):
#    """Make pattern for mps.conv2d primitive"""
#    data = wildcard()
#    weight = wildcard()
#    bias = wildcard()
#    pat = is_op("nn.conv2d")(data, weight)
#    if with_bias:
#        pat = is_op("add")(pat, bias) | is_op("nn.bias_add")(pat, bias)
#    if activation == "relu":
#        pat = is_op("nn.relu")(pat)
#    elif activation == "sigmoid":
#        pat = is_op("sigmoid")(pat)
#    return pat
#
#
#def check_conv(extract):
#    """Check conv pattern is supported by MPS."""
#    call = extract
#    while call.op.name != "nn.conv2d":
#        call = call.args[0]
#    return conv2d_check(call)


@register_pattern_table("mps")
def pattern_table():
    """Get MPS specific fusing patterns collection"""
    mps_patterns = [
        #conv2d_pat,
    ]
    return mps_patterns

