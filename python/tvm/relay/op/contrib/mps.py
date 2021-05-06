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


@tvm.ir.register_op_attr("nn.dense", "target.mps")
def dense(expr):
    """Check if the dense can be used in MPS."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "float32":
        return False
    if not isinstance(args[1], tvm.relay.expr.Constant):
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "float32":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


def make_dense_pattern():
    """Make pattern for bnns.dense primitive"""
    data = wildcard()
    weight = wildcard()
    return is_op("nn.dense")(data, weight)


def check_dense(extract):
    """Check dense pattern is supported by BNNS."""
    call = extract
    while call.op.name != "nn.dense":
        call = call.args[0]
    return dense(call)


@register_pattern_table("mps")
def pattern_table():
    """Get MPS specific fusing patterns collection"""
    dense_pat = (
        "mps.dense",
        make_dense_pattern(),
        check_dense,
    )
    mps_patterns = [
        dense_pat,
    ]
    return mps_patterns

