# mypy: allow-untyped-defs
"""
TileLang codegen backend for PyTorch Inductor.

Generates TileLang (@T.prim_func) kernels targeting NPU (Ascend) via
`tilelang.compile(..., target='npuir')`.

Limitations of initial implementation:
- Supports 1-D pointwise (xnumel only) kernels; reductions raise NotImplementedError.
- Assumes contiguous 1-D tensor layout; complex/strided indexing is not yet handled.
- Requires tilelang package to be installed at runtime.
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config, ir
from ..utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    IndentedBuffer,
    Placeholder,
)
from ..virtualized import ReductionType, StoreMode, V
from .common import (
    BackendFeature,
    CSE,
    CSEVariable,
    OpOverrides,
    SizeArg,
    TensorArg,
)
from .simd import (
    SIMDKernel,
    SIMDScheduling,
    IterationRangesRoot,
    IterationRangesEntry,
)

if TYPE_CHECKING:
    from ..scheduler import Scheduler


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_TORCH_TO_TILELANG_TYPE: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def tilelang_dtype(dtype: torch.dtype) -> str:
    return _TORCH_TO_TILELANG_TYPE.get(dtype, "float32")


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    pass


# ---------------------------------------------------------------------------
# Op overrides — emit Python scalar expressions for T.Parallel body
# ---------------------------------------------------------------------------

class TileLangOverrides(OpOverrides):
    """
    Map inductor element-wise ops to scalar Python expressions compatible
    with TileLang's T.Parallel body (no tl.* functions).
    """

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True):
        if dtype == torch.bool:
            return f"({x} != 0)"
        tl_type = tilelang_dtype(dtype)
        return f"T.cast({x}, '{tl_type}')"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        # TileLang bitcast: fall back to reinterpret-cast via float/int
        tl_src = tilelang_dtype(src_dtype)
        tl_dst = tilelang_dtype(dtype)
        return f"T.reinterpret_cast({x}, '{tl_src}', '{tl_dst}')"

    @staticmethod
    def constant(value, dtype: torch.dtype):
        import torch._prims_common as prim
        py_type = prim.dtype_to_type(dtype)
        return repr(py_type(value))

    @staticmethod
    def abs(x):
        return f"abs({x})"

    @staticmethod
    def neg(x):
        return f"(-{x})"

    @staticmethod
    def exp(x):
        return f"_math.exp({x})"

    @staticmethod
    def exp2(x):
        return f"_math.pow(2.0, {x})"

    @staticmethod
    def expm1(x):
        return f"(_math.exp({x}) - 1.0)"

    @staticmethod
    def log(x):
        return f"_math.log({x})"

    @staticmethod
    def log2(x):
        return f"_math.log2({x})"

    @staticmethod
    def log1p(x):
        return f"_math.log1p({x})"

    @staticmethod
    def sqrt(x):
        return f"_math.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"(1.0 / _math.sqrt({x}))"

    @staticmethod
    def sin(x):
        return f"_math.sin({x})"

    @staticmethod
    def cos(x):
        return f"_math.cos({x})"

    @staticmethod
    def tan(x):
        return f"_math.tan({x})"

    @staticmethod
    def tanh(x):
        return f"_math.tanh({x})"

    @staticmethod
    def asin(x):
        return f"_math.asin({x})"

    @staticmethod
    def acos(x):
        return f"_math.acos({x})"

    @staticmethod
    def atan(x):
        return f"_math.atan({x})"

    @staticmethod
    def atan2(x, y):
        return f"_math.atan2({x}, {y})"

    @staticmethod
    def sigmoid(x):
        return f"(1.0 / (1.0 + _math.exp(-({x}))))"

    @staticmethod
    def relu(x):
        return f"(({x}) if ({x}) > 0.0 else 0.0)"

    @staticmethod
    def minimum(a, b):
        return f"(({a}) if ({a}) < ({b}) else ({b}))"

    @staticmethod
    def maximum(a, b):
        return f"(({a}) if ({a}) > ({b}) else ({b}))"

    @staticmethod
    def where(cond, a, b):
        return f"(({a}) if ({cond}) else ({b}))"

    @staticmethod
    def logical_not(a):
        return f"(not ({a}))"

    @staticmethod
    def logical_and(a, b):
        return f"(({a}) and ({b}))"

    @staticmethod
    def logical_or(a, b):
        return f"(({a}) or ({b}))"

    @staticmethod
    def logical_xor(a, b):
        return f"(bool({a}) != bool({b}))"

    @staticmethod
    def bitwise_and(a, b):
        return f"(({a}) & ({b}))"

    @staticmethod
    def bitwise_or(a, b):
        return f"(({a}) | ({b}))"

    @staticmethod
    def bitwise_xor(a, b):
        return f"(({a}) ^ ({b}))"

    @staticmethod
    def bitwise_not(a):
        return f"(~({a}))"

    @staticmethod
    def sign(x):
        return f"((1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0)))"

    @staticmethod
    def floor(x):
        return f"_math.floor({x})"

    @staticmethod
    def ceil(x):
        return f"_math.ceil({x})"

    @staticmethod
    def trunc(x):
        return f"_math.trunc({x})"

    @staticmethod
    def pow(a, b):
        return f"_math.pow({a}, {b})"

    @staticmethod
    def rand(seed, offset):
        raise NotImplementedError("TileLang backend: rand() not yet supported")

    @staticmethod
    def randint64(seed, offset, low, high):
        raise NotImplementedError("TileLang backend: randint64() not yet supported")

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("TileLang backend: load_seed() not yet supported")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

# Default block size for 1-D tiling.
_TILELANG_DEFAULT_XBLOCK = 128


class TileLangKernel(SIMDKernel[TileLangCSEVariable]):
    """
    Generates TileLang prim_func source for a fused set of pointwise nodes.

    The generated kernel looks like:
        @T.prim_func
        def KERNEL_NAME_prim_fn(in_ptr0: T.Tensor((_xnumel,), 'float32'), ...):
            with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (bid, _):
                _in_ptr0_local = T.alloc_ub((_XBLOCK,), 'float32')
                T.copy(in_ptr0[bid*_XBLOCK : (bid+1)*_XBLOCK], _in_ptr0_local)
                for _tilelang_li in T.Parallel(_XBLOCK):
                    <compute body>
                T.copy(_out_ptr0_local, out_ptr0[bid*_XBLOCK : (bid+1)*_XBLOCK])
    """

    overrides = TileLangOverrides  # type: ignore[assignment]
    kexpr = SIMDKernel.sexpr  # use Python expression printer

    def __init__(self, tiling, **kwargs) -> None:
        super().__init__(tiling, **kwargs)
        # Re-create CSE without the Triton-specific suffix/prefix
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)
        # Buffers that need T.copy before the compute loop (input loads)
        self._tilelang_inputs: dict[str, tuple[str, str, torch.dtype]] = {}
        # Buffers that need T.copy after the compute loop (output stores)
        self._tilelang_outputs: dict[str, tuple[str, str, torch.dtype]] = {}

    # ------------------------------------------------------------------
    # Required overrides from SIMDKernel
    # ------------------------------------------------------------------

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return tilelang_dtype(dtype)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        # Suppress Triton-style range-tree setup code (xoffset, xindex, xmask …).
        # TileLang's T.Kernel + T.Parallel handle block/thread indexing instead.
        pass

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        return "bid"

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        # Not used in TileLang body generation, but must return something
        return f"T.arange(0, {entry.prefix.upper()}BLOCK)"

    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value: Any) -> str:
        return repr(value)

    # ------------------------------------------------------------------
    # load / store / reduction — core of the backend
    # ------------------------------------------------------------------

    def load(self, name: str, index: sympy.Expr) -> TileLangCSEVariable:
        """
        Register `name` as needing a T.copy load and return a CSE variable
        that reads the element at `_tilelang_li` from the local UB buffer.

        NOTE: the `index` sympy expression is intentionally ignored for now.
        This works correctly for 1-D contiguous access. Non-contiguous or
        strided access requires index-aware local buffer mapping (future work).
        """
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        local_name = f"_{var}_local"
        if name not in self._tilelang_inputs:
            self._tilelang_inputs[name] = (var, local_name, dtype)
        line = f"{local_name}[_tilelang_li]"
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
        mode: StoreMode = StoreMode.UPDATE,
    ) -> None:
        """
        Register `name` as needing a T.copy store and write the element at
        `_tilelang_li` in the local UB buffer.
        """
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        local_name = f"_{var}_local"
        if name not in self._tilelang_outputs:
            self._tilelang_outputs[name] = (var, local_name, dtype)
        self.stores.writeline(f"{local_name}[_tilelang_li] = {value}")

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: TileLangCSEVariable,
    ) -> TileLangCSEVariable:
        raise NotImplementedError(
            "TileLang backend: reductions are not yet implemented. "
            "The graph will fall back to the Triton backend for reduction kernels."
        )

    # ------------------------------------------------------------------
    # Kernel source generation
    # ------------------------------------------------------------------

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """
        Return the TileLang prim_func source that goes inside the per-shape
        factory function generated by TileLangScheduling.define_kernel().

        The caller (define_kernel) wraps this in:
            def _prim_factory_<kernel_name>(_xnumel):
                <src returned here>
                return <kernel_name>_prim_fn
        """
        # `self.loads`, `self.compute`, `self.stores` are filled by
        # codegen_node_schedule_with_kernel before this is called.

        xblock = _TILELANG_DEFAULT_XBLOCK
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"

        argdefs, _, signature, _ = self.args.python_argdefs()

        # Build T.Tensor argument list for the prim_func signature
        prim_sig_parts: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if isinstance(sig, TensorArg):
                dtype_str = tilelang_dtype(sig.dtype)
                prim_sig_parts.append(
                    f"{argdef.name}: T.Tensor((_xnumel,), '{dtype_str}')"
                )
            # SizeArg / WorkspaceArg: omitted from prim_func signature;
            # sizes are closure-captured from the factory.

        code = IndentedBuffer()
        code.writeline("import tilelang.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        code.writeline(f"_XBLOCK = {xblock}")
        code.writeline("")
        code.writeline(f"@T.prim_func")
        code.writeline(f"def {prim_fn_name}(")
        with code.indent():
            for i, part in enumerate(prim_sig_parts):
                comma = "," if i < len(prim_sig_parts) - 1 else ""
                code.writeline(f"{part}{comma}")
        code.writeline("):")

        with code.indent():
            code.writeline(
                f"with T.Kernel(T.ceildiv(_xnumel, _XBLOCK), is_npu=True) as (bid, _):"
            )
            with code.indent():
                # Allocate UB (on-chip buffer) for each input
                for _buf_name, (var, local_name, dtype) in self._tilelang_inputs.items():
                    code.writeline(
                        f"{local_name} = T.alloc_ub((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                    )
                # Allocate UB for outputs (if not already allocated as input)
                _input_locals = {v for _, v, _ in self._tilelang_inputs.values()}
                for _buf_name, (var, local_name, dtype) in self._tilelang_outputs.items():
                    if local_name not in _input_locals:
                        code.writeline(
                            f"{local_name} = T.alloc_ub((_XBLOCK,), '{tilelang_dtype(dtype)}')"
                        )
                code.writeline("")
                # T.copy: global → local for all inputs
                for _buf_name, (var, local_name, _dtype) in self._tilelang_inputs.items():
                    code.writeline(
                        f"T.copy({var}[bid * _XBLOCK : (bid + 1) * _XBLOCK], {local_name})"
                    )
                code.writeline("")
                # T.Parallel compute body
                code.writeline("for _tilelang_li in T.Parallel(_XBLOCK):")
                with code.indent():
                    if self.loads.getvalue().strip():
                        code.splice(self.loads)
                    if self.compute.getvalue().strip():
                        code.splice(self.compute)
                    if self.stores.getvalue().strip():
                        code.splice(self.stores)
                    else:
                        code.writeline("pass")
                code.writeline("")
                # T.copy: local → global for all outputs
                for _buf_name, (var, local_name, _dtype) in self._tilelang_outputs.items():
                    code.writeline(
                        f"T.copy({local_name}, {var}[bid * _XBLOCK : (bid + 1) * _XBLOCK])"
                    )

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[ir.IRNode] = None) -> None:
        """
        Emit the call to `name(...)` in the generated wrapper code.
        The kernel function signature is:
            name(tensor_arg0, ..., xnumel)
        """
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_call_args = [
            a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)
        ]
        numel_call_args = [
            str(tree.numel) for tree in self.active_range_trees()
        ]
        all_call_args = tensor_call_args + numel_call_args
        wrapper.writeline(f"{name}({', '.join(all_call_args)})")

    def create_cse_var(self, *args, **kwargs) -> TileLangCSEVariable:
        return TileLangCSEVariable(*args, **kwargs)

    # SIMDKernel hooks that are Triton-specific; make them no-ops.
    def should_use_persistent_reduction(self) -> bool:
        return False

    def should_use_cooperative_reduction(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TileLangScheduling(SIMDScheduling):
    """
    Inductor scheduling backend that emits TileLang kernels for NPU.

    Plug in via:
        register_backend_for_device("npu", TileLangScheduling, PythonWrapperCodegen)
    """

    kernel_type: type[Any] = TileLangKernel

    # Conservative feature set — expand as the backend matures.
    backend_features: OrderedSet[BackendFeature] = OrderedSet(
        [BackendFeature.INPLACE_BUFFERS]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def codegen_comment(self, node_schedule) -> None:
        wrapper = V.graph.wrapper_code
        origins, _ = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)

    def codegen_sync(self) -> None:
        # Emit a device sync after NPU kernel launch.
        V.graph.wrapper_code.writeline("torch.npu.synchronize()")

    def define_kernel(self, src_code: str, node_schedule, kernel: TileLangKernel) -> str:
        """
        Wrap the prim_func source emitted by TileLangKernel.codegen_kernel()
        in a shape-keyed caching function and inject it into the module header.

        Generated module-level code:
            import tilelang as _tilelang_<suffix>

            def _prim_factory_<name>(_xnumel):
                <src_code>          # defines <name>_prim_fn
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(tensor_arg0, ..., xnumel):
                _key = (int(xnumel),)
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<suffix>.compile(
                        _prim_factory_<name>(_key[0]), target='npuir')
                _<name>_cache[_key](tensor_arg0, ...)
        """
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        suffix = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[src_code] = kernel_name

        # Replace placeholder with real kernel name
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        # Collect call-site information from the kernel object
        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [
            a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)
        ]
        active_trees = kernel.active_range_trees()
        numel_arg_names = [f"{t.prefix}numel" for t in active_trees]
        outer_arg_list = tensor_call_args + numel_arg_names

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata = f"{origins}\n{detailed_origins}"

        # Build the module-level code block
        import_alias = f"_tilelang_{suffix}"
        cache_var = f"_{kernel_name}_cache"
        factory_fn = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang kernel — {metadata.strip()}")
        code.writeline(f"import tilelang as {import_alias}")
        code.writeline("")

        # prim_func factory function (closes over _xnumel for dynamic shapes)
        if numel_arg_names:
            factory_params = ", ".join(f"_{n}" for n in numel_arg_names)
        else:
            factory_params = "_dummy=None"
        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            # _xnumel = _xnumel (make available to the prim_func body)
            if numel_arg_names:
                for n in numel_arg_names:
                    code.writeline(f"_xnumel = _{n}")  # expose as _xnumel
                    break  # use first (xnumel) for initial 1-D support
            else:
                code.writeline("_xnumel = 1")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")
        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                key_parts = ", ".join(f"int({n})" for n in numel_arg_names)
                code.writeline(f"_key = ({key_parts},)")
            else:
                code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                if numel_arg_names:
                    factory_call_args = ", ".join(
                        f"_key[{i}]" for i in range(len(numel_arg_names))
                    )
                else:
                    factory_call_args = ""
                code.writeline(f"{cache_var}[_key] = {import_alias}.compile(")
                with code.indent():
                    code.writeline(
                        f"{factory_fn}({factory_call_args}), target='npuir'"
                    )
                code.writeline(")")
            code.writeline(
                f"{cache_var}[_key]({', '.join(tensor_call_args)})"
            )

        wrapper.header.splice(code.getvalue())
        return kernel_name
