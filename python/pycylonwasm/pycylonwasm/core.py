# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WASM runtime management using wasmtime.

This module handles loading and interacting with the cylon-wasm module.
Implements the wasm-bindgen ABI for string passing and result handling.
"""

import os
import struct
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from wasmtime import Store, Module, Instance, Linker, FuncType, ValType, Func
except ImportError:
    raise ImportError(
        "wasmtime is required for pycylonwasm. "
        "Install with: pip install wasmtime"
    )


# Default path to WASM binary
def _get_default_wasm_path() -> str:
    """Get the default path to the WASM binary."""
    env_path = os.environ.get("CYLON_WASM_PATH")
    if env_path:
        return env_path
    # Relative to this file: pycylonwasm/ -> python/ -> cylon/
    return str(Path(__file__).parent.parent.parent.parent / "rust" / "cylon-wasm" / "pkg" / "cylon_wasm_bg.wasm")


class WasmRuntime:
    """
    Manages the WASM runtime and provides access to cylon-wasm functions.

    This class handles:
    - Loading the WASM module
    - Memory management for string passing (wasm-bindgen ABI)
    - Calling exported functions
    """

    def __init__(self, wasm_path: Optional[str] = None):
        """
        Initialize the WASM runtime.

        Args:
            wasm_path: Path to the cylon_wasm_bg.wasm file.
                      If None, uses CYLON_WASM_PATH env var or default location.
        """
        self.wasm_path = wasm_path or _get_default_wasm_path()
        self._store: Optional[Store] = None
        self._instance: Optional[Instance] = None
        self._initialized = False

        # Cached exports
        self._memory = None
        self._malloc = None
        self._realloc = None
        self._free = None

    def __enter__(self) -> "WasmRuntime":
        """Context manager entry - initialize runtime."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup."""
        # wasmtime handles cleanup automatically via Python GC
        self._initialized = False
        self._store = None
        self._instance = None
        self._memory = None

    def _define_wbg_stubs(self, linker: Linker) -> None:
        """
        Define stub functions for wasm-bindgen imports.

        wasm-bindgen generates JS glue code that provides these functions.
        For Python, we provide minimal stubs that allow the module to load.
        """
        # Error storage for capturing WASM errors
        self._last_error = None

        # Stub for creating JS Error objects: () -> anyref
        def wbg_new():
            # Return None as null externref
            return None

        # Stub for getting error stack: (i32, anyref) -> ()
        def wbg_stack(ptr: int, err_ref):
            pass

        # Stub for console.error: (i32, i32) -> ()
        def wbg_error(ptr: int, len: int):
            pass

        # Stub for initializing externref table: () -> ()
        def wbg_init_externref():
            pass

        # Stub for JS value cast: (i32, i32) -> anyref
        def wbg_cast(a: int, b: int):
            return None

        # Define the imports in the "wbg" namespace
        linker.define_func(
            "wbg", "__wbg_new_8a6f238a6ece86ea",
            FuncType([], [ValType.externref()]),
            wbg_new
        )
        linker.define_func(
            "wbg", "__wbg_stack_0ed75d68575b0f3c",
            FuncType([ValType.i32(), ValType.externref()], []),
            wbg_stack
        )
        linker.define_func(
            "wbg", "__wbg_error_7534b8e9a36f1ab4",
            FuncType([ValType.i32(), ValType.i32()], []),
            wbg_error
        )
        linker.define_func(
            "wbg", "__wbindgen_init_externref_table",
            FuncType([], []),
            wbg_init_externref
        )
        linker.define_func(
            "wbg", "__wbindgen_cast_2241b6af4c4b2941",
            FuncType([ValType.i32(), ValType.i32()], [ValType.externref()]),
            wbg_cast
        )

    def initialize(self) -> None:
        """Load and initialize the WASM module."""
        if self._initialized:
            return

        if not os.path.exists(self.wasm_path):
            raise FileNotFoundError(
                f"WASM binary not found at {self.wasm_path}. "
                f"Build with: cd rust/cylon-wasm && wasm-pack build --target web"
            )

        self._store = Store()
        module = Module.from_file(self._store.engine, self.wasm_path)

        # Create linker and define wasm-bindgen stub imports
        linker = Linker(self._store.engine)
        self._define_wbg_stubs(linker)

        # Instantiate module
        self._instance = linker.instantiate(self._store, module)

        # Cache commonly used exports
        exports = self._instance.exports(self._store)
        self._memory = exports["memory"]
        self._malloc = exports["__wbindgen_malloc"]
        self._realloc = exports["__wbindgen_realloc"]
        self._free = exports["__wbindgen_free"]

        # Call init if available (sets up panic hook)
        init_func = exports.get("init")
        if init_func:
            init_func(self._store)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure the runtime is initialized."""
        if not self._initialized:
            self.initialize()

    def _get_export(self, name: str):
        """Get an exported function by name."""
        self._ensure_initialized()
        return self._instance.exports(self._store).get(name)

    def _read_memory(self, ptr: int, length: int) -> bytes:
        """Read bytes from WASM memory."""
        self._ensure_initialized()
        data = self._memory.data_ptr(self._store)
        return bytes(data[ptr:ptr + length])

    def _write_memory(self, ptr: int, data: bytes) -> None:
        """Write bytes to WASM memory."""
        self._ensure_initialized()
        mem = self._memory.data_ptr(self._store)
        for i, b in enumerate(data):
            mem[ptr + i] = b

    def _alloc_string(self, s: str) -> Tuple[int, int]:
        """
        Allocate a string in WASM memory (wasm-bindgen ABI).

        Returns:
            Tuple of (pointer, length)
        """
        self._ensure_initialized()
        encoded = s.encode("utf-8")
        length = len(encoded)

        # Allocate memory: malloc(size, align)
        ptr = self._malloc(self._store, length, 1)

        # Write string to memory
        self._write_memory(ptr, encoded)

        return ptr, length

    def _read_string(self, ptr: int, length: int) -> str:
        """Read a string from WASM memory."""
        data = self._read_memory(ptr, length)
        return data.decode("utf-8")

    def _free_memory(self, ptr: int, length: int) -> None:
        """Free allocated memory: free(ptr, size, align)."""
        if ptr and self._free:
            self._free(self._store, ptr, length, 1)

    # =========================================================================
    # Simple exports (no string returns)
    # =========================================================================

    def simd_available(self) -> bool:
        """Check if SIMD is available."""
        self._ensure_initialized()
        func = self._get_export("simd_available")
        result = func(self._store)
        return result != 0

    def version(self) -> str:
        """Get the cylon-wasm version."""
        self._ensure_initialized()
        func = self._get_export("version")
        result = func(self._store)
        # Result is [ptr, len]
        ptr, length = result[0], result[1]
        try:
            return self._read_string(ptr, length)
        finally:
            self._free_memory(ptr, length)

    def sum_f32(self, data: List[float]) -> float:
        """Compute sum of f32 array."""
        self._ensure_initialized()
        import array
        arr = array.array('f', data)
        arr_bytes = arr.tobytes()

        # Allocate and copy
        ptr = self._malloc(self._store, len(data) * 4, 4)
        self._write_memory(ptr, arr_bytes)

        func = self._get_export("sum_f32")
        result = func(self._store, ptr, len(data))
        return result

    def sum_f64(self, data: List[float]) -> float:
        """Compute sum of f64 array."""
        self._ensure_initialized()
        import array
        arr = array.array('d', data)
        arr_bytes = arr.tobytes()

        # Allocate and copy
        ptr = self._malloc(self._store, len(data) * 8, 8)
        self._write_memory(ptr, arr_bytes)

        func = self._get_export("sum_f64")
        result = func(self._store, ptr, len(data))
        return result

    # =========================================================================
    # String-returning exports (JSON API)
    # =========================================================================

    def _alloc_bytes(self, data: bytes) -> Tuple[int, int]:
        """
        Allocate bytes in WASM memory.

        Returns:
            Tuple of (pointer, length)
        """
        self._ensure_initialized()
        length = len(data)

        # Allocate memory: malloc(size, align)
        ptr = self._malloc(self._store, length, 1)

        # Write bytes to memory
        self._write_memory(ptr, data)

        return ptr, length

    def _read_bytes(self, ptr: int, length: int) -> bytes:
        """Read bytes from WASM memory."""
        return self._read_memory(ptr, length)

    # =========================================================================
    # Arrow IPC API (efficient binary format)
    # =========================================================================

    def _call_ipc_func_2arg(self, func_name: str, data1: bytes, data2: bytes) -> bytes:
        """Call a WASM function with 2 Arrow IPC args, returns Arrow IPC."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr1, len1 = self._alloc_bytes(data1)
        ptr2, len2 = self._alloc_bytes(data2)
        result_ptr = 0
        result_len = 0

        try:
            result = func(self._store, ptr1, len1, ptr2, len2)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_bytes(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    def _call_ipc_with_config(self, func_name: str, data: bytes, config_json: str) -> bytes:
        """Call a WASM function with Arrow IPC data and JSON config."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr1, len1 = self._alloc_bytes(data)
        ptr2, len2 = self._alloc_string(config_json)
        result_ptr = 0
        result_len = 0

        try:
            result = func(self._store, ptr1, len1, ptr2, len2)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_bytes(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    def _call_ipc_2tables_with_config(self, func_name: str, left: bytes, right: bytes, config_json: str) -> bytes:
        """Call a WASM function with 2 Arrow IPC tables and JSON config."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr1, len1 = self._alloc_bytes(left)
        ptr2, len2 = self._alloc_bytes(right)
        ptr3, len3 = self._alloc_string(config_json)
        result_ptr = 0
        result_len = 0

        try:
            result = func(self._store, ptr1, len1, ptr2, len2, ptr3, len3)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_bytes(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    # Arrow IPC API functions

    def table_info_ipc(self, data: bytes) -> str:
        """Get table info from Arrow IPC data. Returns JSON."""
        self._ensure_initialized()
        func = self._get_export("table_info")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length)
            result_ptr = result[0]
            result_len = result[1]
            return self._read_string(result_ptr, result_len)
        finally:
            pass

    def json_to_ipc(self, json_str: str) -> bytes:
        """Convert JSON table to Arrow IPC format."""
        self._ensure_initialized()
        func = self._get_export("json_to_ipc")
        ptr, length = self._alloc_string(json_str)
        try:
            result = func(self._store, ptr, length)
            result_ptr = result[0]
            result_len = result[1]
            return self._read_bytes(result_ptr, result_len)
        finally:
            pass

    def ipc_to_json(self, data: bytes) -> str:
        """Convert Arrow IPC to JSON (for debugging)."""
        self._ensure_initialized()
        func = self._get_export("ipc_to_json")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length)
            result_ptr = result[0]
            result_len = result[1]
            return self._read_string(result_ptr, result_len)
        finally:
            pass

    def join_tables_ipc(self, left: bytes, right: bytes, config_json: str) -> bytes:
        """Join two tables (Arrow IPC format)."""
        return self._call_ipc_2tables_with_config("join_tables", left, right, config_json)

    def filter_table_ipc(self, data: bytes, config_json: str) -> bytes:
        """Filter table rows (Arrow IPC format)."""
        return self._call_ipc_with_config("filter_table", data, config_json)

    def groupby_table_ipc(self, data: bytes, config_json: str) -> bytes:
        """GroupBy with aggregations (Arrow IPC format)."""
        return self._call_ipc_with_config("groupby_table", data, config_json)

    def project_table_ipc(self, data: bytes, columns_json: str) -> bytes:
        """Project (select) columns (Arrow IPC format)."""
        return self._call_ipc_with_config("project_table", data, columns_json)

    def sort_table_ipc(self, data: bytes, column: int, ascending: bool) -> bytes:
        """Sort table by single column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("sort_table")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column, 1 if ascending else 0)
            result_ptr = result[0]
            result_len = result[1]
            return self._read_bytes(result_ptr, result_len)
        finally:
            pass

    def union_tables_ipc(self, left: bytes, right: bytes) -> bytes:
        """Union two tables (Arrow IPC format)."""
        return self._call_ipc_func_2arg("union_tables", left, right)

    def intersect_tables_ipc(self, left: bytes, right: bytes) -> bytes:
        """Intersect two tables (Arrow IPC format)."""
        return self._call_ipc_func_2arg("intersect_tables", left, right)

    def subtract_tables_ipc(self, left: bytes, right: bytes) -> bytes:
        """Subtract tables (Arrow IPC format)."""
        return self._call_ipc_func_2arg("subtract_tables", left, right)

    def unique_table_ipc(self, data: bytes, columns_json: str, keep_first: bool) -> bytes:
        """Remove duplicate rows (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("unique_table")
        ptr1, len1 = self._alloc_bytes(data)
        ptr2, len2 = self._alloc_string(columns_json)
        try:
            result = func(self._store, ptr1, len1, ptr2, len2, 1 if keep_first else 0)
            result_ptr = result[0]
            result_len = result[1]
            return self._read_bytes(result_ptr, result_len)
        finally:
            pass

    def compute_sum_ipc(self, data: bytes, column: int) -> float:
        """Compute sum of column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("compute_sum")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column)
            return result[0]
        finally:
            pass

    def compute_mean_ipc(self, data: bytes, column: int) -> float:
        """Compute mean of column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("compute_mean")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column)
            return result[0]
        finally:
            pass

    def compute_min_ipc(self, data: bytes, column: int) -> float:
        """Compute min of column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("compute_min")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column)
            return result[0]
        finally:
            pass

    def compute_max_ipc(self, data: bytes, column: int) -> float:
        """Compute max of column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("compute_max")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column)
            return result[0]
        finally:
            pass

    def compute_count_ipc(self, data: bytes, column: int) -> int:
        """Compute count of column (Arrow IPC format)."""
        self._ensure_initialized()
        func = self._get_export("compute_count")
        ptr, length = self._alloc_bytes(data)
        try:
            result = func(self._store, ptr, length, column)
            return int(result[0])
        finally:
            pass

    # =========================================================================
    # JSON API (legacy, kept for compatibility)
    # =========================================================================

    def _call_json_func_1arg(self, func_name: str, json_arg: str) -> str:
        """Call a WASM function with 1 JSON string arg, returns JSON string."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr, length = self._alloc_string(json_arg)
        result_ptr = 0
        result_len = 0

        try:
            # Call: func(ptr, len) -> [result_ptr, result_len, error_idx, has_error]
            result = func(self._store, ptr, length)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                # Error occurred - try to read error message if available
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_string(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    def _call_json_func_2arg(self, func_name: str, json_arg1: str, json_arg2: str) -> str:
        """Call a WASM function with 2 JSON string args, returns JSON string."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr1, len1 = self._alloc_string(json_arg1)
        ptr2, len2 = self._alloc_string(json_arg2)
        result_ptr = 0
        result_len = 0

        try:
            # Call: func(ptr1, len1, ptr2, len2) -> [result_ptr, result_len, error_idx, has_error]
            result = func(self._store, ptr1, len1, ptr2, len2)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_string(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    def _call_json_func_3arg(self, func_name: str, arg1: str, arg2: str, arg3: str) -> str:
        """Call a WASM function with 3 JSON string args, returns JSON string."""
        self._ensure_initialized()
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        ptr1, len1 = self._alloc_string(arg1)
        ptr2, len2 = self._alloc_string(arg2)
        ptr3, len3 = self._alloc_string(arg3)
        result_ptr = 0
        result_len = 0

        try:
            result = func(self._store, ptr1, len1, ptr2, len2, ptr3, len3)
            result_ptr = result[0]
            result_len = result[1]
            has_error = result[3] if len(result) > 3 else False

            if has_error:
                raise RuntimeError(f"WASM error in {func_name}")

            return self._read_string(result_ptr, result_len)
        finally:
            self._free_memory(result_ptr, result_len)

    def aggregate(self, table_json: str, column: int, op: str) -> float:
        """
        Compute aggregation over a column.

        Args:
            table_json: Table as JSON string
            column: Column index
            op: Aggregation operation (sum, mean, min, max, count)

        Returns:
            Aggregation result as float
        """
        self._ensure_initialized()
        func = self._get_export("aggregate")

        ptr1, len1 = self._alloc_string(table_json)
        ptr2, len2 = self._alloc_string(op)

        try:
            # aggregate(table_ptr, table_len, column, op_ptr, op_len) -> [result, error_idx, has_error]
            result = func(self._store, ptr1, len1, column, ptr2, len2)
            has_error = result[2] if len(result) > 2 else False

            if has_error:
                raise RuntimeError(f"WASM error in aggregate")

            return result[0]
        finally:
            pass  # Input strings are not freed by wasm-bindgen for us

    def table_info(self, table_json: str) -> str:
        """Get table info as JSON."""
        return self._call_json_func_1arg("table_info", table_json)

    def filter_table(self, table_json: str, config_json: str) -> str:
        """Filter table rows."""
        return self._call_json_func_2arg("filter_table", table_json, config_json)

    def groupby_table(self, table_json: str, config_json: str) -> str:
        """Group table and compute aggregations."""
        return self._call_json_func_2arg("groupby_table", table_json, config_json)

    def join_tables(self, left_json: str, right_json: str, config_json: str) -> str:
        """Join two tables."""
        return self._call_json_func_3arg("join_tables", left_json, right_json, config_json)

    def select_columns(self, table_json: str, columns_json: str) -> str:
        """Select specific columns from table."""
        return self._call_json_func_2arg("select_columns", table_json, columns_json)

    def concat_tables(self, top_json: str, bottom_json: str) -> str:
        """Concatenate two tables vertically."""
        return self._call_json_func_2arg("concat_tables", top_json, bottom_json)

    def execute_pipeline(self, table_json: str, pipeline_json: str) -> str:
        """Execute a pipeline of operations."""
        return self._call_json_func_2arg("execute_pipeline", table_json, pipeline_json)


def create_runtime(wasm_path: Optional[str] = None) -> WasmRuntime:
    """
    Create a new WASM runtime instance.

    Args:
        wasm_path: Path to cylon_wasm_bg.wasm file.
                   If None, uses CYLON_WASM_PATH env var or default location.

    Returns:
        Initialized WasmRuntime instance.

    Example:
        # As context manager (recommended)
        with create_runtime() as runtime:
            result = runtime.join_tables(left, right, config)

        # Or manual management
        runtime = create_runtime()
        try:
            result = runtime.filter_table(table, config)
        finally:
            pass  # cleanup handled by GC
    """
    runtime = WasmRuntime(wasm_path)
    runtime.initialize()
    return runtime
