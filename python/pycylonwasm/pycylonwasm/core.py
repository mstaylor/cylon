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
"""

import json
import os
from pathlib import Path
from typing import Optional, Any, Dict

try:
    from wasmtime import Store, Module, Instance, Func, Memory
except ImportError:
    raise ImportError(
        "wasmtime is required for pycylon_wasm. "
        "Install with: pip install wasmtime"
    )


# Default path to WASM binary (can be overridden)
_DEFAULT_WASM_PATH = os.environ.get(
    "CYLON_WASM_PATH",
    str(Path(__file__).parent.parent.parent / "rust" / "cylon-wasm" / "pkg" / "cylon_wasm_bg.wasm")
)

# Global runtime instance
_runtime: Optional["WasmRuntime"] = None


class WasmRuntime:
    """
    Manages the WASM runtime and provides access to cylon-wasm functions.

    This class handles:
    - Loading the WASM module
    - Memory management for string passing
    - Calling exported functions
    """

    def __init__(self, wasm_path: Optional[str] = None):
        """
        Initialize the WASM runtime.

        Args:
            wasm_path: Path to the cylon_wasm_bg.wasm file.
                      If None, uses CYLON_WASM_PATH env var or default location.
        """
        self.wasm_path = wasm_path or _DEFAULT_WASM_PATH
        self._store: Optional[Store] = None
        self._instance: Optional[Instance] = None
        self._memory: Optional[Memory] = None
        self._initialized = False

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

        # Create instance with imports (empty for now, may need JS stubs)
        self._instance = Instance(self._store, module, [])

        # Get memory export
        self._memory = self._instance.exports(self._store).get("memory")

        self._initialized = True

    @property
    def store(self) -> Store:
        """Get the WASM store."""
        if not self._initialized:
            self.initialize()
        return self._store

    @property
    def instance(self) -> Instance:
        """Get the WASM instance."""
        if not self._initialized:
            self.initialize()
        return self._instance

    def _get_export(self, name: str) -> Any:
        """Get an exported function by name."""
        if not self._initialized:
            self.initialize()
        return self._instance.exports(self._store).get(name)

    def _alloc_string(self, s: str) -> tuple:
        """
        Allocate a string in WASM memory.

        Returns:
            Tuple of (pointer, length)
        """
        # Get allocator function
        alloc = self._get_export("__wbindgen_malloc")
        if alloc is None:
            raise RuntimeError("WASM module missing __wbindgen_malloc export")

        encoded = s.encode("utf-8")
        length = len(encoded)

        # Allocate memory
        ptr = alloc(self._store, length, 1)

        # Write string to memory
        memory_data = self._memory.data_ptr(self._store)
        for i, byte in enumerate(encoded):
            memory_data[ptr + i] = byte

        return ptr, length

    def _read_string(self, ptr: int, length: int) -> str:
        """Read a string from WASM memory."""
        memory_data = self._memory.data_ptr(self._store)
        bytes_data = bytes(memory_data[ptr:ptr + length])
        return bytes_data.decode("utf-8")

    def _free(self, ptr: int, length: int) -> None:
        """Free allocated memory."""
        free = self._get_export("__wbindgen_free")
        if free:
            free(self._store, ptr, length, 1)

    def call_json_function(self, func_name: str, *json_args: str) -> str:
        """
        Call a WASM function that takes JSON strings and returns JSON.

        Args:
            func_name: Name of the exported function
            *json_args: JSON string arguments

        Returns:
            JSON string result
        """
        func = self._get_export(func_name)
        if func is None:
            raise RuntimeError(f"WASM function '{func_name}' not found")

        # For wasm-bindgen generated code, we need to handle the ABI
        # This is a simplified version - full implementation would use
        # wasm-bindgen's JS shim or implement the full ABI

        # Allocate input strings
        ptrs = []
        for arg in json_args:
            ptr, length = self._alloc_string(arg)
            ptrs.append((ptr, length))

        try:
            # Call function (ABI depends on wasm-bindgen version)
            # This is a placeholder - actual implementation needs ABI handling
            result_ptr = func(self._store, *[p for ptr_len in ptrs for p in ptr_len])

            # Read result (simplified)
            # Actual implementation would parse wasm-bindgen return format
            return "{}"  # Placeholder

        finally:
            # Free input strings
            for ptr, length in ptrs:
                self._free(ptr, length)

    def version(self) -> str:
        """Get the cylon-wasm version."""
        version_func = self._get_export("version")
        if version_func:
            # Call and decode result
            pass
        return "0.1.0"

    def simd_available(self) -> bool:
        """Check if SIMD is available."""
        simd_func = self._get_export("simd_available")
        if simd_func:
            return bool(simd_func(self._store))
        return False


def get_runtime() -> WasmRuntime:
    """
    Get the global WASM runtime instance.

    Creates and initializes the runtime on first call.
    """
    global _runtime
    if _runtime is None:
        _runtime = WasmRuntime()
        _runtime.initialize()
    return _runtime


def set_wasm_path(path: str) -> None:
    """
    Set the path to the WASM binary.

    Must be called before get_runtime() or any operations.

    Args:
        path: Path to cylon_wasm_bg.wasm file
    """
    global _runtime
    if _runtime is not None and _runtime._initialized:
        raise RuntimeError("Cannot change WASM path after runtime is initialized")
    _runtime = WasmRuntime(path)
