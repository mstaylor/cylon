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
PyCylon WASM - Python wrapper for Cylon WASM operations.

This module provides a Python interface to Cylon DataFrame operations
running in WebAssembly via wasmtime.

Example:
    from pycylonwasm import Table, join, groupby, filter_table

    # Create tables
    left = Table({
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"]
    })
    right = Table({
        "id": [1, 2, 4],
        "dept": ["eng", "sales", "marketing"]
    })

    # Join
    result = join(left, right, on="id", how="inner")

    # GroupBy
    grouped = left.groupby("id").agg({"name": "count"})

    # Filter
    filtered = left.filter(left["id"] > 1)
"""

from .core import WasmRuntime, get_runtime, set_wasm_path
from .table import Table
from .operations import join, groupby, filter_table, aggregate

__version__ = "0.1.0"

__all__ = [
    "WasmRuntime",
    "get_runtime",
    "set_wasm_path",
    "Table",
    "join",
    "groupby",
    "filter_table",
    "aggregate",
]
