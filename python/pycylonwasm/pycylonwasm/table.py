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
Table class for pycylon_wasm.

Provides a pandas-like interface for tables that can be processed via WASM.
"""

import json
from typing import Dict, List, Any, Optional, Union, Sequence


def _infer_column_type(values: Sequence) -> str:
    """Infer the column type from values."""
    # Filter out None values for type inference
    non_null = [v for v in values if v is not None]

    if not non_null:
        return "Int32"  # Default for all-null columns

    sample = non_null[0]

    if isinstance(sample, bool):
        return "Boolean"
    elif isinstance(sample, int):
        # Check if values fit in Int32
        if all(isinstance(v, int) and -2147483648 <= v <= 2147483647 for v in non_null):
            return "Int32"
        return "Int64"
    elif isinstance(sample, float):
        return "Float64"
    elif isinstance(sample, str):
        return "String"
    else:
        raise TypeError(f"Unsupported type: {type(sample)}")


def _convert_values(values: Sequence, col_type: str) -> List:
    """Convert values to the appropriate type with None handling."""
    result = []
    for v in values:
        if v is None:
            result.append(None)
        elif col_type == "Boolean":
            result.append(bool(v))
        elif col_type in ("Int32", "Int64"):
            result.append(int(v))
        elif col_type in ("Float32", "Float64"):
            result.append(float(v))
        elif col_type == "String":
            result.append(str(v))
        else:
            result.append(v)
    return result


class Table:
    """
    A table/DataFrame representation for WASM operations.

    Stores data in a format compatible with cylon-wasm JSON serialization.

    Example:
        # From dict
        t = Table({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        # From columns
        t = Table.from_columns(
            columns=["id", "name"],
            data=[
                {"type": "Int32", "data": [1, 2, 3]},
                {"type": "String", "data": ["a", "b", "c"]}
            ]
        )

        # Access
        print(t.num_rows)
        print(t.columns)
        print(t["id"])
    """

    def __init__(self, data: Optional[Dict[str, Sequence]] = None):
        """
        Create a table from a dictionary.

        Args:
            data: Dict mapping column names to value lists.
                  Values can be int, float, str, bool, or None.
        """
        self._columns: List[str] = []
        self._data: List[Dict[str, Any]] = []
        self._num_rows = 0

        if data:
            self._from_dict(data)

    def _from_dict(self, data: Dict[str, Sequence]) -> None:
        """Initialize from dictionary."""
        if not data:
            return

        # Validate all columns have same length
        lengths = [len(v) for v in data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All columns must have the same length")

        self._num_rows = lengths[0] if lengths else 0
        self._columns = list(data.keys())
        self._data = []

        for col_name, values in data.items():
            col_type = _infer_column_type(values)
            converted = _convert_values(values, col_type)
            self._data.append({
                "type": col_type,
                "data": converted
            })

    @classmethod
    def from_columns(cls, columns: List[str], data: List[Dict[str, Any]]) -> "Table":
        """
        Create table from column definitions.

        Args:
            columns: List of column names
            data: List of column data dicts with "type" and "data" keys
        """
        table = cls()
        table._columns = columns
        table._data = data
        table._num_rows = len(data[0]["data"]) if data else 0
        return table

    @classmethod
    def from_json(cls, json_str: str) -> "Table":
        """Create table from JSON string."""
        parsed = json.loads(json_str)
        return cls.from_columns(parsed["columns"], parsed["data"])

    def to_json(self) -> str:
        """Serialize table to JSON string."""
        return json.dumps({
            "columns": self._columns,
            "data": self._data
        })

    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary of columns."""
        return {
            col: self._data[i]["data"]
            for i, col in enumerate(self._columns)
        }

    @property
    def num_rows(self) -> int:
        """Number of rows."""
        return self._num_rows

    @property
    def num_columns(self) -> int:
        """Number of columns."""
        return len(self._columns)

    @property
    def columns(self) -> List[str]:
        """Column names."""
        return self._columns.copy()

    @property
    def dtypes(self) -> Dict[str, str]:
        """Column data types."""
        return {
            col: self._data[i]["type"]
            for i, col in enumerate(self._columns)
        }

    def column_index(self, name: str) -> int:
        """Get column index by name."""
        try:
            return self._columns.index(name)
        except ValueError:
            raise KeyError(f"Column '{name}' not found")

    def __getitem__(self, key: Union[str, int]) -> List:
        """Get column by name or index."""
        if isinstance(key, str):
            idx = self.column_index(key)
        else:
            idx = key
        return self._data[idx]["data"]

    def __len__(self) -> int:
        """Number of rows."""
        return self._num_rows

    def __repr__(self) -> str:
        """String representation."""
        return f"Table({self._columns}, {self._num_rows} rows)"

    def head(self, n: int = 5) -> "Table":
        """Return first n rows."""
        new_data = []
        for col_data in self._data:
            new_data.append({
                "type": col_data["type"],
                "data": col_data["data"][:n]
            })
        return Table.from_columns(self._columns.copy(), new_data)

    def select(self, columns: List[str]) -> "Table":
        """Select specific columns."""
        indices = [self.column_index(c) for c in columns]
        new_data = [self._data[i] for i in indices]
        return Table.from_columns(columns, new_data)

    def rename(self, mapping: Dict[str, str]) -> "Table":
        """Rename columns."""
        new_columns = [mapping.get(c, c) for c in self._columns]
        return Table.from_columns(new_columns, self._data.copy())

    def to_pandas(self):
        """Convert to pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.to_dict())
        except ImportError:
            raise ImportError("pandas is required for to_pandas()")

    @classmethod
    def from_pandas(cls, df) -> "Table":
        """Create from pandas DataFrame."""
        return cls(df.to_dict(orient="list"))

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        rows = min(10, self._num_rows)

        html = "<table><thead><tr>"
        for col in self._columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"

        for i in range(rows):
            html += "<tr>"
            for col_data in self._data:
                val = col_data["data"][i]
                html += f"<td>{val}</td>"
            html += "</tr>"

        html += "</tbody></table>"

        if self._num_rows > 10:
            html += f"<p>... {self._num_rows - 10} more rows</p>"

        return html
