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
DataFrame operations using WASM backend.

Provides join, groupby, filter, and aggregation operations.
"""

import json
from typing import Dict, List, Optional, Union, Any

from .table import Table
from .core import get_runtime


# =============================================================================
# Join Operations
# =============================================================================

def join(
    left: Table,
    right: Table,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, int, List[Union[str, int]]]] = None,
    right_on: Optional[Union[str, int, List[Union[str, int]]]] = None,
    how: str = "inner",
    left_suffix: str = "_l",
    right_suffix: str = "_r",
) -> Table:
    """
    Join two tables.

    Args:
        left: Left table
        right: Right table
        on: Column name(s) to join on (must exist in both tables)
        left_on: Column name(s) or index(es) from left table
        right_on: Column name(s) or index(es) from right table
        how: Join type - "inner", "left", "right", or "full_outer"
        left_suffix: Suffix for duplicate column names from left table
        right_suffix: Suffix for duplicate column names from right table

    Returns:
        Joined table

    Example:
        result = join(employees, departments, on="dept_id", how="inner")
        result = join(t1, t2, left_on="id", right_on="employee_id")
    """
    # Resolve column specifications
    if on is not None:
        if isinstance(on, str):
            on = [on]
        left_on = [left.column_index(c) for c in on]
        right_on = [right.column_index(c) for c in on]
    else:
        if left_on is None or right_on is None:
            raise ValueError("Must specify 'on' or both 'left_on' and 'right_on'")

        # Convert to lists
        if not isinstance(left_on, list):
            left_on = [left_on]
        if not isinstance(right_on, list):
            right_on = [right_on]

        # Convert names to indices
        left_on = [
            left.column_index(c) if isinstance(c, str) else c
            for c in left_on
        ]
        right_on = [
            right.column_index(c) if isinstance(c, str) else c
            for c in right_on
        ]

    # Map how to join_type
    join_type_map = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "full_outer",
        "full_outer": "full_outer",
        "full": "full_outer",
    }
    join_type = join_type_map.get(how.lower())
    if join_type is None:
        raise ValueError(f"Unknown join type: {how}")

    # Build config
    config = {
        "join_type": join_type,
        "left_on": left_on,
        "right_on": right_on,
        "left_suffix": left_suffix,
        "right_suffix": right_suffix,
    }

    # Call WASM function
    runtime = get_runtime()
    result_json = runtime.call_json_function(
        "join_tables",
        left.to_json(),
        right.to_json(),
        json.dumps(config)
    )

    return Table.from_json(result_json)


# =============================================================================
# GroupBy Operations
# =============================================================================

class GroupByResult:
    """
    Intermediate result for groupby operations.

    Example:
        grouped = table.groupby("category")
        result = grouped.agg({"value": "sum", "count": "count"})
    """

    def __init__(self, table: Table, keys: List[int]):
        self._table = table
        self._keys = keys

    def agg(self, aggregations: Dict[str, str]) -> Table:
        """
        Apply aggregations.

        Args:
            aggregations: Dict mapping column names to aggregation operations.
                         Operations: "sum", "mean", "min", "max", "count"

        Returns:
            Aggregated table
        """
        agg_list = []
        for col_name, op in aggregations.items():
            col_idx = self._table.column_index(col_name)
            agg_list.append({
                "column": col_idx,
                "op": op.lower(),
                "alias": f"{col_name}_{op.lower()}"
            })

        config = {
            "keys": self._keys,
            "aggregations": agg_list,
        }

        runtime = get_runtime()
        result_json = runtime.call_json_function(
            "groupby_table",
            self._table.to_json(),
            json.dumps(config)
        )

        return Table.from_json(result_json)

    def sum(self, column: str) -> Table:
        """Compute sum."""
        return self.agg({column: "sum"})

    def mean(self, column: str) -> Table:
        """Compute mean."""
        return self.agg({column: "mean"})

    def min(self, column: str) -> Table:
        """Compute min."""
        return self.agg({column: "min"})

    def max(self, column: str) -> Table:
        """Compute max."""
        return self.agg({column: "max"})

    def count(self) -> Table:
        """Compute count."""
        # Use first non-key column for count
        for i, col in enumerate(self._table.columns):
            if i not in self._keys:
                return self.agg({col: "count"})
        # Fallback to first column
        return self.agg({self._table.columns[0]: "count"})


def groupby(
    table: Table,
    keys: Union[str, int, List[Union[str, int]]],
) -> GroupByResult:
    """
    Group table by key columns.

    Args:
        table: Input table
        keys: Column name(s) or index(es) to group by

    Returns:
        GroupByResult for chaining aggregations

    Example:
        result = groupby(table, "category").agg({"value": "sum"})
        result = groupby(table, ["region", "year"]).agg({"sales": "sum", "sales": "mean"})
    """
    if not isinstance(keys, list):
        keys = [keys]

    key_indices = [
        table.column_index(k) if isinstance(k, str) else k
        for k in keys
    ]

    return GroupByResult(table, key_indices)


# =============================================================================
# Filter Operations
# =============================================================================

def filter_table(
    table: Table,
    predicates: List[Dict[str, Any]],
    logic: str = "and",
) -> Table:
    """
    Filter table rows based on predicates.

    Args:
        table: Input table
        predicates: List of predicate dicts with keys:
                   - column: column name or index
                   - op: "eq", "ne", "lt", "le", "gt", "ge"
                   - value: comparison value
        logic: "and" or "or" for combining predicates

    Returns:
        Filtered table

    Example:
        result = filter_table(table, [
            {"column": "age", "op": "gt", "value": 18},
            {"column": "status", "op": "eq", "value": "active"}
        ], logic="and")
    """
    # Convert column names to indices
    converted_predicates = []
    for pred in predicates:
        col = pred["column"]
        if isinstance(col, str):
            col = table.column_index(col)
        converted_predicates.append({
            "column": col,
            "op": pred["op"],
            "value": pred["value"],
        })

    config = {
        "predicates": converted_predicates,
        "logic": logic.lower(),
    }

    runtime = get_runtime()
    result_json = runtime.call_json_function(
        "filter_table",
        table.to_json(),
        json.dumps(config)
    )

    return Table.from_json(result_json)


# =============================================================================
# Aggregation Operations
# =============================================================================

def aggregate(
    table: Table,
    column: Union[str, int],
    op: str,
) -> float:
    """
    Compute single aggregation over a column.

    Args:
        table: Input table
        column: Column name or index
        op: Aggregation operation - "sum", "mean", "min", "max", "count"

    Returns:
        Aggregation result

    Example:
        total = aggregate(table, "sales", "sum")
        average = aggregate(table, "price", "mean")
    """
    if isinstance(column, str):
        column = table.column_index(column)

    runtime = get_runtime()
    # For aggregate, we could call the WASM function directly
    # or use the JSON API

    # Using JSON API approach
    config = {
        "keys": [],  # No grouping
        "aggregations": [{"column": column, "op": op.lower()}]
    }

    result_json = runtime.call_json_function(
        "aggregate",
        table.to_json(),
        str(column),
        op.lower()
    )

    # Parse result
    return float(result_json)


# =============================================================================
# Convenience methods for Table class
# =============================================================================

def _add_table_methods():
    """Add operation methods to Table class."""

    def _groupby(self, keys):
        return groupby(self, keys)

    def _filter(self, predicates, logic="and"):
        return filter_table(self, predicates, logic)

    def _join(self, right, **kwargs):
        return join(self, right, **kwargs)

    Table.groupby = _groupby
    Table.filter = _filter
    Table.join = _join


# Add methods on module load
_add_table_methods()
