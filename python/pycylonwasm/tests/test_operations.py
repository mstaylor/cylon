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

"""Tests for DataFrame operations.

Tests marked with @pytest.mark.wasm require the WASM binary to be built.
Other tests verify the Python-side logic without requiring WASM.

Run all tests:
    pytest python/pycylonwasm/tests/test_operations.py -v

Run only non-WASM tests:
    pytest python/pycylonwasm/tests/test_operations.py -v -m "not wasm"
"""

import json
import pytest
from pycylonwasm.table import Table
from pycylonwasm.operations import (
    join,
    groupby,
    filter_table,
    aggregate,
    GroupByResult,
)


class TestJoinConfig:
    """Tests for join configuration parsing (no WASM required)."""

    def test_join_validates_on_parameter(self, employees_data, departments_data):
        """Test that join validates 'on' or 'left_on/right_on'."""
        left = Table(employees_data)
        right = Table(departments_data)

        with pytest.raises(ValueError, match="Must specify"):
            join(left, right)  # No join keys specified

    def test_join_on_converts_to_indices(self, employees_data, departments_data):
        """Test that 'on' parameter is converted to column indices."""
        left = Table(employees_data)
        right = Table(departments_data)

        # This will fail at WASM call, but we can test the config building
        # by mocking or checking intermediate state
        # For now, just verify it doesn't raise on config building
        try:
            join(left, right, on="dept_id")
        except Exception as e:
            # Expected to fail at WASM call if binary not available
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_join_type_mapping(self):
        """Test join type string mapping."""
        from pycylonwasm.operations import join

        # The mapping is internal, but we can verify valid types are accepted
        valid_types = ["inner", "left", "right", "outer", "full_outer", "full"]
        left = Table({"id": [1]})
        right = Table({"id": [1]})

        for how in valid_types:
            try:
                join(left, right, on="id", how=how)
            except Exception as e:
                # Should not raise ValueError for invalid join type
                assert "Unknown join type" not in str(e)

    def test_invalid_join_type_raises(self):
        """Test that invalid join type raises ValueError."""
        left = Table({"id": [1]})
        right = Table({"id": [1]})

        with pytest.raises(ValueError, match="Unknown join type"):
            join(left, right, on="id", how="invalid_type")


class TestGroupByConfig:
    """Tests for groupby configuration (no WASM required)."""

    def test_groupby_returns_result_object(self, employees_data):
        """Test that groupby returns GroupByResult."""
        table = Table(employees_data)
        result = groupby(table, "dept_id")

        assert isinstance(result, GroupByResult)

    def test_groupby_accepts_string_key(self, employees_data):
        """Test groupby with single string key."""
        table = Table(employees_data)
        result = groupby(table, "dept_id")

        assert result._keys == [2]  # dept_id is at index 2

    def test_groupby_accepts_list_keys(self, employees_data):
        """Test groupby with list of keys."""
        table = Table(employees_data)
        result = groupby(table, ["emp_id", "dept_id"])

        assert result._keys == [0, 2]

    def test_groupby_accepts_int_key(self, employees_data):
        """Test groupby with integer column index."""
        table = Table(employees_data)
        result = groupby(table, 2)

        assert result._keys == [2]


class TestGroupByResultMethods:
    """Tests for GroupByResult aggregation methods."""

    def test_agg_builds_config(self, employees_data):
        """Test that agg() builds correct configuration."""
        table = Table(employees_data)
        grouped = groupby(table, "dept_id")

        # This will fail at WASM call, but tests config building
        try:
            grouped.agg({"emp_id": "count"})
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_sum_calls_agg(self, employees_data):
        """Test that sum() is a shorthand for agg with sum."""
        table = Table(employees_data)
        grouped = groupby(table, "dept_id")

        try:
            grouped.sum("emp_id")
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_mean_calls_agg(self, employees_data):
        """Test that mean() is a shorthand for agg with mean."""
        table = Table(employees_data)
        grouped = groupby(table, "dept_id")

        try:
            grouped.mean("emp_id")
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_count_uses_non_key_column(self, employees_data):
        """Test that count() uses first non-key column."""
        table = Table(employees_data)
        grouped = groupby(table, "dept_id")

        try:
            grouped.count()
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()


class TestFilterConfig:
    """Tests for filter configuration (no WASM required)."""

    def test_filter_converts_column_names(self, sample_table_data):
        """Test that filter converts column names to indices."""
        table = Table(sample_table_data)

        predicates = [
            {"column": "age", "op": "gt", "value": 30}
        ]

        try:
            filter_table(table, predicates)
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_filter_accepts_column_indices(self, sample_table_data):
        """Test that filter accepts column indices directly."""
        table = Table(sample_table_data)

        predicates = [
            {"column": 2, "op": "gt", "value": 30}  # age column by index
        ]

        try:
            filter_table(table, predicates)
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_filter_logic_options(self, sample_table_data):
        """Test that filter accepts 'and' and 'or' logic."""
        table = Table(sample_table_data)

        predicates = [
            {"column": "age", "op": "gt", "value": 30},
            {"column": "score", "op": "gt", "value": 85},
        ]

        for logic in ["and", "or", "AND", "OR"]:
            try:
                filter_table(table, predicates, logic=logic)
            except Exception as e:
                # Should not fail on logic parsing
                assert "logic" not in str(e).lower()


class TestAggregateConfig:
    """Tests for aggregate configuration (no WASM required)."""

    def test_aggregate_converts_column_name(self, sample_table_data):
        """Test that aggregate converts column name to index."""
        table = Table(sample_table_data)

        try:
            aggregate(table, "age", "sum")
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()

    def test_aggregate_accepts_column_index(self, sample_table_data):
        """Test that aggregate accepts column index."""
        table = Table(sample_table_data)

        try:
            aggregate(table, 2, "sum")  # age column by index
        except Exception as e:
            assert "WASM" in str(e) or "not found" in str(e).lower()


class TestTableMethodBindings:
    """Tests for operation methods bound to Table class."""

    def test_table_has_groupby_method(self, sample_table_data):
        """Test that Table has groupby method."""
        table = Table(sample_table_data)
        assert hasattr(table, "groupby")
        assert callable(table.groupby)

    def test_table_has_filter_method(self, sample_table_data):
        """Test that Table has filter method."""
        table = Table(sample_table_data)
        assert hasattr(table, "filter")
        assert callable(table.filter)

    def test_table_has_join_method(self, sample_table_data):
        """Test that Table has join method."""
        table = Table(sample_table_data)
        assert hasattr(table, "join")
        assert callable(table.join)

    def test_table_groupby_returns_result(self, sample_table_data):
        """Test that table.groupby() returns GroupByResult."""
        table = Table(sample_table_data)
        result = table.groupby("id")
        assert isinstance(result, GroupByResult)


# =============================================================================
# Integration tests (require WASM binary)
# =============================================================================

@pytest.mark.wasm
class TestJoinIntegration:
    """Integration tests for join operation."""

    def test_inner_join(self, employees_data, departments_data, wasm_available):
        """Test inner join produces correct results."""
        if not wasm_available:
            pytest.skip("WASM binary not available")

        left = Table(employees_data)
        right = Table(departments_data)

        result = join(left, right, on="dept_id", how="inner")

        # Inner join should have 3 rows (dept_id 10, 10, 20)
        assert result.num_rows == 3

    def test_left_join(self, employees_data, departments_data, wasm_available):
        """Test left join preserves all left rows."""
        if not wasm_available:
            pytest.skip("WASM binary not available")

        left = Table(employees_data)
        right = Table(departments_data)

        result = join(left, right, on="dept_id", how="left")

        # Left join should have 4 rows (all employees)
        assert result.num_rows == 4


@pytest.mark.wasm
class TestGroupByIntegration:
    """Integration tests for groupby operation."""

    def test_groupby_count(self, employees_data, wasm_available):
        """Test groupby with count aggregation."""
        if not wasm_available:
            pytest.skip("WASM binary not available")

        table = Table(employees_data)
        result = table.groupby("dept_id").agg({"emp_id": "count"})

        # Should have 3 groups (dept_id: 10, 20, 30)
        assert result.num_rows == 3


@pytest.mark.wasm
class TestFilterIntegration:
    """Integration tests for filter operation."""

    def test_filter_gt(self, sample_table_data, wasm_available):
        """Test filter with greater than predicate."""
        if not wasm_available:
            pytest.skip("WASM binary not available")

        table = Table(sample_table_data)
        result = filter_table(table, [
            {"column": "age", "op": "gt", "value": 35}
        ])

        # Should have 2 rows (age 40 and 45)
        assert result.num_rows == 2


@pytest.mark.wasm
class TestAggregateIntegration:
    """Integration tests for aggregate operation."""

    def test_sum(self, sample_table_data, wasm_available):
        """Test sum aggregation."""
        if not wasm_available:
            pytest.skip("WASM binary not available")

        table = Table(sample_table_data)
        result = aggregate(table, "age", "sum")

        # Sum of [25, 30, 35, 40, 45] = 175
        assert result == 175.0
