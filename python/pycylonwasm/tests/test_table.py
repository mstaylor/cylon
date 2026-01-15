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

"""Tests for Table class.

These tests do not require the WASM binary as they test pure Python functionality.

Run with:
    pytest python/pycylonwasm/tests/test_table.py -v
"""

import json
import pytest
from pycylonwasm.table import Table


class TestTableCreation:
    """Tests for Table creation methods."""

    def test_create_from_dict(self, sample_table_data):
        """Test creating table from dictionary."""
        table = Table(sample_table_data)

        assert table.num_rows == 5
        assert table.num_columns == 4
        assert table.columns == ["id", "name", "age", "score"]

    def test_create_empty_table(self):
        """Test creating empty table."""
        table = Table()

        assert table.num_rows == 0
        assert table.num_columns == 0
        assert table.columns == []

    def test_create_from_columns(self):
        """Test creating table from column definitions."""
        columns = ["a", "b"]
        data = [
            {"type": "Int32", "data": [1, 2, 3]},
            {"type": "String", "data": ["x", "y", "z"]},
        ]

        table = Table.from_columns(columns, data)

        assert table.num_rows == 3
        assert table.num_columns == 2
        assert table.columns == ["a", "b"]

    def test_mismatched_column_lengths_raises(self):
        """Test that mismatched column lengths raise error."""
        with pytest.raises(ValueError, match="same length"):
            Table({
                "a": [1, 2, 3],
                "b": [1, 2],  # Different length
            })


class TestTableTypeInference:
    """Tests for automatic type inference."""

    def test_infer_int32(self):
        """Test Int32 type inference."""
        table = Table({"values": [1, 2, 3]})
        assert table.dtypes["values"] == "Int32"

    def test_infer_int64_for_large_values(self):
        """Test Int64 type inference for large values."""
        table = Table({"values": [1, 2, 2**31]})  # Exceeds Int32 max
        assert table.dtypes["values"] == "Int64"

    def test_infer_float64(self):
        """Test Float64 type inference."""
        table = Table({"values": [1.5, 2.5, 3.5]})
        assert table.dtypes["values"] == "Float64"

    def test_infer_string(self):
        """Test String type inference."""
        table = Table({"values": ["a", "b", "c"]})
        assert table.dtypes["values"] == "String"

    def test_infer_boolean(self):
        """Test Boolean type inference."""
        table = Table({"values": [True, False, True]})
        assert table.dtypes["values"] == "Boolean"

    def test_infer_with_nulls(self):
        """Test type inference with null values."""
        table = Table({"values": [1, None, 3]})
        assert table.dtypes["values"] == "Int32"
        assert table["values"] == [1, None, 3]


class TestTableAccess:
    """Tests for accessing table data."""

    def test_getitem_by_name(self, sample_table_data):
        """Test accessing column by name."""
        table = Table(sample_table_data)

        assert table["id"] == [1, 2, 3, 4, 5]
        assert table["name"] == ["Alice", "Bob", "Carol", "Dave", "Eve"]

    def test_getitem_by_index(self, sample_table_data):
        """Test accessing column by index."""
        table = Table(sample_table_data)

        assert table[0] == [1, 2, 3, 4, 5]  # id column
        assert table[1] == ["Alice", "Bob", "Carol", "Dave", "Eve"]  # name column

    def test_getitem_invalid_name_raises(self, sample_table_data):
        """Test accessing invalid column raises KeyError."""
        table = Table(sample_table_data)

        with pytest.raises(KeyError, match="not found"):
            _ = table["invalid_column"]

    def test_column_index(self, sample_table_data):
        """Test getting column index by name."""
        table = Table(sample_table_data)

        assert table.column_index("id") == 0
        assert table.column_index("name") == 1
        assert table.column_index("age") == 2

    def test_len(self, sample_table_data):
        """Test len() returns number of rows."""
        table = Table(sample_table_data)
        assert len(table) == 5


class TestTableSerialization:
    """Tests for table serialization."""

    def test_to_json(self, sample_table_data):
        """Test JSON serialization."""
        table = Table(sample_table_data)
        json_str = table.to_json()

        parsed = json.loads(json_str)
        assert "columns" in parsed
        assert "data" in parsed
        assert parsed["columns"] == ["id", "name", "age", "score"]
        assert len(parsed["data"]) == 4

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = json.dumps({
            "columns": ["a", "b"],
            "data": [
                {"type": "Int32", "data": [1, 2]},
                {"type": "String", "data": ["x", "y"]},
            ]
        })

        table = Table.from_json(json_str)

        assert table.columns == ["a", "b"]
        assert table.num_rows == 2
        assert table["a"] == [1, 2]
        assert table["b"] == ["x", "y"]

    def test_roundtrip_json(self, sample_table_data):
        """Test JSON roundtrip preserves data."""
        original = Table(sample_table_data)
        json_str = original.to_json()
        restored = Table.from_json(json_str)

        assert restored.columns == original.columns
        assert restored.num_rows == original.num_rows
        for col in original.columns:
            assert restored[col] == original[col]

    def test_to_dict(self, sample_table_data):
        """Test conversion to dictionary."""
        table = Table(sample_table_data)
        result = table.to_dict()

        assert result == sample_table_data


class TestTableOperations:
    """Tests for table operations."""

    def test_head(self, sample_table_data):
        """Test head() returns first n rows."""
        table = Table(sample_table_data)
        head = table.head(2)

        assert head.num_rows == 2
        assert head["id"] == [1, 2]
        assert head["name"] == ["Alice", "Bob"]

    def test_head_default(self, sample_table_data):
        """Test head() default is 5 rows."""
        table = Table(sample_table_data)
        head = table.head()

        assert head.num_rows == 5  # All rows since we only have 5

    def test_select(self, sample_table_data):
        """Test select() returns subset of columns."""
        table = Table(sample_table_data)
        selected = table.select(["name", "age"])

        assert selected.columns == ["name", "age"]
        assert selected.num_columns == 2
        assert selected["name"] == ["Alice", "Bob", "Carol", "Dave", "Eve"]

    def test_rename(self, sample_table_data):
        """Test rename() changes column names."""
        table = Table(sample_table_data)
        renamed = table.rename({"id": "user_id", "name": "user_name"})

        assert "user_id" in renamed.columns
        assert "user_name" in renamed.columns
        assert "id" not in renamed.columns
        assert renamed["user_id"] == [1, 2, 3, 4, 5]


class TestTableRepr:
    """Tests for table string representations."""

    def test_repr(self, sample_table_data):
        """Test __repr__ output."""
        table = Table(sample_table_data)
        repr_str = repr(table)

        assert "Table" in repr_str
        assert "5 rows" in repr_str

    def test_repr_html(self, sample_table_data):
        """Test HTML representation for Jupyter."""
        table = Table(sample_table_data)
        html = table._repr_html_()

        assert "<table>" in html
        assert "<th>id</th>" in html
        assert "<td>Alice</td>" in html


class TestTableWithNulls:
    """Tests for tables with null values."""

    def test_null_values_preserved(self, table_with_nulls):
        """Test that null values are preserved."""
        table = Table(table_with_nulls)

        assert table["value"] == [10, None, 30, None]
        assert table["name"] == ["a", "b", None, "d"]

    def test_null_values_in_json(self, table_with_nulls):
        """Test null values in JSON serialization."""
        table = Table(table_with_nulls)
        json_str = table.to_json()
        restored = Table.from_json(json_str)

        assert restored["value"] == [10, None, 30, None]
