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

"""Tests for Lambda handler.

These tests verify the handler's request parsing and response formatting.
Integration tests that require WASM are marked with @pytest.mark.wasm.

Run with:
    pytest python/pycylonwasm/tests/test_lambda_handler.py -v
"""

import json
import sys
import pytest
from pathlib import Path

# Add lambda handler to path
# Path: tests/ -> pycylonwasm/ -> python/ -> cylon/
lambda_path = Path(__file__).parent.parent.parent.parent / "target" / "aws" / "scripts" / "lambda"
sys.path.insert(0, str(lambda_path))

from wasm_handler import handler, _handle_join, _handle_groupby, _handle_filter, _handle_aggregate


class TestHandlerValidation:
    """Tests for handler input validation."""

    def test_missing_operation_returns_400(self):
        """Test that missing operation field returns 400."""
        event = {}
        result = handler(event, None)

        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "operation" in body["error"].lower()

    def test_unknown_operation_returns_400(self):
        """Test that unknown operation returns 400."""
        event = {"operation": "unknown_op"}
        result = handler(event, None)

        assert result["statusCode"] == 400
        body = json.loads(result["body"])
        assert "unknown" in body["error"].lower()

    def test_valid_operations_recognized(self):
        """Test that valid operation names are recognized."""
        valid_ops = ["join", "groupby", "filter", "aggregate"]

        for op in valid_ops:
            event = {"operation": op}
            result = handler(event, None)

            # Should not return "unknown operation" error
            if result["statusCode"] == 400:
                body = json.loads(result["body"])
                assert "unknown operation" not in body["error"].lower()


class TestJoinHandler:
    """Tests for join operation handler."""

    def test_join_requires_left_table(self):
        """Test that join requires left table."""
        event = {
            "operation": "join",
            "right": {"columns": [], "data": []},
            "config": {}
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "left" in body["error"].lower() or "right" in body["error"].lower()

    def test_join_requires_right_table(self):
        """Test that join requires right table."""
        event = {
            "operation": "join",
            "left": {"columns": [], "data": []},
            "config": {}
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "left" in body["error"].lower() or "right" in body["error"].lower()


class TestGroupByHandler:
    """Tests for groupby operation handler."""

    def test_groupby_requires_table(self):
        """Test that groupby requires table."""
        event = {
            "operation": "groupby",
            "config": {"keys": [0], "aggregations": {}}
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "table" in body["error"].lower()


class TestFilterHandler:
    """Tests for filter operation handler."""

    def test_filter_requires_table(self):
        """Test that filter requires table."""
        event = {
            "operation": "filter",
            "config": {"predicates": []}
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "table" in body["error"].lower()


class TestAggregateHandler:
    """Tests for aggregate operation handler."""

    def test_aggregate_requires_table(self):
        """Test that aggregate requires table."""
        event = {
            "operation": "aggregate",
            "column": 0,
            "op": "sum"
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "table" in body["error"].lower()

    def test_aggregate_requires_column(self):
        """Test that aggregate requires column."""
        event = {
            "operation": "aggregate",
            "table": {"columns": ["a"], "data": [{"type": "Int32", "data": [1]}]},
            "op": "sum"
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "column" in body["error"].lower()

    def test_aggregate_requires_op(self):
        """Test that aggregate requires op."""
        event = {
            "operation": "aggregate",
            "table": {"columns": ["a"], "data": [{"type": "Int32", "data": [1]}]},
            "column": 0
        }
        result = handler(event, None)

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "op" in body["error"].lower()


class TestEventFormat:
    """Tests for proper event format handling."""

    def test_join_event_format(self):
        """Test expected join event format."""
        event = {
            "operation": "join",
            "left": {
                "columns": ["id", "name"],
                "data": [
                    {"type": "Int32", "data": [1, 2]},
                    {"type": "String", "data": ["a", "b"]}
                ]
            },
            "right": {
                "columns": ["id", "value"],
                "data": [
                    {"type": "Int32", "data": [1, 3]},
                    {"type": "Int32", "data": [100, 300]}
                ]
            },
            "config": {
                "join_type": "inner",
                "left_on": [0],
                "right_on": [0]
            }
        }

        # Event format is valid, will fail at WASM call
        result = handler(event, None)
        # Either succeeds (200) or fails at WASM level (500), not format error (400)
        assert result["statusCode"] in [200, 500]

    def test_groupby_event_format(self):
        """Test expected groupby event format."""
        event = {
            "operation": "groupby",
            "table": {
                "columns": ["category", "value"],
                "data": [
                    {"type": "String", "data": ["a", "a", "b"]},
                    {"type": "Int32", "data": [10, 20, 30]}
                ]
            },
            "config": {
                "keys": [0],
                "aggregations": {"value": "sum"}
            }
        }

        result = handler(event, None)
        assert result["statusCode"] in [200, 500]

    def test_filter_event_format(self):
        """Test expected filter event format."""
        event = {
            "operation": "filter",
            "table": {
                "columns": ["id", "value"],
                "data": [
                    {"type": "Int32", "data": [1, 2, 3]},
                    {"type": "Int32", "data": [10, 20, 30]}
                ]
            },
            "config": {
                "predicates": [
                    {"column": 1, "op": "gt", "value": 15}
                ],
                "logic": "and"
            }
        }

        result = handler(event, None)
        assert result["statusCode"] in [200, 500]

    def test_aggregate_event_format(self):
        """Test expected aggregate event format."""
        event = {
            "operation": "aggregate",
            "table": {
                "columns": ["value"],
                "data": [
                    {"type": "Int32", "data": [10, 20, 30]}
                ]
            },
            "column": 0,
            "op": "sum"
        }

        result = handler(event, None)
        assert result["statusCode"] in [200, 500]
