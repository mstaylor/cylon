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
AWS Lambda handler for Cylon WASM operations.

Supports join, groupby, filter, and aggregate operations via WASM.

Environment variables:
    CYLON_WASM_PATH: Path to cylon_wasm_bg.wasm file (optional)

Example event:
    {
        "operation": "join",
        "left": {"columns": [...], "data": [...]},
        "right": {"columns": [...], "data": [...]},
        "config": {"join_type": "inner", "left_on": [0], "right_on": [0]}
    }
"""

import json
import os
import logging
from typing import Any, Dict

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy import to allow Lambda cold start optimization
_pycylonwasm = None


def _get_pycylon():
    """Lazy load pycylonwasm module."""
    global _pycylonwasm
    if _pycylonwasm is None:
        import pycylonwasm
        _pycylonwasm = pycylonwasm
    return _pycylonwasm


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for WASM DataFrame operations.

    Args:
        event: Lambda event with operation and data
        context: Lambda context (unused)

    Returns:
        Response dict with statusCode and body
    """
    try:
        operation = event.get("operation")
        if not operation:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'operation' field"})
            }

        pycylon = _get_pycylon()
        result = None

        if operation == "join":
            result = _handle_join(pycylon, event)
        elif operation == "groupby":
            result = _handle_groupby(pycylon, event)
        elif operation == "filter":
            result = _handle_filter(pycylon, event)
        elif operation == "aggregate":
            result = _handle_aggregate(pycylon, event)
        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown operation: {operation}"})
            }

        return {
            "statusCode": 200,
            "body": result if isinstance(result, str) else json.dumps(result)
        }

    except Exception as e:
        logger.exception("Error processing request")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def _handle_join(pycylon, event: Dict) -> str:
    """Handle join operation."""
    left_data = event.get("left")
    right_data = event.get("right")
    config = event.get("config", {})

    if not left_data or not right_data:
        raise ValueError("Join requires 'left' and 'right' tables")

    left = pycylon.Table.from_json(json.dumps(left_data))
    right = pycylon.Table.from_json(json.dumps(right_data))

    result = pycylon.join(
        left, right,
        left_on=config.get("left_on"),
        right_on=config.get("right_on"),
        how=config.get("join_type", "inner"),
        left_suffix=config.get("left_suffix", "_l"),
        right_suffix=config.get("right_suffix", "_r"),
    )

    return result.to_json()


def _handle_groupby(pycylon, event: Dict) -> str:
    """Handle groupby operation."""
    table_data = event.get("table")
    config = event.get("config", {})

    if not table_data:
        raise ValueError("GroupBy requires 'table'")

    table = pycylon.Table.from_json(json.dumps(table_data))
    keys = config.get("keys", [])
    aggregations = config.get("aggregations", {})

    grouped = pycylon.groupby(table, keys)
    result = grouped.agg(aggregations)

    return result.to_json()


def _handle_filter(pycylon, event: Dict) -> str:
    """Handle filter operation."""
    table_data = event.get("table")
    config = event.get("config", {})

    if not table_data:
        raise ValueError("Filter requires 'table'")

    table = pycylon.Table.from_json(json.dumps(table_data))
    predicates = config.get("predicates", [])
    logic = config.get("logic", "and")

    result = pycylon.filter_table(table, predicates, logic)

    return result.to_json()


def _handle_aggregate(pycylon, event: Dict) -> Dict:
    """Handle aggregate operation."""
    table_data = event.get("table")
    column = event.get("column")
    op = event.get("op")

    if not table_data:
        raise ValueError("Aggregate requires 'table'")
    if column is None:
        raise ValueError("Aggregate requires 'column'")
    if not op:
        raise ValueError("Aggregate requires 'op'")

    table = pycylon.Table.from_json(json.dumps(table_data))
    result = pycylon.aggregate(table, column, op)

    return {"result": result}


# For local testing
if __name__ == "__main__":
    # Test join
    test_event = {
        "operation": "join",
        "left": {
            "columns": ["id", "name"],
            "data": [
                {"type": "Int32", "data": [1, 2, 3]},
                {"type": "String", "data": ["Alice", "Bob", "Carol"]}
            ]
        },
        "right": {
            "columns": ["id", "dept"],
            "data": [
                {"type": "Int32", "data": [1, 2, 4]},
                {"type": "String", "data": ["Eng", "Sales", "Marketing"]}
            ]
        },
        "config": {
            "join_type": "inner",
            "left_on": [0],
            "right_on": [0]
        }
    }

    result = handler(test_event, None)
    print(json.dumps(result, indent=2))
