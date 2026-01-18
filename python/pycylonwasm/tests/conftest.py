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

"""Pytest fixtures for pycylonwasm tests."""

import os
import pytest
from pathlib import Path


@pytest.fixture
def sample_table_data():
    """Sample table data for testing."""
    return {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [85.5, 90.0, 78.5, 92.0, 88.5],
    }


@pytest.fixture
def employees_data():
    """Employee table data."""
    return {
        "emp_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Carol", "Dave"],
        "dept_id": [10, 20, 10, 30],
    }


@pytest.fixture
def departments_data():
    """Department table data."""
    return {
        "dept_id": [10, 20, 40],
        "dept_name": ["Engineering", "Sales", "Marketing"],
    }


@pytest.fixture
def table_with_nulls():
    """Table data with null values."""
    return {
        "id": [1, 2, 3, 4],
        "value": [10, None, 30, None],
        "name": ["a", "b", None, "d"],
    }


@pytest.fixture
def wasm_path():
    """Path to WASM binary, if available."""
    # Check environment variable first
    env_path = os.environ.get("CYLON_WASM_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Check default location relative to this file
    # Path: tests/ -> pycylonwasm/ -> python/ -> cylon/
    default_path = Path(__file__).parent.parent.parent.parent / "rust" / "cylon-wasm" / "pkg" / "cylon_wasm_bg.wasm"
    if default_path.exists():
        return str(default_path)

    return None


@pytest.fixture
def wasm_available(wasm_path):
    """Check if WASM binary is available."""
    return wasm_path is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "wasm: mark test as requiring WASM binary"
    )
