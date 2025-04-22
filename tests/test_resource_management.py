import json
import os
import tempfile
from pathlib import Path
from time import sleep
from unittest import mock

import pytest

from pytest_isolate.plugin import allocate_resources, get_resource_events
from pytest_isolate.resource_management import (
    log_resource_allocation,
    parse_resource_list,
)


@pytest.fixture
def temp_resource_files():
    """Create temporary files for resource manager testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_file = Path(tmpdir) / "resources.lock"
        state_file = Path(tmpdir) / "resources.json"
        yield lock_file, state_file


@pytest.fixture
def mock_pynvml():
    """Mock pynvml module for testing"""
    pynvml_mock = mock.MagicMock()
    pynvml_mock.nvmlDeviceGetCount.return_value = 4

    # Mock the import so it returns our mock when imported in plugin.py
    with mock.patch.dict("sys.modules", {"pynvml": pynvml_mock}):
        yield pynvml_mock


def test_parse_resource_list():
    """Test parsing resources from environment variables."""
    # Test empty input
    assert parse_resource_list(None) == []
    assert parse_resource_list("") == []

    # Test simple cases
    assert parse_resource_list("0") == [0]
    assert parse_resource_list("0,1,2") == [0, 1, 2]

    # Test with spaces
    assert parse_resource_list(" 0, 1, 2 ") == [0, 1, 2]

    # Test with invalid values - no warning in new implementation
    assert parse_resource_list("0,a,2") == []


def test_log_resource_allocation():
    log_resource_allocation(
        [0, 1], "gpu", "foo", start_time=100, file="timeline.events.test.json"
    )

    log_resource_allocation(
        [0, 1], "gpu", "foo", end_time=200, file="timeline.events.test.json"
    )
    events = get_resource_events(file="timeline.events.test.json")
    open("timeline.resources.test.json", "w").write(json.dumps(events))


def test_allocate_resources():
    with allocate_resources({"gpu": 1}, "foo", 100) as allocated:
        assert len(allocated["gpu"]) == 1
        assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


@pytest.mark.isolate(resources={"gpu": 2}, timeout=10)
@pytest.mark.parametrize("dummy", [1, 2])
def test_gpu_marker_2(dummy):
    sleep(0.3)
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 2


@pytest.mark.isolate(resources={"gpu": 1}, timeout=10)
@pytest.mark.parametrize("dummy", [1, 2])
def test_gpu_marker_1(dummy):
    sleep(0.3)
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1


@pytest.mark.xfail
@pytest.mark.isolate(resources={"gpu": 7}, timeout=4)
def test_gpu_marker_7():
    sleep(0.3)
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 7
