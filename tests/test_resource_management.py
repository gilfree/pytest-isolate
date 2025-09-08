import json
import os
import tempfile
from pathlib import Path
from time import sleep
from unittest import mock

import pytest

from pytest_isolate.plugin import allocate_resources, get_resource_events
from pytest_isolate.resource_management import (
    Resource,
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


def test_allocate_resources_fraction():
    with allocate_resources({"gpu": 0.5}, "foo", 100) as allocated:
        assert len(allocated["gpu"]) == 1
        assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))


@pytest.mark.isolate(resources={"gpu": 0.5}, timeout=10)
def test_gpu_marker_fraction():
    sleep(0.3)
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1


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


# Note: Integration tests with @pytest.mark.isolate(resources={"gpu": fraction})
# are not included here because they require actual GPUs to be available.
# The core fractional allocation logic is tested in the unit tests above.


def test_resource_fractional_allocation():
    """Test Resource class fractional allocation logic"""
    # Create a resource with 2 resources (e.g., GPUs)
    resource = Resource(
        env_variable="CUDA_VISIBLE_DEVICES", available=[0, 1], allocated={}
    )

    # Test 1/4 allocations can share the same resource
    res1 = resource.allocate("test1", 1 / 4)  # 4 units
    assert res1 == [0]
    assert resource.resource_usage_count[0] == 4

    res2 = resource.allocate("test2", 1 / 4)  # 4 units
    assert res2 == [0]  # Same resource
    assert resource.resource_usage_count[0] == 8

    res3 = resource.allocate("test3", 1 / 4)  # 4 units
    assert res3 == [0]  # Same resource
    assert resource.resource_usage_count[0] == 12

    res4 = resource.allocate("test4", 1 / 4)  # 4 units
    assert res4 == [0]  # Same resource
    assert resource.resource_usage_count[0] == 16

    # Fifth 1/4 allocation should go to second resource
    res5 = resource.allocate("test5", 1 / 4)  # 4 units
    assert res5 == [1]  # Different resource
    assert resource.resource_usage_count[1] == 4

    # Test release
    resource.release("test1")
    assert resource.resource_usage_count[0] == 12
    assert "test1" not in resource.allocated

    # Test integer allocation (exclusive)
    resource2 = Resource(
        env_variable="CUDA_VISIBLE_DEVICES", available=[0, 1], allocated={}
    )

    res_int = resource2.allocate("test_int", 1)
    assert res_int == [0]
    assert 0 not in resource2.available  # Removed from available

    # Test invalid fractions
    with pytest.raises(ValueError):
        resource.allocate("test_invalid", 1 / 3)


def test_resource_configurable_max_units():
    """Test Resource class with different max_units_per_resource values"""
    # Test with max_units=8
    resource8 = Resource(
        env_variable="TEST_RESOURCE",
        available=[0],
        allocated={},
        max_units_per_resource=8,
    )

    # Should allow 1/8, 1/4, 1/2, 1
    res1 = resource8.allocate("test1", 1 / 8)  # 1 unit
    assert res1 == [0]
    assert resource8.resource_usage_count[0] == 1

    res2 = resource8.allocate("test2", 1 / 4)  # 2 units
    assert res2 == [0]
    assert resource8.resource_usage_count[0] == 3

    res3 = resource8.allocate("test3", 1 / 2)  # 4 units
    assert res3 == [0]
    assert resource8.resource_usage_count[0] == 7

    res4 = resource8.allocate("test4", 1 / 8)  # 1 unit (fills to 8)
    assert res4 == [0]
    assert resource8.resource_usage_count[0] == 8

    # Should reject 1/16 (not valid for max_units=8)
    with pytest.raises(ValueError):
        resource8.allocate("test_invalid", 1 / 16)

    # Test with max_units=4
    resource4 = Resource(
        env_variable="TEST_RESOURCE",
        available=[0],
        allocated={},
        max_units_per_resource=4,
    )

    # Should allow 1/4, 1/2, 1
    res5 = resource4.allocate("test5", 1 / 4)  # 1 unit
    assert res5 == [0]
    assert resource4.resource_usage_count[0] == 1

    # Should reject 1/8 (not valid for max_units=4)
    with pytest.raises(ValueError):
        resource4.allocate("test_invalid", 1 / 8)


def test_resource_mixed_allocation():
    """Test mixing fractional and integer allocations"""
    resource = Resource(
        env_variable="CUDA_VISIBLE_DEVICES", available=[0, 1, 2], allocated={}
    )

    # Fractional allocation
    res1 = resource.allocate("test1", 1 / 2)  # 8 units
    assert res1 == [0]
    assert resource.resource_usage_count[0] == 8

    # Another fractional on same resource
    res2 = resource.allocate("test2", 1 / 2)  # 8 units
    assert res2 == [0]  # Same resource
    assert resource.resource_usage_count[0] == 16  # Full

    # Integer allocation gets different resource
    res3 = resource.allocate("test3", 1)
    assert res3 == [1]
    assert 1 not in resource.available
