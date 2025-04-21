import json
import os
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import filelock

# Constants

DEFAULT_STATE_FILE = Path(tempfile.gettempdir()) / "pytest_isolate_resources.json"
DEFAULT_WAIT_TIMEOUT = 300  # 5 minutes
POLL_INTERVAL = 0.1  # 1 second


@contextmanager
def lock_resource_file():
    with filelock.FileLock(str(DEFAULT_STATE_FILE)+'.lock', timeout=4) as lock:
        yield lock


@dataclass
class Resource:
    env_variable: str
    available: List[int]
    allocated: dict[str, List[int]]

    def allocate(self, test_id: str, count: int) -> List[int]:
        """Allocate resources for a test."""
        if len(self.available) < count:
            return []

        allocated = self.available[:count]
        self.allocated[test_id] = allocated
        self.available = self.available[count:]
        return allocated

    def release(self, test_id: str) -> None:
        """Release resources for a test."""
        if test_id in self.allocated:
            released = self.allocated.pop(test_id)
            self.available.extend(released)
        else:
            raise ValueError(f"Test ID {test_id} not found in allocated resources.")


@dataclass
class StateData:
    resources: dict[str, Resource]

    @classmethod
    def get_instance(cls) -> "StateData":
        if not DEFAULT_STATE_FILE.exists():
            data = {}
        try:
            with open(DEFAULT_STATE_FILE, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
        return StateData(
            resources={name: Resource(**res) for name, res in data.items()}
        )

    def save(self) -> None:
        with open(DEFAULT_STATE_FILE, "w") as f:
            json.dump({name: asdict(res) for name, res in self.resources.items()}, f)


def clean_resources() -> None:
    """Clean up resources by removing lock and state files."""
    if DEFAULT_STATE_FILE.exists():
        os.remove(DEFAULT_STATE_FILE)


def register_resource_provider(
    resource_type: str, env_variable: str, provider_func: Callable[[], List[int]]
) -> None:
    with lock_resource_file():
        existing = provider_func()
        state = StateData.get_instance()
        if resource_type in state.resources:
            warnings.warn(
                f"Resource type {resource_type} already registered. Overwriting."
            )
        state.resources[resource_type] = Resource(
            env_variable=env_variable,
            available=existing,
            allocated={},
        )
        state.save()


def parse_resource_list(env_value: Optional[str]) -> List[int]:
    """Parse a comma-separated list of resource IDs from an environment variable."""
    if not env_value:
        return []
    try:
        vals= [int(x.strip()) for x in env_value.split(",") if x.strip()]
        vals = [x for x in vals if x >= 0]
        return vals
    except ValueError:
        return []


def setup_resource_environment(
    test_id: str,
    count: int,
    resource_type: str,
    wait_timeout: Optional[float] = None,
) -> Optional[List[int]]:
    def try_allocate() -> Optional[List[int]]:
        with lock_resource_file():
            state = StateData.get_instance()
            if resource_type not in state.resources:
                raise ValueError(f"Resource type {resource_type} not registered.")

            resource = state.resources[resource_type]
            allocated = resource.allocate(test_id, count)
            if allocated:
                os.environ[resource.env_variable] = ",".join(map(str, allocated))
                state.save()
                return allocated
        return []

    timeout = wait_timeout or DEFAULT_WAIT_TIMEOUT
    start_time = time.time()
    while time.time() - start_time < timeout:
        allocated = try_allocate()
        if allocated:
            return allocated
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Timeout while waiting for resources of type {resource_type}.")


def cleanup_resource_environment(test_id: str, resource_type: str) -> None:
    with lock_resource_file():
        state = StateData.get_instance()
        if resource_type not in state.resources:
            raise ValueError(f"Resource type {resource_type} not registered.")

        resource = state.resources[resource_type]
        resource.release(test_id)
        os.environ.pop(resource.env_variable, None)
        state.save()
