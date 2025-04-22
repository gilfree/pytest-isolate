import json
import os
import tempfile
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import filelock

from pytest_isolate.tracing import create_event

# Constants

DEFAULT_STATE_FILE = (
    Path(os.environ.get("PYTEST_ISOLATE_STATE_FOLDER", tempfile.gettempdir()))
    / "pytest_isolate_resources.json"
)
DEFAULT_EVENTS_FILE = Path(str(DEFAULT_STATE_FILE).replace(".json", "_events.json"))
POLL_INTERVAL = float(os.environ.get("PYTEST_ISOLATE_POLL_INTERVAL", 0.1))
DEFAULT_LOCK_TIMEOUT = int(os.environ.get("PYTEST_ISOLATE_LOCK_TIMEOUT", 5))
DEFAULT_WAIT_TIMEOUT = 300  # 5 minutes


@contextmanager
def lock_resource_file():
    with filelock.FileLock(
        str(DEFAULT_STATE_FILE) + ".lock", timeout=DEFAULT_LOCK_TIMEOUT
    ) as lock:
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
    if DEFAULT_EVENTS_FILE.exists():
        os.remove(DEFAULT_EVENTS_FILE)
    if DEFAULT_STATE_FILE.with_suffix(".lock").exists():
        os.remove(DEFAULT_STATE_FILE.with_suffix(".lock"))


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
        vals = [int(x.strip()) for x in env_value.split(",") if x.strip()]
        vals = [x for x in vals if x >= 0]
        return vals
    except ValueError:
        return []


def log_resource_allocation(
    allocated,
    resource_type: str,
    test_id: str,
    start_time=None,
    end_time=None,
    file=None,
) -> None:
    file = str(file or DEFAULT_EVENTS_FILE)
    folder = os.path.dirname(file)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(file):
        resource_events = {}
    else:
        with open(file, "r") as f:
            resource_events = json.load(f)
    if test_id not in resource_events:
        resource_events[test_id] = dict(
            allocated=allocated,
            worker_name=resource_type,
            test_name=test_id,
            category="resource_allocation",
        )
    if start_time is not None:
        resource_events[test_id]["start_time"] = start_time
    if end_time is not None:
        resource_events[test_id]["end_time"] = end_time
    with open(file, "w") as f:
        json.dump(resource_events, f)


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
                log_resource_allocation(
                    resource.allocated.get(test_id),
                    resource_type,
                    test_id,
                    start_time=time.time(),
                )
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
        allocated = resource.allocated.get(test_id)
        if allocated:
            log_resource_allocation(
                resource.allocated.get(test_id),
                resource_type,
                test_id,
                end_time=time.time(),
            )
        resource.release(test_id)

        os.environ.pop(resource.env_variable, None)
        state.save()


def get_resource_events(file=None) -> list[dict]:
    # Convert the resource events to the format used in tracing.py,
    # by merging the start and end events.
    file = Path(file or str(DEFAULT_EVENTS_FILE))
    if not file.exists():
        return []
    resource_events = json.load(open(file))
    events = []
    for test_id, event in resource_events.items():
        event_args = dict(**event)
        allocated = event_args.pop("allocated", None)
        for resource_id in allocated:
            resource_event = deepcopy(event_args)
            resource_event["worker_name"] = (
                f"{resource_event['worker_name']}_{resource_id}"
            )
            if "end_time" not in resource_event:
                resource_event["end_time"] = time.time()
            if "start_time" not in resource_event:
                raise ValueError(
                    f"Missing start_time for resource event {resource_event}"
                )
            events.extend(create_event(**resource_event))
    return events
