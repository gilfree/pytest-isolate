# This file is was originally based on pytest forked:
# see: https://github.com/pytest-dev/pytest-forked/blob/master/src/pytest_forked/__init__.py
# Since pytest forked is un unmaintained I decided to make my
# changes in a different project
# Please see README.md


import fcntl
import io
import json
import multiprocessing as mp
import os
import resource
import sys
import warnings
from contextlib import contextmanager
from queue import Empty
from typing import Any, List, Optional, Tuple

import _pytest.capture
import _pytest.warnings
import pytest
from _pytest import runner
from _pytest.runner import runtestprotocol

try:
    import dill
except ImportError:
    pass

try:
    import setproctitle  # noqa F401
except ImportError:
    pass

try:
    from tblib import pickling_support

    pickling_support.install()
except ImportError:
    pass


def forked_subprocess(
    target, args=(), timeout=None, memlimt=None, cpulimit=None
) -> Tuple[Any, int, bool]:
    sub = ForkedSubprocess()
    exitcode, timed_out, result = sub.run_in_subprocess(
        timeout, memlimt, cpulimit, target, args
    )
    return result, exitcode, timed_out


@contextmanager
def limits(max_mem, max_cpu):
    if not max_mem and not max_cpu:
        yield
        return
    (soft_mem, hard_mem) = resource.getrlimit(resource.RLIMIT_DATA)
    (soft_cpu, hard_cpu) = resource.getrlimit(resource.RLIMIT_CPU)
    if max_mem:
        resource.setrlimit(resource.RLIMIT_DATA, (max_mem, hard_mem))
    if max_cpu:
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, hard_cpu))
    try:
        yield
    finally:
        if max_mem:
            resource.setrlimit(resource.RLIMIT_DATA, (soft_mem, hard_mem))
        if max_cpu:
            resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, hard_cpu))
    return


class ForkedSubprocess:
    def __init__(self) -> None:
        self.r_out, self.w_out = os.pipe()
        self.r_err, self.w_err = os.pipe()
        fcntl.fcntl(self.r_err, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(self.r_out, fcntl.F_SETFL, os.O_NONBLOCK)
        self.streams_ready = False
        self.read_out = None
        self.read_err = None

    def parent_open_streams(self):
        if self.streams_ready:
            return
        self.streams_ready = True
        os.close(self.w_out)
        os.close(self.w_err)
        self.read_out = io.open(self.r_out, "rb", 0)
        self.read_err = io.open(self.r_err, "rb", 0)

    def child_redirect_streams(self):
        if self.streams_ready:
            return
        self.streams_ready = True
        os.close(self.r_out)
        os.close(self.r_err)
        os.dup2(self.w_out, 1)
        os.dup2(self.w_err, 2)
        # Not sure why exactly but closing the streams is required
        os.close(self.w_out)
        os.close(self.w_err)

    def run_in_subprocess(self, timeout, memlimit, cpulimit, target, args=()):
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        timed_out = None

        def run_subprocess():
            # Close the read fd, redirect output to write fd
            self.child_redirect_streams()
            try:
                with limits(memlimit, cpulimit):
                    q.put(dill.dumps(target(*args)))
            except BaseException as e:
                q.put(dill.dumps(e))
            sys.stdout.flush()
            sys.stderr.flush()

        ctx = mp.get_context("fork")
        p = ctx.Process(target=run_subprocess)
        p.start()
        self.parent_open_streams()
        time_left = timeout or 1
        result = None
        while time_left > 0:
            try:
                result = dill.loads(q.get(block=True, timeout=min(1, time_left)))
            except Empty:
                if not p.is_alive():
                    break
                if timeout is not None:
                    time_left = time_left - 1
            except Exception as e:
                result = e
                break
            finally:
                if self.streams_ready:
                    out = self.read_out.read()
                    if out:
                        print(out.decode(), file=sys.stdout)
                    err = self.read_out.read()
                    if err:
                        print(err.decode(), file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        if time_left <= 0:
            timed_out = timeout

        if p.is_alive() and (time_left <= 0):
            # On timeout, kill the subprocess
            p.kill()
        p.join()
        return p.exitcode, timed_out, result


def pytest_load_initial_conftests(early_config, parser, args):
    early_config.addinivalue_line(
        "markers",
        "forked: Always fork for this test.",
    )
    early_config.addinivalue_line(
        "markers",
        "isolate: Always isolate this test.",
    )
    early_config.addinivalue_line(
        "markers",
        "timeout: Always isolate this test with timeout.",
    )


@pytest.hookimpl(trylast=True)
def pytest_addoption(parser):
    group = parser.getgroup("isolate")
    group.addoption(
        "--isolate",
        dest="isolate",
        action="store_true",
        default=False,
        help="Isolate each test in a separate process",
    )
    group.addoption(
        "--isolate-timeout",
        dest="isolate_timeout",
        default=None,
        type=float,
        help="Isolate each test in a separate process",
    )
    group.addoption(
        "--isolate-mem-limit",
        dest="isolate_mem_limit",
        default=None,
        help="Limit the memory usage of the test ",
    )
    group.addoption(
        "--isolate-cpu-limit",
        dest="isolate_cpu_limit",
        default=None,
        help="Limit the CPU usage of the test ",
    )
    parser.addini(
        "isolate_timeout", "Default timeout for isolated tests", type="string"
    )
    parser.addini(
        "isolate_mem_limit", "Default memory limit for isolated tests", type="string"
    )
    parser.addini(
        "isolate_cpu_limit", "Default cpu limit for isolated tests", type="string"
    )

    parser.addoption(
        "--timeline",
        dest="timeline",
        action="store_true",
        default=False,
        help="Report test timelines",
    )

    parser.addoption(
        "--timeline-file",
        dest="timeline_file",
        default="./timeline.json",
        help="Path to the timeline file",
    )

    parser.addini("timeline_file", "timeline file", type="string")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if config.pluginmanager.get_plugin("forked"):
        warnings.warn(
            "Isolate is a replacement for pytest-fokred. "
            "Pytest-forked will take precedence while installed, "
            "Forked tests will not have timeout. please uninstall one of the plugins"
        )
    else:
        config.addinivalue_line(
            "markers", "forked: Run this tests in a separate, forked, process"
        )

    if config.pluginmanager.get_plugin("timeout"):
        warnings.warn(
            "Isolate is a replacement for pytest-timeout. "
            "Pytest-timeout will take precedence while installed, "
            "tests with timeout will not be isolated. "
            "Please uninstall one of the plugins"
        )
    else:
        config.addinivalue_line(
            "markers",
            "timeout: Run this tests in a separate, forked, process with timeout",
        )


# Taken from pytest 7.2:
@contextmanager
def catch_warnings(item: pytest.Item):
    """Context manager that catches warnings generated in the contained execution block.

    ``item`` can be None if we are not in the context of an item execution.

    Each warning captured triggers the ``pytest_warning_recorded`` hook.
    """
    config_filters = item.config.getini("filterwarnings")
    cmdline_filters = item.config.known_args_namespace.pythonwarnings or []
    with warnings.catch_warnings(record=True) as log:
        # mypy can't infer that record=True means log is not None; help it.
        assert log is not None

        if not sys.warnoptions:
            # If user is not explicitly configuring warning filters,
            # show deprecation warnings by default (#2908).
            warnings.filterwarnings("always", category=DeprecationWarning)
            warnings.filterwarnings("always", category=PendingDeprecationWarning)

        _pytest.warnings.apply_warning_filters(config_filters, cmdline_filters)

        # apply filters from "filterwarnings" marks
        if item is not None:
            for mark in item.iter_markers(name="filterwarnings"):
                for arg in mark.args:
                    warnings.filterwarnings(
                        *_pytest.warnings.parse_warning_filter(arg, escape=False)
                    )

        yield log


def run_subprocess(item: pytest.Item):
    item.config.pluginmanager.unregister(name="capturemanager")
    try:
        if not os.getenv("PYTEST_ISOLATE_NO_SETPROCTITLE"):
            import setproctitle  # noqa F811

            setproctitle.setproctitle(f"pytest {item.nodeid}")
    except ImportError:
        pass
    try:
        with catch_warnings(item) as warnings:
            reports = runtestprotocol(item, log=False)
        s_reports = []
        for report in reports:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_usage = usage.ru_utime + usage.ru_stime
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_usage += usage.ru_utime + usage.ru_stime
            report.user_properties.append(("cpu_usage", cpu_usage))
            report.cpu_usage = cpu_usage
            s_reports.append(
                item.config.hook.pytest_report_to_serializable(
                    config=item.config, report=report
                )
            )
        return s_reports, warnings
    except BaseException as e:
        return e, None


def run_in_subprocess(
    item: pytest.Item,
    timeout: Optional[float],
    mem_limit: Optional[int],
    cpu_limit: Optional[int],
) -> List[pytest.TestReport]:
    cap = item.config.pluginmanager.getplugin("capturemanager")
    assert isinstance(cap, _pytest.capture.CaptureManager)
    cap.resume_global_capture()
    cap.activate_fixture()
    result, exitcode, timed_out = forked_subprocess(
        run_subprocess, (item,), timeout, mem_limit, cpu_limit
    )
    cap.deactivate_fixture()
    cap.suspend_global_capture(in_=False)
    out, err = cap.read_global_capture()

    warnings_captured = None

    # The result might be an exception, that comes
    # form current process
    if result and not isinstance(result, BaseException):
        # The result came from child process
        result, warnings_captured = result

    if result and not isinstance(result, BaseException):
        results: List[pytest.TestReport] = []
        for s_result in result:
            results.append(
                item.config.hook.pytest_report_from_serializable(
                    config=item.config, data=s_result
                )
            )

        results[-1].sections.append(("Captured stderr", err))
        results[-1].sections.append(("Captured stdout", out))

        if warnings_captured:
            try:
                for warning_message in warnings_captured:
                    item.ihook.pytest_warning_recorded.call_historic(
                        kwargs=dict(
                            warning_message=warning_message,
                            nodeid=item.nodeid,
                            when="call",
                            location=None,
                        )
                    )
            except Exception:
                pass

        return results

    return [report_process_crash(item, out, err, exitcode, timed_out)]


def report_process_crash(
    item: pytest.Item,
    out,
    err,
    exitcode: int,
    timed_out: Optional[int],
):
    from _pytest._code import getfslineno

    path, lineno = getfslineno(item)
    info = ""
    if (path, lineno) != ("", -1):
        info += f"{path}, {lineno}: "

    if timed_out is not None:
        info += f"Timeout > {timed_out}"
    elif exitcode == -24:
        info += "SIGXCPU: CPU time limit exceeded"
    elif exitcode != 0:
        info += f"Crashed with exit code {exitcode}"
    else:
        info += "Exited with no result, memory limit exceeded (probably)"

    call = runner.CallInfo.from_call(lambda: 0 / 0, "???")
    call.excinfo = info
    rep = runner.pytest_runtest_makereport(item, call)
    if out:
        rep.sections.append(("captured stdout", out))
    if err:
        rep.sections.append(("captured stderr", err))

    xfail_marker = item.get_closest_marker("xfail")
    if not xfail_marker:
        return rep

    rep.outcome = "skipped"
    reason = xfail_marker.kwargs.get("reason", "")
    rep.wasxfail = ""
    if reason:
        rep.wasxfail = f"reason: {xfail_marker.kwargs.get('reason','')}; "
    rep.wasxfail += f"pytest-isolate reason: {call.excinfo}"

    warnings.warn(
        "pytest-isolate xfail support is incomplete at the moment and may "
        "output a misleading reason message",
        RuntimeWarning,
    )

    return rep


def get_global_value(item: pytest.Item, name: str):
    default = option = None
    try:
        if item.config.getini(name):
            default = item.config.getini(name)
    except (ValueError, KeyError):
        pass
    try:
        if item.config.getoption(name):
            option = item.config.getoption(name)
    except (ValueError, KeyError):
        pass
    return option or default


def get_marker(item, marker_name, argname=None, pos=None):
    marker = None
    try:
        marker_opt = item.get_closest_marker(marker_name)
        if (
            marker_opt
            and argname
            and marker_opt.kwargs
            and marker_opt.kwargs.get(argname)
        ):
            marker = marker_opt.kwargs.get(argname)
        if (
            marker_opt
            and marker is None
            and pos is not None
            and marker_opt.args
            and len(marker_opt.args) > pos
        ):
            marker = marker_opt.args[0]
    except (ValueError, KeyError):
        pass
    return marker


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    if item.config.pluginmanager.get_plugin("forked"):
        return
    if item.config.pluginmanager.get_plugin("timeout"):
        return
    isolate, timeout, mem_limit, cpu_limit = get_isolation_options(item)

    if (
        isolate is not None
        or mem_limit is not None
        or timeout is not None
        or cpu_limit is not None
    ):
        ihook = item.ihook
        ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        reports = run_in_subprocess(item, timeout, mem_limit, cpu_limit)

        for rep in reports:
            ihook.pytest_runtest_logreport(report=rep)
        ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)

        return True


def get_timeout(item):
    # Should we timeout?
    timeout = get_global_value(item, "timeout")
    timeout_marker = get_marker(item, "timeout", "timeout", 0)

    isolate_timeout = get_global_value(item, "isolate_timeout")
    isolate_timeout_marker = get_marker(item, "isolate", "timeout", 0)

    if isolate_timeout_marker is not None:
        timeout = isolate_timeout_marker
    elif timeout_marker is not None:
        timeout = timeout_marker
    elif isolate_timeout is not None:
        timeout = isolate_timeout

    try:
        timeout = float(timeout)
    except (ValueError, TypeError):
        timeout = None
    return timeout


def get_memory_limit(item):
    # Should we limit memory?
    mem_limit = None
    isolate_mem_limit_marker = get_marker(item, "isolate", "mem_limit", 1)
    isolate_mem_limit = get_global_value(item, "isolate_mem_limit")

    if isolate_mem_limit_marker is not None:
        mem_limit = isolate_mem_limit_marker
    else:
        mem_limit = isolate_mem_limit

    try:
        mem_limit = int(mem_limit)
        if mem_limit < 1 or round(mem_limit) != mem_limit:
            raise ValueError("Memory limit must be in whole bytes")
        mem_limit = int(mem_limit)
    except (ValueError, TypeError):
        mem_limit = None
    return mem_limit


def get_cpu_limit(item):
    # Should we limit cpu?
    cpu_limit = None
    isolate_cpu_limit_marker = get_marker(item, "isolate", "cpu_limit", 2)
    isolate_cpu_limit = get_global_value(item, "isolate_cpu_limit")

    if isolate_cpu_limit_marker is not None:
        cpu_limit = isolate_cpu_limit_marker
    else:
        cpu_limit = isolate_cpu_limit

    try:
        cpu_limit = float(cpu_limit)
        if cpu_limit < 1 or round(cpu_limit) != cpu_limit:
            raise ValueError("Cpu limit must be in whole seconds")
        cpu_limit = int(cpu_limit)
    except (ValueError, TypeError):
        cpu_limit = None
    return cpu_limit


def get_isolation_options(item):
    # Should we fork?
    forked = get_global_value(item, "forked")
    forked_marker = get_marker(item, "forked")
    isolate = get_global_value(item, "isolate")
    isolate_marker = get_marker(item, "isolate")
    if isolate_marker is not None:
        isolate = isolate_marker
    elif forked_marker is not None:
        isolate = forked_marker
    elif isolate is not None:
        pass
    else:
        isolate = forked
    timeout = get_timeout(item)
    mem_limit = get_memory_limit(item)
    cpu_limit = get_cpu_limit(item)

    return isolate, timeout, mem_limit, cpu_limit


@pytest.hookimpl(tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    durations = terminalreporter.config.option.durations
    durations_min = terminalreporter.config.option.durations_min
    verbose = terminalreporter.config.getvalue("verbose")
    tr = terminalreporter
    if durations is not None:
        dlist = []
        for replist in tr.stats.values():
            for rep in replist:
                if hasattr(rep, "cpu_usage"):
                    dlist.append(rep)
        if not dlist:
            return
        dlist.sort(key=lambda x: x.cpu_usage, reverse=True)  # type: ignore[no-any-return]
        if not durations:
            tr.write_sep("=", "highest cpu_usage")
        else:
            tr.write_sep("=", "highest %s cpu_usage" % durations)
            dlist = dlist[:durations]

        for i, rep in enumerate(dlist):
            if verbose < 2 and rep.cpu_usage < durations_min:
                tr.write_line("")
                tr.write_line(
                    "(%s cpu_usage < %gs hidden.  Use -vv to show these durations.)"
                    % (len(dlist) - i, durations_min)
                )
                break
            if rep.when == "teardown":
                tr.write_line(f"{rep.cpu_usage:02.2f}s {rep.nodeid}")

    if config.getoption("timeline"):
        events = []
        for replist in tr.stats.values():
            rep: pytest.TestReport

            for rep in replist:
                if not isinstance(rep, pytest.TestReport) or rep.when != "call":
                    continue
                props = dict(getattr(rep, "user_properties", ()))
                events.extend(
                    create_event(
                        props.get("worker_id", "master"),
                        rep.nodeid,
                        rep.outcome,
                        rep.start,
                        rep.stop,
                    )
                )
        filename = config.getoption("timeline_file")
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        json.dump(events, open(filename, "w"))


@pytest.hookimpl(tryfirst=False, hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    yield
    item.user_properties.append(
        ("worker_id", os.getenv("PYTEST_XDIST_WORKER", "master"))
    )


def create_event(worker_name, test_name, category, start_time, end_time, **kwargs):
    start_event = {
        "name": test_name.rsplit("/")[-1],
        "cat": category,
        "ph": "B",
        "pid": worker_name,
        "tid": 0,
        "ts": start_time * (1000**2),
        "args": kwargs,
    }
    end_event = {
        "name": test_name.rsplit("/")[-1],
        "cat": "pipeline",
        "ph": "E",
        "pid": worker_name,
        "tid": 0,
        "ts": end_time * (1000**2),
    }
    return start_event, end_event
