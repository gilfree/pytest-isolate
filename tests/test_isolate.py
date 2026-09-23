import os
import random
import sys
import warnings
from time import sleep, time

import pytest


@pytest.mark.isolate()
def test_isolate_ok():
    print(f"isolated in {os.getpid()}")
    pass


@pytest.mark.xfail
@pytest.mark.isolate()
def test_isolate_failed():
    pytest.fail("Try fail")


@pytest.mark.xfail
@pytest.mark.isolate(0.1)
def test_isolate_timeout():
    sleep(3)


@pytest.mark.xfail
@pytest.mark.isolate
def test_isolate_segfault():
    import ctypes

    ctypes.string_at(0)


@pytest.mark.xfail
@pytest.mark.isolate
def test_exception():
    raise RuntimeError()


def test_bare_ok():
    print(f"isolated in {os.getpid()}")


@pytest.mark.xfail(strict=True)
def test_bare_fail():
    print("Hello")
    print("Hello")
    print("Hello")
    print(f"isolated in {os.getpid()}", file=sys.stderr)
    sleep(10)


@pytest.mark.isolate(mem_limit=10**9)
def test_rss_limit_ok():
    a = bytearray(10**5)
    print(a[random.randint(0, len(a))])


@pytest.mark.xfail
@pytest.mark.isolate(mem_limit=10**5)
def test_rss_limit_fail():
    a = bytearray(10**6)
    print(a[random.randint(0, len(a))])


@pytest.mark.isolate()
def test_warn():
    warnings.warn("Boo")


@pytest.mark.xfail
@pytest.mark.isolate(cpu_limit=1)
def test_isolate_cpu():
    import numpy as np

    # hog CPU:
    x = 0
    for i in range(100):
        x += (np.random.rand(1000, 1000) @ np.random.rand(1000, 1000)).sum()


def test_slow():
    sleep(0.1)
    pass


# These spawn a pytest subprocess, which costs more than the suite-wide 2 s.
@pytest.mark.isolate(60)
def test_passing_test_does_not_wait_a_wait_delta(pytester):
    """A passing test must not cost a wait_delta. With the bug, three tests at
    wait_delta=5.0 take 15 s."""
    pytester.makepyfile(
        """
        def test_a(): pass
        def test_b(): pass
        def test_c(): pass
        """
    )
    started = time()
    result = pytester.runpytest_subprocess("--isolate", "-o", "wait_delta=5.0")
    elapsed = time() - started

    result.assert_outcomes(passed=3)
    # Interpreter startup dominates what is left, so the margin is wide.
    assert elapsed < 10.0, (
        f"three trivial isolated tests took {elapsed:.1f}s at wait_delta=5.0"
    )


@pytest.mark.isolate(60)
def test_output_survives_the_early_exit(pytester):
    """The child flushes after putting its result, so the early exit must not
    truncate output."""
    pytester.makepyfile(
        """
        import sys

        def test_talks():
            print("OUT-MARKER")
            print("ERR-MARKER", file=sys.stderr)
        """
    )
    result = pytester.runpytest_subprocess("--isolate", "-s",
                                           "-o", "wait_delta=0.05")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*OUT-MARKER*"])


def test_nvml_cuda_check_is_set_by_importing_the_plugin():
    """Set at import: a conftest touching torch.cuda would beat any hook."""
    assert os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"


def test_an_explicit_value_is_left_alone(pytester, monkeypatch):
    """An explicit value survives."""
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    pytester.makepyfile(
        """
        import os

        def test_kept():
            assert os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "0"
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)


def test_the_opt_out_switches_it_off(pytester, monkeypatch):
    monkeypatch.setenv("PYTEST_ISOLATE_NO_NVML_CUDA_CHECK", "1")
    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)
    pytester.makepyfile(
        """
        import os

        def test_unset():
            assert "PYTORCH_NVML_BASED_CUDA_CHECK" not in os.environ
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)



@pytest.mark.isolate(60)
@pytest.mark.parametrize(
    "fixture, write, expected",
    [
        ("capsys", 'print("SYS")', '"SYS\\n"'),
        ("capsysbinary", 'print("SYS")', 'b"SYS\\n"'),
        ("capfd", 'os.write(1, b"FD\\n")', '"FD\\n"'),
        ("capfdbinary", 'os.write(1, b"FD\\n")', 'b"FD\\n"'),
    ],
)
def test_capture_fixtures_work_in_the_child(pytester, fixture, write, expected):
    """The child drops the parent's capturemanager; the capture fixtures still
    need one."""
    pytester.makepyfile(
        f"""
        import os

        def test_reads_its_output({fixture}):
            {write}
            assert {fixture}.readouterr().out == {expected}
        """
    )
    pytester.runpytest_subprocess("--isolate").assert_outcomes(passed=1)


@pytest.mark.isolate(60)
@pytest.mark.skipif(not os.path.exists("/proc/self/status"), reason="needs /proc")
def test_durations_report_peak_memory_of_a_freed_allocation(pytester):
    """Freed before the test ends, so VmData alone would miss it."""
    pytester.makepyfile(
        """
        def test_big():
            block = bytearray(300 * 2**20)
            block[::4096] = b"x" * len(block[::4096])
            del block

        def test_small():
            pass
        """
    )
    result = pytester.runpytest_subprocess("--isolate", "--durations", "0")
    result.assert_outcomes(passed=2)
    lines = result.stdout.lines
    rows = lines[lines.index(next(x for x in lines if "peak memory" in x)) + 1:][:2]
    data, rss = (float(rows[0].split()[i]) for i in (0, 3))
    assert rows[0].endswith("::test_big"), rows
    assert data >= 300 and rss >= 300, rows[0]
