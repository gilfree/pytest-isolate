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
