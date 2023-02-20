import os
import random
import sys
import warnings
from time import sleep

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


@pytest.mark.xfail
def test_bare_fail():
    print("Hello")
    print("Hello")
    print("Hello")
    print(f"isolated in {os.getpid()}", file=sys.stderr)
    sleep(10)


@pytest.mark.isolate(mem_limit=10**9)
def test_rss_limit_ok():
    a = bytearray(10**7)
    print(a[random.randint(0, len(a))])


@pytest.mark.xfail
@pytest.mark.isolate(mem_limit=10**6)
def test_rss_limit_fail():
    a = bytearray(10**7)
    print(a[random.randint(0, len(a))])


@pytest.mark.isolate()
def test_warn():
    warnings.warn("Boo")
