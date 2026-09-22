"""pytest-isolate: run each test in its own forked process."""


class PytestIsolateWarning(UserWarning):
    """Every warning this plugin raises, so it can be filtered on its own.

    Declared here so ``ignore::pytest_isolate.PytestIsolateWarning`` resolves
    without importing the plugin.
    """


__all__ = ["PytestIsolateWarning"]
