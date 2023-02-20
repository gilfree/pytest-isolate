# pytest-isolate

[![Python package](https://github.com/gilfree/pytest-isolate/actions/workflows/python-package.yml/badge.svg)](https://github.com/gilfree/pytest-isolate/actions/workflows/python-package.yml)

[![PyPI - Version](https://img.shields.io/pypi/v/pytest-isolate.svg)](https://pypi.org/project/pytest-isolate)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytest-isolate.svg)](https://pypi.org/project/pytest-isolate)

Isolate each test in a subprocess - pytest forked replacement, based on pytest-forked.

This pytest plugin was generated with Cookiecutter along with `@hackebrot`'s `cookiecutter-pytest-plugin` template.

## Features

* Run each test in an forked, isolated subprocess
* Captures stdout & stderr of crashing processes
* Add Timeout to a forked test
* Limit memory used by test
* Plays nice with pytest-xdist
* Shows warnings, even with xdist!

## Requirements

* pytest

## Installation

You can install "pytest-isolate" via `pip` from `PyPI`

    pip install pytest-isolate

## Usage

    pytest --isolate

To run every test in its own **forked** subprocess.

Or:

    pytest --isolate-timeout 10 --isolate-mem-limit 1000000

To set a timeout to every test in addition to forking.

> Note:
>
> Since this plugin uses `fork`, it will not work on  operating systems without `fork` support (e.g. Windows).

The flags `--timeout` or `--forked` will also be respected such that `pytest-isolate` is a drop-in replacement forked pytest forked and pytest timeout.

If `pytest-forked` or `pytest-timeout` are installed, then
they will take precedence. Uninstall them to use `pytest-isolate`.

Unlike `pytest-timeout`, timeout in `pytest-isolate` is implemented by forking the test to a separate subprocess, and setting timeout for that subprocess.

You can also use a mark to isolate or time limit the memory test:

```python
@pytest.mark.isolate(timeout=10,mem_limit=10**6)
def test_something():
    pass
```

The options can also be set in an pytest configuration file, e.g:

```toml
[tool.pytest.ini_options]
isolate_timeout=10
isolate_mem_limit=1000000
```

## Contributing

Contributions are very welcome. Tests can be run with `tox`, please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the `MIT`, `pytest-isolate` is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.
