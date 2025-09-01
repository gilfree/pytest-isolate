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
* Limit CPU time used by test
* Manage GPU resources with CUDA_VISIBLE_DEVICES, with support for fractional requests (1/2, 1/4, 1/8, 1/16)
* Plays nice with pytest-xdist
* Shows warnings, even with xdist!
* *Create visual timeline of test execution (isolated or not)*

## Requirements

* pytest

### Optional Dependencies

For GPU resource management:

* pynvml (optional, for automatic GPU detection)

## Support

* Operating Systems: Linux (tested), macOS (not tested but should work). Windows is not supported,
  and will probably not work as we are based on process forking.
* Python Versions: 3.9, 3.10, 3.11, 3.12
  
## Installation

You can install "pytest-isolate" via `pip` from `PyPI`

    pip install pytest-isolate

For GPU resource management support:

    pip install pytest-isolate[gpu]

## Usage

    pytest --isolate

To run every test in its own **forked** subprocess.

Or:

    pytest --isolate-timeout 10 --isolate-mem-limit 1000000 --isolate-cpu-limit 10

To set a timeout to every test in addition to forking, and limit to 10 cpu seconds.

Or:

    pytest --timeline

With possible combination of the above, to generate a timeline of test execution. The
timeline can be viewed in chrome://tracing.

To disable the pulgin, you can use the `--no-isolate` option:

    pytest --no-isolate

> ***Note:***
>
> Since this plugin uses `fork`, it will not work on  operating systems without `fork` support (e.g. Windows).

The flags `--timeout` or `--forked` will also be respected such that `pytest-isolate` is a drop-in replacement forked pytest forked and pytest timeout.

If `pytest-forked` or `pytest-timeout` are installed, then
they will take precedence. Uninstall them to use `pytest-isolate`.

Unlike `pytest-timeout`, timeout in `pytest-isolate` is implemented by forking the test to a separate subprocess, and setting timeout for that subprocess.

### Using the isolate marker

You can use a mark to isolate or time limit the memory and/or cpu usage test:

```python
@pytest.mark.isolate(timeout=10, mem_limit=10**6, cpu_limit=10)
def test_something():
    pass
```

The `isolate` marker can also be used to request gpus for a test on a gpu machine:

```python
# Request 2 GPUs for this test
@pytest.mark.isolate(resources={'gpu': 2})
def test_with_gpus():
    # The test will have CUDA_VISIBLE_DEVICES set to the allocated GPU IDs
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")) == 2

# Request 1/2 of a GPU for this test, up to 2 tests that request 1/2 of a GPU will be allocated the same GPU
@pytest.mark.isolate(resources={'gpu': 1/2})
def test_with_half_gpu():
    # The test will have CUDA_VISIBLE_DEVICES set to the allocated GPU IDs
    assert len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")) == 1
```

### Configuration Options

The options can be set in an pytest configuration file, e.g:

```toml
[tool.pytest.ini_options]
isolate_timeout=10
isolate_mem_limit=1000000
isolate_cpu_limit=10
```

## CUDA_VISIBLE_DEVICES Handling

If `CUDA_VISIBLE_DEVICES` is already set when pytest starts, the plugin will respect this setting and only allocate from the GPUs specified there. This works even without pynvml installed.

For example:

* If `CUDA_VISIBLE_DEVICES=0,1,2` is set, tests will only use GPUs 0, 1, and 2.
* If `CUDA_VISIBLE_DEVICES=` is set (empty), no GPUs will be used.

## Contributing

Contributions are very welcome. Tests can be run with `tox`, please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the `MIT`, `pytest-isolate` is free and open source software

## Issues

If you encounter any problems, please file an issue along with a detailed description.
