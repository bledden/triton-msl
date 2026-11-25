"""``MetalUtils.load_binary`` hands Metal a file and takes it away again.

Metal loads a library from a URL, so the bytes Triton passes are written to a temporary
file first. One file per kernel per process was left in the user's temporary directory
(resources review, Nov 24: 1 file for an operator, 10 for a compiled GPT-2). The file is
needed while the library is loaded and its pipelines are built, and by nothing after
``load_binary`` returns or raises; a path the caller passed is the caller's file.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import triton  # noqa: F401 -- complete backend discovery before importing driver

from triton_msl.backend import driver

BINARY = b"not a real metallib: the device below only reads it back"


class _Pipeline:
    __slots__ = ()

    def maxTotalThreadsPerThreadgroup(self):
        return 1024


class _Device:
    """A device that notes whether the file is there, with the bytes, each time Metal would read it."""

    def __init__(self, library_error=None, missing=False, pipeline_error=None):
        self.library_error, self.missing, self.pipeline_error = library_error, missing, pipeline_error
        self.path = None
        self.seen = []

    def _read(self, step):
        with open(self.path, "rb") as file:
            self.seen.append((step, file.read()))

    def newLibraryWithURL_error_(self, url, _):
        self.path = url
        self._read("library")
        return (None, self.library_error) if self.library_error else (self, None)

    def newFunctionWithName_(self, name):
        self._read("function " + name)
        if name.endswith("__mmdirect"):
            return name
        return None if self.missing else name

    def functionNames(self):
        return SimpleNamespace(count=lambda: 0)

    def newComputePipelineStateWithFunction_error_(self, function, _):
        self._read("pipeline " + function)
        if self.pipeline_error and not function.endswith("__mmdirect"):
            return None, self.pipeline_error
        return _Pipeline(), None


@pytest.fixture
def loader(monkeypatch, tmp_path):
    """(MetalUtils around a recording device factory, the private temporary directory)."""
    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: p)))
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))

    def make(**faults):
        utils = driver.MetalUtils()
        utils._device = _Device(**faults)
        return utils

    return make, tmp_path


def test_a_loaded_binary_leaves_no_file_and_was_there_for_every_step(loader):
    make, directory = loader
    utils = make()
    library, pipeline, *_ = utils.load_binary("kernel", BINARY, 0)
    try:
        assert isinstance(pipeline, _Pipeline)
        steps = [step for step, _ in utils._device.seen]
        assert steps == [
            "library",
            "function kernel",
            "pipeline kernel",
            "function kernel__mmdirect",
            "pipeline kernel__mmdirect",
        ]
        assert all(content == BINARY for _, content in utils._device.seen)
        assert os.listdir(directory) == []
    finally:
        utils.unload_module(library)


@pytest.mark.parametrize(
    "faults, message",
    [
        ({"library_error": "no such library"}, "Failed to load metallib"),
        ({"missing": True}, "not found in metallib"),
        ({"pipeline_error": "no pipeline"}, "Failed to create pipeline state"),
    ],
)
def test_a_binary_that_fails_to_load_leaves_no_file(loader, faults, message):
    make, directory = loader
    utils = make(**faults)
    with pytest.raises(RuntimeError, match=message):
        utils.load_binary("kernel", BINARY, 0)
    assert utils._device.seen and utils._device.seen[0] == ("library", BINARY)
    assert os.listdir(directory) == []


def test_a_bytearray_is_a_binary_too(loader):
    make, directory = loader
    utils = make()
    library, *_ = utils.load_binary("kernel", bytearray(BINARY), 0)
    utils.unload_module(library)
    assert os.listdir(directory) == []


def test_a_path_the_caller_passed_is_left_alone(loader):
    make, directory = loader
    theirs = directory / "theirs.metallib"
    theirs.write_bytes(BINARY)
    utils = make()
    library, *_ = utils.load_binary("kernel", str(theirs), 0)
    utils.unload_module(library)
    assert utils._device.path == str(theirs)
    assert theirs.read_bytes() == BINARY
    assert os.listdir(directory) == ["theirs.metallib"]


class _Handle:
    """The temporary file's own handle, failing at one step."""

    def __init__(self, real, fail):
        self._real, self._fail, self.name, self.written = real, fail, real.name, []

    def write(self, data):
        if self._fail == "write":
            raise OSError("no space left on device")
        self.written.append(bytes(data))
        return self._real.write(data)

    def close(self):
        self._real.close()
        if self._fail == "close":
            raise OSError("close failed")


def _handles(monkeypatch, fail=None):
    made = []
    real = driver.tempfile.NamedTemporaryFile

    def named(*args, **kwargs):
        made.append(_Handle(real(*args, **kwargs), fail))
        return made[-1]

    monkeypatch.setattr(driver.tempfile, "NamedTemporaryFile", named)
    return made


def test_the_binary_is_written_through_the_temporary_files_own_handle(loader, monkeypatch):
    make, directory = loader
    made = _handles(monkeypatch)
    utils = make()
    library, *_ = utils.load_binary("kernel", BINARY, 0)
    utils.unload_module(library)
    assert [handle.written for handle in made] == [[BINARY]]
    assert os.listdir(directory) == []


@pytest.mark.parametrize("step", ["write", "close"])
def test_a_binary_that_cannot_be_written_leaves_no_file_and_reaches_no_device(loader, monkeypatch, step):
    make, directory = loader
    _handles(monkeypatch, fail=step)
    utils = make()
    with pytest.raises(OSError, match="no space left|close failed"):
        utils.load_binary("kernel", BINARY, 0)
    assert utils._device.seen == []
    assert os.listdir(directory) == []


def test_a_file_that_cannot_be_removed_is_reported_and_the_library_is_still_returned(loader, monkeypatch):
    make, directory = loader
    utils = make()

    def refuse(path):
        raise PermissionError(13, "Permission denied", path)

    with monkeypatch.context() as patch:
        patch.setattr(driver.os, "unlink", refuse)
        with pytest.warns(RuntimeWarning, match=r"temporary Metal library .*\.metallib could not be removed"):
            library, pipeline, *_ = utils.load_binary("kernel", BINARY, 0)
    utils.unload_module(library)
    assert isinstance(pipeline, _Pipeline)
    assert len(os.listdir(directory)) == 1


def test_a_removal_failure_does_not_hide_the_load_failure(loader, monkeypatch):
    make, _ = loader
    utils = make(library_error="no such library")

    def refuse(path):
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(driver.os, "unlink", refuse)
    with (
        pytest.warns(RuntimeWarning, match="could not be removed"),
        pytest.raises(RuntimeError, match="Failed to load metallib"),
    ):
        utils.load_binary("kernel", BINARY, 0)


_TWICE = """
#include <metal_stdlib>
using namespace metal;
kernel void twice(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    y[gid] = 2.0f * x[gid];
}
"""


def test_the_device_computes_with_a_pipeline_whose_file_is_gone(runner, monkeypatch, tmp_path):
    """On the real device: nothing is left, and the pipeline still runs after the file is removed."""
    with open(runner.compile(_TWICE, "twice"), "rb") as file:
        binary = file.read()
    monkeypatch.setattr(driver.tempfile, "tempdir", str(tmp_path))
    utils = driver.MetalUtils()
    utils._device = runner.device  # the buffers below and the pipeline belong to one device
    library, pipeline, *_ = utils.load_binary("twice", binary, 0)
    try:
        assert os.listdir(tmp_path) == []
        values = [float(index) - 100.0 for index in range(256)]
        x, y = runner.make_float_buffer(values), runner.make_empty_buffer(256)
        runner.run(pipeline, [x, y], 256, block_size=256)
        assert runner.read_float_buffer(y, 256) == [2.0 * value for value in values]
    finally:
        utils.unload_module(library)
