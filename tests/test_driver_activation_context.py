"""Activation uses the compiler's selection boundary; CPU-only Metal endpoints."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import triton.backends

from triton_msl.backend import _toolchain_contract as tc, driver


@pytest.fixture
def activation(monkeypatch, tmp_path):
    bundle = tmp_path / "Selected.xctoolchain"
    selector = bundle / "usr/bin/metal"
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"selector")
    resolver = tmp_path / "xcrun"
    resolver.write_bytes(b"resolver")
    events = {"queries": [], "legacy": [], "devices": []}
    monkeypatch.setattr(driver.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(tc, "_compiler_context", None)
    monkeypatch.setattr(tc.shutil, "which", lambda _: str(resolver))

    def query(command, **kwargs):
        events["queries"].append((command, kwargs))
        return str(selector)

    def legacy(command, **kwargs):
        events["legacy"].append(command)
        return subprocess.CompletedProcess(command, 0)

    def device():
        events["devices"].append(True)
        return object()

    monkeypatch.setattr(tc.subprocess, "check_output", query)
    monkeypatch.setattr(driver.subprocess, "run", legacy)
    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=device))
    return events, resolver, selector


def test_non_darwin_does_not_select_or_query_device(activation, monkeypatch):
    events, _, _ = activation
    monkeypatch.setattr(driver.platform, "system", lambda: "Linux")
    assert driver.MetalDriver.is_active() is False
    assert events == {"queries": [], "legacy": [], "devices": []}


def test_activation_reuses_compiler_context_without_inventory(activation, monkeypatch):
    events, resolver, _ = activation

    def forbidden(*args, **kwargs):
        pytest.fail("activation requested full toolchain identity")

    monkeypatch.setattr(tc, "toolchain_identity", forbidden)
    assert driver.MetalDriver.is_active() is True
    assert driver.MetalDriver.is_active() is True
    assert events["legacy"] == []
    assert events["queries"] == [
        ([str(resolver.resolve()), "-sdk", "macosx", "--find", "metal"], {"text": True, "stderr": subprocess.PIPE})
    ]
    assert len(events["devices"]) == 2


@pytest.mark.parametrize("name", ["PATH", "DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS"])
def test_changed_selection_refuses_before_device(activation, monkeypatch, name):
    events, _, _ = activation
    tc.compiler_process_context()
    monkeypatch.setenv(name, "/changed-selection-for-activation")
    with pytest.warns(RuntimeWarning, match="selection changed.*restart"):
        assert driver.MetalDriver.is_active() is False
    assert len(events["queries"]) == 1
    assert events["legacy"] == []
    assert events["devices"] == []


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("resolver absent"),
        PermissionError("resolver inaccessible"),
        subprocess.CalledProcessError(1, "selected resolver"),
        subprocess.TimeoutExpired("selected resolver", 5),
        RuntimeError("unproved selected layout"),
    ],
)
def test_context_failure_is_inactive_before_device(activation, monkeypatch, error):
    events, _, _ = activation

    def unavailable():
        raise error

    monkeypatch.setattr(tc, "compiler_process_context", unavailable)
    with pytest.warns(RuntimeWarning, match="cannot select a usable Metal compiler") as caught:
        assert driver.MetalDriver.is_active() is False
    assert str(error) in str(caught[0].message)
    assert events == {"queries": [], "legacy": [], "devices": []}


@pytest.mark.parametrize("selection", ["", "unbundled"])
def test_unproved_real_context_is_inactive(activation, monkeypatch, tmp_path, selection):
    events, _, _ = activation
    unbundled = tmp_path / "metal"
    unbundled.write_bytes(b"unbundled selector")
    monkeypatch.setattr(tc.subprocess, "check_output", lambda *a, **k: "" if not selection else str(unbundled))
    with pytest.warns(RuntimeWarning, match="empty Metal compiler selection|unproved metal toolchain layout"):
        assert driver.MetalDriver.is_active() is False
    assert events["devices"] == []
    assert tc._compiler_context is None


def test_missing_metal_binding_is_inactive(activation, monkeypatch):
    monkeypatch.setitem(sys.modules, "Metal", None)
    assert driver.MetalDriver.is_active() is False


def test_no_device_is_inactive(activation, monkeypatch):
    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=lambda: None))
    assert driver.MetalDriver.is_active() is False


def test_device_runtime_error_is_not_mislabeled_as_selection_failure(activation, monkeypatch):
    error = RuntimeError("device query failed")

    def failed():
        raise error

    monkeypatch.setitem(sys.modules, "Metal", SimpleNamespace(MTLCreateSystemDefaultDevice=failed))
    with pytest.raises(RuntimeError) as caught:
        driver.MetalDriver.is_active()
    assert caught.value is error


def test_unexpected_context_programming_error_propagates(activation, monkeypatch):
    def broken():
        raise TypeError("invalid context implementation")

    monkeypatch.setattr(tc, "compiler_process_context", broken)
    with pytest.raises(TypeError, match="invalid context implementation"):
        driver.MetalDriver.is_active()
