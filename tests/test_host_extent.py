"""CPU admission/identity controls. Real native behavior is checked when equipped."""
import json
from types import SimpleNamespace

import pytest

from triton_msl.backend import _cache_contract as contract, _host_extent as extent

PAGE = 16384


class Storage:
    def __init__(self):
        self.ptr, self.size = 16 * PAGE, PAGE + 1024

    def data_ptr(self):
        return self.ptr

    def nbytes(self):
        return self.size


@pytest.mark.parametrize("ptr,size,state,expected", [
    (16 * PAGE, 1024, True, "mirror"), (16 * PAGE + 64, PAGE + 1024, True, "mirror"),
    (16 * PAGE, PAGE, True, "existing"), (16 * PAGE, PAGE, False, "existing"),
    (16 * PAGE, PAGE + 1024, False, "mirror"), (0, PAGE + 1024, True, "mirror"),
    (16 * PAGE, (1 << 64) - 1, True, "mirror"),
    (16 * PAGE, PAGE + 1024, "unavailable", "mirror"),
])
def test_no_query_paths(ptr, size, state, expected):
    def forbidden(*args):
        raise AssertionError("allocator query on an ineligible path")
    assert extent.select(Storage(), ptr, size, state, forbidden)[0] == expected


def test_live_owner_and_fresh_replacement():
    st = Storage()
    query = lambda p, n: (((n + PAGE - 1) // PAGE) * PAGE, "admitted")
    ptr, size = st.ptr, st.size
    assert extent.select(st, ptr, size, True, query)[0] == "borrow"
    st.ptr += PAGE
    assert extent.select(st, ptr, size, True, query)[0] == "refuse"
    assert extent.select(st, st.ptr, size, True, query)[0] == "borrow"
    def changes(p, n):
        st.size += 4
        return query(p, n)
    assert extent.select(st, st.ptr, size, True, changes)[0] == "refuse"
    assert extent.select(st, st.ptr, st.size, True, query)[0] == "borrow"


def test_unavailable_or_refused_helper_preserves_mirror():
    st = Storage()
    assert extent.select(st, st.ptr, st.size, True, None)[2] == "unavailable"
    assert extent.select(st, st.ptr, st.size, True, lambda p, n: (0, "vm_region"))[2] == "vm_region"


def test_owner_storage_replacement_during_query_refuses():
    st = Storage()
    owner = SimpleNamespace(untyped_storage=lambda: st)
    replacement = Storage()
    replacement.ptr += PAGE
    def change(p, n):
        owner.untyped_storage = lambda: replacement
        return 2 * PAGE, "admitted"
    assert extent.select(st, st.ptr, st.size, True, change, owner=owner)[0] == "refuse"


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("value", [None, "0", "1"])
def test_policy_stamp_and_fast_encoder(monkeypatch, ready, value):
    monkeypatch.setattr(contract, "_HOST_EXTENT_READY", ready)
    monkeypatch.setattr(contract, "_policy_snapshot", None)
    if value is None:
        monkeypatch.delenv("TRITON_MSL_HOST_EXTENT", raising=False)
    else:
        monkeypatch.setenv("TRITON_MSL_HOST_EXTENT", value)
    policy = contract.effective_policy()
    expected = (True if ready else "unavailable") if value == "1" else False
    assert policy["HOST_EXTENT"] == expected
    assert policy.keys() == contract._STAMP_POLICY_FIELDS
    source = dict(schema=contract.SOURCE_SCHEMA, implementation="p", frameworks="f", label="l", policy=policy)
    contract._encode_known_stamp.cache_clear()
    first = contract._encode_execution(source, "toolchain")
    assert contract._encode_execution(source, "toolchain") == first
    assert json.loads(first)["source"]["policy"]["HOST_EXTENT"] == expected
    assert contract._encode_known_stamp.cache_info().hits == 1


def test_trace_is_copied_and_context_local():
    import contextvars
    extent.clear_launch()
    metadata = {}
    extent.record({"route": "mirror", "reason": "geometry"}, metadata)
    read = extent.last_launch()
    read[0]["route"] = "borrow"
    assert extent.last_launch()[0]["route"] == "mirror"
    assert contextvars.Context().run(extent.last_launch) is None
    assert metadata["host_extent"][0]["reason"] == "geometry"
    extent.clear_launch()
    assert extent.last_launch() is None


def test_real_native_extent_and_interior():
    native = contract._validation_native
    if native is None or not hasattr(native, "host_extent"):
        pytest.skip("requires the extent-equipped validation helper")
    import torch
    if not native.host_extent_ready:
        assert native.host_extent(0, 0) == (0, "unavailable")
        return
    tensor = torch.empty((1048832,), dtype=torch.float32)
    st = tensor.untyped_storage()
    span, reason = native.host_extent(st.data_ptr(), st.nbytes())
    assert (span, reason) == (4210688, "admitted")
    assert native.host_extent(st.data_ptr() + PAGE, PAGE + 1024) == (0, "no_malloc_extent")
