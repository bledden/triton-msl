"""A `while` loop whose body needs the whole block of lanes computes what the counted loop computes.

Two things stood in the way. A value carried by a `while` loop was not counted among the
values of its kernel, so anything in the body that asked how it is held (a sum, a largest
value, a flip of it) was refused with "native per-result metadata is missing for SSA
value ...". And a `while` loop is written into the shader as `for (;;) {`, its test, then
`if (!(test)) break;`, and that `break` switched off the reuse of threadgroup arrays for
the whole kernel: a flip of 1024 lanes keeps twenty arrays, 61440 bytes where 32768 are
allowed, and the launch failed while the pipeline was being built, with a RuntimeError
that named no reason a caller could act on.

The exit at the head of such a loop is now read as what it is, the test of a loop that
tests before each trip. A kernel whose arrays still do not fit is refused before any
pipeline is asked for, as the out-of-resources error an autotuner already understands.
"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

import triton_msl  # noqa: F401
from triton_msl.codegen import _lowerer_helpers
from triton_msl.codegen._lowerer_helpers import _alias_shared_memory
from triton_msl.errors import MetalNonRecoverableError, MetalResourceError

needs_metal = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires actual Metal")

DEVICES = ["mps", "cpu"]
GRID = 3
TWO_SUMS, RUNNING_SUM, LOOP_IN_LOOP, HELD_ACROSS, STORED_EACH_TRIP = range(5)
BODIES = [
    "two sums and a flip",
    "a running sum of the flip",
    "a loop inside the loop",
    "values held across the loop",
    "stored on each trip",
]
BRANCHED, SUM_IN_THE_TEST = range(2)


@triton.jit
def _looped(Z, R, TRIPS, N: tl.constexpr, BODY: tl.constexpr, COUNTED: tl.constexpr):
    # Each program owns 4 * N cells of Z and of R and touches its first N only.
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    zb = Z + pid * 4 * N
    rb = R + pid * 4 * N
    j = 0
    x = tl.load(zb + n)
    if BODY == 0:
        if COUNTED:
            for j2 in range(TRIPS):
                x = (tl.flip(x, 0) + tl.sum(x, axis=0) - tl.max(x, axis=0) + j2) % 1009
        else:
            while j < TRIPS:
                x = (tl.flip(x, 0) + tl.sum(x, axis=0) - tl.max(x, axis=0) + j) % 1009
                j += 1
        tl.store(rb + n, x)
    elif BODY == 1:
        if COUNTED:
            for j2 in range(TRIPS):
                x = tl.cumsum(tl.flip(x, 0), 0) % 1009 + j2
        else:
            while j < TRIPS:
                x = tl.cumsum(tl.flip(x, 0), 0) % 1009 + j
                j += 1
        tl.store(rb + n, x)
    elif BODY == 2:
        if COUNTED:
            for j2 in range(TRIPS):
                x = tl.flip(x, 0) + j2
                for i2 in range(2):
                    x = (tl.flip(x, 0) * 3 + i2) % 1009
        else:
            while j < TRIPS:
                x = tl.flip(x, 0) + j
                i = 0
                while i < 2:
                    x = (tl.flip(x, 0) * 3 + i) % 1009
                    i += 1
                j += 1
        tl.store(rb + n, x)
    elif BODY == 3:
        a = tl.flip(x, 0)
        s = tl.sum(x, axis=0) % 1009
        if COUNTED:
            for j2 in range(TRIPS):
                x = (tl.flip(x, 0) + a + j2) % 1009
        else:
            while j < TRIPS:
                x = (tl.flip(x, 0) + a + j) % 1009
                j += 1
        tl.store(rb + n, tl.flip(x + a, 0) + s)
    else:
        if COUNTED:
            for j2 in range(TRIPS):
                tl.store(zb + n, (tl.flip(tl.load(zb + n), 0) + tl.sum(tl.load(zb + n), axis=0) + j2) % 1009)
        else:
            while j < TRIPS:
                tl.store(zb + n, (tl.flip(tl.load(zb + n), 0) + tl.sum(tl.load(zb + n), axis=0) + j) % 1009)
                j += 1


@triton.jit
def _too_many_arrays(Z, R, TRIPS, N: tl.constexpr, FORM: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    zb = Z + pid * 4 * N
    rb = R + pid * 4 * N
    j = 0
    x = tl.load(zb + n)
    if FORM == 0:  # the flip runs on some trips only, so no trip hands its arrays to the next
        while j < TRIPS:
            if j % 2 == 0:
                x = tl.flip(x, 0) + 1
            else:
                x = x * 2 % 1009
            x = x + tl.sum(x, axis=0) % 7
            j += 1
    else:  # the loop's test itself needs the whole block of lanes
        while tl.max(x, axis=0) // 4096 + j < TRIPS:
            x = (tl.flip(x, 0) + j) % 1009
            j += 1
    tl.store(rb + n, x)


@triton.jit
def _row_sum_of_what_came_back(Z, R, TRIPS, N: tl.constexpr, COUNTED: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, 4)
    k = tl.arange(0, N // 4)
    off = i[:, None] * (N // 4) + k[None, :]
    zb = Z + pid * 4 * N
    old = off * 0
    j = 0
    if COUNTED:
        for j2 in range(TRIPS):
            old = tl.atomic_xchg(zb + off, (tl.sum(old, 1) % 1009)[:, None] + off + j2)
    else:
        while j < TRIPS:
            old = tl.atomic_xchg(zb + off, (tl.sum(old, 1) % 1009)[:, None] + off + j)
            j += 1
    tl.store(R + pid * 4 * N + off, old)


@triton.jit
def _moving_pointer(Z, R, TRIPS, N: tl.constexpr, EXCHANGED: tl.constexpr):
    # At most three trips are launched, so the pointer stays inside the program's 4 * N cells.
    pid = tl.program_id(0)
    n = tl.arange(0, N)
    p = Z + pid * 4 * N + n
    old = n * 0
    j = 0
    while j < TRIPS:
        if EXCHANGED:
            old = tl.atomic_xchg(p, old * 2 + n + j)
        else:
            old = tl.load(p) + old
            tl.store(p, old * 2 + n + j)
        p = p + N
        j += 1
    tl.store(R + pid * 4 * N + n, old)


def _cells(lanes):
    z = (np.arange(GRID * 4 * lanes + 5) * 3 % 997).astype(np.int32)
    r = np.full(GRID * 4 * lanes + 5, -7, dtype=np.int32)
    return z, r


def _looped_expected(lanes, trips, body):
    z0, r0 = _cells(lanes)
    z, r = z0.astype(np.int64), r0.astype(np.int64)
    n = np.arange(lanes)
    for pid in range(GRID):
        b = pid * 4 * lanes
        x = z[b + n].copy()
        if body == TWO_SUMS:
            for j in range(trips):
                x = (x[::-1] + x.sum() - x.max() + j) % 1009
        elif body == RUNNING_SUM:
            for j in range(trips):
                x = np.cumsum(x[::-1]) % 1009 + j
        elif body == LOOP_IN_LOOP:
            for j in range(trips):
                x = x[::-1] + j
                for i in range(2):
                    x = (x[::-1] * 3 + i) % 1009
        elif body == HELD_ACROSS:
            a, s = x[::-1].copy(), x.sum() % 1009
            for j in range(trips):
                x = (x[::-1] + a + j) % 1009
            x = (x + a)[::-1] + s
        else:
            for j in range(trips):
                z[b + n] = (z[b + n][::-1] + z[b + n].sum() + j) % 1009
            continue
        r[b + n] = x
    return z.astype(np.int32), r.astype(np.int32)


def _launch(kernel, lanes, trips, device, **constants):
    z0, r0 = _cells(lanes)
    z, r = torch.tensor(z0, device=device), torch.tensor(r0, device=device)
    kernel[(GRID,)](z, r, trips, N=lanes, **constants)
    if device == "mps":
        torch.mps.synchronize()
    return z.cpu().numpy(), r.cpu().numpy()


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("counted", [0, 1], ids=["while", "counted"])
@pytest.mark.parametrize("body", range(len(BODIES)), ids=BODIES)
@pytest.mark.parametrize("trips", [0, 1, 2, 5])
@pytest.mark.parametrize("lanes", [16, 256, 1024])
def test_a_while_loop_computes_what_the_counted_loop_computes(device, counted, body, trips, lanes):
    z, r = _launch(_looped, lanes, trips, device, BODY=body, COUNTED=counted)
    want_z, want_r = _looped_expected(lanes, trips, body)
    assert np.array_equal(z, want_z)
    assert np.array_equal(r, want_r)


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("counted", [0, 1], ids=["while", "counted"])
@pytest.mark.parametrize("trips", [0, 1, 3])
@pytest.mark.parametrize("lanes", [256, 1024])
def test_a_row_sum_of_what_an_exchange_returned_is_carried_by_either_loop(device, counted, trips, lanes):
    z0, r0 = _cells(lanes)
    z, r = _launch(_row_sum_of_what_came_back, lanes, trips, device, COUNTED=counted)
    want_z, want_r = z0.astype(np.int64), r0.astype(np.int64)
    off = np.arange(4)[:, None] * (lanes // 4) + np.arange(lanes // 4)[None, :]
    for pid in range(GRID):
        old = off * 0
        for j in range(trips):
            new = (old.sum(1) % 1009)[:, None] + off + j
            old = want_z[pid * 4 * lanes + off].copy()
            want_z[pid * 4 * lanes + off] = new
        want_r[pid * 4 * lanes + off] = old
    assert np.array_equal(z, want_z.astype(np.int32))
    assert np.array_equal(r, want_r.astype(np.int32))


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("exchanged", [1, 0], ids=["exchanged", "loaded and stored"])
@pytest.mark.parametrize("trips", [0, 1, 3])
@pytest.mark.parametrize("lanes", [16, 1024])
def test_a_pointer_moved_by_a_while_loop_is_refused_for_its_order(device, exchanged, trips, lanes):
    """How many trips the loop takes is not known before launch, so neither is what it reaches."""
    z0, r0 = _cells(lanes)
    z, r = torch.tensor(z0, device=device), torch.tensor(r0, device=device)
    with pytest.raises(MetalNonRecoverableError, match="lane order: .* a loop whose trips"):
        _moving_pointer[(GRID,)](z, r, trips, N=lanes, EXCHANGED=exchanged)
    assert np.array_equal(z.cpu().numpy(), z0)
    assert np.array_equal(r.cpu().numpy(), r0)


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [BRANCHED, SUM_IN_THE_TEST], ids=["a flip on some trips", "a sum in the test"])
def test_arrays_that_cannot_fit_are_refused_as_out_of_resources(device, form):
    z0, r0 = _cells(1024)
    z, r = torch.tensor(z0, device=device), torch.tensor(r0, device=device)
    with pytest.raises(OutOfResources, match="threadgroup memory bytes") as refusal:
        _too_many_arrays[(GRID,)](z, r, 2, N=1024, FORM=form)
    assert refusal.value.required > refusal.value.limit == 32768
    assert np.array_equal(z.cpu().numpy(), z0)
    assert np.array_equal(r.cpu().numpy(), r0)


@needs_metal
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("form", [BRANCHED, SUM_IN_THE_TEST], ids=["a flip on some trips", "a sum in the test"])
@pytest.mark.parametrize("trips", [0, 1, 2, 5])
def test_the_same_kernels_compute_where_their_arrays_fit(device, form, trips):
    lanes = 256
    z0, r0 = _cells(lanes)
    z, r = _launch(_too_many_arrays, lanes, trips, device, FORM=form)
    want = r0.astype(np.int64)
    n = np.arange(lanes)
    for pid in range(GRID):
        x = z0[pid * 4 * lanes + n].astype(np.int64)
        for j in range(trips):
            if form == BRANCHED:
                x = (x[::-1] + 1) if j % 2 == 0 else (x * 2 % 1009)
                x = x + x.sum() % 7
            else:
                x = (x[::-1] + j) % 1009
        want[pid * 4 * lanes + n] = x
    assert np.array_equal(z, z0)
    assert np.array_equal(r, want.astype(np.int32))


# The text of a loop as the generic lowerer writes a `while`: its test, the exit, the body.
BARRIER = "threadgroup_barrier(mem_flags::mem_threadgroup);"
BODY = f"""a[lid] = v;
        {BARRIER}
        float old = a[127 - lid];
        {BARRIER}
        b[lid] = old;
        {BARRIER}
        v = b[127 - lid];
        {BARRIER}"""


def _shader(loop):
    return f"""kernel void probe(uint lid [[thread_index_in_threadgroup]]) {{
    threadgroup float a[128];
    threadgroup float b[128];
    float v = 0;
    int j = 0;
    {loop}
}}"""


def _one_array(loop):
    aliases = {}
    text = _alias_shared_memory(_shader(loop), allocation_aliases=aliases)
    shared = "threadgroup float b[128];" not in text
    assert (aliases.get("b", "b") == aliases.get("a", "a")) is shared
    return shared


def test_the_counted_loop_this_is_measured_against_shares_its_arrays():
    assert _one_array(f"for (int t = 0; t < 4; ++t) {{\n        {BODY}\n    }}")


def test_an_exit_at_the_head_of_a_loop_is_the_test_of_that_loop():
    head = "bool mask_1 = j < 4;\n        if (!(mask_1)) break;"
    assert _one_array(f"for (;;) {{\n        {head}\n        {BODY}\n        j += 1;\n    }}")
    assert _one_array(f"for (;;) {{\n        if (!(mask_1)) break;\n        {BODY}\n    }}")


@pytest.mark.parametrize(
    "loop",
    [
        # the exit is not the first thing the loop decides
        f"for (;;) {{\n        {BODY}\n        if (!(mask_1)) break;\n    }}",
        f"for (;;) {{\n        a[lid] = v;\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        {BARRIER}\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (j > 9) {{ j = 0; }}\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (j > 9) continue;\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
        # a second way out, wherever it is
        f"for (;;) {{\n        if (!(mask_1)) break;\n        if (!(mask_2)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (!(mask_1)) break;\n        {BODY}\n        if (!(mask_2)) break;\n    }}",
        f"for (;;) {{\n        if (!(mask_1)) break;\n        {BODY}\n        if (j > 9) return;\n    }}",
        f"for (;;) {{\n        if (!(mask_1)) break;\n        {BODY}\n        if (j > 9) continue;\n    }}",
        # an exit of another spelling, or of another kind of loop
        f"for (;;) {{\n        if (mask_1) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (!(j < 4)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (!(mask_1)) return;\n        {BODY}\n    }}",
        f"for (int t = 0; t < 4; ++t) {{\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
        f"for (;;) {{\n        if (lid < 64) {{\n        if (!(mask_1)) break;\n        }}\n        {BODY}\n    }}",
        f"while (true) {{\n        if (!(mask_1)) break;\n        {BODY}\n    }}",
    ],
)
def test_any_other_way_out_of_a_loop_keeps_every_array_apart(loop):
    assert not _one_array(loop)


def test_an_exit_at_the_head_does_not_excuse_an_exit_elsewhere_in_the_kernel():
    head = f"for (;;) {{\n        if (!(mask_1)) break;\n        {BODY}\n    }}"
    assert not _one_array("if (lid > 200) return;\n    " + head)
    assert not _one_array(head + "\n    for (int t = 0; t < 4; ++t) {\n        if (t == 2) break;\n    }")


def _declared(*arrays):
    lines = "\n".join(f"    threadgroup {kind} shared_{i}[{count}];" for i, (kind, count) in enumerate(arrays))
    return "kernel void probe() {\n" + lines + "\n}"


def test_the_bytes_of_the_declared_arrays_are_counted_against_the_limit():
    assert _lowerer_helpers._check_generic_shared_memory_budget(_declared(("float", 4096), ("int", 4096))) == 32768
    assert (
        _lowerer_helpers._check_generic_shared_memory_budget(_declared(("half", 1024), ("uchar", 3), ("bool", 1)))
        == 2052
    )
    assert _lowerer_helpers._check_generic_shared_memory_budget("kernel void probe() {\n}") == 0
    with pytest.raises(MetalResourceError) as refusal:
        _lowerer_helpers._check_generic_shared_memory_budget(_declared(("float", 4096), ("int", 4096), ("uchar", 1)))
    assert (refusal.value.required, refusal.value.limit) == (32769, 32768)
    assert refusal.value.resource == "threadgroup memory bytes"
    assert not isinstance(refusal.value, MetalNonRecoverableError)


def test_only_a_proof_of_excess_is_refused():
    """A declaration this cannot size, or one that is not code, counts as nothing."""
    unknown = _declared(("float", 8192)).replace("float", "float4x4")
    assert _lowerer_helpers._check_generic_shared_memory_budget(unknown) == 0
    commented = (
        "kernel void probe() {\n    // threadgroup float shared_0[9999];\n"
        "    /*\n    threadgroup float shared_1[9999];\n    */\n}"
    )
    assert _lowerer_helpers._check_generic_shared_memory_budget(commented) == 0
