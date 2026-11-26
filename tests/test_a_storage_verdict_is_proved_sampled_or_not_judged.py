"""Every launch the access-plan storage proof passes carries its verdict: proved, sampled or not judged.

The proof is exact only where that is cheap: an access that does not follow the program id, an
address that is a lane part plus a program part under a mask that does not follow the program,
and any access of a launch within 262,144 address instances (every program is read). Everything
past that is SAMPLED at the first two and last two programs of each grid axis and trips of each
loop: a cell found outside there refuses the launch, as it did when the proof first shipped,
and a launch with nothing found there is dispatched and labelled sampled, never proved.

A walk of every program up to 4,194,304 instances replaced the sampling for one commit. It cost
milliseconds on every first sight of a grid and scalar set, and it stopped reading an access
whose one program alone is past the budget (a K loop of 300 trips over a 32 by 32 tile): a
short operand the sampling had refused was dispatched. Both are undone here.

The verdict belongs to the launch: a remembered launch reports the verdict of its first sight,
another launch being proved does not clear it, and it is read from the launcher
(``compiled.run.storage_verdict``), per specialisation.

Launch decisions run with both submission methods replaced; nothing is submitted.
"""

import pytest
import torch
import triton
import triton.language as tl

from tests.test_a_loop_free_dot_proves_its_storage import OUTSIDE, _forget_tripped_lease, _install_tripwire
from tests.test_the_access_plan_storage_proof_reads_every_program import B16, _rows, _source, _split
from triton_msl.errors import MetalNonRecoverableError

requires_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal GPU needed")
pytestmark = requires_gpu


@triton.jit
def _masked_loop(A, B, C, M, N, K, SAM, SAK, SBK, SBN, SCM, SCN, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # A masked K loop on the K-loop template with no storage plan and no fast replacement.
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(
            A + rm[:, None] * SAM + (k + rk)[None, :] * SAK,
            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + (k + rk)[:, None] * SBK + rn[None, :] * SBN,
            mask=((k + rk)[:, None] < K) & (rn[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(C + rm[:, None] * SCM + rn[None, :] * SCN, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _launch(monkeypatch, kernel, grid, tensors, scalars, constants):
    """("dispatched" | "refused", the launcher's verdict) with the submission methods replaced."""
    handle = []
    with monkeypatch.context() as patch:
        hits = _install_tripwire(patch)
        try:
            handle.append(kernel[grid](*tensors, *scalars, **constants))
            outcome = "dispatched"
        except MetalNonRecoverableError as refusal:
            outcome = "refused" if OUTSIDE in str(refusal) else str(refusal)
        except Exception:
            if not hits:
                raise
            outcome = "dispatched"
        finally:
            _forget_tripped_lease(hits)
    return outcome


def _verdict(kernel, scalars, constants):
    """The storage verdict of the launcher this specialisation last launched through."""
    for cache in kernel.device_caches.values():
        for compiled in cache[0].values():
            found = getattr(compiled.run, "storage_verdict", None)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("depth", [8192, 9600])
def test_a_long_k_loop_with_a_short_operand_is_refused(depth, compile_shader, monkeypatch):
    """One program, 32 by 32 tiles, 256 or 300 trips: 262,144 or 307,200 address instances for A alone."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    scalars = [32, 32, depth, 2 * depth, 2, 64, 2, 64, 2]
    sizes = [31 * 2 * depth + 2 * (depth - 1) + 1, 64 * (depth - 1) + 62 + 1, 31 * 64 + 62 + 1]
    constants = dict(BM=32, BN=32, BK=32)
    whole = [torch.zeros(size, device="mps") for size in sizes]
    assert _launch(monkeypatch, _masked_loop, (1, 1), whole, scalars, constants) == "dispatched"
    for short in (sizes[0] - 1, sizes[0] // 2):
        tensors = [torch.zeros(short, device="mps")] + whole[1:]
        assert _launch(monkeypatch, _masked_loop, (1, 1), tensors, scalars, constants) == "refused"


@pytest.mark.parametrize("compile_shader", ["1", "0"])
@pytest.mark.parametrize("kernel,programs,n", [(_rows, 1100, 16), (_split, 1102, 1760)], ids=["over_launched", "split"])
def test_past_the_budget_a_launch_is_sampled_and_within_it_refused(kernel, programs, n, compile_shader, monkeypatch):
    """1,100 programs of 256 lanes are 281,600 instances: past the budget. Ten programs are within it."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    scalars = [160, n, 16, 16, 16, 16 if kernel is _rows else 1760]
    sizes = [span[1] + 1 for span in _source(kernel, programs, *scalars)]
    whole = [torch.zeros(size, device="mps") for size in sizes]
    assert _launch(monkeypatch, kernel, (programs,), whole, scalars, B16) == "dispatched"
    assert _verdict(kernel, scalars, B16)[0] == "sampled"
    # Past the budget only the ends of the grid are read: a tensor short in the middle is not seen.
    middle = [torch.zeros(sizes[0] - 1, device="mps")] + whole[1:]
    assert _launch(monkeypatch, kernel, (programs,), middle, scalars, B16) == "dispatched"
    assert _verdict(kernel, scalars, B16)[0] == "sampled"
    # What the ends show is refused: a tensor that does not hold the second program's rows.
    first = [torch.zeros(16 * 16 + 8, device="mps")] + whole[1:]
    assert _launch(monkeypatch, kernel, (programs,), first, scalars, B16) == "refused"
    # Within the budget every program is read, and the launch is proved or refused.
    few = 10 if kernel is _rows else 12
    scalars = [160, 16, 16, 16, 16, 16 if kernel is _rows else 1760]
    sizes = [span[1] + 1 for span in _source(kernel, few, *scalars)]
    whole = [torch.zeros(size, device="mps") for size in sizes]
    assert _launch(monkeypatch, kernel, (few,), whole, scalars, B16) == "dispatched"
    assert _verdict(kernel, scalars, B16) == ("proved", ())
    for role in (0, 2):
        tensors = list(whole)
        tensors[role] = torch.zeros(sizes[role] - 1, device="mps")
        assert _launch(monkeypatch, kernel, (few,), tensors, scalars, B16) == "refused"


def test_a_remembered_launch_keeps_the_verdict_of_its_first_sight(monkeypatch):
    """X sampled, Y proved, X again from the memo: still sampled, and Y still proved."""
    scalars = [160, 16, 16, 16, 16, 16]
    big = [torch.zeros(span[1] + 1, device="mps") for span in _source(_rows, 20000, *scalars)]
    small = [torch.zeros(span[1] + 1, device="mps") for span in _source(_rows, 3, *scalars)]
    seen = []
    for tensors, programs in ((big, 20000), (small, 3), (big, 20000), (small, 3), (big, 20000)):
        assert _launch(monkeypatch, _rows, (programs,), tensors, scalars, B16) == "dispatched"
        seen.append(_verdict(_rows, scalars, B16)[0])
    assert seen == ["sampled", "proved", "sampled", "proved", "sampled"]
    kinds = {why for _, _, why in _verdict(_rows, scalars, B16)[1]}
    assert kinds == {"sampled"}


@pytest.mark.parametrize("compile_shader", ["1", "0"])
def test_a_refused_launch_reports_refused_not_the_launch_before_it(compile_shader, monkeypatch):
    """The verdict is of the last launch: after a refusal it names the access that left its storage."""
    monkeypatch.setenv("TRITON_MSL_COMPILE_SHADER", compile_shader)
    scalars = [160, 16, 16, 16, 16, 16]
    sizes = [span[1] + 1 for span in _source(_rows, 10, *scalars)]
    whole = [torch.zeros(size, device="mps") for size in sizes]
    assert _launch(monkeypatch, _rows, (10,), whole, scalars, B16) == "dispatched"
    assert _verdict(_rows, scalars, B16) == ("proved", ())
    short = [torch.zeros(sizes[0] - 1, device="mps")] + whole[1:]
    assert _launch(monkeypatch, _rows, (10,), short, scalars, B16) == "refused"
    assert _verdict(_rows, scalars, B16) == ("refused", (("L", 0, 0, sizes[0] - 1, 0, sizes[0] - 1),))
    assert _launch(monkeypatch, _rows, (10,), whole, scalars, B16) == "dispatched"
    assert _verdict(_rows, scalars, B16) == ("proved", ())


def test_a_template_plan_launch_has_no_access_plan_verdict(monkeypatch):
    """A kernel whose storage its template plan judges: ``storage_verdict`` stays None, passed or refused."""
    from tests.test_a_loop_free_dot_proves_its_storage import REFUSED, _loop_free

    constants = dict(BM=16, BN=16, BK=16)
    scalars = [16, 16, 16, 16, 16]
    whole = [torch.zeros(256, device="mps") for _ in range(3)]
    handle = []
    for tensors, outcome in ((whole, "dispatched"), ([torch.zeros(255, device="mps")] + whole[1:], REFUSED)):
        with monkeypatch.context() as patch:
            hits = _install_tripwire(patch)
            try:
                handle.append(_loop_free[(1, 1)](*tensors, *scalars, **constants))
                seen = "dispatched"
            except MetalNonRecoverableError as refusal:
                seen = REFUSED if REFUSED in str(refusal) else str(refusal)
            except Exception:
                if not hits:
                    raise
                seen = "dispatched"
            finally:
                _forget_tripped_lease(hits)
        assert seen == outcome
        launchers = [compiled.run for cache in _loop_free.device_caches.values() for compiled in cache[0].values()]
        assert launchers and all(launcher.storage_verdict is None for launcher in launchers)
