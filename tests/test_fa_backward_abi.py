"""CPU pre-submit ABI checks; fake device metadata never initializes Metal."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton.backends
from triton_msl import fa_backward as fa


class MetadataMPS(torch.Tensor):
    @property
    def device(self):
        return getattr(self, "_reported_device", torch.device("mps", 0))


class NoPreparation(MetadataMPS):
    def reshape(self, *args, **kwargs):
        raise AssertionError("reshape reached before refusal")

    def float(self, *args, **kwargs):
        raise AssertionError("conversion reached before refusal")


def tensor(shape=(2, 16, 64), dtype=torch.float32, device="mps:0"):
    value = torch.empty(shape, dtype=dtype, device="cpu").as_subclass(NoPreparation)
    value._reported_device = torch.device(device)
    return value


@pytest.fixture
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("autograd/eager/runtime work reached for invalid fixed ABI")

    monkeypatch.setattr(fa._FlashAttentionFn, "apply", forbidden)
    monkeypatch.setattr(fa, "_attention_output", forbidden)
    monkeypatch.setattr(fa, "_logsumexp", forbidden)
    monkeypatch.setattr(fa, "_dispatch_backward", forbidden)


@pytest.mark.parametrize("position", range(3), ids=["q", "k", "v"])
@pytest.mark.parametrize("value", [None, 1, (2, 16, 64)])
def test_non_tensor_rejected_before_preparation(forbid_execution, position, value):
    args = [tensor() for _ in range(3)]
    args[position] = value
    with pytest.raises(TypeError, match="Tensor"):
        fa.flash_attention(*args)


@pytest.mark.parametrize("shape", [(), (64,), (2, 16, 32), (2, 17, 64), (2, 0, 64), (0, 16, 64)])
def test_invalid_shared_shape_rejected_before_preparation(forbid_execution, shape):
    with pytest.raises(ValueError):
        fa.flash_attention(*(tensor(shape) for _ in range(3)))


@pytest.mark.parametrize("position", [1, 2], ids=["k", "v"])
@pytest.mark.parametrize("shape", [(64,), (1, 32, 64), (1, 16, 64), (2, 32, 32), (1, 2, 16, 64)])
def test_peer_shape_is_checked_before_equal_numel_can_reshape(forbid_execution, position, shape):
    args = [tensor() for _ in range(3)]
    args[position] = tensor(shape)
    with pytest.raises(ValueError, match="rank|shape"):
        fa.flash_attention(*args)


def test_leading_dimensions_cannot_be_silently_regrouped(forbid_execution):
    with pytest.raises(ValueError, match="shape"):
        fa.flash_attention(tensor((1, 2, 16, 64)), tensor((2, 1, 16, 64)), tensor((1, 2, 16, 64)))


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.complex64, torch.bool])
def test_unsupported_dtype_rejected_before_preparation(forbid_execution, dtype):
    with pytest.raises(TypeError, match="dtype"):
        fa.flash_attention(*(tensor(dtype=dtype) for _ in range(3)))


@pytest.mark.parametrize("position", [1, 2], ids=["k", "v"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_peer_dtype_must_match_before_q_driven_cast(forbid_execution, position, dtype):
    args = [tensor() for _ in range(3)]
    args[position] = tensor(dtype=dtype)
    with pytest.raises(TypeError, match="dtype"):
        fa.flash_attention(*args)


@pytest.mark.parametrize("position", range(3), ids=["q", "k", "v"])
@pytest.mark.parametrize("device", ["cpu", "meta", "mps:1"])
def test_same_mps_device_is_required_before_preparation(forbid_execution, position, device):
    args = [tensor() for _ in range(3)]
    args[position] = tensor(device=device)
    with pytest.raises(ValueError, match="device|MPS|mps"):
        fa.flash_attention(*args)


def test_all_cpu_inputs_are_refused_before_eager_work(forbid_execution):
    with pytest.raises(ValueError, match="MPS|mps"):
        fa.flash_attention(*(tensor(device="cpu") for _ in range(3)))


def test_non_strided_layout_is_refused_before_preparation(forbid_execution):
    class SparseMetadata(NoPreparation):
        @property
        def layout(self):
            return torch.sparse_coo

    args = [tensor() for _ in range(3)]
    args[1] = torch.empty(2, 16, 64).as_subclass(SparseMetadata)
    with pytest.raises(ValueError, match="strided|layout"):
        fa.flash_attention(*args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 64), (2, 16, 64), (1, 2, 16, 64)])
@pytest.mark.parametrize("layout", ["contiguous", "slice", "transpose", "expanded"])
@pytest.mark.parametrize("causal", [False, True])
def test_supported_shape_dtype_and_views_keep_cpu_oracle_and_gradients(monkeypatch, dtype, shape, layout, causal):
    generator = torch.Generator(device="cpu").manual_seed(747)
    leaves = []
    inputs = []
    for _ in range(3):
        if layout == "slice":
            source = torch.randn((*shape[:-1], shape[-1] * 2), generator=generator, dtype=dtype) * 0.25
            source = source[..., ::2]
        elif layout == "transpose":
            source = torch.randn((*shape[:-2], shape[-1], shape[-2]), generator=generator, dtype=dtype) * 0.25
            source = source.transpose(-1, -2)
        elif layout == "expanded":
            source = torch.randn((*shape[:-2], 1, shape[-1]), generator=generator, dtype=dtype) * 0.25
            source = source.expand(shape)
        else:
            source = torch.randn(shape, generator=generator, dtype=dtype) * 0.25
        value = source.detach().requires_grad_()
        leaves.append(value)
        inputs.append(value.as_subclass(MetadataMPS))
    assert all(t.is_contiguous() == (layout == "contiguous") for t in leaves)
    seen = []

    def cpu_apply(q, k, v, scale, causal):
        seen.append((tuple(q.shape), q.dtype, k.dtype, v.dtype, scale, causal))
        return F.scaled_dot_product_attention(
            *(x.as_subclass(torch.Tensor) for x in (q, k, v)), scale=scale, is_causal=causal
        )

    monkeypatch.setattr(fa._FlashAttentionFn, "apply", cpu_apply)
    references = [x.detach().clone().requires_grad_() for x in leaves]
    scale = 0.2
    # Keep the CPU oracle on the same documented math backend: rank-4 CPU
    # SDPA otherwise selects a different fused implementation and can round
    # half gradients one ulp differently from the rank-3 wrapper boundary.
    with sdpa_kernel(SDPBackend.MATH):
        actual = fa.flash_attention(*inputs, scale=scale, causal=causal)
        expected = F.scaled_dot_product_attention(*(x.float() for x in references), scale=scale, is_causal=causal).to(
            dtype
        )
        gradient = torch.randn(shape, generator=generator, dtype=dtype)
        actual.backward(gradient)
        expected.backward(gradient)
    assert actual.shape == shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for value, reference in zip(leaves, references):
        assert value.grad.dtype == dtype
        torch.testing.assert_close(value.grad, reference.grad, rtol=2e-5, atol=2e-6)
    assert seen == [
        (
            (int(torch.tensor(shape[:-2]).prod()) if shape[:-2] else 1, shape[-2], 64),
            torch.float32,
            torch.float32,
            torch.float32,
            scale,
            causal,
        )
    ]


def test_same_numel_broadcast_case_refuses_instead_of_regrouping_heads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched peers entered autograd")

    monkeypatch.setattr(fa._FlashAttentionFn, "apply", forbidden)
    q = torch.zeros(2, 16, 64)
    k = torch.zeros(1, 32, 64)
    v = torch.cat((torch.zeros(1, 16, 64), torch.ones(1, 16, 64)), dim=1)
    reference = F.scaled_dot_product_attention(q, k, v)
    # Zero scores assign equal weight to16 zero and16 one value rows.
    assert torch.all(reference == 0.5)
    with pytest.raises(ValueError, match="shape"):
        fa.flash_attention(*(x.as_subclass(MetadataMPS) for x in (q, k, v)))
