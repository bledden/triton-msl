"""Real CPU allocations only. A snapshot never permits dispatch."""
import copy
from dataclasses import FrozenInstanceError
import itertools
from pathlib import Path
import struct

import pytest
import torch
from triton._C.libtriton import ir
from triton_msl.backend.compiler import MetalOptions
from triton_msl.backend import _source_program_admission as admission
from triton_msl.codegen import msl_emitter


@pytest.fixture(scope='module', params=['masked', 'padded'])
def product(request):
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(Path(__file__).parent / 'fixtures/source_program' /
                                      (request.param + '.ttgir')), context)
    metadata = {}
    source = msl_emitter.emit_msl(module, metadata, MetalOptions())
    return metadata['batched_dot_bounds'], source


def tensors(n=32, h=1, batch=1, offset=0):
    shapes = [(batch, h, n, n, 32), (batch, h, n, n)]
    shapes += [(batch, h, n, n, 32)] * 3
    shapes += [(batch, h, n, n), (batch, n, n)]
    result = []
    for root, shape in enumerate(shapes):
        size = 1
        for dim in shape:
            size *= dim
        backing = torch.zeros(size + offset + 3, dtype=torch.bool if root == 6 else torch.float32)
        tensor = backing[offset:offset + size].reshape(shape)
        result.append(tensor)
    return tuple(result) + (0.125, -float('inf'), n, h)


def capture(product, arguments=None, grid=(1, 32, 1)):
    descriptor, source = product
    return admission.prepare_cpu(descriptor, source, tensors() if arguments is None else arguments, grid)


@pytest.mark.parametrize('n,h,batch', itertools.product((32, 64), (1, 2), (1, 2)))
def test_real_allocations_and_independent_address_extrema(product, n, h, batch):
    arguments = tensors(n, h, batch, offset=2)
    snapshot = capture(product, arguments, (n // 32, n, batch * h))
    assert len(snapshot.bindings) == 11
    assert all(type(p) is bytes for p in snapshot.bindings)
    assert snapshot.copied_bytes == sum(t.numel() * t.element_size() for t in arguments[:7]) + 16
    assert snapshot.observed_f32_values == sum(t.numel() for t in arguments[2:6]) + 2
    assert snapshot.bindings[7:] == (struct.pack('<f', 0.125), bytes.fromhex('000080ff'),
                                     struct.pack('<i', n), struct.pack('<i', h))
    # Independent source formulas at every corner. On this positive full-tile
    # domain every factor is monotone; div(z,H) is monotone for positive H.
    extrema = {root: [] for root in range(7)}
    for x, y, z, r, c, iv in itertools.product((0, n // 32 - 1), (0, n - 1),
                                               (0, batch * h - 1), (0, 31), (0, 31), (0, n - 32)):
        j = 32 * x + r
        base = z * n * n * 32 + y * n * 32
        values = (base + j * 32 + c, z * n * n + y * n + j,
                  base + j * 32 + c, base + r + (iv + c) * 32,
                  base + (iv + r) * 32 + c, z * n * n + j * n + iv + c,
                  (z // h) * n * n + y * n + iv + r)
        for root, value in enumerate(values):
            extrema[root].append(value)
    for root, low, high, width in snapshot.reach:
        assert (low, high) == (min(extrema[root]), max(extrema[root]))
        assert high + 1 == arguments[root].numel()
        assert snapshot.allocations[root].offset == 2 * width


@pytest.mark.parametrize('root', range(7))
def test_short_view_refuses(product, root):
    args = list(tensors())
    args[root] = args[root].flatten()[:-1]
    with pytest.raises(admission.AdmissionRefusal, match='shape'):
        capture(product, tuple(args))


@pytest.mark.parametrize('pair', [(0, 2), (2, 3), (3, 4), (1, 5)])
def test_shared_owner_refuses_even_disjoint_views(product, pair):
    args = list(tensors())
    a, b = pair
    size = args[a].numel()
    owner = torch.zeros(size * 2, dtype=torch.float32)
    args[a] = owner[:size].reshape(args[a].shape)
    args[b] = owner[size:].reshape(args[b].shape)
    with pytest.raises(admission.AdmissionRefusal, match='overlapping'):
        capture(product, tuple(args))


@pytest.mark.parametrize('root', [2, 3, 4, 5])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_inputs_refuse(product, root, value):
    args = tensors()
    args[root].flatten()[-1] = value
    with pytest.raises(admission.AdmissionRefusal, match='nonfinite input'):
        capture(product, args)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1e100, True, 1])
def test_scale_refuses_nonfinite_overflow_and_nonexact_types(product, value):
    args = list(tensors())
    args[7] = value
    with pytest.raises(admission.AdmissionRefusal, match='scalar'):
        capture(product, tuple(args))


def test_qualified_negative_infinity_sentinel_is_retained_exactly(product):
    snapshot = capture(product)
    assert snapshot.bindings[8] == bytes.fromhex('000080ff')
    assert struct.unpack('<I', snapshot.bindings[8])[0] == 0xff800000


@pytest.mark.parametrize('value', [-10000.0, 0.0, -0.0, 1.0, -1e100,
                                  float('inf'), float('nan'), True, 1])
def test_unqualified_sentinel_refuses(product, value):
    args = list(tensors())
    args[8] = value
    with pytest.raises(admission.AdmissionRefusal, match='sentinel'):
        capture(product, tuple(args))


def test_output_nan_and_first_block_mask_are_not_repaired(product):
    args = tensors()
    args[0].fill_(float('nan'))
    args[1].fill_(float('nan'))
    snapshot = capture(product, args)
    assert snapshot.bindings[0] == args[0].view(torch.uint8).numpy().tobytes()
    assert snapshot.bindings[6] == bytes(args[6].numel())


def test_snapshot_survives_caller_mutation_and_resize_without_reusing_memory(product):
    args = tensors()
    snapshot = capture(product, args)
    before = snapshot.bindings
    args[2].fill_(float('nan'))
    args[3].resize_(1)
    assert snapshot.bindings == before
    assert all(type(p) is bytes for p in snapshot.bindings)
    with pytest.raises(FrozenInstanceError):
        snapshot.branch = 'other'
    with pytest.raises(admission.AdmissionRefusal, match='not execution authority'):
        admission.require_execution_qualification(snapshot)


@pytest.mark.parametrize('damage', ['stride', 'dtype', 'grad', 'subclass', 'meta', 'boolbits'])
def test_unsupported_tensor_forms_refuse(product, damage):
    args = list(tensors())
    match = {'stride': 'shape or strides', 'dtype': 'dtype', 'grad': 'view policy',
             'subclass': 'exact tensor', 'meta': 'capture not qualified', 'boolbits': 'noncanonical bool'}[damage]
    if damage == 'stride':
        args[2] = args[2].transpose(-1, -2)
    elif damage == 'dtype':
        args[2] = args[2].half()
    elif damage == 'grad':
        args[2].requires_grad_()
    elif damage == 'subclass':
        class Custom(torch.Tensor):
            pass
        args[2] = args[2].as_subclass(Custom)
    elif damage == 'meta':
        args[2] = torch.empty(args[2].shape, device='meta')
    else:
        args[6].view(torch.uint8).flatten()[0] = 2
    with pytest.raises(admission.AdmissionRefusal, match=match):
        capture(product, tuple(args))


@pytest.mark.parametrize('grid', [(1, 16, 1), (2, 32, 1), (1, 32, 3), (True, 32, 1), [1, 32, 1]])
def test_partial_or_malformed_grid_refuses(product, grid):
    with pytest.raises(admission.AdmissionRefusal, match='grid'):
        capture(product, grid=grid)


@pytest.mark.parametrize('damage', ['links', 'cast', 'root', 'shape', 'source', 'domain', 'field', 'custom'])
def test_descriptor_mutations_refuse(product, damage):
    descriptor, source = copy.deepcopy(product)
    d = descriptor[1]
    if damage == 'links':
        d['addresses'][0]['links'] = (('const', 0),)
    elif damage == 'cast':
        d['addresses'][0]['bool_byte_cast'] = True
    elif damage == 'root':
        d['addresses'][0]['root'] = d['addresses'][1]['root']
    elif damage == 'shape':
        d['addresses'][0]['shape'] = (16, 32)
    elif damage == 'source':
        source += '\n'
    elif damage == 'domain':
        d['full_tile_domain']['N'] = [16, 32]
    elif damage == 'field':
        d['trusted'] = True
    else:
        class Custom(dict):
            def items(self):
                pytest.fail('custom callback ran')
        descriptor[1] = Custom(d)
    with pytest.raises(admission.AdmissionRefusal):
        capture((descriptor, source))


def test_intervening_tensor_change_is_refused(product, monkeypatch):
    args = tensors()
    original = admission._facts
    calls = 0
    def mutate(*a):
        nonlocal calls
        calls += 1
        if calls == 8:
            args[2].add_(1)
        return original(*a)
    monkeypatch.setattr(admission, '_facts', mutate)
    with pytest.raises(admission.AdmissionRefusal, match='changed during capture'):
        capture(product, args)


def test_intervening_descriptor_change_is_refused(product, monkeypatch):
    descriptor, source = copy.deepcopy(product)
    original = admission._facts
    def mutate(*a):
        result = original(*a)
        descriptor[1]['entry'] = 'changed'
        return result
    monkeypatch.setattr(admission, '_facts', mutate)
    with pytest.raises(admission.AdmissionRefusal):
        capture((descriptor, source))


def test_i32_proof_does_not_hide_wrap_or_negative_link():
    for expr in [('const', 1 << 31), ('mul', ('const', 1 << 30), ('const', 2)),
                 ('add', ('const', (1 << 31) - 1), ('const', 1))]:
        with pytest.raises(admission.AdmissionRefusal, match='i32'):
            admission._range(expr, 32, 1, 1)


@pytest.mark.parametrize('field,value', [('schema', True), ('max_imprecise_accumulation', False),
                                        ('outputs', (False, True)), ('threads', (32, True, True))])
def test_bool_is_not_an_integer_metadata_leaf(product, field, value):
    descriptor, source = copy.deepcopy(product)
    descriptor[1][field] = value
    with pytest.raises(admission.AdmissionRefusal, match='descriptor policy'):
        capture((descriptor, source))


def test_short_actual_storage_refuses_before_copy(product):
    args = tensors()
    args[2].untyped_storage().resize_(4)
    with pytest.raises(admission.AdmissionRefusal, match='allocation reach'):
        capture(product, args)


def test_snapshot_does_not_claim_finite_intermediates(product):
    args = tensors()
    args[2].fill_(3.0e38)
    args[3].fill_(3.0e38)
    snapshot = capture(product, args)
    assert snapshot.observed_f32_values > 0
    with pytest.raises(admission.AdmissionRefusal, match='not execution authority'):
        admission.require_execution_qualification(snapshot)


def test_descriptor_tree_is_bounded():
    value = [0] * 64
    value = [value] * 64
    with pytest.raises(admission.AdmissionRefusal, match='node budget'):
        admission._plain([value, value])
