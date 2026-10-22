"""Unwired CPU snapshot preparation for the proved saved-LSE program.

This is NOT a launch permit. The driver still refuses this descriptor. The
caller must first validate packed metadata and live execution identity. MPS
capture, framework compilation, publication and general numerical accuracy are
not qualified here. In particular, finite inputs need not have finite results.
"""
from dataclasses import dataclass
import hashlib
import math
import struct

from triton_msl.codegen._source_program.address import expected


class AdmissionRefusal(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise AdmissionRefusal(reason)


def _plain(value, depth=0, budget=None):
    """Snapshot a bounded metadata tree without invoking custom containers."""
    if budget is None:
        budget = [4096]
    budget[0] -= 1
    require(budget[0] >= 0, 'descriptor node budget')
    require(depth <= 32, 'descriptor depth')
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) in (list, tuple):
        require(len(value) <= 64, 'descriptor sequence length')
        return tuple(_plain(v, depth + 1, budget) for v in value)
    if type(value) is dict:
        require(len(value) <= 32 and all(type(k) is str for k in value), 'descriptor keys')
        return {k: _plain(v, depth + 1, budget) for k, v in value.items()}
    raise AdmissionRefusal('descriptor value type')


def _same(a, b):
    """Metadata equality must not confuse bool and int leaves."""
    if type(a) is not type(b):
        return False
    if type(a) is tuple:
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) is dict:
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _descriptor(value, source):
    d = _plain(value)
    require(type(d) is tuple and len(d) == 2 and d[0] == 'source_program_pending_v1', 'descriptor kind')
    d = d[1]
    require(type(d) is dict, 'descriptor payload')
    require(set(d) == {'schema', 'entry', 'source_sha256', 'branch', 'bindings', 'outputs',
                      'scalar_roles', 'threads', 'full_tile_domain', 'addresses',
                      'native_ieee_enum', 'max_imprecise_accumulation',
                      'private_bytes_declared', 'runtime_admission'}, 'descriptor fields')
    require(type(source) is str and type(d['entry']) is str and d['entry'].isascii()
            and d['entry'].isidentifier(), 'source or entry')
    require(hashlib.sha256(source.encode()).hexdigest() == d['source_sha256']
            and ('kernel void ' + d['entry'] + '(') in source, 'source binding')
    policy = {'schema': 1, 'bindings': 11, 'outputs': (0, 1), 'threads': (32, 1, 1),
              'native_ieee_enum': 2, 'max_imprecise_accumulation': 0}
    require(all(_same(d[k], v) for k, v in policy.items()), 'descriptor policy')
    require(d['branch'] in ('padded', 'masked'), 'descriptor branch')
    require(_same(d['full_tile_domain'], {'N': (32, 64), 'H': (1, 2), 'batch': (1, 2),
                                        'DIM': 32, 'BLOCK': 32}), 'descriptor domain')
    require(_same(d['scalar_roles'], ((7, 'f32', 'scale'), (8, 'f32', 'sentinel'),
                                     (9, 'i32', 'N'), (10, 'i32', 'H'))), 'scalar roles')
    require(type(d['private_bytes_declared']) is int and d['private_bytes_declared'] > 0
            and d['runtime_admission'] == 'blocked: numerical, storage and dispatch qualification pending',
            'pending policy')
    records = d['addresses']
    require(type(records) is tuple and len(records) == 7, 'address population')
    wanted = expected()
    seen = set()
    for record in records:
        require(type(record) is dict and set(record) == {'root', 'shape', 'element', 'links',
                                                        'bool_byte_cast'}, 'address fields')
        root = record['root']
        require(type(root) is int and root in wanted and root not in seen, 'address root')
        seen.add(root)
        shape, element, links = wanted[root]
        if root == 1 and d['branch'] == 'padded':
            shape = (32, 1)
        require(_same(record['shape'], shape) and record['element'] == element
                and _same(record['links'], links) and record['bool_byte_cast'] is (root == 6),
                'address program differs')
    return d


def _range(expr, n, h, batch):
    """Nonnegative i32 enclosure; reject wrap rather than silently widening it.

    Every node is checked before a link is sign-extended. The finite admitted
    domain has no wrapping nodes. This is not a general affine approximation.
    """
    op = expr[0]
    if op == 'const':
        result = (expr[1], expr[1])
    elif op == 'arg':
        value = {9: n, 10: h}[expr[1]]
        result = (value, value)
    elif op == 'pid':
        result = (0, (n // 32, n, batch * h)[expr[1]] - 1)
    elif op == 'lane':
        result = (0, 31)
    elif op == 'iv':
        result = (0, n - 32)
    else:
        a, b = _range(expr[1], n, h, batch), _range(expr[2], n, h, batch)
        if op == 'add':
            result = (a[0] + b[0], a[1] + b[1])
        elif op == 'mul':
            result = (a[0] * b[0], a[1] * b[1])
        else:
            require(op == 'divsi' and b[0] > 0, 'undefined address division')
            result = (a[0] // b[1], a[1] // b[0])
    require(0 <= result[0] <= result[1] < (1 << 31), 'address i32 wrap or negative range')
    return result


@dataclass(frozen=True)
class Allocation:
    root: int
    base: int
    capacity: int
    offset: int
    extent: int
    version: int


@dataclass(frozen=True)
class Snapshot:
    """Immutable bytes only; no mutable tensor owner or callable escapes."""
    source_sha256: str
    branch: str
    grid: tuple
    allocations: tuple
    bindings: tuple
    reach: tuple
    copied_bytes: int
    observed_f32_values: int


def _facts(torch, tensor, root, shape):
    require(type(tensor) is torch.Tensor, 'exact tensor required')
    require(tensor.device.type == 'cpu', 'MPS capture not qualified')
    require(tensor.layout is torch.strided and tensor.dtype is (torch.bool if root == 6 else torch.float32),
            'binding dtype or layout')
    require(not tensor.requires_grad and not tensor.is_conj() and not tensor.is_neg(), 'tensor view policy')
    require(tuple(tensor.shape) == shape and tensor.is_contiguous(), 'binding shape or strides')
    storage = tensor.untyped_storage()
    require(type(storage) is torch.UntypedStorage, 'exact storage required')
    base, capacity = storage.data_ptr(), storage.nbytes()
    offset = tensor.storage_offset() * tensor.element_size()
    extent = tensor.numel() * tensor.element_size()
    require(0 < base < (1 << 64) and 0 < capacity <= (1 << 26)
            and base + capacity <= (1 << 64) and 0 <= offset
            and offset + extent <= capacity and tensor.data_ptr() == base + offset,
            'allocation reach or identity')
    return Allocation(root, base, capacity, offset, extent, tensor._version), storage


def prepare_cpu(checked_descriptor, source, arguments, grid):
    """Observe finite immutable CPU bindings, without compiling or dispatching.

    No concurrent mutation contract: version/fact checks detect ordinary races,
    not arbitrary native writes. Later execution must use these exact bytes.
    """
    d = _descriptor(checked_descriptor, source)
    require(type(arguments) is tuple and len(arguments) == 11, 'argument population')
    require(type(grid) is tuple and len(grid) == 3 and all(type(x) is int for x in grid), 'grid type')
    scale, sentinel, n, h = arguments[7:]
    require(type(n) is int and n in (32, 64) and type(h) is int and h in (1, 2), 'scalar domain')
    require(grid[0] == n // 32 and grid[1] == n and grid[2] in (h, 2 * h), 'full grid required')
    batch = grid[2] // h
    scalar_bytes = []
    for value in (scale,):
        require(type(value) is float and math.isfinite(value), 'finite exact scalar required')
        try:
            packed = struct.pack('<f', value)
        except (OverflowError, struct.error) as exc:
            raise AdmissionRefusal('scalar outside finite f32') from exc
        require(math.isfinite(struct.unpack('<f', packed)[0]), 'scalar outside finite f32')
        scalar_bytes.append(packed)
    # The qualified recurrence uses -infinity, not a finite masking surrogate.
    # Check the Python value before packing so finite overflow cannot enter.
    require(type(sentinel) is float and sentinel == -math.inf,
            'sentinel must be exact negative infinity')
    sentinel_bytes = struct.pack('<f', sentinel)
    require(sentinel_bytes == bytes.fromhex('000080ff'), 'sentinel bits must be ff800000')
    scalar_bytes.append(sentinel_bytes)
    scalar_bytes.extend((struct.pack('<i', n), struct.pack('<i', h)))
    shapes = ((batch, h, n, n, 32), (batch, h, n, n),
              *((batch, h, n, n, 32),) * 3, (batch, h, n, n), (batch, n, n))
    import torch
    facts, owners = [], []
    for root, (tensor, shape) in enumerate(zip(arguments[:7], shapes)):
        fact, owner = _facts(torch, tensor, root, shape)
        facts.append(fact)
        owners.append(owner)
    for i, a in enumerate(facts):
        for b in facts[:i]:
            require(a.base + a.capacity <= b.base or b.base + b.capacity <= a.base,
                    'overlapping backing allocations')
    reach = []
    for record in d['addresses']:
        ranges = [_range(expr, n, h, batch) for expr in record['links']]
        low, high = sum(r[0] for r in ranges), sum(r[1] for r in ranges)
        root = record['root']
        width = 1 if root == 6 else 4
        fact = facts[root]
        require(0 <= low * width and (high + 1) * width <= fact.extent
                and fact.offset + (high + 1) * width <= fact.capacity, 'proved address escapes allocation')
        reach.append((root, low, high, width))
    bindings = []
    observed = 2
    for root, tensor in enumerate(arguments[:7]):
        # numpy shares a view; tobytes is the actual immutable copy. Never retain
        # that array or reuse caller memory after this observation.
        payload = tensor.detach().view(torch.uint8).numpy().tobytes()
        require(len(payload) == facts[root].extent, 'snapshot extent')
        if root in (2, 3, 4, 5):
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', payload)), 'nonfinite input')
            observed += len(payload) // 4
        elif root == 6:
            require(all(v in (0, 1) for v in payload), 'noncanonical bool')
        bindings.append(payload)
    for root, (tensor, shape) in enumerate(zip(arguments[:7], shapes)):
        after, owner = _facts(torch, tensor, root, shape)
        require(after == facts[root] and owner is owners[root], 'allocation changed during capture')
    require(_descriptor(checked_descriptor, source) == d, 'descriptor changed during capture')
    bindings.extend(scalar_bytes)
    return Snapshot(d['source_sha256'], d['branch'], grid, tuple(facts), tuple(bindings),
                    tuple(reach), sum(map(len, bindings)), observed)


def require_execution_qualification(*args, **kwargs):
    raise AdmissionRefusal('snapshot is not execution authority; framework compilation remains unqualified')
