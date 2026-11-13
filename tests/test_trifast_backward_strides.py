"""Derived-stride regression through the real reporter frontend, CPU only."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from triton_msl.codegen import generic_lowerer


@pytest.fixture
def strides():
    tree = ast.parse(Path(generic_lowerer.__file__).read_text())
    detect = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == '_detect_biased_fa_backward')
    functions = [n for n in detect.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'uniform_stride', 'split_stride'}]
    args = {'n': SimpleNamespace(index=3, is_ptr=False, type_str='i32')}
    ops = {}
    ns = {'arg_by_id': args, 'op_by_id': ops, 'C1': 'c1'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<stride regression>', 'exec'), ns)
    return ns, args, ops


@pytest.mark.parametrize('value', [0, 1, 32, 4294967295, -1, 4294967296, True, 1.0])
def test_constant_stride_range(strides, value):
    ns, _, ops = strides
    ops['c'] = SimpleNamespace(op='arith.constant', type_str='i32', attrs={'value': value}, operand_ids=[])
    expected = ('c1' if value == 1 else ('literal', value)) if type(value) is int and 0 <= value <= 4294967295 else None
    assert ns['uniform_stride']('c') == expected


@pytest.mark.parametrize('kind', ['tt.get_program_id', 'tt.make_range', 'tt.load', 'arith.addi', 'arith.divsi'])
def test_coordinate_is_not_uniform_stride(strides, kind):
    ns, _, ops = strides
    ops['x'] = SimpleNamespace(op=kind, type_str='i32', attrs={}, operand_ids=['n'])
    assert ns['uniform_stride']('x') is None


def test_stride_types_cycles_and_commuted_product(strides):
    ns, args, ops = strides
    for name, pointer, type_ in [('p', True, '!tt.ptr<i32>'), ('f', False, 'f32'), ('wide', False, 'i64')]:
        args[name] = SimpleNamespace(index=4, is_ptr=pointer, type_str=type_)
        assert ns['uniform_stride'](name) is None
    ops['c'] = SimpleNamespace(op='arith.constant', type_str='i32', attrs={'value': 32}, operand_ids=[])
    ops['product'] = SimpleNamespace(op='arith.muli', type_str='i32', attrs={}, operand_ids=['n', 'c'])
    assert ns['uniform_stride']('product') == ('mul', 3, ('literal', 32))
    ops['coordinate'] = SimpleNamespace(op='tt.make_range', type_str='i32', attrs={}, operand_ids=[])
    for pair in [('product', 'coordinate'), ('coordinate', 'product')]:
        term = SimpleNamespace(op='arith.muli', type_str='i32', attrs={}, operand_ids=pair)
        assert ns['split_stride'](term) == ('coordinate', ('mul', 3, ('literal', 32)))
    ops['cycle'] = SimpleNamespace(op='tt.splat', type_str='i32', attrs={}, operand_ids=['cycle'])
    assert ns['uniform_stride']('cycle') is None


@pytest.mark.parametrize('name', ['_bwd_q', '_bwd_kv', '_bwd_b'])
def test_reporter_derived_stride_graph_emits(name):
    from test_fa_bwd_routing import _build_lowerer
    path = Path(__file__).parent / 'fixtures/trifast_reporter/kernels.py'
    spec = importlib.util.spec_from_file_location('_reporter_stride_fixture', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, name)
    constants = dict(DIM=32, BLOCK_J=32, BLOCK_K=32, CLOSEST_N=1024, PADDED=False)
    signature = {n: ('*i1' if n in ('mask_ptr', 'm_ptr') else '*fp32' if n.endswith('_ptr')
                     else 'fp32' if n in ('sm_scale', 'neg_inf') else 'i32')
                 for n in fn.arg_names if n not in constants}
    assert name in _build_lowerer(fn, signature, constants).lower()
