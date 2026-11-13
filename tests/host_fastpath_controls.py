"""Cold-child host fastpath controls; explicitly collected by the parent test.

No driver import: extract the installed scalar function verbatim and compare
native admission with its complete Python body on independently made fixtures.
"""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import struct
import sys
import types
import pytest

@pytest.fixture(scope='module')
def scalar():
    assert 'triton_msl.backend.driver' not in sys.modules
    import triton_msl
    from triton_msl.backend import _launch_signature as binding
    root = Path(triton_msl.__file__).resolve().parent
    path = root / 'backend/driver.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_COMPILE_SHADER_SAFE_SCALAR_SIGS' for t in node.targets)) or
             (isinstance(node, ast.FunctionDef) and node.name == '_compile_shader_scalars_ok')]
    assert len(nodes) == 2
    arms = []
    for name in ('complete_python', 'selected'):
        ns = {'__name__': name, '__builtins__': vars(builtins), '_SCALAR_GATE': None}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        arms.append(ns)
    native = binding._binder_native
    if native is not None:
        assert native.configure_scalar(arms[1]['_compile_shader_scalars_ok']) is True
        arms[1]['_SCALAR_GATE'] = native.scalar_gate
    print(json.dumps({'installed_root': str(root), 'driver_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'binder': None if native is None else str(Path(native.__file__).resolve())}))
    return arms, native, binding

class Owner:
    pass

class Pointer(list):
    data_ptr = list.copy

def case(kind='plain'):
    events = []
    owner = Owner()
    value = int('257')
    args = [Pointer(), value]
    owner.arg_names = ['p', 'n']
    owner.signature = {'p': '*fp32', 'n': 'i32'}
    owner.constexpr_indices = set()
    checked = (args.copy(), ['*fp32', owner.signature['n']], [0, 1], [None, struct.pack('<i', value)])
    if kind == 'range': args[1] = 1 << 31
    elif kind == 'value': args[1] = int('258')
    elif kind == 'declaration': owner.signature['n'] = 'i8'; args[1] = 1000
    elif kind == 'position': checked[2][1] = 0
    elif kind == 'payload': checked[3][1] = bytearray(b'proof')
    elif kind == 'short': checked[1].clear()
    elif kind == 'constants': owner.constexpr_indices = {1}
    elif kind == 'property':
        class Watched:
            @property
            def arg_names(self): events.append('names'); return ['p', 'n']
        changed = Watched(); changed.signature = owner.signature; changed.constexpr_indices = set(); owner = changed
    elif kind == 'mapping':
        class Mapping(dict):
            def get(self, key): events.append(('get', key)); return super().get(key)
        owner.signature = Mapping(owner.signature)
    elif kind == 'foreign':
        class Key:
            def __hash__(self): return hash('n')
            def __eq__(self, other): events.append(('eq', other)); return False
        owner.signature = {Key(): 'unused', **owner.signature}
    elif kind == 'pointer_callback':
        class P:
            @property
            def data_ptr(self): events.append('pointer'); owner.signature['n'] = 'fp16'; return lambda: 0
        args[0] = P(); checked[0][0] = args[0]
    return owner, args, checked, events

@pytest.mark.parametrize('kind', ['plain', 'range', 'value', 'declaration', 'position', 'payload', 'short', 'constants', 'property', 'mapping', 'foreign', 'pointer_callback'])
def test_scalar_differential(scalar, kind):
    arms, native, _ = scalar
    outcomes = []
    before = None if native is None else native.scalar_statistics()
    for ns in arms:
        owner, args, checked, events = case(kind)
        outcomes.append((ns['_compile_shader_scalars_ok'](owner, args, checked=checked), events))
    assert outcomes[0] == outcomes[1]
    if kind == 'plain' and native is not None:
        assert native.scalar_statistics()['hits'] == before['hits'] + 1

@pytest.mark.parametrize('kind', ['value', 'position', 'payload'])
def test_changed_proof_reaches_observable_repacking(scalar, monkeypatch, kind):
    arms, _, binding = scalar
    for ns in arms:
        events = []
        def pack(*args): events.append(args); raise RuntimeError('independent repack witness')
        monkeypatch.setattr(binding, 'scalar_bytes', pack)
        owner, args, checked, _ = case(kind)
        assert ns['_compile_shader_scalars_ok'](owner, args, checked=checked) is False
        assert len(events) == 1

@pytest.mark.parametrize('mode', ['trace', 'profile'])
def test_observers_decline_and_recover(scalar, mode):
    arms, native, _ = scalar
    if native is None: pytest.skip('native route absent')
    fn = arms[1]['_compile_shader_scalars_ok']
    owner, args, checked, _ = case()
    before = native.scalar_statistics(); events = []
    def observe(*args): events.append(args[1]); return observe
    setter = getattr(sys, 'set' + mode)
    assert getattr(sys, 'get' + mode)() is None
    setter(observe)
    try: assert fn(owner, args, checked=checked)
    finally: setter(None)
    after = native.scalar_statistics()
    assert after['hits'] == before['hits'] and after['misses'] == before['misses'] + 1
    assert events
    assert fn(owner, args, checked=checked)
    assert native.scalar_statistics()['hits'] == after['hits'] + 1

@pytest.mark.parametrize('scope', ['local', 'global'])
def test_monitoring_declines_and_recovers(scalar, scope):
    arms, native, _ = scalar
    if native is None: pytest.skip('native route absent')
    fn = arms[1]['_compile_shader_scalars_ok']; owner, args, checked, _ = case()
    tool = next((i for i in range(6) if sys.monitoring.get_tool(i) is None), None)
    if tool is None: pytest.skip('monitoring tools occupied')
    events = []; event = sys.monitoring.events.LINE
    sys.monitoring.use_tool_id(tool, 'host-stack-independent-review')
    sys.monitoring.register_callback(tool, event, lambda *args: events.append(args))
    before = native.scalar_statistics()
    try:
        if scope == 'local': sys.monitoring.set_local_events(tool, fn.__code__, event)
        else: sys.monitoring.set_events(tool, event)
        assert fn(owner, args, checked=checked)
    finally:
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.set_local_events(tool, fn.__code__, 0)
        sys.monitoring.register_callback(tool, event, None)
        sys.monitoring.free_tool_id(tool)
    after = native.scalar_statistics()
    assert events and after['hits'] == before['hits'] and after['misses'] == before['misses'] + 1
    assert fn(owner, args, checked=checked)
    assert native.scalar_statistics()['hits'] == after['hits'] + 1

def test_binder_uses_shared_census():
    from triton_msl.backend import _launch_signature as binding
    from triton_msl.backend import _cache_contract as cache
    native, validation = binding._binder_native, cache._validation_native
    if native is None or validation is None: pytest.skip('optional native helpers absent')
    names, signature = ['p', 'n'], {'p': '*fp32', 'n': 'i32'}
    plan = binding.make_binding_plan(names, signature)
    validation.key_cache_reset()
    before = validation.key_cache_info()
    for _ in range(100):
        assert native.bind_flat((None, 1), names, signature, plan) is not NotImplemented
    after = validation.key_cache_info()
    assert after['hits'] - before['hits'] >= 390
    assert after['watch_installs'] - before['watch_installs'] >= 4


def test_packed_cached_keys_do_not_escape_into_foreign_comparisons():
    from triton_msl.backend import _launch_contract as contract
    if contract.PACKED_COPY_IMPLEMENTATION != 'native': pytest.skip('optional packed helper absent')
    namespace = dict(vars(contract))
    fn = types.FunctionType(contract._checked_packed_copy.__code__, namespace)
    namespace['_checked_packed_copy'] = fn
    namespace['_checked_packed_copy_python'] = types.FunctionType(contract._checked_packed_copy_python.__code__, namespace)
    plan = (list, ((int, 1),))
    assert fn([1], plan) == [1]  # exercise Unicode-only cached-name route first
    name = '_checked_packed_copy'; events = []; sentinel = RuntimeError('foreign lookup')
    class Key(str):
        __hash__ = str.__hash__
        def __eq__(self, other): events.append(other); raise sentinel
    del namespace[name]
    namespace[Key(name)] = None
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught: fn([1], plan)
        assert caught.value is sentinel
    assert len(events) == 2 and events[0] is not events[1]
