"""CPU-only execution of the real launcher's classification AST, no driver import."""
import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton_msl

DRIVER = Path(triton_msl.__file__).resolve().parent / 'backend/driver.py'


def classifier(*, consumer=builtins.all, standard=builtins.all):
    tree = ast.parse(DRIVER.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__call__' and any(isinstance(x, ast.Name) and x.id == 'all_mps' for x in ast.walk(n)))
    body = None
    for node in ast.walk(method):
        statements = getattr(node, 'body', [])
        if not isinstance(statements, list):
            continue
        for i, statement in enumerate(statements):
            if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_mps' for t in statement.targets):
                body = statements[i:i+2]
                assert isinstance(body[1], ast.If)
                break
        if body is not None:
            break
    assert body is not None
    function = ast.parse('def classify(tensors, _torch):\n pass').body[0]
    function.body = body + [ast.Return(value=ast.Name(id='all_mps', ctx=ast.Load()))]
    namespace = {'all': consumer, '_standard_all': standard}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(DRIVER), 'exec'), namespace)
    return namespace['classify']


class Tensor:
    def __init__(self, device): self.device = device


@pytest.mark.parametrize('values,expected', [([], False), ([Tensor('mps')], True), ([Tensor('mps:0'),Tensor('mps')],True), ([Tensor('cpu')],False), ([object()],False), ([Tensor('mps'),object()],False)])
def test_classification(values, expected):
    assert classifier()(values, SimpleNamespace(Tensor=Tensor)) is expected


def test_device_read_conversion_and_short_circuit_order():
    events = []
    class Device:
        def __init__(self, text): self.text=text
        def __str__(self): events.append(('str',self.text)); return self.text
    class Value(Tensor):
        @property
        def device(self): events.append(('get',self.text)); return Device(self.text)
        def __init__(self,text): self.text=text
    assert classifier()([Value('mps'),Value('cpu'),Value('unread')],SimpleNamespace(Tensor=Tensor)) is False
    assert events == [('get','mps'),('str','mps'),('get','cpu'),('str','cpu')]


def test_type_is_reselected_after_device_callback():
    framework=SimpleNamespace(Tensor=Tensor)
    class Device:
        def __str__(self): framework.Tensor=int; return 'mps'
    assert classifier()([Tensor(Device()),Tensor('mps')],framework) is False


@pytest.mark.parametrize('consume',[False,True])
def test_custom_consumer_stays_lazy_and_preserves_result(consume):
    events=[];sentinel=object()
    class Value(Tensor):
        @property
        def device(self): events.append('device');return 'mps'
        def __init__(self):pass
    def custom(values):
        events.append('all')
        if consume:assert next(values) is True
        return sentinel
    assert classifier(consumer=custom)([Value()],SimpleNamespace(Tensor=Tensor)) is sentinel
    assert events==(['all','device'] if consume else ['all'])


def test_empty_input_does_not_call_custom_consumer():
    def forbidden(values):raise AssertionError('consumer called')
    assert classifier(consumer=forbidden)([],SimpleNamespace(Tensor=Tensor)) is False


def test_device_exception_propagates_once():
    error=ValueError('device');events=[]
    class Value(Tensor):
        @property
        def device(self):events.append('device');raise error
        def __init__(self):pass
    with pytest.raises(ValueError) as caught:classifier()([Value()],SimpleNamespace(Tensor=Tensor))
    assert caught.value is error and events==['device']


def test_no_standard_consumer_does_not_treat_none_as_builtin():
    fn = classifier(consumer=None, standard=None)
    with pytest.raises(TypeError):fn([Tensor('mps')],SimpleNamespace(Tensor=Tensor))


def test_stop_iteration_keeps_generator_error_and_cause():
    error = StopIteration('device callback')
    class Device:
        def __str__(self): raise error
    with pytest.raises(RuntimeError, match='generator raised StopIteration') as caught:
        classifier()([Tensor(Device())], SimpleNamespace(Tensor=Tensor))
    assert caught.value.__cause__ is error


def test_standard_consumer_avoids_generator_frame():
    import sys
    fn = classifier()
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename == str(DRIVER):
            calls.append(frame.f_code.co_name)
    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        assert fn([Tensor('mps')], SimpleNamespace(Tensor=Tensor)) is True
    finally:
        sys.setprofile(previous)
    assert '<genexpr>' not in calls and 'classify' in calls
