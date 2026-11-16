"""CPU native/Python binder differentials, admission and callback ordering."""

import ast
import builtins
import gc
from pathlib import Path
import random
import struct
from types import MappingProxyType
import weakref

import pytest
from tests.native_helper_policy import require_selected_native
import torch
import triton.backends
from triton_msl.backend import _launch_signature as s


@pytest.fixture
def native():
    require_selected_native(s._binder_native, "binder")
    assert s._binder_native is not None, "native-specific controls require the built image"
    return s._binder_native


def outcome(fn):
    try:
        return ("value", fn())
    except Exception as exc:
        return ("error", type(exc), str(exc))


def equivalent(left, right):
    assert left[0] == right[0]
    if left[0] == "error":
        assert left[1:] == right[1:]
    else:
        a, b = left[1], right[1]
        assert all(x is y for x, y in zip(a[0], b[0])) and len(a[0]) == len(b[0])
        assert a[1:] == b[1:]


def check(args, names, sig, plan):
    equivalent(
        outcome(lambda: s.bind_arguments(args, names, sig, _plan=plan)),
        outcome(lambda: s.bind_arguments_with_plan(args, names, sig, plan)),
    )


@pytest.mark.parametrize("value", [0, 1, -1, -(1 << 31), (1 << 31) - 1, True, False])
def test_native_i32_exact_width_and_fresh_outputs(native, value):
    names = ["p", "n", "BLOCK"]
    sig = dict(p="*fp32", n="i32", BLOCK="constexpr")
    plan = s.make_binding_plan(names, sig)
    pointer = torch.empty(1, device="cpu")
    args = (pointer, value, object())
    a = native.bind_flat(args, names, sig, plan)
    b = native.bind_flat(args, names, sig, plan)
    assert a is not NotImplemented and b is not NotImplemented
    assert a[0][0] is pointer and a[0][1] is value
    assert a[1:] == (["*fp32", "i32"], [0, 1], [None, struct.pack("<i", value)])
    assert all(x is not y for x, y in zip(a, b))
    assert a[3][1] is not b[3][1]
    a[0].clear()
    assert b[0]
    check(args, names, sig, plan)


def test_random_flat_and_original_positions(native):
    rng = random.Random(770)
    p = torch.empty(2, device="cpu")
    for _ in range(500):
        names = [f"a{i}" for i in range(rng.randrange(1, 45))]
        sig = {n: rng.choice(["i32", "i32", "*fp32", "constexpr"]) for n in names}
        sig[names[-1]] = "i32"
        plan = s.make_binding_plan(names, sig)
        args = tuple(
            rng.randrange(-(1 << 31), 1 << 31)
            if sig[n] == "i32"
            else rng.choice([p, None])
            if sig[n] == "*fp32"
            else object()
            for n in names
        )
        assert native.bind_flat(args, names, sig, plan) is not NotImplemented
        check(args, names, sig, plan)


@pytest.mark.parametrize("value", [1 << 31, -(1 << 31) - 1, 1 << 100, 1.0, None, "1", (1,)])
def test_unadmitted_i32_uses_exact_python_refusal(native, value):
    names = ["v"]
    sig = {"v": "i32"}
    plan = s.make_binding_plan(names, sig)
    assert native.bind_flat((value,), names, sig, plan) is NotImplemented
    check((value,), names, sig, plan)


@pytest.mark.parametrize(
    "ty,value",
    [("i8", 128), ("u32", 1 << 31), ("i64", 1 << 40), ("fp16", 1.25), ("fp32", -0.0), ("fp64", 1.0), ("bf16", 1.25)],
)
def test_changed_live_width_falls_back(native, ty, value):
    names = ["v"]
    sig = {"v": "i32"}
    plan = s.make_binding_plan(names, sig)
    sig["v"] = ty
    assert native.bind_flat((value,), names, sig, plan) is NotImplemented
    check((value,), names, sig, plan)


def test_exact_tensor_property_mutation_no_native_prefix(native, monkeypatch):
    pointer = torch.empty(1, device="cpu")
    events = []
    names = ["p", "n"]
    sig = {"p": "*fp32", "n": "i32"}
    plan = s.make_binding_plan(names, sig)
    assert native.bind_flat((pointer, 128), names, sig, plan) is not NotImplemented

    def getter(self):
        events.append("get")
        sig["n"] = "i8"
        return lambda: 0

    monkeypatch.setattr(torch.Tensor, "data_ptr", property(getter))
    assert native.bind_flat((pointer, 128), names, sig, plan) is NotImplemented and events == []
    left = outcome(lambda: s.bind_arguments((pointer, 128), names, sig, _plan=plan))
    assert events == ["get"]
    events.clear()
    sig["n"] = "i32"
    right = outcome(lambda: s.bind_arguments_with_plan((pointer, 128), names, sig, plan))
    assert events == ["get"]
    equivalent(left, right)


def test_custom_scalar_and_callback_order_not_replayed(native):
    events = []

    class Pointer:
        @property
        def data_ptr(self):
            events.append("pointer")
            return lambda: 0

    class Integer(int):
        def __eq__(self, other):
            events.append(("eq", other))
            return super().__eq__(other)

    names = ["p", "v"]
    sig = dict(p="*fp32", v="i1")
    plan = s.make_binding_plan(["i"], {"i": "i32"})
    args = (Pointer(), Integer(1))
    assert native.bind_flat(args, names, sig, plan) is NotImplemented and not events
    left = outcome(lambda: s.bind_arguments(args, names, sig, _plan=plan))
    before = list(events)
    events.clear()
    right = outcome(lambda: s.bind_arguments_with_plan(args, names, sig, plan))
    assert events == before and events.count("pointer") == 1
    equivalent(left, right)


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("int", tuple),
        ("tuple", int),
        ("type", lambda value: object),
        ("isinstance", lambda *args: False),
        ("getattr", lambda *args: None),
        ("callable", lambda *args: False),
        ("zip", lambda *args: iter(())),
        ("enumerate", lambda *args: iter(())),
    ],
)
def test_live_builtin_bindings_preserved(native, monkeypatch, name, replacement):
    names = ["p", "v"]
    sig = dict(p="*fp32", v="i32")
    plan = s.make_binding_plan(names, sig)
    args = (torch.empty(1, device="cpu"), 1)
    monkeypatch.setattr(s, name, replacement, raising=False)
    assert native.bind_flat(args, names, sig, plan) is NotImplemented
    check(args, names, sig, plan)


def test_missing_or_custom_map_key_declines_without_equality(native):
    events = []

    class Key(str):
        def __eq__(self, other):
            events.append(other)
            return super().__eq__(other)

        __hash__ = str.__hash__

    names = ["v"]
    sig = {Key("v"): "i32"}
    plan = s.make_binding_plan(names, sig)
    assert native.bind_flat((1,), names, sig, plan) is NotImplemented and not events
    check((1,), names, sig, plan)
    sig = {}
    check((1,), names, sig, plan)


def test_mutated_packer_map_and_forged_plan_preserved(native):
    names = ["v"]
    sig = {"v": "i32"}
    plan = s.make_binding_plan(names, sig)
    events = []
    wrong = s._BindingPlan(MappingProxyType({"i32": (False, lambda v: events.append(v) or b"custom")}), plan.native)
    assert native.bind_flat((1,), names, sig, wrong) is NotImplemented and not events
    check((1,), names, sig, wrong)
    assert events == [1, 1]
    raw = {"i32": (False, struct.Struct("<i").pack)}
    proxy, cap = native.make_parts(raw)
    owned = s._BindingPlan(proxy, cap)
    raw["i32"] = (False, lambda v: events.append(v) or b"changed")
    assert native.bind_flat((2,), names, sig, owned) is NotImplemented
    check((2,), names, sig, owned)


def test_plan_descriptor_changes_and_binder_replacement(native, monkeypatch):
    names = ["v"]
    sig = {"v": "i32"}
    plan = s.make_binding_plan(names, sig)
    calls = []

    def replacement(*args, **kwargs):
        calls.append("binder")
        return ("custom",)

    monkeypatch.setattr(s, "bind_arguments", replacement)
    assert native.bind_flat((1,), names, sig, plan) is NotImplemented and not calls
    assert s.bind_arguments_with_plan((1,), names, sig, plan) == ("custom",) and calls == ["binder"]


def test_pointer_instance_override_and_subclass_getattribute(native):
    names = ["p", "v"]
    sig = {"p": "*fp32", "v": "i32"}
    plan = s.make_binding_plan(names, sig)
    p = torch.empty(1, device="cpu")
    p.data_ptr = lambda: 0
    assert native.bind_flat((p, 1), names, sig, plan) is NotImplemented
    check((p, 1), names, sig, plan)

    class Custom(torch.Tensor):
        def __getattribute__(self, name):
            return super().__getattribute__(name)

    p = torch.empty(1, device="cpu").as_subclass(Custom)
    assert native.bind_flat((p, 1), names, sig, plan) is NotImplemented
    check((p, 1), names, sig, plan)


def test_null_pointer_and_strong_leaf_ownership(native):
    names = ["p", "n"]
    sig = {"p": "*fp32", "n": "i32"}
    plan = s.make_binding_plan(names, sig)
    assert native.bind_flat((None, 1), names, sig, plan)[0] == [None, 1]
    p = torch.empty(1, device="cpu")
    ref = weakref.ref(p)
    result = native.bind_flat((p, 1), names, sig, plan)
    del p
    gc.collect()
    assert ref() is result[0][0]
    del result
    gc.collect()
    assert ref() is None


def test_tuple_pointer_refusal_precedes_c_method_descriptor(native):
    class TuplePointer(tuple):
        __slots__ = ()
        data_ptr = tuple.count

    names = ["p", "n"]
    sig = {"p": "*fp32", "n": "i32"}
    plan = s.make_binding_plan(names, sig)
    args = (TuplePointer((1,)), 1)
    assert native.bind_flat(args, names, sig, plan) is NotImplemented
    check(args, names, sig, plan)


def test_native_path_counters(native):
    names = ["v"]
    sig = {"v": "i32"}
    plan = s.make_binding_plan(names, sig)
    before = native.statistics()
    assert native.bind_flat((1,), names, sig, plan) is not NotImplemented
    assert native.bind_flat((1.0,), names, sig, plan) is NotImplemented
    with pytest.raises(TypeError):
        native.bind_flat()
    after = native.statistics()
    assert {key: after[key] - before[key] for key in before} == dict(hits=1, misses=1, errors=1)


@pytest.mark.parametrize("format", ["<q", ">i", "<b", "<f", "<ii", "0s"])
def test_live_struct_reinitialization_preserves_payload_and_refusal(native, format):
    names = ["p", "n"]
    sig = {"p": "*fp32", "n": "i32"}
    plan = s.make_binding_plan(names, sig)
    args = (None, 1)
    packer = plan.packers["i32"][1]
    assert native.bind_flat(args, names, sig, plan) is not NotImplemented
    packer.__self__.__init__(format)
    assert native.bind_flat(args, names, sig, plan) is NotImplemented
    check(args, names, sig, plan)
    packer.__self__.__init__("<i")
    assert native.bind_flat(args, names, sig, plan) is not NotImplemented
    check(args, names, sig, plan)
    with pytest.raises(struct.error):
        packer.__self__.__init__("invalid")
    check(args, names, sig, plan)


@pytest.mark.parametrize("mutation", ["signature", "struct", "pointer"])
def test_gc_threshold_mutation_after_native_return_is_live_next_call(native, mutation):
    # CPython 3.14 schedules allocation-triggered cyclic GC for the evaluator.
    # This witnesses a post-return callback, NOT a callback inside C allocation.
    old_threshold = gc.get_threshold()
    was_enabled = gc.isenabled()
    witnessed = 0
    try:
        for delta in range(1, 9):
            gc.collect()
            gc.disable()
            names = ["p", "n"]
            sig = {"p": "*fp32", "n": "i32"}
            pointer = torch.empty(1, device="cpu")
            args = (pointer, 1)
            plan = s.make_binding_plan(names, sig)
            events = []
            initial_hits = native.statistics()["hits"]

            def callback(phase, info):
                if phase == "start":
                    events.append(native.statistics()["hits"] - initial_hits)
                    if mutation == "signature":
                        sig["n"] = "i64"
                    elif mutation == "struct":
                        plan.packers["i32"][1].__self__.__init__("<q")
                    else:
                        pointer.data_ptr = None

            gc.callbacks.append(callback)
            try:
                count = gc.get_count()[0]
                gc.set_threshold(count + delta, 100000, 100000)
                gc.enable()
                result = native.bind_flat(args, names, sig, plan)
                gc.disable()
            finally:
                gc.callbacks.remove(callback)
            if events == [1]:
                witnessed += 1
                assert result is not NotImplemented
                assert result[1:] == (["*fp32", "i32"], [0, 1], [None, struct.pack("<i", 1)])
                assert native.bind_flat(args, names, sig, plan) is NotImplemented
                check(args, names, sig, plan)
        assert witnessed, "no post-native-return GC mutation was witnessed"
    finally:
        gc.set_threshold(*old_threshold)
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


def test_nested_live_names_and_large_arity_fallback(native):
    plan = s.make_binding_plan(["v"], {"v": "i32"})
    for args, names, sig in [
        (((None, 7),), ["v"], {"v": ("*fp32", "i32")}),
        ((2, 3), ["a", "b"], {"a": "i32", "b": "i32"}),
        (tuple(range(129)), [str(i) for i in range(129)], {str(i): "i32" for i in range(129)}),
    ]:
        names.reverse()
        check(args, names, sig, plan)


def test_build_is_explicit_and_unsupported_remains_python():
    source = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "binder_extension_enabled")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    select = scope[node.name]
    assert select("1", "cpython", (3, 14), "darwin", "arm64", False)
    assert not select("0", "cpython", (3, 14), "darwin", "arm64", False)
    for version in [(3, 13), (3, 15)]:
        assert not select("auto", "cpython", version, "darwin", "arm64", False)
        with pytest.raises(RuntimeError):
            select("1", "cpython", version, "darwin", "arm64", False)


def python_plan_shim():
    scope = {}
    exec(
        "def bind_arguments_with_plan(args, names, signature, plan):\n    return bind_arguments(args, names, signature, _plan=plan)",
        vars(s),
        scope,
    )
    return scope["bind_arguments_with_plan"]


def test_c_dispatch_preserves_python_call_protocol(native):
    oracle = python_plan_shim()
    names = ["n"]
    sig = {"n": "i32"}
    plan = s.make_binding_plan(names, sig)
    calls = [
        (((1,), names, sig, plan), {}),
        ((), dict(args=(1,), names=names, signature=sig, plan=plan)),
        (((1,), names), dict(signature=sig, plan=plan)),
        ((), {}),
        (((1,), names, sig, plan, 5), {}),
        (((1,), names, sig, plan), dict(plan=plan)),
        (((1,), names, sig), dict(unexpected=1)),
    ]
    for args, kwargs in calls:
        equivalent(
            outcome(lambda: oracle(*args, **kwargs)), outcome(lambda: s.bind_arguments_with_plan(*args, **kwargs))
        )


def test_initial_unsupported_plan_remains_live_python(native):
    names = ["n", "f"]
    sig = {"n": "i32", "f": "fp32"}
    plan = s.make_binding_plan(names, sig)
    assert plan.native is None
    check((1, 1.5), names, sig, plan)
    sig["f"] = "i32"
    assert native.bind_flat((1, 2), names, sig, plan) is NotImplemented
    check((1, 2), names, sig, plan)


def test_foreign_global_lookup_runs_once_at_python_fallback(native):
    oracle = python_plan_shim()
    names = ["n"]
    sig = {"n": "i32"}
    plan = s.make_binding_plan(names, sig)
    events = []

    class Key(str):
        def __eq__(self, other):
            events.append("lookup")
            sig["n"] = "i8"
            return super().__eq__(other)

        __hash__ = str.__hash__

    key = Key("bind_arguments")
    saved = vars(s).pop("bind_arguments")
    try:
        vars(s)[key] = saved
        left = outcome(lambda: oracle((128,), names, sig, plan))
        before = list(events)
        events.clear()
        sig["n"] = "i32"
        right = outcome(lambda: s.bind_arguments_with_plan((128,), names, sig, plan))
        assert events == before == ["lookup"]
        equivalent(left, right)
    finally:
        vars(s).pop(key)
        vars(s)["bind_arguments"] = saved


@pytest.mark.parametrize("mode", ["none", "deleted", "substitute"])
def test_live_native_helper_opt_out_keeps_python_behavior(native, monkeypatch, mode):
    names = ["n"]
    sig = {"n": "i32"}
    plan = s.make_binding_plan(names, sig)
    if mode == "deleted":
        monkeypatch.delattr(s, "_binder_native")
    else:
        monkeypatch.setattr(s, "_binder_native", None if mode == "none" else object())
    before = native.statistics()
    check((1,), names, sig, plan)
    after = native.statistics()
    assert after["hits"] == before["hits"] and after["misses"] == before["misses"] + 1


@pytest.mark.parametrize("mode", ["false", "binder", "builtins", "raise"])
def test_foreign_fallback_lookup_side_effects_are_not_replayed(native, mode):
    oracle = python_plan_shim()
    names = ["n"]
    sig = {"n": "i64"}
    plan = s.make_binding_plan(names, sig)
    events = []
    global_saved = vars(s)["bind_arguments"]
    sentinel = object()
    builtin_saved = vars(builtins).get("bind_arguments", sentinel)

    def replacement(*args, **kwargs):
        return ("replacement",)

    class Key(str):
        def __eq__(self, other):
            events.append(other is oracle.__code__.co_names[0])
            if mode == "raise":
                raise LookupError("foreign key marker")
            if mode == "binder":
                vars(s).pop(key)
                vars(s)["bind_arguments"] = replacement
            elif mode == "builtins":
                vars(builtins)["bind_arguments"] = replacement
            return False

        __hash__ = str.__hash__

    key = Key("bind_arguments")

    def clear_lookup():
        for existing in list(vars(s)):
            if existing is key or (type(existing) is str and existing == "bind_arguments"):
                vars(s).pop(existing)
        vars(builtins).pop("bind_arguments", None)

    try:
        clear_lookup()
        vars(s)[key] = global_saved
        left = outcome(lambda: oracle((1,), names, sig, plan))
        left_events = list(events)
        clear_lookup()
        vars(s)[key] = global_saved
        events.clear()
        right = outcome(lambda: s.bind_arguments_with_plan((1,), names, sig, plan))
        assert left == right and events == left_events == [True]
    finally:
        clear_lookup()
        vars(s)["bind_arguments"] = global_saved
        if builtin_saved is not sentinel:
            vars(builtins)["bind_arguments"] = builtin_saved


def test_missing_fallback_nameerror_attributes(native, monkeypatch):
    oracle = python_plan_shim()
    names = ["n"]
    sig = {"n": "i64"}
    plan = s.make_binding_plan(names, sig)
    monkeypatch.delattr(s, "bind_arguments")
    monkeypatch.delattr(builtins, "bind_arguments", raising=False)
    errors = []
    for fn in (oracle, s.bind_arguments_with_plan):
        with pytest.raises(NameError) as caught:
            fn((1,), names, sig, plan)
        errors.append((caught.value.args, caught.value.name, str(caught.value)))
    assert errors[0] == errors[1]
