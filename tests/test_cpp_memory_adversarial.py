"""Adversarial CPU-only regressions for the C++ masked-load repair.

Native LLVM IR is produced by the loaded ``triton_msl._triton_msl_cpp`` helper
(``run_to_llvm`` on hand-written TTGIR, or ``MetalBackend.make_llir`` on real
Triton TTGIR). Nothing is dispatched. Load/phi counts pin the current pre-optimization lowering,
not a general prohibition on legal load reuse or store forwarding. Every test that inspects the mask branch
also records whether the ordinary opt-in router would admit the same text, so
"reachable through the router" and "reachable by direct native call" stay
distinguishable in the JUnit properties.

Expected outcome per helper is stated in each test's docstring:
  BITES      fails on the parent helper (unconditional load + select), passes repaired
  RETENTION  passes on both helpers (current pre-optimization lowering shape)
  MODIFIERS  volatile is preserved; cache/eviction hints are documented no-ops
"""

import re

import pytest
import triton  # noqa: F401  (backend discovery before triton_msl imports)

cpp = pytest.importorskip("triton_msl._triton_msl_cpp")

# --------------------------------------------------------------------------- IR helpers


def function_body(ir):
    return ir[ir.index("define") : ir.index("!llvm.module.flags")]


def blocks_of(body):
    """{label: text}; the entry block is labelled 'entry'."""
    parts = re.split(r"^([\w.]+):[^\n]*\n", body, flags=re.M)
    out = {"entry": parts[0]}
    for label, text in zip(parts[1::2], parts[2::2]):
        out[label] = text
    return out


def cond_branches(body):
    """[(cond, true_label, false_label)] in textual order."""
    return re.findall(r"br i1 (%[\w.]+), label %([\w.]+), label %([\w.]+)", body)


def preds(blocks):
    p = {k: set() for k in blocks}
    for label, text in blocks.items():
        for m in re.finditer(r"br i1 %[\w.]+, label %([\w.]+), label %([\w.]+)", text):
            p[m.group(1)].add(label)
            p[m.group(2)].add(label)
        for m in re.finditer(r"br label %([\w.]+)", text):
            p[m.group(1)].add(label)
    return p


def mask_values(body):
    """SSA names of i1 values that are lane-mask predicates: an `icmp` on i32
    whose operands transitively depend on the lane id `%lid`, plus anything
    derived from such predicates by i1 logic. Loop-induction compares that do
    not depend on `%lid` are excluded, so a loop condition cannot pose as a mask."""
    defs = {}
    for m in re.finditer(r"^\s*(%[\w.]+) = ([^\n]*)$", body, re.M):
        defs[m.group(1)] = m.group(2)

    memo = {}

    def depends_on_lid(name):
        if name == "%lid":
            return True
        if name in memo:
            return memo[name]
        memo[name] = False
        text = defs.get(name, "")
        if text.startswith("phi "):
            return False  # loop-carried values are not lane predicates
        memo[name] = any(depends_on_lid(op) for op in re.findall(r"%[\w.]+", text))
        return memo[name]

    names = {n for n, text in defs.items() if text.startswith("icmp ") and depends_on_lid(n)}
    changed = True
    while changed:
        changed = False
        for n, text in defs.items():
            m = re.match(r"(?:xor|and|or) i1 (%[\w.]+), (%[\w.]+|true|false)", text)
            if m and n not in names and (m.group(1) in names or m.group(2) in names):
                names.add(n)
                changed = True
    return names


def every_float_load_is_mask_guarded(body):
    """Each `load float` block is the TRUE target of a conditional branch on a
    mask-derived i1, and has that branch block as its only predecessor. This
    is the structural property that suppresses inactive reads regardless of
    how much other control flow (scf.if, loops) surrounds the load."""
    blocks = blocks_of(body)
    p = preds(blocks)
    masks = mask_values(body)
    loads = [l for l, t in blocks.items() if re.search(r"\bload (?:volatile )?float\b", t)]
    assert loads, body
    for label in loads:
        assert len(p[label]) == 1, (label, p[label], body)
        (pred,) = p[label]
        branches = [(c, t, f) for c, t, f in cond_branches(blocks[pred]) if t == label]
        assert branches, (label, blocks[pred], body)
        cond, _t, f = branches[-1]
        assert cond in masks, (label, cond, masks, body)
        assert not re.search(r"\bload (?:volatile )?float\b", blocks[f]), (label, f, body)
    return loads


def lower(text):
    return function_body(cpp.run_to_llvm(text))


def router_admits(text):
    from triton_msl.backend.compiler import MetalBackend
    from triton_msl.backend.cpp_families import cpp_refusal_reason

    return cpp_refusal_reason(text) is None and not MetalBackend._has_complex_ops(text)


# --------------------------------------------------------------------------- TTGIR corpus

HEAD = """module {
  tt.func public @k(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %extent: i32, %flag: i32) {
    %p = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %z = arith.constant dense<7.0> : tensor<128xf32>
    %n = tt.splat %extent : i32 -> tensor<128xi32>
    %i = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %m = arith.cmpi slt, %i, %n : tensor<128xi32>
"""
TAIL = """    tt.return
  }
}"""


def kernel(body):
    return HEAD + body + TAIL


NESTED_IF = kernel("""    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %flag, %c0 : i32
    %x = scf.if %cond -> (tensor<128xf32>) {
      %v = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
      scf.yield %v : tensor<128xf32>
    } else {
      scf.yield %z : tensor<128xf32>
    }
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

NESTED_IF_NO_OTHER = kernel("""    %c0 = arith.constant 0 : i32
    %cond = arith.cmpi sgt, %flag, %c0 : i32
    %x = scf.if %cond -> (tensor<128xf32>) {
      %v = tt.load %p, %m : tensor<128x!tt.ptr<f32>>
      scf.yield %v : tensor<128xf32>
    } else {
      scf.yield %z : tensor<128xf32>
    }
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

RUNTIME_LOOP = kernel("""    %lo = arith.constant 0 : i32
    %st = arith.constant 1 : i32
    %x = scf.for %kk = %lo to %flag step %st iter_args(%c = %z) -> (tensor<128xf32>) : i32 {
      %v = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
      %s = arith.addf %c, %v : tensor<128xf32>
      scf.yield %s : tensor<128xf32>
    }
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

ZERO_TRIP_CONST_LOOP = kernel("""    %lo = arith.constant 0 : i32
    %hi = arith.constant 0 : i32
    %st = arith.constant 1 : i32
    %x = scf.for %kk = %lo to %hi step %st iter_args(%c = %z) -> (tensor<128xf32>) : i32 {
      %v = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
      scf.yield %v : tensor<128xf32>
    }
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

DEPENDENT_OTHER = kernel("""    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    %two = arith.constant dense<2.0> : tensor<128xf32>
    %o = arith.mulf %x, %two : tensor<128xf32>
    %nm = arith.cmpi sge, %i, %n : tensor<128xi32>
    %y = tt.load %p, %nm, %o : tensor<128x!tt.ptr<f32>>
    tt.store %q, %y : tensor<128x!tt.ptr<f32>>
""")

LOAD_THEN_STORE_DIFFERENT_MASKS = kernel("""    %c1 = arith.constant dense<1> : tensor<128xi32>
    %n1 = arith.subi %n, %c1 : tensor<128xi32>
    %sm = arith.cmpi slt, %i, %n1 : tensor<128xi32>
    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x, %sm : tensor<128x!tt.ptr<f32>>
""")

NEGATED_MASK = kernel("""    %t = arith.constant dense<true> : tensor<128xi1>
    %nm = arith.xori %m, %t : tensor<128xi1>
    %x = tt.load %p, %nm, %z : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

CONJOINED_MASK = kernel("""    %c4 = arith.constant dense<4> : tensor<128xi32>
    %m2 = arith.cmpi sge, %i, %c4 : tensor<128xi32>
    %both = arith.andi %m, %m2 : tensor<128xi1>
    %x = tt.load %p, %both, %z : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

STORE_BETWEEN_SAME_PTR = kernel("""    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    tt.store %p, %z, %m : tensor<128x!tt.ptr<f32>>
    %y = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    %s = arith.addf %x, %y : tensor<128xf32>
    tt.store %q, %s : tensor<128x!tt.ptr<f32>>
""")

TWO_PTRS_SAME_MASK = kernel("""    %c1 = arith.constant dense<1> : tensor<128xi32>
    %p1 = tt.addptr %p, %c1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    %y = tt.load %p1, %m, %z : tensor<128x!tt.ptr<f32>>
    %s = arith.addf %x, %y : tensor<128xf32>
    tt.store %q, %s : tensor<128x!tt.ptr<f32>>
""")

SAME_PTR_UNEQUAL_MASKS = kernel("""    %c1 = arith.constant dense<1> : tensor<128xi32>
    %n1 = arith.addi %n, %c1 : tensor<128xi32>
    %m1 = arith.cmpi slt, %i, %n1 : tensor<128xi32>
    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    %y = tt.load %p, %m1, %z : tensor<128x!tt.ptr<f32>>
    %s = arith.addf %x, %y : tensor<128xf32>
    tt.store %q, %s : tensor<128x!tt.ptr<f32>>
""")

SAME_PTR_SAME_MASK_DIFFERENT_OTHER = kernel("""    %z2 = arith.constant dense<9.0> : tensor<128xf32>
    %x = tt.load %p, %m, %z : tensor<128x!tt.ptr<f32>>
    %y = tt.load %p, %m, %z2 : tensor<128x!tt.ptr<f32>>
    %s = arith.addf %x, %y : tensor<128xf32>
    tt.store %q, %s : tensor<128x!tt.ptr<f32>>
""")

VOLATILE_LOAD = kernel("""    %x = tt.load %p, %m, %z {isVolatile = true} : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x : tensor<128x!tt.ptr<f32>>
""")

CACHE_MODIFIED_LOAD = kernel("""    %x = tt.load %p, %m, %z cacheModifier = cg evictionPolicy = evict_last : tensor<128x!tt.ptr<f32>>
    tt.store %q, %x cacheModifier = wt : tensor<128x!tt.ptr<f32>>
""")


def _record(record_property, text, body):
    record_property("router_admits", str(router_admits(text)))
    record_property("float_loads", str(len(re.findall(r"\bload (?:volatile )?float\b", body))))
    record_property("cond_branches", str(len(cond_branches(body))))
    record_property("selects", str(len(re.findall(r"\bselect i1\b", body))))


# --------------------------------------------------------------------------- 1. inactive loads under control flow


@pytest.mark.parametrize(
    "text,label",
    [(NESTED_IF, "with_other"), (NESTED_IF_NO_OTHER, "without_other")],
    ids=["with_other", "without_other"],
)
def test_masked_load_inside_scf_if_is_guarded_by_mask_not_only_by_condition(record_property, text, label):
    """BITES. The load must sit on the mask's true edge *inside* the scf.if branch;
    the parent emits the load unconditionally inside the branch and selects after."""
    body = lower(text)
    _record(record_property, text, body)
    loads = every_float_load_is_mask_guarded(body)
    assert len(loads) == 1, body
    assert "select i1" not in body, body
    blocks = blocks_of(body)
    bypass = [f for _c, _t, f in cond_branches(body) if _t == loads[0]][0]
    assert ("7.000000e+00" if label == "with_other" else "undef") in blocks[bypass] or "undef" in body, body


def test_masked_load_inside_runtime_loop_is_guarded_on_every_iteration(record_property):
    """BITES. Inside a runtime-trip scf.for the load block must still be the mask
    branch's true target (not merely the loop body), and no select remains."""
    body = lower(RUNTIME_LOOP)
    _record(record_property, RUNTIME_LOOP, body)
    every_float_load_is_mask_guarded(body)
    assert "select i1" not in body, body
    assert "fadd float" in body, body


def test_zero_trip_constant_loop_body_load_is_still_mask_guarded(record_property):
    """BITES. A constant zero-trip loop is not folded by the native passes today
    (retention of that fact is recorded, not required); whatever survives, a load in
    its body must be mask-guarded so a future bound change cannot expose it."""
    body = lower(ZERO_TRIP_CONST_LOOP)
    _record(record_property, ZERO_TRIP_CONST_LOOP, body)
    record_property("loop_folded", str("load float" not in body))
    if "load float" in body:
        every_float_load_is_mask_guarded(body)
        assert "select i1" not in body, body


def test_dependent_other_flows_from_first_masked_load_without_hoisting_arithmetic(record_property):
    """BITES for the guard; RETENTION for value flow. `other` of the second load is
    computed from the first load's merged value; the mulf must live after the first
    merge (not inside the first load block) and reach the second bypass phi."""
    body = lower(DEPENDENT_OTHER)
    _record(record_property, DEPENDENT_OTHER, body)
    loads = every_float_load_is_mask_guarded(body)
    assert len(loads) == 2, body
    blocks = blocks_of(body)
    mul = re.search(r"(%[\w.]+) = fmul float (%[\w.]+), 2\.0+e\+00", body)
    assert mul, body
    assert not any(mul.group(0) in blocks[l] for l in loads), body
    first_phi = re.search(r"(%[\w.]+) = phi float", body).group(1)
    assert mul.group(2) == first_phi, body
    second_bypass = [f for _c, _t, f in cond_branches(body) if _t == loads[1]][0]
    assert mul.group(1) in blocks[second_bypass] or re.search(rf"phi float .*{re.escape(mul.group(1))}", body), body


def test_masked_store_after_masked_load_keeps_independent_guards(record_property):
    """BITES (load) + RETENTION (store). Different masks for the load and the
    store: exactly one load on the load-mask edge, exactly one store on the
    store-mask edge, and the store must not be nested under the load's guard."""
    body = lower(LOAD_THEN_STORE_DIFFERENT_MASKS)
    _record(record_property, LOAD_THEN_STORE_DIFFERENT_MASKS, body)
    loads = every_float_load_is_mask_guarded(body)
    blocks = blocks_of(body)
    stores = [l for l, t in blocks.items() if re.search(r"\bstore float\b", t)]
    assert len(loads) == 1 and len(stores) == 1, body
    branches = cond_branches(body)
    assert len(branches) == 2, body
    store_branch = [b for b in branches if b[1] == stores[0]]
    assert store_branch and store_branch[0][0] != [b for b in branches if b[1] == loads[0]][0][0], body


@pytest.mark.parametrize("text", [NEGATED_MASK, CONJOINED_MASK], ids=["negated", "conjoined"])
def test_mask_expressions_derived_by_i1_logic_still_guard_the_load(record_property, text):
    """BITES. xor/and over comparison masks are still dynamic masks."""
    body = lower(text)
    _record(record_property, text, body)
    every_float_load_is_mask_guarded(body)
    assert "select i1" not in body, body


# --------------------------------------------------------------------------- 2. negatives against unsound load combining


def test_two_loads_of_same_pointer_with_intervening_store_are_not_combined(record_property):
    """RETENTION at this lowering stage. Both loads and the intervening store
    survive. Later store forwarding may remove a read, but must not reuse the
    first load's value as though the store had not happened."""
    body = lower(STORE_BETWEEN_SAME_PTR)
    _record(record_property, STORE_BETWEEN_SAME_PTR, body)
    assert len(re.findall(r"\bload (?:volatile )?float\b", body)) == 2, body
    assert len(re.findall(r"\bstore float\b", body)) == 2, body
    phis = re.findall(r"(%[\w.]+) = phi float", body)
    add = re.search(r"fadd float (%[\w.]+), (%[\w.]+)", body)
    assert add and add.group(1) != add.group(2), body


@pytest.mark.parametrize(
    "text",
    [TWO_PTRS_SAME_MASK, SAME_PTR_UNEQUAL_MASKS, SAME_PTR_SAME_MASK_DIFFERENT_OTHER],
    ids=["distinct_pointers", "unequal_masks", "different_other"],
)
def test_distinct_loads_are_not_merged(record_property, text):
    """RETENTION. Two loads that differ in pointer, mask or `other` remain two
    loads with two merges at this lowering stage. An optimizer may share a read
    if it independently preserves mask and inactive-value semantics."""
    body = lower(text)
    _record(record_property, text, body)
    assert len(re.findall(r"\bload (?:volatile )?float\b", body)) == 2, body
    merges = len(re.findall(r"\bphi float\b", body)) + len(re.findall(r"\bselect i1\b", body))
    assert merges >= 2, body


def test_guarded_variant_of_negatives_has_one_guard_per_load():
    """BITES. On the repaired helper every load in the combining negatives is
    individually guarded; on the parent these are unguarded loads + selects."""
    for text in (
        STORE_BETWEEN_SAME_PTR,
        TWO_PTRS_SAME_MASK,
        SAME_PTR_UNEQUAL_MASKS,
        SAME_PTR_SAME_MASK_DIFFERENT_OTHER,
    ):
        body = lower(text)
        loads = every_float_load_is_mask_guarded(body)
        assert len(loads) == 2, body


# --------------------------------------------------------------------------- 3. memory modifiers


def test_volatile_load_is_represented_or_refused(record_property):
    """The opt-in route keeps source volatility on the guarded LLVM read."""
    body = lower(VOLATILE_LOAD)
    _record(record_property, VOLATILE_LOAD, body)
    record_property("router_admits_volatile", str(router_admits(VOLATILE_LOAD)))
    assert router_admits(VOLATILE_LOAD)
    assert "load volatile float" in body, body


@pytest.mark.parametrize("mode", ["dynamic", "true", "false", "unmasked"])
def test_volatile_load_keeps_mask_and_constant_elision(mode):
    text = VOLATILE_LOAD
    if mode in {"true", "false"}:
        text = text.replace(
            "%m = arith.cmpi slt, %i, %n : tensor<128xi32>",
            f"%m = arith.constant dense<{mode}> : tensor<128xi1>",
        )
    elif mode == "unmasked":
        text = text.replace("tt.load %p, %m, %z", "tt.load %p")
    body = lower(text)
    if mode == "false":
        assert not re.search(r"\bload (?:volatile )?float\b", body), body
    else:
        assert len(re.findall(r"\bload volatile float\b", body)) == 1, body
        if mode == "dynamic":
            every_float_load_is_mask_guarded(body)
        else:
            assert "br i1" not in body, body


@pytest.mark.parametrize("element", ["float", "i32", "i64"])
@pytest.mark.parametrize("volatile", [False, True])
def test_typed_pointer_conversion_retains_volatile_and_element_type(element, volatile):
    from triton_msl.backend.compiler import MetalBackend

    qualifier = "volatile " if volatile else ""
    opaque = f"""define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1) {{
  %p = addrspacecast ptr addrspace(1) %0 to ptr
  %q = addrspacecast ptr addrspace(1) %1 to ptr
  %a = getelementptr {element}, ptr %p, i32 1
  %b = getelementptr {element}, ptr %q, i32 1
  %v = load {qualifier}{element}, ptr %a, align 4
  store {element} %v, ptr %b, align 4
  ret void
}}
"""
    typed = MetalBackend._opaque_to_typed_ptrs(opaque)
    assert f"load {qualifier}{element}, {element} addrspace(1)* %a" in typed
    assert f"store {element} %v, {element} addrspace(1)* %b" in typed
    assert not re.search(r"\bptr\b", typed), typed


def test_cache_and_eviction_hints_do_not_change_memory_semantics(record_property):
    """NVIDIA performance hints are documented no-ops on this C++ Metal route."""
    body = lower(CACHE_MODIFIED_LOAD)
    plain = CACHE_MODIFIED_LOAD.replace(" cacheModifier = cg evictionPolicy = evict_last", "").replace(
        " cacheModifier = wt", ""
    )
    _record(record_property, CACHE_MODIFIED_LOAD, body)
    assert router_admits(CACHE_MODIFIED_LOAD)
    assert body == lower(plain)
    every_float_load_is_mask_guarded(body)


def test_modifier_bearing_masked_loads_are_still_mask_guarded(record_property):
    """BITES. Whatever happens to the modifiers, the mask guard must not be
    lost on a modifier-bearing load."""
    for text in (VOLATILE_LOAD, CACHE_MODIFIED_LOAD):
        body = lower(text)
        every_float_load_is_mask_guarded(body)
        assert "select i1" not in body, body


def test_mask_guard_is_also_present_through_the_ordinary_router_path(record_property, monkeypatch):
    """BITES. Real Triton TTGIR (blocked layouts, loc info, program-id based
    indices, volatile and cache modifiers) through MetalBackend.make_llir, i.e.
    exactly what the opt-in router feeds the native pass, not a hand-written module."""
    import triton.language as tl
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from triton_msl.backend.compiler import MetalBackend

    @triton.jit
    def _modified_loads(X, O, N, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(X + i, i < N, other=0.0, volatile=True)
        b = tl.load(X + i, i < N, other=0.0, cache_modifier=".cg", eviction_policy="evict_last")
        tl.store(O + i, a + b, i < N, cache_modifier=".wt")

    # This test ends at LLVM IR. Do not probe an actual device/toolchain.
    monkeypatch.setattr("triton_msl.backend.device_detect.air_version_components", lambda: (2, 8, 4, 0))
    target = GPUTarget("metal", "apple-m4", 32)
    backend = MetalBackend(target)
    options = backend.parse_options({"num_warps": 4})
    src = ASTSource(
        fn=_modified_loads,
        signature={"X": "*fp32", "O": "*fp32", "N": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 128},
    )
    context = ir.context()
    ir.load_dialects(context)
    mod = src.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(), context)
    metadata = {}
    mod = backend.make_ttir(mod, metadata, options)
    mod = backend.make_ttgir(mod, metadata, options)
    text = str(mod)
    assert "isVolatile = true" in text and "cacheModifier = cg" in text, text
    assert router_admits(text)
    record_property("router_admits", "True")
    body = function_body(MetalBackend.make_llir(text, {"name": "k"}, options))
    loads = every_float_load_is_mask_guarded(body)
    assert len(loads) == 2, body
    assert "select i1" not in body, body
    assert "load volatile float" in body, body
    assert not re.search(r"\bptr\b", body), body
    record_property("volatile_preserved", "True")
