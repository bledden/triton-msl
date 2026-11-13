"""Cooperative lowering selected only after the real closed source proof.

No public/caller-record construction path and no runtime or library loading.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class View:
    name: str
    transpose: bool = False

    def at(self, r, c):
        if self.transpose:
            r, c = c, r
        return f"{self.name}[{r} * 33u + {c}]"


def dot(target, left, right, initial):
    # One shared implementation for every design dot, preserving supplied C.
    return [
        f"{target} = {initial};",
        "for (uint k = 0u; k < 32u; ++k) {",
        f"float product = {left.at('r', 'k')} * {right.at('k', 'c')};",
        f"{target} = {target} + product;",
        "}",
    ]


def cells(body):
    return [
        "for (uint slot = 0u; slot < 8u; ++slot) {",
        "uint linear = tid + 128u * slot;",
        "uint r = linear / 32u;",
        "uint c = linear % 32u;",
        *body,
        "}",
    ]


def render(program, symbol):
    from triton_msl.codegen._source_program import renderer

    # Semantic.prove closes the entire forward DAG, including ordered selects,
    # source constants, supplied dot accumulators, load masks and loop carries.
    # This is a fixed lowering of that closed program, not recognition by hash.
    addresses = {x["record"].root: x["record"] for x in program["addresses"]}
    assert set(addresses) == set(range(7))

    def load(role, coords):
        return f"b{role}[" + renderer.offset(addresses[role], coords) + "]"

    inv = f"as_type<float>(0x{program['constant_bits']['inv_ln2']:08x}u)"
    ln = f"as_type<float>(0x{program['constant_bits']['ln2']:08x}u)"
    if type(symbol) is not str or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("invalid source-program entry symbol")
    sig = [
        f"device {'' if i < 2 else 'const '}{'uchar' if i == 6 else 'float'}* b{i} [[buffer({i})]]" for i in range(7)
    ]
    sig += [f"constant {'float' if i < 9 else 'int'}& a{i} [[buffer({i})]]" for i in range(7, 11)]
    sig += ["uint3 pid [[threadgroup_position_in_grid]]", "uint tid [[thread_index_in_threadgroup]]"]
    code = [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "#pragma clang fp contract(off) reassociate(off)",
        "// Exact 128-thread full-tile lowering. External caller must enforce metadata.",
        "kernel void " + symbol + "(" + ",\n".join(sig) + ") {",
        "threadgroup float Q[1056];",
        "threadgroup float S[1056];",
        "threadgroup float KV[1056];",
        "threadgroup float M[32];",
        "threadgroup float L[32];",
        "threadgroup float A[32];",
        "threadgroup uchar Mask[32];",
        "// EXECUTOR_BEGIN",
        "float acc[8];",
        f"float scale = a7 * {inv};",
    ]
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
    code += cells(["acc[slot] = 0.0f;", f"Q[r * 33u + c] = {load(2, ('r', 'c'))} * scale;"])
    code += ["if (tid < 32u) {", "M[tid] = as_type<float>(0xff800000u);", "L[tid] = 0.0f;", "}", barrier]
    code += ["for (uint iv = 0u; iv < as_type<uint>(a9); iv += 32u) {"]
    code += cells([f"KV[r * 33u + c] = {load(3, ('r', 'c'))};", f"S[r * 33u + c] = {load(5, ('r', 'c'))} * {inv};"])
    code += ["if (tid < 32u) {", f"Mask[tid] = {load(6, ('tid',))};", "}", barrier]
    code += cells(
        ["float score;"]
        + dot("score", View("Q"), View("KV"), "S[r * 33u + c]")
        + [
            "score = (Mask[c] != uchar(0) ? a8 : score);",
            "score = ((as_type<int>(r + pid.x * 32u) < a9 && as_type<int>(c + iv) < a9) ? score : a8);",
            "S[r * 33u + c] = score;",
        ]
    )
    code += [
        barrier,
        "if (tid < 32u) {",
        "float maximum = S[tid * 33u];",
        "for (uint k = 1u; k < 32u; ++k) {",
        "maximum = fmax(maximum, S[tid * 33u + k]);",
        "}",
        "maximum = fmax(M[tid], maximum);",
        "float difference = M[tid] - maximum;",
        "A[tid] = precise::exp2(difference);",
        "M[tid] = maximum;",
        "}",
        barrier,
    ]
    code += cells(["float difference = S[r * 33u + c] - M[r];", "S[r * 33u + c] = precise::exp2(difference);"])
    code += [
        barrier,
        "if (tid < 32u) {",
        "float total = S[tid * 33u];",
        "for (uint k = 1u; k < 32u; ++k) {",
        "total = total + S[tid * 33u + k];",
        "}",
        "float rescaled = L[tid] * A[tid];",
        "L[tid] = rescaled + total;",
        "}",
    ]
    code += cells([f"KV[r * 33u + c] = {load(4, ('r', 'c'))};"])
    code += [barrier]
    code += cells(["float carry = acc[slot] * A[r];"] + dot("acc[slot]", View("S"), View("KV"), "carry"))
    code += [barrier, "}"]
    stores = [f"{load(0, ('r', 'c'))} = acc[slot] / L[r];"]
    if not program["padded"]:
        stores = ["if (as_type<int>(r + pid.x * 32u) < a9) {", *stores, "}"]
    code += cells(stores)
    code += ["if (tid < 32u) {"]
    if not program["padded"]:
        code += ["if (as_type<int>(tid + pid.x * 32u) < a9) {"]
    code += [
        f"float scaled_maximum = M[tid] * {ln};",
        "float log_l = precise::log(L[tid]);",
        f"{load(1, ('tid', '0u') if program['padded'] else ('tid',))} = scaled_maximum + log_l;",
    ]
    if not program["padded"]:
        code += ["}"]
    code += ["}", "}"]
    return "\n".join(code) + "\n", {"declared_private_bytes_upper_bound": 32, "shared_bytes": 13088, "threads": 128}
