"""Measured cooperative attention emission for an explicit validation option.

Only a caller holding the unchanged wide_row_plan proof may use this builder.
No detector, default route or launch ABI is changed by importing this module.
"""

from dataclasses import dataclass
import hashlib
import re

from ._lowerer_helpers import _check_replay_shared_memory_budget
from .msl_emitter import Arg, KernelBuilder


@dataclass(frozen=True)
class CooperativeEmission:
    source: str
    entry: str
    threads: int
    shared: int
    execution_width: int


def make_descriptor(lowerer, plan, scalar_source, candidate):
    """Attention descriptor union member; existing packed field 9 seals all bytes.

    The primary metallib and stash remain scalar32/0. This separate sibling is
    compiled only by the opt-in zero-copy dispatcher, never mistaken for the
    scalar stash or implicitly substituted in an external metallib loader.
    """
    names = [arg.name for arg in lowerer.kb.args]
    role_names = [plan[key] for key in ("q", "k", "v", "bias", "out", "nq", "nk")]
    if len(names) != 7 or len(set(role_names)) != 7:
        return None  # Retain scalar for ABIs outside the measured seven slots.
    return {
        "kind": "cooperative-row-v1",
        "primary": lowerer.kb.name,
        "entry": candidate.entry,
        "source": candidate.source,
        "source_sha256": hashlib.sha256(candidate.source.encode()).hexdigest(),
        "scalar_sha256": hashlib.sha256(scalar_source.encode()).hexdigest(),
        "threads": candidate.threads,
        "shared": candidate.shared,
        "execution_width": candidate.execution_width,
        "scalar_threads": 32,
        "scalar_shared": 0,
        "arg_count": len(names),
        "roles": [names.index(plan[key]) for key in ("q", "k", "v", "bias", "out", "nq", "nk")],
    }


def build_candidate(lowerer, plan):
    """Build the measured sibling without mutating the scalar lowerer's state.

    This checks implementation scope, not source admission. wide_row_plan is
    responsible for proving the source's addresses, reductions and rounding.
    """
    if (plan["rows"], plan["cols"], plan["depth"], plan["causal"]) != (32, 64, 64, False):
        return None
    scalar = lowerer.kb
    if scalar is None:
        raise ValueError("cooperative sibling requires the proved scalar builder")
    kb = KernelBuilder(scalar.name + "__attncoop16", block_size=512)
    # Source argument names are arbitrary. Rename collisions in this sibling
    # only, preserving the positional ABI and the scalar builder byte-for-byte.
    roles = {"Q", "K", "V", "Bias", "Out", "NQ", "NK"}
    reserved = set(re.findall(r"\b[A-Za-z_]\w*\b", _BODY)) - roles
    reserved.update({"pid", "lid", "tid", "sgitg", "tiisg"})
    used = reserved | {arg.name for arg in scalar.args}
    renamed = {}
    for index, arg in enumerate(scalar.args):
        name = arg.name
        if name in reserved:
            name = f"_tmattn_arg{index}"
            while name in used:
                name += "_"
            used.add(name)
        renamed[arg.name] = name
        kb.args.append(Arg(name, arg.dtype, is_ptr=arg.is_ptr, const=arg.const))
    kb._needs_simd_qualifiers = True
    kb.declare_threadgroup_array("Qs", dtype="fp32", size=32 * 64)
    kb.declare_threadgroup_array("S", dtype="fp32", size=32 * 64)
    shared = _check_replay_shared_memory_budget(kb._threadgroup_arrays, {})
    assert shared == 16384
    names = {
        name: renamed[plan[key]]
        for name, key in (
            ("Q", "q"),
            ("K", "k"),
            ("V", "v"),
            ("Bias", "bias"),
            ("Out", "out"),
            ("NQ", "nq"),
            ("NK", "nk"),
        )
    }
    body = re.sub(r"\b(?:Q|K|V|Bias|Out|NQ|NK)\b", lambda m: names[m[0]], _BODY)
    for line in body.splitlines():
        kb.raw_line(line)
    return CooperativeEmission(kb.build(), kb.name, 512, shared, 32)


# Mirrored from the independently reviewed SB4 shader. Only argument identifiers,
# SIMD attribute variable names and tile flattening change. Keep FP statements,
# ordered chains and the final device barrier intact.
_BODY = r"""
        // Thread mapping by construction from the SIMD attributes (execution width must be 32).
        uint _p523_slot = 2u * sgitg + tiisg / 16u;
        uint _p523_lane = tiisg % 16u;
        uint _p515_row = pid * 32u + _p523_slot;
        bool _p523_active = int(_p515_row) < NQ;
        // phase 0 -- rounded scaled Q once per row, in the reference's two-statement form.
        if (_p523_active) {
            #pragma unroll
            for (uint _p523_m_q = 0u; _p523_m_q < 4u; ++_p523_m_q) {
                uint _p515_kd_q = _p523_lane + 16u * _p523_m_q;
                float _p515_q_q = static_cast<float>(Q[_p515_row * 64u + _p515_kd_q]);
                _p515_q_q = static_cast<float>(static_cast<half>(_p515_q_q * 0.125f));
                Qs[_p523_slot * 64u + _p515_kd_q] = _p515_q_q;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup); // SB0
        float _p515_acc[4];
        #pragma unroll
        for (uint _p523_c_z = 0u; _p523_c_z < 4u; ++_p523_c_z) {
            _p515_acc[_p523_c_z] = 0.0f;
        }
        float _p515_denom = 0.0f;
        float _p515_maximum_d = -INFINITY;
        float _p515_newmax_d = -INFINITY;
        float _p515_alpha_d = 0.0f;
        for (int _p515_start_d = 0; _p515_start_d < NK; _p515_start_d += 64) {
            if (_p523_active) {
                float _p515_score_d0[4];
                int _p515_key_d0[4];
                bool _p515_live_d0[4];
                #pragma unroll
                for (uint _p523_m_s = 0u; _p523_m_s < 4u; ++_p523_m_s) {
                    _p515_key_d0[_p523_m_s] = _p515_start_d + int(_p523_lane + 16u * _p523_m_s);
                    _p515_live_d0[_p523_m_s] = _p515_key_d0[_p523_m_s] < NK;
                    _p515_score_d0[_p523_m_s] = -INFINITY;
                    if (_p515_live_d0[_p523_m_s]) {
                        _p515_score_d0[_p523_m_s] = Bias[_p515_row * uint(NK) + uint(_p515_key_d0[_p523_m_s])];
                    }
                }
                for (uint _p515_kd_d0 = 0u; _p515_kd_d0 < 64u; ++_p515_kd_d0) {
                    float _p515_q_d0 = Qs[_p523_slot * 64u + _p515_kd_d0];
                    #pragma unroll
                    for (uint _p523_m_k = 0u; _p523_m_k < 4u; ++_p523_m_k) {
                        if (_p515_live_d0[_p523_m_k]) {
                            _p515_score_d0[_p523_m_k] += _p515_q_d0 * static_cast<float>(K[uint(_p515_key_d0[_p523_m_k]) * 64u + _p515_kd_d0]);
                        }
                    }
                }
                #pragma unroll
                for (uint _p523_m_w = 0u; _p523_m_w < 4u; ++_p523_m_w) {
                    S[_p523_slot * 64u + _p523_lane + 16u * _p523_m_w] = _p515_score_d0[_p523_m_w];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup); // SB1
            if (_p523_active) {
                float _p515_block_max_d = -INFINITY;
                for (uint _p515_j_d0 = 0u; _p515_j_d0 < 64u; ++_p515_j_d0) {
                    _p515_block_max_d = fmax(_p515_block_max_d, S[_p523_slot * 64u + _p515_j_d0]);
                }
                _p515_newmax_d = fmax(_p515_maximum_d, _p515_block_max_d);
                _p515_alpha_d = exp(_p515_maximum_d - _p515_newmax_d);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup); // SB2 (WAR guard before the p write)
            if (_p523_active) {
                #pragma unroll
                for (uint _p523_m_p = 0u; _p523_m_p < 4u; ++_p523_m_p) {
                    uint _p553_j_p = _p523_lane + 16u * _p523_m_p;
                    S[_p523_slot * 64u + _p553_j_p] = exp(S[_p523_slot * 64u + _p553_j_p] - _p515_newmax_d);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup); // SB3
            if (_p523_active) {
                float _p515_p_sum_d = 0.0f;
                for (uint _p515_j_d1 = 0u; _p515_j_d1 < 64u; ++_p515_j_d1) {
                    _p515_p_sum_d += S[_p523_slot * 64u + _p515_j_d1];
                }
                _p515_denom = _p515_denom * _p515_alpha_d + _p515_p_sum_d;
                #pragma unroll
                for (uint _p523_c_a = 0u; _p523_c_a < 4u; ++_p523_c_a) {
                    _p515_acc[_p523_c_a] *= _p515_alpha_d;
                }
                for (uint _p515_j_a1 = 0u; _p515_j_a1 < 64u; ++_p515_j_a1) {
                    int _p515_key_a1 = _p515_start_d + int(_p515_j_a1);
                    if (_p515_key_a1 < NK) {
                        float _p553_p = S[_p523_slot * 64u + _p515_j_a1];
                        #pragma unroll
                        for (uint _p523_c_v = 0u; _p523_c_v < 4u; ++_p523_c_v) {
                            uint _p515_od = _p523_lane + 16u * _p523_c_v;
                            _p515_acc[_p523_c_v] += _p553_p * static_cast<float>(V[uint(_p515_key_a1) * 64u + _p515_od]);
                        }
                    }
                }
                _p515_maximum_d = _p515_newmax_d;
            }
            // SB4: fence block n's reads of S (sum scan, PV) against block n+1's S writes. Lockstep
            // execution is not an MSL guarantee, so it is not load-bearing here.
            simdgroup_barrier(mem_flags::mem_threadgroup); // SB4
        }
        if (_p523_active) {
            #pragma unroll
            for (uint _p523_c_d = 0u; _p523_c_d < 4u; ++_p523_c_d) {
                _p515_acc[_p523_c_d] = _p515_acc[_p523_c_d] / _p515_denom;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (_p523_active) {
            #pragma unroll
            for (uint _p523_c_o = 0u; _p523_c_o < 4u; ++_p523_c_o) {
                uint _p523_od = _p523_lane + 16u * _p523_c_o;
                Out[_p515_row * 64u + _p523_od] = _p515_acc[_p523_c_o];
            }
        }
"""
