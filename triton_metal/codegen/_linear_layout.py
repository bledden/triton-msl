"""Linear-layout computation helper for ``ttg.linear`` and friends.

A linear layout maps a hardware position ``(register, lane, warp[, block])``
to a logical tensor index, via XOR of basis vectors. See the LinearLayout
documentation in ``triton/include/triton/Tools/LinearLayout.h`` for the full
formalism.

This module gives us enough of the layout machinery to:
  - parse ``#ttg.linear<{...}>`` and ``#ttg.blocked<{...}>`` attributes from
    TTGIR text
  - compute the logical position of every (register, lane, warp) triple
  - emit MSL expressions for the same computation at codegen time

Used by ``_lower_convert_layout`` to emit a shared-memory shuffle that
correctly redistributes data between two arbitrary layouts of the same
1-D tensor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LinearLayout:
    """A linear layout from (register, lane, warp[, block]) to 1-D logical position.

    Each ``*_basis`` is a list of integers — basis vector i is the value
    of L when input bit i is set and all other input bits are 0. The
    layout is then computed as the XOR of all basis vectors corresponding
    to set bits in the input.
    """

    register_basis: List[int] = field(default_factory=list)
    lane_basis: List[int] = field(default_factory=list)
    warp_basis: List[int] = field(default_factory=list)
    block_basis: List[int] = field(default_factory=list)

    @property
    def num_registers_per_thread(self) -> int:
        """Number of elements held per thread (= 2 ** len(register_basis))."""
        return 1 << len(self.register_basis)

    @property
    def num_lanes(self) -> int:
        """Threads per warp (= 2 ** len(lane_basis))."""
        return 1 << len(self.lane_basis)

    @property
    def num_warps(self) -> int:
        """Warps per CTA (= 2 ** len(warp_basis))."""
        return 1 << len(self.warp_basis)

    @property
    def num_blocks(self) -> int:
        return 1 << len(self.block_basis)

    @property
    def total_elements(self) -> int:
        return (self.num_registers_per_thread * self.num_lanes
                * self.num_warps * self.num_blocks)

    def position(self, register: int, lane: int, warp: int, block: int = 0) -> int:
        """Compute the logical 1-D tensor position for a hardware location.

        Equivalent to L(register, lane, warp, block).
        """
        out = 0
        for i, basis in enumerate(self.register_basis):
            if (register >> i) & 1:
                out ^= basis
        for i, basis in enumerate(self.lane_basis):
            if (lane >> i) & 1:
                out ^= basis
        for i, basis in enumerate(self.warp_basis):
            if (warp >> i) & 1:
                out ^= basis
        for i, basis in enumerate(self.block_basis):
            if (block >> i) & 1:
                out ^= basis
        return out

    def msl_position_expr(self, reg_var: str, lane_var: str, warp_var: str,
                          block_var: str = "0") -> str:
        """Generate an MSL expression that computes ``position(reg, lane, warp, block)``.

        The expression assumes ``reg_var``/``lane_var``/``warp_var`` are
        ``uint`` variables. Result is ``int`` (signed because tensor
        positions can be used as offsets).
        """
        # XOR of (basis_i & ((var >> i) & 1) ? basis_i : 0).
        # Compact: ``-((var >> i) & 1) & basis_i`` produces ``basis_i`` when
        # bit i is set, ``0`` otherwise. XOR-summing gives the layout value.
        terms = []
        for var, bases in (
            (reg_var, self.register_basis),
            (lane_var, self.lane_basis),
            (warp_var, self.warp_basis),
            (block_var, self.block_basis),
        ):
            for i, b in enumerate(bases):
                if b == 0:
                    continue
                terms.append(f"((-(int)(({var} >> {i}u) & 1u)) & {b})")
        if not terms:
            return "0"
        # XOR-sum.
        return " ^ ".join(terms)


def parse_linear_layout(mod_text: str, layout_name: str) -> Optional[LinearLayout]:
    """Parse a ``#ttg.linear<{...}>`` attribute by name.

    Looks for a definition of the form
        #<name> = #ttg.linear<{register = [[a], [b], ...],
                              lane = [[c], [d], ...],
                              warp = [[e], ...],
                              block = []}>

    Each basis vector is a list of integers (1 element for 1-D tensors,
    multiple for higher-dim). For our 1-D ``convert_layout`` use we
    extract the single integer per basis. If any basis vector is multi-
    dimensional we return None (signalling "not handled here" so the
    generic passthrough path is used).
    """
    pattern = (
        r"#"
        + re.escape(layout_name)
        + r"\s*=\s*#ttg\.linear<\{(.+?)\}>"
    )
    m = re.search(pattern, mod_text, re.DOTALL)
    if not m:
        return None
    body = m.group(1)

    def _parse_field(name: str) -> Optional[List[int]]:
        # Find ``name = [`` and walk forward until the matching ``]`` —
        # naive regex can\'t handle the nested brackets in
        # ``register = [[32], [64], [16]]``.
        anchor = re.search(rf"{name}\s*=\s*\[", body)
        if not anchor:
            return None
        i = anchor.end()
        depth = 1
        end = None
        while i < len(body):
            c = body[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end is None:
            return None
        inner = body[anchor.end():end].strip()
        if not inner:
            return []
        out = []
        for vec in re.finditer(r"\[\s*([^\]]+?)\s*\]", inner):
            parts = [p.strip() for p in vec.group(1).split(",")]
            if len(parts) != 1:
                return None  # multi-dim basis vector — not handled
            out.append(int(parts[0]))
        return out

    reg = _parse_field("register")
    lane = _parse_field("lane")
    warp = _parse_field("warp")
    block = _parse_field("block")

    if any(b is None for b in (reg, lane, warp, block)):
        return None

    return LinearLayout(
        register_basis=reg,
        lane_basis=lane,
        warp_basis=warp,
        block_basis=block,
    )


def blocked_to_linear(size_per_thread: List[int], threads_per_warp: List[int],
                      warps_per_cta: List[int], order: List[int],
                      tensor_shape: Tuple[int, ...]) -> Optional[LinearLayout]:
    """Convert a 1-D blocked layout to its LinearLayout equivalent.

    Only handles the 1-D case used by ``test_trans_reshape``\\'s output
    layout. Returns None for multi-dimensional layouts.
    """
    if (len(size_per_thread) != 1 or len(threads_per_warp) != 1
            or len(warps_per_cta) != 1 or len(order) != 1
            or len(tensor_shape) != 1):
        return None
    spt = size_per_thread[0]
    tpw = threads_per_warp[0]
    wpc = warps_per_cta[0]
    n = tensor_shape[0]

    # Tile size = spt * tpw * wpc. If n > tile_size, the layout repeats
    # ("tiles") with a stride of tile_size. The tiling lives at the high
    # bits of the register index. With n / tile_size copies, log2 of that
    # ratio is the count of "tile-repeat" register bits.
    tile_size = spt * tpw * wpc
    if n % tile_size != 0:
        return None
    n_tiles = n // tile_size
    if n_tiles & (n_tiles - 1):
        return None  # not a power of 2

    register_basis = []
    # First spt register bits walk the in-thread contiguous block (stride 1)
    bits_in_spt = (spt - 1).bit_length()
    for i in range(bits_in_spt):
        register_basis.append(1 << i)
    # Then tile-repeat register bits at stride tile_size
    bits_in_tiles = (n_tiles - 1).bit_length()
    for i in range(bits_in_tiles):
        register_basis.append(tile_size << i)

    # Lane bits: stride spt across lanes
    bits_in_tpw = (tpw - 1).bit_length()
    lane_basis = [spt << i for i in range(bits_in_tpw)]

    # Warp bits: stride spt * tpw across warps
    bits_in_wpc = (wpc - 1).bit_length()
    warp_basis = [(spt * tpw) << i for i in range(bits_in_wpc)]

    return LinearLayout(
        register_basis=register_basis,
        lane_basis=lane_basis,
        warp_basis=warp_basis,
        block_basis=[],
    )
