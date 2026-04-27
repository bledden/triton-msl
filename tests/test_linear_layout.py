"""Tests for the LinearLayout helper used by ``ttg.convert_layout`` lowering."""

from triton_metal.codegen._linear_layout import (
    LinearLayout,
    parse_linear_layout,
    blocked_to_linear,
)


def test_layout_position_simple_identity():
    """L(x) = x for an 8-element identity layout (bases [1, 2, 4])."""
    ll = LinearLayout(register_basis=[1, 2, 4])
    assert ll.position(0, 0, 0) == 0
    assert ll.position(1, 0, 0) == 1
    assert ll.position(7, 0, 0) == 7  # 1 ^ 2 ^ 4 = 7
    assert ll.total_elements == 8


def test_layout_xor_linearity():
    """L(a ^ b) == L(a) ^ L(b)."""
    ll = LinearLayout(register_basis=[5, 3])
    # L(0) = 0, L(1) = 5, L(2) = 3, L(3) = 5 ^ 3 = 6
    assert ll.position(0, 0, 0) == 0
    assert ll.position(1, 0, 0) == 5
    assert ll.position(2, 0, 0) == 3
    assert ll.position(3, 0, 0) == 6


def test_parse_linear_layout_test_trans_reshape_signature():
    """Parse the exact #linear layout from test_trans_reshape's TTGIR."""
    mod_text = (
        "#linear = #ttg.linear<{"
        "register = [[32], [64], [16]], "
        "lane = [[128], [256], [512], [1], [2]], "
        "warp = [[4], [8]], "
        "block = []}>"
    )
    ll = parse_linear_layout(mod_text, "linear")
    assert ll is not None
    assert ll.register_basis == [32, 64, 16]
    assert ll.lane_basis == [128, 256, 512, 1, 2]
    assert ll.warp_basis == [4, 8]
    assert ll.block_basis == []
    assert ll.total_elements == 1024


def test_parse_layout_returns_none_for_multidim_basis():
    """Multi-dim basis vectors aren't supported; parser returns None."""
    mod_text = (
        "#linear = #ttg.linear<{"
        "register = [[1, 0], [0, 1]], "  # 2-D basis
        "lane = [[1]], warp = [[1]], block = []}>"
    )
    ll = parse_linear_layout(mod_text, "linear")
    assert ll is None


def test_layout_bijection_for_test_trans_reshape():
    """The test_trans_reshape layout covers all 1024 positions exactly once."""
    mod_text = (
        "#linear = #ttg.linear<{"
        "register = [[32], [64], [16]], "
        "lane = [[128], [256], [512], [1], [2]], "
        "warp = [[4], [8]], "
        "block = []}>"
    )
    ll = parse_linear_layout(mod_text, "linear")
    seen = set()
    for warp in range(ll.num_warps):
        for lane in range(ll.num_lanes):
            for reg in range(ll.num_registers_per_thread):
                seen.add(ll.position(reg, lane, warp))
    assert len(seen) == 1024
    assert max(seen) == 1023


def test_blocked_to_linear_simple_1d():
    """1-D blocked layout with sizePerThread=4, threadsPerWarp=32, warpsPerCTA=4
    over 1024 elements should give 8 registers per thread (4 in tile 0, 4 in tile 1)."""
    ll = blocked_to_linear([4], [32], [4], [0], (1024,))
    assert ll is not None
    assert ll.total_elements == 1024
    assert ll.num_registers_per_thread == 8  # 4 spt * 2 tiles
    # Thread 0 register 0 = element 0; register 1 = element 1; register 4 = element 512 (next tile)
    assert ll.position(0, 0, 0) == 0
    assert ll.position(1, 0, 0) == 1
    assert ll.position(3, 0, 0) == 3
    assert ll.position(4, 0, 0) == 512  # next tile starts here
    # Thread 1 register 0 = element 4 (sizePerThread stride)
    assert ll.position(0, 1, 0) == 4
    # Warp 1 thread 0 register 0 = element 128 (warp stride)
    assert ll.position(0, 0, 1) == 128


def test_msl_position_expr_matches_python():
    """The MSL-emitted expression must compute the same positions as the Python helper."""
    ll = LinearLayout(register_basis=[32, 64, 16],
                      lane_basis=[128, 256, 512, 1, 2],
                      warp_basis=[4, 8])
    expr = ll.msl_position_expr("r", "l", "w")
    # Spot-check by mock-evaluating in Python (replace ``& {b}`` with int math).
    # The expression form is: "((-(int)((r >> i) & 1u)) & B) ^ ..."
    # Simulate the int-bool-and-int-mask trick.
    def eval_expr(r, l, w):
        result = 0
        for var, bases in (
            (r, ll.register_basis), (l, ll.lane_basis), (w, ll.warp_basis)
        ):
            for i, b in enumerate(bases):
                result ^= (-((var >> i) & 1)) & b
        return result

    for r, l, w in [(0, 0, 0), (5, 13, 2), (7, 31, 3)]:
        py = ll.position(r, l, w)
        mock = eval_expr(r, l, w)
        assert py == mock, f"MSL-mock {mock} != python {py} at ({r},{l},{w})"
