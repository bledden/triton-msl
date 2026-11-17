"""Device-free evaluation of proven source program-id domains."""


def matmul_grid_rectangles(plan, grid, value):
    """At most three exact half-open source-coordinate rectangles, or ValueError.

    ``value`` resolves an immutable source scalar reference using the launch's
    captured payload. This function performs no tensor/device introspection.
    """
    bm, bn, mapping, axes, grid_refs, load_refs, full_k = plan
    if (
        type(bm) is not int
        or type(bn) is not int
        or min(bm, bn) <= 0
        or type(axes) not in (tuple, list)
        or len(axes) != 2
        or any(type(v) is not bool for v in axes)
        or type(full_k) is not bool
        or len(grid_refs) != 2
        or len(load_refs) != 2
        or type(grid) not in (tuple, list)
        or len(grid) != 3
        or any(type(v) is not int or not 0 < v <= 0x7FFFFFFF for v in grid)
    ):
        raise ValueError("malformed source grid plan")

    def tiles(ref, block):
        extent = value(ref)
        # Grid arithmetic adds in signed i32 before division. Prove
        # nonzero divisors and no wrap rather than evaluating another map.
        if not 0 < extent <= 0x7FFFFFFF - block + 1:
            raise ValueError("source grid divisor can wrap or divide by zero")
        return (extent + block - 1) // block

    programs = grid[0]
    rects = []

    def add(m0, m1, n0, n1):
        if m1 > m0 and n1 > n0:
            rects.append((m0 * bm, m1 * bm, n0 * bn, n1 * bn))

    if mapping == "2d":
        mt, nt = (grid[i] if axes[i] else 1 for i in range(2))
        add(0, mt, 0, nt)
    elif mapping == "1d":
        npm = tiles(grid_refs[0], bm)
        complete, partial = divmod(programs, npm)
        add(0, npm, 0, complete)
        add(0, partial, complete, complete + 1)
    elif type(mapping) in (tuple, list) and len(mapping) == 2 and mapping[0] == "flat_n" and mapping[1] == bn:
        npn = tiles(grid_refs[1], bn)
        complete, partial = divmod(programs, npn)
        add(0, complete, 0, npn)
        add(complete, complete + 1, 0, partial)
    elif type(mapping) in (tuple, list) and len(mapping) == 3 and mapping[0] == "grouped" and mapping[2] == bn:
        group = mapping[1]
        npm, npn = tiles(grid_refs[0], bm), tiles(grid_refs[1], bn)
        if type(group) is not int or group <= 0 or group * npn > 0x7FFFFFFF:
            raise ValueError("source grouped span can wrap")
        span = group * npn
        gid = (programs - 1) // span
        first = gid * group
        gm = min(npm - first, group)
        if gm <= 0:
            raise ValueError("source grouped mapping divides by a nonpositive group")
        add(0, first, 0, npn)
        complete, partial = divmod(programs - gid * span, gm)
        add(first, first + gm, 0, complete)
        add(first, first + partial, complete, complete + 1)
    else:
        raise ValueError("unproved source grid mapping")
    if max(max(m1, n1) for _, m1, _, n1 in rects) > 0x7FFFFFFF:
        raise ValueError("source tile index can wrap")
    return tuple(rects)


def matmul_grid_domain(plan, grid, value):
    """Exclusive row/column maxima; useful for independent operand domains."""
    rects = matmul_grid_rectangles(plan, grid, value)
    return max(r[1] for r in rects), max(r[3] for r in rects)
