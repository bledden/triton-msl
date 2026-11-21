# Changelog

## 0.4.0

Compatibility and safety update. It runs on the Xcode 27 toolchain, which the
published 0.3.0 refuses, and fixes defects found in 0.3.0. No general speedup is
claimed. The user guide is `docs/releases/0.4.0.md`.

### Matmul source domains

- Derive ordinary and direct-entry row/column clipping from source masks rather
  than arguments named `M` and `N`. Unmasked stores write whole launched tiles;
  operand load masks and flattened/grouped grid divisors retain separate roles.
- Validate grid-dependent source storage and integer address ranges before either
  entry runs, including aligned K. Incomplete flat/grouped grids use an exact union
  of at most three rectangles, preserving valid sparse output allocations.
- Complete all single-program operand reads before ordinary contiguous or scalar
  fallback stores can replace aliased input cells. The direct entry continues to
  require an output whose storage no input shares.
- Keep the source's load-then-store order when a matmul output overlaps an
  operand. One program loads every operand before it stores, so a
  single-program launch that writes its result over `A` or `B` is defined. The
  fast replacement spreads that program over several threadgroups, which
  store finished tiles while others still load: a 256 by 256 in-place product
  was wrong in 4 or 5 of 300 launches on an M4 Max. A launch whose output
  bytes intersect an operand's now stays on the compiled entry or the host
  path, and that includes the split-K replacement for a deep product. The
  quantized replacements (int8 and int4 GEMM and GEMV) had the same race,
  between 1 and 11 of 200 launches for an in-place GEMV depending on the run;
  they have no compiled entry to fall back to, so an output that overlaps any
  operand now refuses, in one message that also lists the route's other
  conditions. This is a change for an in-place quantized launch that ran
  before. Tensors in one storage that share no cell the launch addresses keep
  the replacement: the routes compare the cells addressed through each
  pointer, and whole storages where a route does not know that count.
- Offer the fast matmul replacement only where it is that source domain. It
  computes exactly `args[3]` by `args[4]`, so a store clip, load clip, load
  modulus or flat-grid divisor that is a separate argument now stays on the
  ordinary entry, as does an unmasked store over a partial tile. Before this, a
  kernel whose store mask named a smaller extent was written through to `M`.
- Replay a compile-time constant load clip on a tiled index. 0.3.0 accepted the
  constant and then loaded whole tiles, so rows past it kept their product.
- Take a matmul template's reduction extent from the K-loop's own bounds, never
  from an argument merely named `K`. The templates reduce `[0, extent)`, so a
  loop is admitted when it starts at 0 and either steps `BLOCK_K` to an argument
  or constant, or counts `cdiv(extent, BLOCK_K)` or a constant number of blocks.
  A later start, `K - BK`, `K // 2`, `min(K, X)`, a floor count, a separate
  count argument, and K-clipped loads beside a loop over something else now
  refuse on the template routes. Before this they ran as `range(0, K, BK)`. A
  floor count with `K` a multiple of `BLOCK_K` was right by coincidence and is
  among the refusals; spell it `tl.cdiv(K, BK)` or step by `BK`.
- Reduce a single, non-looped `tl.dot` over its own `BLOCK_K` tile. With an
  argument named `K` larger than the tile it reduced over `K`.
- Give the direct matmul entry a depth only when the loop proves which argument
  it is. Constant-bound loops keep the ordinary entry.
- Refuse a store mask the templates cannot spell: a second clip on one axis, a
  non-strict `<=` clip, and a constant clip on a flat 1-D grid. One `<` clip per
  axis was kept and the rest dropped. The source-replaying K-loop epilogue route
  executes the source loop and store mask as written and keeps all of these.
- Refuse a store clip on any index other than the one the store addresses,
  for matmul and for the head_dim 64 and 128 FlashAttention templates.
  `(rm + 1) < M`, `(rm - 16) < M`, `rm * 2 < M` and a column index used as the
  row clip all ran as `rm < M`, and `(rn < N)[None, :]` ran as a row clip. A
  constant clip is whole-tile only on the bare `tl.arange` tile. The epilogue
  route keeps the matmul forms; `rm[:, None] < M` and `(rm < M)[:, None]` both
  keep the template.
- These changes require fresh integrated hardware and installed-artifact
  qualification. Earlier source-tree matmul timings do not measure their cost.

### Attention template addresses

- Prove every source load and store address before a biased, triangle,
  backward, varlen or MLA attention template stands in for the kernel. The
  templates rebuild each tensor's address from its strides and their own
  coordinates; before this, only the strides were read from the source. A row
  or depth index that was shifted or scaled, a batch or head offset on the
  wrong tensor, a loop that advanced differently and a key or value tile that
  ignored the loop variable all ran as the canonical address. This held on the
  biased 2-D and 3-D forward routes, the biased backward routes, the varlen
  route and the MLA route; the dense route already refused these. On the
  backward, varlen and MLA routes each address is now evaluated exactly on
  random program ids and stride values and must equal the template's. On the
  biased forward routes each address term is matched against the template's
  row, key, depth and batch index. Anything else refuses. Strides are bound to
  the index they multiply, so a kernel that passes its strides in another
  order computes correctly, and the head index spelled `pid - (pid // H) * H`
  is the template's `pid % H`.
- Stop the FlashAttention output clip from being accepted on the column axis.
  The sequence length bounds rows only.
- Refuse, at launch, a FlashAttention store without the row clip over a
  partial last row tile: an unmasked store, or one masked on the columns alone
  (`offs_d[None, :] < HEAD_DIM`), which bounds no row. The templates store the
  rows below the sequence length; the source stores the whole tile. Mask the
  store with the row boundary or launch whole tiles.
- Refuse, at launch, an attention template launch whose strides, head count
  or lengths the template would read differently from the source. The
  templates bind every integer argument as a 32-bit unsigned value and add
  addresses in 32-bit arithmetic. A negative stride (a view walked from its
  last row) was read as a large positive one: on the dense, small-query,
  varlen, biased and backward routes the result was wrong and the stores
  landed outside the view. A stride, head count or length argument that is
  negative or does not fit 31 bits now refuses when a template receives it,
  as does a tensor whose storage exceeds 2**32 elements, and a view that
  starts inside its storage with more than 2**30 elements from its first one
  on: there a wrapped 32-bit source offset can name a cell before the view,
  which the template's unsigned index cannot reach. Head dimension 32
  with 32-row blocks takes the general path and computes negative strides as
  before, and so does a dense kernel specialized for a 64-bit argument.
- Read the varlen cumulative sequence lengths at the width the tensor holds.
  The templates indexed them as 32-bit integers, so an int64 tensor, which is
  what `cumsum` returns, gave every second entry a zero. Signed and unsigned
  32-bit and 64-bit tensors now compute; other widths refuse.
- Compute each varlen sequence length as the source compares it. The source
  subtracts two cumulative lengths and masks `index < length`. Cumulative
  lengths that decrease give a negative length, which selects no row and no
  key; the template read it as a large unsigned length, attended every key
  and stored every row. The same holds for a difference that overflows the
  tensor's width (int32 entries of opposite sign near the ends of the range):
  the source's wrapped difference is negative, and the template now takes the
  sign of that wrapped difference instead of comparing the two entries. A
  length from an unsigned tensor wraps, as it does in the source. A 64-bit
  length above 32 bits selects every launched index rather than keeping its
  low word. A kernel that compares one length signed in one mask and unsigned
  in another refuses.
- Compare a 64-bit length at 64 bits in the generic lowering's staged masks.
  A packed-sequence kernel that widens its int32 cumulative entries before it
  subtracts them (`tl.load(cu + b).to(tl.int64)`) takes the generic lowering.
  Its per-lane masks compared the length at 64 bits, but the masks rebuilt
  for the staged operand fill and the staged store cast both sides to 32
  bits, so a length of 2**31 or more read as negative and the launch stored
  nothing where the source stores every launched row. The same kernel with
  its rows spelled `arange - program_id` or `program_id + arange - 1` stored
  row -1 of a later program 2**32 elements from its place: the rebuilt
  address added a 32-bit row offset, in which the unsigned program id had
  wrapped, to a pointer that already carried the 64-bit start. A 64-bit
  comparison or address is now rebuilt operation by operation at the widths
  the source gives it, sign and zero extensions included, and refuses when a
  part of it is not one of the operations the rebuild renders. 32-bit masks
  and addresses are emitted as before.
- Address a varlen sequence from a signed start. The source adds the
  cumulative entry to the row index in signed arithmetic, so a negative entry
  addresses rows before the tensor's first row, which is defined on a view
  into a larger storage. The templates held the start in 32 unsigned bits and
  addressed about 2**32 elements past the view. The varlen templates now form
  each offset in 32 wrapping bits and read it as signed, as the source's
  pointer add reads a 32-bit offset: a negative entry addresses the rows
  before the view, and an int32 or uint32 entry whose product with the row
  stride wraps addresses the cell the wrapped product names. An int64 entry
  agrees with the source wherever the source's cell lies in the storage. A
  varlen launch refuses a tensor whose storage reaches more than 2**31
  elements before or after its first element, which a 32-bit offset cannot
  name.
- Store the biased log-sum-exp at its tensor's element type. The templates
  wrote it through a 32-bit float pointer, so a half or bfloat16 tensor
  received four bytes for every two it holds: wrong values, and writes past
  the rows the kernel stores.
- Attend whole key tiles on the varlen route. The source loop is
  `range(0, max_seqlen, 32)` masked by the sequence's key count, so with
  `max_seqlen = 40` it attends up to 64 keys. The templates stopped at
  `max_seqlen` itself and dropped the keys between it and the tile end.
- Refuse, at launch, an MLA kernel whose sequence length is not a whole number
  of 32-row tiles. The MLA template has no partial-tile form: the source reads
  or zero-scores its whole last key tile and, unmasked, stores its whole last
  row tile, and the template did neither.
- Rebuild a staged operand's address and mask from a closed set of forms.
  The cooperative fill of a staged `tl.dot` operand recomputes the address
  and mask of every cell of the tile. A value with no defining operation was
  taken as one number for the whole tile, which holds for a scalar argument
  or a scalar a loop carries. A tensor a loop carries, such as a key index
  advanced by `kn += BLOCK_N`, is one lane's register, so every column read
  one lane's index: a kernel written that way differed from the source in 20
  of 20 launches on an M4 Max, with no exception. A value extended from 8 or
  16 bits lost the wrap of its own width inside the 32-bit sum. A carried
  tensor is now rendered as its start plus the trip count times a step that
  is fixed over the trips, a narrow extension without a row or column index
  is cast to its own type first, and each pointer addition keeps its own
  sum. Forms outside that set are refused when the kernel is lowered: a
  carried tensor that is doubled, added to twice per trip, decremented with
  `-=`, stepped by a kernel argument or 8 bits wide. A narrow product and a
  narrow value holding the row index were refused before and still are.

### Attention template masks

- Prove where each source mask lands before an attention template stands in
  for the kernel. The templates rebuild every mask themselves: zero for the
  operand rows past a sequence boundary, negative infinity for the keys past
  it, the lower triangle on a causal route. The lowering checked which index a
  source mask compared and against which bound, but not the axis the
  comparison was broadcast along. On a square tile `mask_m[None, :]` has the
  same shape as `mask_m[:, None]` and masks other cells, and such a kernel ran
  on the template and returned the canonical result. This held for load,
  store and score-select masks on the biased 2-D and 3-D forward routes and
  the varlen route, and for the causal select on the MLA route. Each mask
  must now vary along the template's axis for its role, or the kernel
  refuses. The dense route already refused these.
- Prove the varlen score select. The template gives every key past a
  sequence's last a score of negative infinity. The source select was not
  checked at all, so a kernel without it, with its branches exchanged, with a
  fill other than negative infinity, with another bound or, on the causal
  route, with another triangle ran as the canonical kernel. The select, or a
  chain of selects, must now keep the score inside the key boundary exactly
  once, with the row boundary and the causal triangle where the route has
  them, and supply negative infinity. A finite fill such as `-1e9` refuses,
  including one whose value equals the bit word of negative infinity at
  another float type (`64512.0`): the template supplies negative infinity,
  and the two differ when a row has no key left. A boundary select anywhere
  else in the kernel refuses.
- Apply the biased route's loaded key mask per key. A source that broadcast
  the loaded mask along the rows (`m_block[:, None]`) masked query rows, and
  the template masked keys.
- Refuse an unmasked V load that reads past the sequence. The templates
  supply zero for the V rows past the key boundary; an unmasked source load
  reads whatever the tensor holds there, and a non-finite value reaches the
  output through a zero probability. The varlen route refuses the kernel. The
  biased routes refuse, at launch, a sequence length that is not a whole
  number of key tiles, and compute whole-tile launches as before.
- On the biased backward routes, refuse a masked load that supplies a value
  other than zero and an unmasked log-sum-exp load. The templates supply zero
  for every operand row past the sequence length. The loaded key mask may
  supply any constant: a key past the sequence gets the sentinel score
  whatever the mask holds there.
- On the biased backward routes, prove the loaded bias and key mask along
  the axes their addresses walk. A bias tile loaded transposed
  (`Bias[key, row]`), a bias load masked by the key boundary on its rows or
  by a key bound one short, and a key-mask load masked by the query-row
  boundary or by a key bound one short ran as the canonical kernel. With a
  sequence length that is not a whole number of tiles they returned other
  gradients; on whole tiles every such mask is all true and the result
  agreed. Each now refuses, whatever the length. A key mask broadcast along
  the rows in the score select already refused.
- Count the keys the source loop visits on the biased fp32 route, and no
  others. The template walks 64 keys at a time. Over a source loop of 32 keys
  and an odd number of key tiles it gave the 32 keys past the last source
  iteration the sentinel score, where the source never visits them. With a
  finite runtime sentinel, a row with every key masked, or a sentinel that is
  not very negative, returned other values. Those keys now score negative
  infinity. A source block of 128 keys or more visits whole 64-key tiles past
  the template's last one, each a padding key that carries the sentinel; the
  template now walks them. Every power-of-two key block therefore stays on
  this template.
- Fit the tiled biased template into Metal's 32 KB of threadgroup memory. A
  half or bfloat16 source whose key block is not 64, and any source with
  strides the 64-key template cannot take, runs on the tiled template, which
  walks the source's own block. Its scores and accumulator are fixed by the
  source; the staged head-dim slice now narrows until the working set fits.
  bfloat16 at head dim 64 or 128 with a 64-key block, half types at head
  dim 32 or 64 with a 128-key block, and float32 or half tensors with a
  head-dim stride other than 1 at a 64- or 128-key block failed at pipeline
  creation and now compute.
  Where no slice fits (head dim 128 with a 128-key block, a 256-key block)
  the kernel refuses by name with the size it would need.
- Accept a boundary written bound first (`N > m_idxs`, `seqlen_k > kn`) on the
  load, store and select masks of the biased and varlen routes and on the
  load masks of the MLA route. These spellings refused before. The MLA causal
  select written bound first still refuses.
- Name the proof that failed when a softmax kernel with several dots refuses
  after its four pointer roles resolved. The message ended at "refusing
  rather than guess"; it now adds the unproved field. A kernel refused before
  its roles resolve keeps the shorter message, which names no attention role.
- Refuse by name a dense attention kernel whose K, V and Out tensors do not
  share one float element type, or whose Q is bfloat beside half tiles. These
  reached the Metal compiler and failed there. A Q of another float type that
  the template casts still runs.

### Outputs that share memory with an input

- Refuse an attention launch whose output shares memory with another tensor
  argument. A source program loads its rows, keys and values before it
  stores, so such a launch is defined in the source. The attention lowerings
  do not keep that order, and two shapes returned other values in 40 of 40
  launches on an M4 Max. A dense
  head dim 64 launch, float32 or half, with a 64-row block and the output
  over the keys or the values runs as two 32-row passes, and rows 32 and up
  were computed from rows the first pass had stored. The dQ backward template
  stores delta before it loads the log-sum-exp, so with delta over that
  tensor every dQ cell was wrong. A biased launch with the log-sum-exp over
  the output's first column was wrong as well. The two biased source replays
  keep the order and keep computing in place: each runs one threadgroup per
  program with a barrier between its last load and its stores. The wide row
  replay does so by construction; for the narrow one the lowering checks the
  emitted shader and publishes the output if the text does not show it. Every
  other forward attention kernel (two or more dots, an exponential and a max)
  and every backward attention template now refuses, by argument name and
  before anything runs, when a tensor it stores through overlaps another
  tensor argument. This is a change for in-place launches that computed
  before, such as the output over the queries: on those routes most in-place
  probe launches returned the source's result before this work, and they now
  refuse together with the ones that did not. A two-dot kernel with
  an exponential and a row max that is not softmax attention is refused in
  place too. Inputs that share memory with each other, such as one tensor as
  query, key and value, are not compared and still compute. A two-dot kernel
  without a max stays on the generic lowering and is not refused.
- The attention refusal does not go by the views as passed. A kernel addresses
  its tensors through its own stride and length arguments, which need not be
  the views' strides or extents, so cells the views keep apart are not cells
  the kernel keeps apart: slot views `buf[:, i, :]` passed with contiguous
  stride arguments returned 1117 to 1364 wrong cells of 4096, and an output
  placed inside rows that a shorter view's arguments address was computed from
  overwritten inputs. An output whose storage is different from every other
  tensor argument's computes. For an output that is a view of another tensor
  argument's storage, the launcher bounds the cells the source reaches through
  each tensor argument, one range of cells for each, from the launch's own
  strides, lengths and grid. The launch runs when every range lies inside its
  storage and the range of each stored argument meets no other argument's, so
  tensors laid as separate blocks of one storage compute. It is refused when
  that is not shown, and an argument whose storage cannot be read counts as
  shared. This still gives up launches that were correct. Tensors interleaved
  in one buffer, such as query, key, value and output as slots of one packed
  buffer or a log-sum-exp column beside its output, share no cell, but their
  ranges meet. A varlen launch places its rows by the values of a tensor,
  which the launcher does not read, so its cells have no bound. Write those
  outputs to a tensor of their own. The matmul and quantized routes compare
  byte ranges where the route knows how many elements it addresses through
  each operand, and whole storages otherwise.
- Refuse an ordinary launch whose lanes are not proved to keep the source's
  order. A Triton program finishes each operation over every lane before the
  next one starts. The ordinary shader keeps each lane's own order only, so
  where two lanes of one program reach the same element and one of them
  stores it, the result depended on which lane ran first, with no exception.
  On an M4 Max an in-place reverse that stores through the reversed index was
  wrong in 100 of 100 launches at 256 elements and above, an in-place
  transpose at 16 by 16 and above, and a counter cell stored by every lane
  counted more than once at large blocks. 0.2.0 and 0.3.0 lower these kernels
  the same way. Before a launch is submitted, the launcher now groups the
  tensor arguments by the storage they share and looks, per group, for a
  proof that no two lanes of one program meet at an element one of them
  stores. Without a proof it raises `MetalNonRecoverableError` with a message
  that starts `lane order:` and names the arguments and the reason; nothing
  is submitted and no tensor is changed. There is no switch that turns the
  check off. A store of one element is now done by one lane. A launch with
  more addresses than one comparison takes is compared in runs of programs,
  each cell within its own program. `x[:, 1:] += y` as `torch.compile`
  writes it is admitted this way from 1500 by 1501 up to 8192 by 4096. The
  first launch of a size pays for the comparison (measured once each, with
  other work on the machine: 156 ms and 1586 ms) and the answer is kept for
  later launches. A launch that would need more than 16 runs is refused,
  which that kernel reaches at 8192 by 8192.

  An access under a branch is counted whichever way the branch goes, with one
  exception. A condition made of the launch's integer scalars and constants is
  one number for every program, lane and trip. The launcher computes it as the
  shader does, wrapping at the type's width, and leaves out what is under the
  way the launch does not take: a kernel that reverses a row in place only
  `if S > 0` is refused where `S` is positive and computes where it is not. A
  scalar that a counted loop advances by one fixed amount each trip is read
  the same way. A condition that follows a loaded value, the program id, a
  loop's trip, any other carried value, a float scalar, a division by zero or
  the smallest integer divided by -1 is not read, and what is under it is
  counted in.

  This contains the wrong results; it does not make these kernels compute.
  What is refused:

  - A kernel that permutes or shifts a tensor in place within one program:
    reverse, transpose, shift by one, a gather through an index tensor, two
    lanes per cell (`i // 2`, `i % M`). Some of these returned the source's
    result at small sizes before and are refused at every size now. Write
    the result to a separate tensor.
  - A cell that every lane loads and the program also stores: a scale factor
    the block reads and the program then increments, a counter kept in cell
    0 of the output, a column of per-row values updated beside a full block
    that reads it. These are refused with separate tensors too, because the
    cell itself is what the lanes share.
  - A block that is stored and then read back through another index later in
    the same program, such as a sum over neighbours after a store, at 256
    elements and above.
  - Two different atomics on one tensor in one program (add then max).
  - A product written as one `tl.dot` without a loop over K, with its output
    over an operand. It was wrong at 64 by 64 and right at 16 and 32 before;
    it is refused at every size. The looped matmul in place is covered under
    "Matmul source domains" and still computes for one program.
  - An exchange, a compare-and-swap, or an atomic whose returned value is
    used, when two lanes reach one cell from different trips of a loop. The
    source orders the trips; the shader does not. Where the lanes of one
    such operation meet on a single trip, or outside a loop, the launch is
    not refused: the source fixes no order among the lanes of one operation,
    and the shader's is not always the interpreter's lowest lane first (4 to
    8 of 60 launches differed at 256 lanes, none at 32 or 2048). Part
    of these returned the source's result in every probe launch before. In a
    `while` loop the trips are not known before the launch, so they compute
    only where the address is the same on every trip and keeps each lane to
    cells of its own; otherwise they are refused, also when the launch makes
    one trip. Where two lanes met on different trips in a kernel of more than
    1024 lanes these were wrong before. An integer atomic add, and an atomic
    max or min, whose returned value is not used still compute, in a loop too.
  - Two float `tl.atomic_add` operations, as two statements or as one in a
    loop, that reach one cell from different lanes. The shader applies them
    lane by lane, so the cell received the source's terms in another order,
    and the order of float additions is their rounding: `1e8 + 1 - 1e8` gave
    1.0 where the source gives 0.0. The check reads addresses, not values,
    so these are refused also where no order rounds: a float histogram or
    scatter-add accumulated over the trips of a loop is refused. Nor does the
    check evaluate a mask made of loaded values, an index that is loaded, or
    a branch on anything but the launch's integer scalars. Two float adds
    that only such a mask or branch keeps to one lane per cell are refused,
    with a reason that names the mask or branch; adds through a loaded index
    are refused because the address is not known before the launch. An add
    under a branch on integer scalars is counted where the launch takes the
    branch and left out where it does not. In a `while` loop a float add
    computes only where its address is the same on every trip and keeps each
    lane to its own cell. These returned the source's terms, in one order or
    the other, before: with values whose sum depends on the order the result
    was wrong at 256 lanes and above, and in part of the launches at 64 and
    128 lanes. Still computing: integer adds, which are exact in either
    order, a float add of each lane to its own cell, and one float add
    outside a loop, whose lanes the source does not order. Accumulate into
    integers, or make each trip its own launch.
  - A store over a block narrower than the kernel's widest block when two of
    its elements reach one cell, or when it shares a tensor with a
    block-wide store: a narrow store inside a loop whose trips reach one
    cell from different lanes, a narrow copy one cell along its own storage,
    a narrow store before or after a block-wide store into the same tensor,
    and a narrow load of cells a block-wide store of the same program wrote.
    The last three returned the source's result in every probe launch
    before, at every size tried.
  - Launches the check cannot follow although the shader matched the source
    in every probe launch. With separate tensors: a store inside a loop
    whose trip count is loaded from memory; ten nested calls of functions
    that are not inlined. With shared memory: a pointer chosen
    by a condition, a `tl.where` or a `while` loop; an address computed
    through floating point or loaded from the shared storage itself (a
    histogram whose bins lie behind its indices in one storage); a cell
    replaced by the sum of its own block; a
    masked store beside an unmasked load of the cells it stores, where the
    only lanes that read those cells are lanes the mask turns off.

  Still computing in place: a kernel in which each lane loads and stores its
  own elements, also over a block narrower than the kernel's widest
  (`x[:64] *= 2` beside a 128-wide store to another tensor), views of one
  storage that a mask or the index arithmetic keeps apart, and an
  accumulation of per-row totals into the output (`out[row] += sum`).
  Views of one storage with different element sizes, such as a `float32`
  input and an `int16` output in one buffer, are compared byte by byte and
  compute when they share no byte that one of them stores. A byte that views
  of two element sizes both reach, one of them storing it, is refused
  whichever lanes reach it, and so is a host array that does not start at a
  multiple of its element size.
  Conflicts between
  different programs of one launch are races in the source and are not
  looked at. The check is not applied to a route that replaces the kernel
  with a template (matmul, quantized matmul, attention), which has the
  refusals above, or to a launcher built from metadata without an access
  plan. On the opt-in C++ route only launches provable from the source alone
  are admitted.

### Compatibility

- Support the Xcode 27 toolchain. On it, 0.3.0 refuses every kernel with
  `cannot establish Metal compiler/SDK identity: unsupported native load command 0x27`:
  its Mach-O dependency parser rejects the `LC_DYLD_ENVIRONMENT` load command.
  0.4.0 accepts that command only in a main executable and only for
  `DYLD_VERSIONED_FRAMEWORK_PATH`. Any other embedded variable still refuses.
- Qualify the Metal language-standard probe by creating a compute pipeline, not only
  by loading a library (GitHub issue #12). Where a standard's library loads but
  pipeline creation fails, 0.3.0 selected that standard and kernels then failed.
  0.4.0 moves on to the next older standard.
- Keep the declared Python 3.10 support. Two cleanup paths added during 0.4.0
  development called Python 3.11 APIs (`sys.exception()` and
  `BaseException.add_note`). They now use `sys.exc_info()` and add the note only
  when the exception has `add_note`; the original exception still propagates.
  Python 3.10 itself was not run: the check is a static source test and unit tests
  on Python 3.14.
- The package licence expression is now `MIT AND BSD-3-Clause`. The `applegpu`
  static decoder is vendored as `triton_msl._vendor.applegpu` under its BSD-3-Clause
  licence, so the disassembly helper works from an installed wheel. 0.3.0 looked
  for a repository-relative `third_party/applegpu`, which a wheel does not contain,
  and reported the decoder unavailable.

### Start-up identity record

- In 0.3.0, every new Python process recomputed the compiler, SDK and framework
  identities from full file bytes before its first kernel. 0.4.0 keeps that full
  computation for the first process, whenever anything it checks has changed, and
  on request. Otherwise a later process validates a small per-user record and
  reuses the recorded identity. The record holds the identity the full computation
  produced, so cache keys are the same in both modes unless a change listed
  below goes unseen.
- `TRITON_MSL_IDENTITY_RECORD=0` forces the full computation in every process and
  reads or writes no record. `TRITON_MSL_IDENTITY_RECORD=refresh` recomputes and
  rewrites the record. The default is on.
- Records live in `~/.cache/triton_msl/identity-v1-uid<uid>/` unless
  `TRITON_MSL_IDENTITY_DIR` is set. They are private to the user: the directory is
  mode 0700 and the files 0600.
- `python -m triton_msl.tools.identity status|refresh|clear|verify` manages the
  records. `verify` runs the full computation and compares. `refresh` and `verify`
  reach the framework record of the command's own process only; for an application,
  use `clear` or run it once with `TRITON_MSL_IDENTITY_RECORD=refresh`.
- The record is used only where file metadata is on APFS and the record folder
  carries no access control list; otherwise the full computation runs.
- The record relies on file metadata, so it does not see what only the full
  computation sees: a byte change that leaves a file's size, times and inode
  unchanged (for example a write through a memory mapping held open across the
  change); a change to an SDK file other than the SDK's version files; a privileged
  change to file times or to the raw device; and a record altered by the same user.
  `docs/CACHE_CONTRACTS.md` has the details.
- Start-up, measured on one M4 Max with one workload (a new process makes its
  first `@triton.jit` vector add; compiled kernels cached; 24 processes each way):
  the median first call took 7.03 s with the record switched off and 0.96 s with
  the record in use on the native wheel, and 7.06 s and 0.93 s on the pure wheel.
  The toolchain files were already in the operating system's file cache and the
  machine was doing other work. The record does not apply to the first process,
  which does the full computation and writes it; in four single observations the
  first call of a first process took 7.73 to 14.82 s. Other machines, other
  workloads and a cold file cache were not measured. The table is in
  `docs/releases/0.4.0.md`.

### Fixes

- A signed 32- or 64-bit sum, difference or product that leaves its type's range
  now wraps, as Triton's integer arithmetic does. The generic lowering wrote these
  as signed shader arithmetic, and the Metal shader compiler reads a signed sum as
  one that never overflows: `x + 1 > x` was true where `x` is the largest int32,
  `(x * 3) // 3` was turned back into `x`, and a branch on `S + 1 > 3` was taken
  for the largest `S`. Nothing was refused or reported. The sources of 0.2.0 and
  0.3.0 write the same shader text. These operations are now computed on the
  unsigned type of the same width, which holds the same bits and wraps, and the
  result is converted back. Narrower integers and unsigned ones were right before.
  The text of every generically lowered shader with integer arithmetic changes.
- Opt-in C++ route (`TRITON_MSL_USE_CPP=1` with a separately built extension):
  a masked `tt.load` was lowered to an unconditional load followed by a select, so
  inactive lanes could read outside their allocation. The load is now control
  dependent on the mask. The typed-pointer rewrite on that route now keeps
  `volatile` on loads; 0.3.0 dropped it.
- Host-copy route: if submission or waiting raised, 0.3.0 dropped its references to
  the launch's buffers and owners while GPU work could still be pending. 0.4.0 retains
  them until the GPU's access has provably ended: the failure came before any commit
  attempt, or the command buffer reports completion or error. Until then, later host
  launches raise `triton_msl.errors.PostSubmitError`. See "Host dispatch failures
  with uncertain completion" in `docs/USAGE.md`.
- Host-copy route: pooled buffer leases are released when preparation raises.
  0.3.0 released them only on the normal path.
- Kernels with more than 31 arguments now run on the host-copy route when the
  packed argument layout is certified by the compiler. 0.3.0 refused them there; the
  MPS route already supported them.
- Compiler and probe subprocesses run with a fixed working directory and absolute
  operands. 0.3.0 ran them in the application's working directory.
- A changed or unusable compiler selection raises the typed
  `triton_msl.errors.MetalToolchainSelectionError`.
- Framework discovery now retains the import machinery whose `id()` values key its
  memo. 0.3.0 did not, so after that machinery was replaced and an address reused,
  the memo could return stale results.
- Loaded Metal pipelines are owned by the loader that acquired them and released
  after its matching unload. 0.3.0 kept them in a process-global table for the life
  of the process.
- A masked store of one value per row of a two-dimensional block ignored its mask.
  `torch.compile` writes a reduction over the last axis this way, and when the row
  count is not a multiple of the row block the last program wrote the rows past the
  end, outside the output, with no exception. The store now carries the mask and is
  done by one lane per row.
- A masked load or store inside a function that is not inlined
  (`@triton.jit(noinline=True)`) was lowered without its mask: the store landed where
  the source stores nothing, and the load read memory where the source reads `other`.
- Host tensors: memory written only through a function that is not inlined, or by
  `tl.atomic_cas`, was not copied back after the launch, so a cpu tensor came back
  unchanged with no exception.
- A kernel that needs one thread per lane over a block of more than 1024 lanes ran
  its first 1024 lanes only and left the rest of the output as it was. Metal runs at
  most 1024 threads in a threadgroup. A kernel that needs them only for its
  `tl.debug_barrier()` now runs a loop over its lanes, as the same kernel does without
  the barrier. The barrier then orders threads and not lanes, so a launch in which two
  lanes of a program meet at an element one of them stores is refused before it runs,
  for its lane order, and a launch whose lanes keep to separate elements computes
  (2048 to 8192 lanes were tried, between separate tensors and in place). A kernel that
  needs a thread per lane for another reason beside its barrier (a running sum, a flip,
  a sort, a transposed or gathered tile) is refused by block size when it is lowered,
  also where its mask keeps every active lane below 1024. Use a block of at most 1024
  there.
- Quantized weight-only matmul (int8 and int4, GEMV and GEMM): the kernels these
  routes accept have no masks, so as written they compute whole blocks, and every
  route clipped the result at the `M`, `N` and `K` it was passed. With sizes that are
  not whole blocks the output differed from the source: rows or columns the source
  stores were left untouched, and a last `K` block was cut short. A launch now
  computes the whole blocks the source addresses when they lie inside the tensors'
  storages, on the fast template where only an extent changes and on a scalar
  template otherwise, so it stores whole tiles past `N` or `M` as the source does.
  When the whole blocks reach outside a storage the program is undefined as written
  and the launch keeps the clipped template, provided the cells at `M`, `N` and `K`
  are inside every storage; otherwise it is refused. Inside a storage is measured in
  the storage's own bytes on every route: an expanded one-element scale, zero point,
  input or weight has more elements than cells and was measured by its element count
  on the per-group and symmetric routes, which would have launched the template on
  cells outside the tensor. A negative stride argument is refused on these routes. A
  per-group kernel whose `K`
  block spans more than one scale group is refused by name: the source takes one
  scale per block and the templates take one per element. A refusal on these routes
  now names its reason.
- A value reduced along the rows of a two-dimensional block was read wrongly by a
  later operation. `tl.sum(v, axis=1)` of an (R, C) block is held with row r's result
  on lane r*C. A second reduction, `tl.argmax` or `tl.argmin` of the row totals, the
  totals laid along the other axis, an outer product of two such vectors and a gather
  through each row's largest index took lanes 0 to R-1 as the R results, so they read
  the first row's total C times and returned a wrong value with no exception. Every
  one-dimensional value now records how the lanes hold it, and each later use takes
  one lane per row. Where that is not proved the kernel is refused when it is
  lowered: `tl.argmax`, `tl.argmin` and scans over a block whose rows are spread over
  more lanes than the block, a broadcast whose users read different lane spacings,
  and a pointer converted from such an index that reaches anything other than a load.
  A three-dimensional block reduced to two dimensions and stored uses the
  three-dimensional template only inside the address form it was proved for.
- A store or an atomic over a block smaller than the kernel's widest block (a 4 by 8
  tile beside a 32 by 32 one, 64 elements beside a 128-wide copy) ran on lanes that
  did not hold its elements. A narrow tile copy stored wrong cells, a narrow
  one-dimensional store through `j % H`, `j * 2` or a loaded index wrote cells the
  source never stores, and a narrow atomic in a kernel of more than 1024 lanes was
  repeated for every 1024 lanes of the block, all with no exception. The lowering
  now proves, from how each index, value and mask is held on the lanes, that one lane
  per element holds that element's address, value and mask, and guards the access
  with it. Where the proof does not reach, the kernel is refused when it is lowered
  and the message names the operation, the block and the reason. Refused this way
  where the result was wrong before: a one-dimensional running total laid along the
  rows of a narrow tile, and a narrow tile copy into an output that does not start
  on a 16-byte boundary, at some tile shapes. Refused this way although the lowering
  before it returned the source's result in probe launches: an atomic over a narrow
  block whose returned value is stored, and a tile indexed through `tl.reshape` of a
  range in a three-dimensional kernel.
- A load over a block smaller than the kernel's widest block ran on every lane. On
  the lanes past its own range the index is one the source never forms, so the load
  read up to a block's worth of elements beyond the ones the source loads, beyond the
  tensor when the tensor ends there, and a second load through a loaded index read
  wherever the stray value pointed. Such a load is now made only on the lanes whose
  range indices lie inside their ranges; an index written with a modulo is inside its
  range on every lane and needs no condition. Not covered: a load inside a called
  function that is not inlined; no launch that reaches one with a narrow range was
  found.
- A tile flattened with `tl.reshape(t, [R * C])` and stored through a one-dimensional
  range beside a wider block was stored by every lane of the kernel. Two to eight lanes
  met in each cell, and the lanes past the tile carried values of tile positions that
  do not exist, so the cells held wrong values with no exception: a 2 by 64 tile beside
  a 256-wide copy, a 16 by 16 tile beside a 512-wide one. The lowering now proves that
  lane i holds element i of the flattened tile on the tile's own lanes, and a store, a
  masked store, a store of the tile plus a one-dimensional load and an atomic add take
  those lanes. The stores that were wrong compute on tensors that start on a 16-byte
  boundary: tiles of 128 to 1024 elements beside blocks of 256 to 2048 lanes were tried,
  and of those the 128 by 1 tile is refused. Through a view that starts off that
  boundary some of these stores are refused instead (2 by 64 from 256 lanes on, 16 by 16
  from 512). The same proof covers a row reshaped to a column and a range reshaped to a
  row in a store. An atomic over a one-column tile is not covered by it: where the
  kernel's lanes are looped over, beside 2048 lanes or more and at some shapes beside
  1024, such a tile is held one row per group of lanes, the atomic read row `lane / N`
  on its first lanes and added the first rows again (an 8 by 1 tile beside 2048 lanes).
  The atomic now asks of its address, value and mask what a
  narrow store asks, and is refused when the kernel is lowered where they are not held
  element by lane; the 351 probe launches that were wrong are refused and the 506 that
  computed still compute. Refused when the kernel is lowered, where the
  result was wrong before: a flattened tile whose values are computed from its indices
  with no load. Refused as before at the shapes tried, with the reason: the same
  stores of a tile of 64 elements or fewer (`1-D store layout was not declared by its
  producer`), and a `tl.sum` of a flattened tile beside a block of 256 lanes or more
  (`would be accumulated with wrapped (repeated) elements`). The sum of a tile of up to
  64 elements beside a block of up to 128 lanes computes.
- A float16 or bfloat16 result was a float32 number until it was stored. The ordinary
  lowering holds every float in a Metal `float` register and narrowed a value only at a
  store or an explicit cast, so the second of two chained operations read an unrounded
  first one. `(x + y) - x` on float16 tensors gave the float32 remainder in 64 of 64
  cells where the source gives 0.0; 2048 with 1 added on each of eight loop trips gave
  2056 where the source stays at 2048; a `tl.sum` of 64 cells of 4000 widened to float32
  gave 256000 where the source's float16 sum is infinite; a comparison
  (`tl.where(x + y > x, ...)`) chose by the unrounded value; an integer converted to
  bfloat16 went by way of float32 and was rounded twice; a float16 constant was read
  as the float32 number its decimal names (`x * 0.1`); and `tl.math.fma` was rounded
  twice. All with no exception. Each result the source declares float16 or bfloat16 is
  now rounded to that type where it is produced: arithmetic, the math functions, a
  conversion from an integer, a constant, a loop-carried value, and the result of a
  reduction, a scan or a `tl.dot`. An operation the source promotes to float32 is left
  in float32, and a reduction, a scan and a `tl.dot` still add in float32 inside and
  round their result once. A float16 fused multiply-add is Metal's half one. A division
  in a kernel that holds 16-bit floats goes through a guarded quotient: the Metal
  compiler turned the narrowed quotient of a division by a constant into a half
  multiply by the rounded reciprocal, and 19,836 float16 values of `x / 3.0` differed.
  Refused when the kernel is lowered: a bfloat16 `tl.math.fma`, which Metal computes in
  float; an operation with several results that computes a narrow value by arithmetic;
  and a matmul that carries a float16 or bfloat16 accumulator over its K loop
  (`acc += tl.dot(a, b, out_dtype=tl.float16)`), which was run with one rounding after
  the last block and differed from the source's rounding per block in 555 of 4096
  cells of a 64 by 64 probe. The values of float16 and bfloat16 kernels with chained
  narrow operations change with this fix. The attention and matmul templates keep the
  float32 accumulators their sources declare.
- A loop-carried tile with more cells than the kernel has threads (32 by 64 in a group
  of 1024) is held in threadgroup memory and updated in place, and four things were
  wrong there with no exception. `a[:, None] - acc` and `a[:, None] / acc` were computed
  as `acc - a[:, None]` and `acc / a[:, None]`, in the loop and after it. A second
  operand of the tile's own shape was read as one value per row. The old cells stayed
  readable by name after they were overwritten. And a value computed one cell per
  thread from the tile (`tl.maximum(acc, 40.0)` after the dot) was never written back,
  so the next trip and the loop's result read the cells as they stood before: 4096 of
  4096 cells of a 32 by 64 probe. The update is now emitted with its operands in the
  source's order, and only where the IR shows the other operand is one value per row,
  the tile is made anew before every run of the update and nothing reads the old
  cells after it. Every other form is refused when the kernel is lowered, and the
  message gives the reason. Refused although it returned the source's result before,
  where the loop's result was then divided by a column: two updates through one
  broadcast column, `(acc * a[:, None]) * a[:, None]`.
- A float16 or bfloat16 atomic on a view that starts an odd number of elements into its
  storage updated the cell before the one named. Metal has no 16-bit atomic, so the
  cell is updated by a compare-and-swap of the 4-byte word that holds it, and the word
  was found from the element's index, which is the right word only when the tensor
  itself starts a word. Every add on such a view landed one cell early, on the device
  and on a host tensor alike, and a negative index became an unsigned one far outside
  the storage. The word is now the one the element's address lies in.
- A row-wise sum chosen by a branch, carried by a `while` loop or followed by a long
  chain of operations lost its place on the lanes, with no exception. A sum of an N by
  W / N tile in a W-lane kernel is held one row per W / N lanes, and where a narrow
  store or an atomic of N cells reads it a conversion moves it to one row per lane. The
  conversion asks where its source is held. For the result of an `if` or of a `while`
  loop nothing answered, and behind more than 256 values the search stopped: the rows
  were then taken from the first N lanes, which all hold the first rows. With 2 rows on
  1024 lanes both cells got row 0. The search now follows a branch to the values its two
  sides yield and a `while` loop to the value it starts from and the value its body
  hands on, and goes on to every producer. The rows are moved from the lanes that hold
  them, or the kernel is refused when it is lowered where the sources are held
  differently. The length of the chain no longer matters: the resolver stopped 24
  operations deep, so 25 additions after the sum were refused where 24 computed, and
  it now meets every value once (1, 25, 60, 130 and 400 operations compute alike).
- A loop that hands one carried value on in another's place gave both the same value,
  with no exception. `a, b = b, a` in a counted loop or a `while` loop gives each the
  other's old value; the carried variables were written one after another, so the
  second read the first as just written. A three-way rotation, `a, b = a + b, a` and
  `c = b; b = a; a = a + 1` went the same way; `a, b = b, a + b` did not. A yielded
  value that is, or reads, a carried variable written before it is now copied before
  any of them is written. A `while` loop that carries a vector of more elements than the
  kernel has threads did not compile; it is now refused by name when the kernel is
  lowered (a counted loop carries one).
- A `while` loop whose body needs the whole block of lanes was refused, or failed
  while its pipeline was built, where the same body in a counted loop computed. A value
  carried by a `while` loop was not counted among the values of its kernel, so a sum, a
  largest value or a flip of it inside the loop was refused with "native per-result
  metadata is missing for SSA value ...". And the exit of a `while` loop is written
  into the shader as a `break` at the head of the loop, which switched off the reuse of
  threadgroup arrays for the whole kernel: a flip of 1024 lanes kept twenty arrays,
  61440 bytes where 32768 are allowed, and the launch failed with a `RuntimeError` from
  pipeline creation. An exit that is the first thing its loop decides is now read as
  the test of a loop that tests before each trip. Every other `break`, `continue` or
  `return` keeps all arrays apart, as before. A generically lowered kernel whose arrays
  still do not fit is refused before a pipeline is asked for, as Triton's
  `OutOfResources` with the bytes required and the limit.
- The row-wise sort template stood for kernels that are not a row-wise sort. The
  template replaces the whole kernel with "load row r, sort it, store it", and it was
  chosen from a signature that other kernels fit. `tl.flip(v, 1)` of a 2 by N tile
  came back sorted instead of reversed. A sort whose load or store carries an offset
  or a program term read and wrote program 0's rows for every program of the grid. A
  mask on the load or store and arithmetic on the sorted values were dropped. All of
  these returned with no exception. The template is now chosen by a proof that the
  kernel is exactly one unmasked load, the network `tl.sort` writes and one unmasked
  store at the template's addresses; every other kernel goes to the ordinary
  lowering, which computes it or refuses it. The ordinary lowering runs a tile of at
  most 1024 elements, so a larger sort that is not that plain form is refused when the
  kernel is lowered, and the message gives that reason. Four such forms returned the
  source's result through the template before, at 64 by 32 and 512 by 8: a store that
  converts the sorted values, a load of the columns in reverse, a mask that is true on
  every lane, and a sort of a sort. The template also runs the source's own
  network now. It used to compare signed, so unsigned values above the signed range
  were sorted as negative ones, and where neither of a pair is greater (a NaN, a zero
  of each sign) it kept both values where the source leaves one value in both cells.

### Additions

- Source-program export: an installed `triton-msl-export` command (also
  `python -m triton_msl.tools.export_source_program`) writes the Metal source, an
  optional metallib and a manifest for a proved saved-LSE attention program, for use
  by an external Metal driver. The 128-thread cooperative lowering is opt-in
  (`--cooperative`, or `MetalOptions(experimental_source_program_cooperative=True)`).
  The scalar source-program emission is not opt-in: a kernel whose arguments are
  seven pointers followed by four scalars and that uses `tt.dot` with `exp2`, `log`
  or `maxnumf` enters the source-program proof by default. If the proof succeeds,
  the regular Python launcher refuses to run the result with
  `MetalNonRecoverableError`. If it fails, compilation refuses with no fallback.
- Generic backward stride lowering for the reporter's triangle-attention backward
  kernels, which use derived contiguous strides and boolean masks.
- Add seven dense attention fragment compositions behind the new default-off
  `MetalOptions(experimental_fa_fragment_lowering=...)` option (default `"off"`),
  with source and precision admission checks and a public GPU test for each.
- Admit full-storage, zero-offset dense rank-two fp16/fp32 transposes to the direct
  MPS route. Offset, partial, overlapping and gappy transposes keep the host route.
  Four transposed-B matmul cases fall from 1.08–1.26 ms to 0.21–0.26 ms per public
  call plus sync on an M4 Max in both installed modes, against the November 3
  development wheels. This is a descriptive workflow result, not a release
  performance qualification, and it was not compared with 0.3.0.
- Tile-aligned K-loop matmuls with an fp32 output, contiguous or with a transposed
  fp32 B, launch a sub-tiled direct entry: each 32-lane SIMD group computes whole
  32×32 sub-tiles with matrix fragments read in place, at 32 to 128 threads per
  group. The transposed-B form no longer stages its operands for these launches.
  Unaligned extents, an output that shares storage with another pointer argument,
  a grid with more tiles than the output holds and other layouts keep the ordinary
  entry. See `docs/USAGE.md`.
- The strided scalar matmul entry for tiles larger than 32×32 now loads its whole
  tile before it stores, so a single program whose output is one of its operands
  no longer reads cells another thread has already overwritten.
- New default-off options: `TRITON_MSL_HOST_EXTENT=1` (host storage extent
  borrowing), `TRITON_MSL_CPP_SCALAR_CFG=1` (LLVM scalar cleanup on the opt-in C++
  route), `MetalOptions(experimental_cooperative_attention=True)` and
  `MetalOptions(experimental_tiled_k_layout=True)`. See `docs/USAGE.md`.
- Compose four host fast paths: guard reuse, shared binder census, scalar
  eligibility and packed-key reuse. Installed correctness passes. In the November 9
  nine-case comparison with the November 3 development wheels, the native wheel's
  mean call time is higher in all nine cases (+1.47% to +10.07%) and the pure wheel
  is mixed (−8.14% to +13.23%). Two workers per arm: the panel establishes no
  general speedup and attributes the difference to no cause.
- Carry forward the pure-Python environment-snapshot and tensor-classification
  changes.

### Performance

- An October 18 comparison of an October 11 installed development build against
  released 0.3.0 resolved none of its 36 cases as faster or slower. It used an older
  retained Metal compiler, because 0.3.0 cannot compile with the current toolchain.
  The final 0.4.0 build has not been timed against 0.3.0.
- The per-launch host cost over 0.2.0 that shipped in 0.3.0 remains. In the same
  comparison, 10 of 36 cases were resolved as slower than 0.2.0 and the rest were
  unresolved; on small single-launch workloads the paired excess was about 18–24 µs
  for the native wheel and about 45–54 µs for the pure wheel.

### Validation

Run on November 14 on local builds of the release source (one M4 Max, macOS 26.7,
Xcode 27.2, CPython 3.14):

- Project suite with the compiled helpers: 7,386 passed, 62 skipped, 0 failed.
  From a tree without the helpers: 7,042 passed, 119 skipped; the 287 tests that
  need a compiled helper failed (127) or errored at setup (160) there.
- Upstream Triton `test_core.py`: 5,780 passed, 3,562 skipped, 0 failed of 9,342,
  with each wheel's package files.
- Installed device-free selection: native 1,480 passed and 6 skipped; pure 1,388
  passed and 98 skipped; 0 failed.

### Not tested

- M1, M2, M3 and M5 hardware; all testing used one M4 Max.
- Python versions other than CPython 3.14. The native wheel is built for 3.14
  only, and the pure wheel was not run on Python 3.10 to 3.13.
- Older macOS releases and toolchains.

## 0.3.0 - 2026-09-16

Supersedes unpublished rc1–rc4 candidates. Fresh validation on the formatted 0.3.0 candidate at `73ba428` records 6,079 project passes / 16 skips, exact 9,342-node upstream identity (5,780 passed / 3,562 skipped), installed acceptance 1,468 native / 252 pure with no failures/skips, and four unchanged M4 Max floors (packets 906–909). Final-version artifacts are freshly built, reproduced from the sdist and rebound to these tested runtime payloads. The full warm comparisons in packets 879/890/895 and the vendor receipts remain pre-format measurements; they were not rerun on formatted bytes, and no warm alert is cleared.

The formatting pass changed 228 Python files: 226 syntax trees were identical excluding source locations, and two differed only in docstring whitespace. The reviewed codegen corpus retained 38 identical emissions (35 distinct shaders) and 29 identical refusals; this is bounded corpus evidence, not a proof for every kernel. Two test-only source-mutation fixtures were repaired to match the formatter's spacing without changing their counterexamples or assertions. Fresh gates were required because code objects and cache identities changed. The supplemental installed flip witness's stale producer-hash pin was rebound to the AST-identical formatted test file, then the complete installed selections reran.

The formatted cold subset computes correctly for add, compiled MLP and GPT-2 in public-main, native and pure arms. Native observations are 9.48 s / 10.02 s / 5.13 s versus main's 0.63 s / 0.86 s / 2.81 s. These are single first-specialization observations; GPT-2 follows MLP in the same worker. Imports/model construction are outside the timer, and the full warm-matrix timing loops were omitted. They do not establish a formatting-related speedup or clear the cold-start regression. The complete scope and pure-wheel observations are in `docs/RELEASE_CANDIDATE_LIMITATIONS.md`.

The 814→871 audit repaired codegen admission, native type ownership, transpose orientation, reduction ordering, partial-K extent replay, flip detection/side effects, runtime ABI/state checks and diagnostics; unsupported forms now refuse where a faithful bounded lowering is not proved. Exact scopes are pinned in the capability register. M1 remains best-effort post-merge and the optional C++ route remains deferred/off by default. Performance costs remain open and unaccepted.

- Requalify the attention comparison on the current M4 Max stack (890): 84 native/pure
  cases pass the declared checks. Native wins all 36 dense medians against current SDPA,
  but takes 23–44% longer than MLX; historical MLX near-parity is not sustained. At D128,
  batch 2 / 8 heads / N1024–4096, fp16 full is 1.94–2.18× SDPA and fp32 full
  1.36–1.41×; pure is separately 1.18–1.50× and 0.94–1.16×. Pure has 34/36 dense
  median wins and substantial block variation. These are API-call measurements, not a
  recreation of the August stack or a clearance of main-relative regressions. See the
  current-comparator and same-stack attribution sections in the release limitations.
- Add a same-stack main/final follow-up (895): 28 backend cases pass; native final/main
  paired-block latency ratios are 1.085–1.291 on 12 dense D128 cases and 1.125/1.131
  on full/causal MLA N2048. Main already trails current MLX on the dense medians;
  the final backend adds a measurable cost. Two long-context dense comparisons have
  blocks crossing parity. This isolates a current-stack backend difference, not its
  individual code-hunk causes or the entire historical August-to-current change.

- Record candidate-bound A40/MI300X source-portability checks (2026-09-15): all 11 required
  exact-byte predicates pass across Metal/native, Metal/pure and both remote backends.
  The original default-precision runs remain 14/16 required numerical passes per remote
  backend: two attention rows miss the unchanged oracle tolerance. Two additive
  explicit-IEEE attention contexts pass on all three vendors at the same 3e-5 absolute
  and relative tolerances; largest cross-vendor difference is 1.79e-7. This corrects the
  validation's precision contract, not production defaults or tolerances. Original failed
  results remain failed and are not replaced by the companion results. See `PORTABILITY.md`.

- Adopt the explicitly scoped option-A saved-identity path: recognized unchanged-state hits
  use callback-free reads and omit key/descriptor/metadata observation callbacks and their
  associated Python audit hooks, including side effects. `TRITON_MSL_IDENTITY_FAST_PATH=0`
  runs the complete evaluator on every invocation, including after a warm record. Supported
  between-call policy/provider changes remain checked; neither public-launch validation nor
  argument/ABI validation or immediate assertion reporting is removed.
- Cache only the exact-dictionary/exact-string-key predicate with a public CPython dictionary
  watcher; mutations dirty it, deallocation removes it, and unavailable watcher slots or full
  tables retain the original scan. Correct multi-instance invalidation and bind the live
  environment instance dictionary before inspecting its saved children. Native modules must
  respect watcher-ID ownership; clearing a foreign ID can defeat invalidation without detection
  on the cached-hit path and is outside the supported contract.
- Cache immutable probe-schema/key-domain preparation without caching dictionary values.
  Bound certificates by watcher epoch/context/live size and preserve all value reads. Reclaim
  obsolete cache-only programs on a miss before evaluator inputs are read; never run cleanup
  on a hot hit. This fixes ordinary rebuilds exhausting the 16-slot cache. Held/cyclic programs
  conservatively retain slots and scan if necessary. Component screens save about 0.5–0.6 µs
  per validation; no workload alert is cleared by those screens.
- Ship three CPU host helpers in the CPython 3.14 macOS 15+ arm64 wheel: validation, packed
  copying and binding. The pure wheel contains none. Missing and discoverable-broken helpers
  retain distinct fallback/error behavior. Validation/packed source builds support 3.13/3.14;
  the binder is 3.14-only, and no runtime-qualified 3.13 native wheel is included.

- Complete attempted-submission tracking across nested dispatch helpers, preventing fallback
  replay after an uncertain or failed invocation. This supersedes the earlier hook-only repair.
- Implement supported retained device assertions with a launch-local error flag, uniform stop
  before guarded accesses, and host-visible failure before returning results. Unsupported forms
  still refuse; assertion-bearing launches synchronize to inspect the flag.
- Record exact 871 native and pure-Python comparisons against main `182c1820`: all 30 required
  candidate rows source-correct versus 18 correct / six wrong / six refused on main. Native
  retains 13 alerts above 10%; pure retains 14; KDA prefill is a separate 1.1016× alert. Separate tables and µs/request accounting are
  in the limitations document. No ratio is assigned against wrong/refusing main; no residual
  cost is accepted or declared irreducible. Cross-session ratio changes do not isolate code
  effects. The varlen comparison includes a changed retained route, not a same-route speedup.
- Reduce the per-validation key/probe scan overhead (not the seconds-scale fresh-process toolchain/framework content-hash inventory) and certify standard top-level namespace-path
  no-op recalculation, including real MLX imports. Changed or unsupported namespace state
  retains the complete evaluator. This is a validator recovery, not full MLX qualification.
- Batch nonfinite-safe scratch rescaling on the half-input/default-float attention path
  without reintroducing diagonal multiplication or skipping scalar arithmetic. Other
  precision paths retain their existing emission. Full-call results remain workload-specific.
- Disclose the historical, pre-P1 MLA length-context measurements separately from retained
  exact871 N=32 results. Completion-wait timing is not a GPU timestamp; the historical
  longer-context observations do not establish a current exact871 larger-N residual.
- Explicitly qualify the shipped half-accumulate docstring's historical speed/error numbers
  in README and limitations: unvalidated for this candidate, no universal error bound,
  and no new opt-in-mode credit from the default-float P1 tests. Runtime text is unchanged.
- Preserve the separate historical latency experiment versus earlier candidate 439: **+15.1%** for the
  standalone retained-assertion workload and **+11.9%** for GPT-2 small. Both are open performance
  alerts in that experiment; the other nine tested workloads were within ±1.2%. Those figures
  are not the current main-relative summary. These local paired comparisons
  do not qualify historical throughput/speedup claims. GPT-2 has checks on **2 of 43** backend
  launches. Later diagnostics separately identify host-validation and immediate-observation
  components; the historical whole-call figures are not an additive per-kernel decomposition.
- Recover distinct broadcast index expressions, including comparison/select-derived coordinates;
  retain recursive rejection of unsupported assertions in callees.
- Recover canonical grouped and N-fastest flat-grid matmul coordinates without replacing the
  source grid or silently computing unlaunched tiles.
- Bind runtime float scalars with their declared ABI. Recover issue #11's fp32 32×32 K-loop
  matmul with runtime alpha/beta and column bias by replaying the source epilogue.
- Recover PR #7's widened-P/V decode spelling at 8-/16-row tiles, keeping query and key bounds
  distinct, including folded Q=1; preserve Q's fp16 scale-rounding point. Add source replay for
  bias-without-LSE at 32×32 tiles, independent lengths, causal masking and fp16/fp32 inputs.
  These are bounded source contracts, not arbitrary attention spellings or tile combinations.
- Make the optional value-head dimension mean symmetric attention consistently when absent.
- Strengthen exceptional-value tests, distinguishing exact classification from Welford's
  order-robust admissible bounds; no universal error or exceptional-value guarantee is claimed.

- **Fix a silently-wrong generic two-dimensional reduction present since 0.2.0.** The threadgroup
  scratch allocator previously merged arrays by textual first/last use and could place a reduction's result on
  its own input array; reducers in later SIMD groups then overwrote rows that reducers in the first
  SIMD group were still reading. On one M4 Max this produced wrong rows in roughly 2% of launches for
  row-softmax and row-sum tiles with more than 32 rows (for example 128×8 and 64×16), on the 0.2.0
  source and on the earlier rc4 candidate alike. Results are now held in registers and written after
  a barrier on both reduction axes, with emission pins for both. Attention kernels with 32-row tiles
  (one SIMD group of reducers) were not observed to fail; larger generic axis-1 reductions
  could be exposed when emitted with this alias pattern. The earlier rc4 artifact is superseded for this reason.
- Refuse MLX direct extraction when the returned shader still references original thread-parameter
  names that the generated preamble does not supply (issue #2's direct-extractor escape). Quoted text
  and comments are excluded from the check; the diagnostic names the discarded parameters.
- Replay K-loop matmul epilogues for f16/bf16/f32 storage and rectangular 16/32/64 tiles with f32 dot
  accumulation and f32 epilogue arithmetic (nine recovered dyadic-exact GPU cases plus the fp32 32×32 control).
  Activations, clamps, fused multiply-adds and non-tile-boundary output masks still refuse.
- Recover six biased-attention combinations with 8/16/32 query rows, 32 keys, D32/64,
  fp16/bf16 storage and f32 or explicit f16 probabilities. A separate full-row replay also
  computes the tested fp16/D64 32×64 noncausal and 64×32 causal score tiles with independent
  query/key lengths. It preserves typed f32 negative-infinity sentinels and completes all
  program input reads before output stores, including the tested overlapping Bias/Out
  allocation. D128, explicit bf16 dot operands and arbitrary score-result scaling remain
  outside the demonstrated envelope. The later accepted score-reuse repair evaluates each
  wide score tile once per key block instead of 130 times, preserving the tested source
  arithmetic, aliases and exceptional-value behavior. Current performance remains unqualified.
- Repair threadgroup scratch reuse at the allocator: aliases now require proved synchronization
  between lifetimes, including control-flow and loop-backedge checks. This closes packet 516's
  reduce→scan and join→split races and the structural arg-reduction/atomic handoff concerns.
  The reduce→scan defect is public since 0.2.0; join→split was introduced in the unpublished
  lineage (commit 6ab6db7, already an ancestor of local main). B/F/G were structural concerns,
  not witnessed numeric failures. On the retained bounded corpus, the repair has zero audit
  flags across 1,230 unique emitted kernels and no newly over-budget allocation; a focused
  D/E GPU replay went from 60/60 wrong parent launches to 0/80 wrong repaired/control launches.
  These finite observations do not prove all possible kernels race-free.
- Fix public argmin/argmax axis1 output-address corruption: blocked one-dimensional stores now
  preserve program/runtime offsets and remap the pointer and mask to the result's row.
  Runtime strides, selects, masks and negative offsets have exact index/sentinel checks.
  Nonuniform/unparsed tensor constants and unproved data-dependent address leaves now refuse
  at this store boundary; true/false splat masks remain supported. This is distinct from the
  allocator's unwitnessed arg-reduction scratch race.
- Reject unknown/nonuniform constant literals at generic constant lowering instead of silently
  treating them as zero or uniformly true. A valid alternating boolean mask was confirmed to
  compile as an all-true mask even on a direct one-dimensional store, outside the arg-reduction
  repair. Numeric scalar/splat literals, true/false masks and typed special-float words remain
  supported; general nonuniform tensor constants are not newly implemented.
- Refuse, with a typed capacity error that autotuning may prune, source-replay kernels whose
  threadgroup allocations exceed 32 KiB after scratch reuse (previously a pipeline-creation failure at
  load time); exactly 32 KiB remains admitted. The check covers the bounded replay families only.

Thanks to Hocine Benkelaya / NeuroBrix for PRs #5–#7 and issues #8–#11, their reproducers and
design observations. The implementation here is independently developed against the current
compiler; reporter-derived regression kernels are identified in their test files. No contributor
PR has been merged by this work. Earlier rc entries below are historical development snapshots,
not evidence that their original repair drafts were complete.

## 0.3.0rc3 - 2026-09-10

Supersedes rc2 (never published). Independent review of rc2 found the dispatch fix and its
coverage incomplete; this round closes them:

- **Submission-boundary contract made complete and phase-aware.** A failure at/after a kernel is
  submitted now fails loud across *every* fast-dispatch helper — elementwise, FlashAttention, MLA,
  fast-matmul (incl. split-K's two dispatches), and the quantized helpers (incl. the per-group
  fast→scalar fallback, which previously re-dispatched after a caught post-submit failure). A
  genuine *pre-submission* eligibility miss still falls through to the host path; non-dispatchable
  args (e.g. `triton.reinterpret` wrappers) are excluded before dispatch rather than via an
  ambiguous dispatch exception.
- **Regressions now cover the real driver path** (`[32]→[64]` atomic via a post-submit hook failure)
  plus a per-helper phase matrix.
- **Welford / variance exceptional-value tests strengthened** to source-derived per-output
  classification (both mean and m2), a double-precision clean oracle, finite-sibling exactness, the
  −Inf case, and checker self-validation.

## 0.3.0rc2 - 2026-09-10

Supersedes 0.3.0rc1 (never published). Two correctness additions found during rc1 validation:

- **Dispatch fallback no longer double-applies.** A fast-path failure *after* a kernel was
  enqueued (e.g. from the launch-exit hook) was silently re-run on the host path, double-applying
  atomics and in-place updates. It now fails loud instead; pre-enqueue misses still fall through to
  the correct host path. Pinned by `tests/test_dispatch_post_submit_no_double_apply.py`.
- **Exceptional-value coverage for variance / Welford.** NaN / +-Inf inputs to layernorm stay
  row-local (finite rows bit-identical) and single-pass Welford propagates non-finite inputs rather
  than returning a plausible-but-wrong finite statistic. Pinned by
  `tests/test_welford_nonfinite_inputs.py`.

## 0.3.0rc1 - 2026-09-10

**Release candidate, for validation only.** This is a pre-release: pip's default resolution
excludes it unless you pass `--pre` or pin the version explicitly. It exists so the reporter of GitHub issue #4 and other
Apple Silicon users can exercise the correctness work before the stable release. Read
[docs/RELEASE_CANDIDATE_LIMITATIONS.md](docs/RELEASE_CANDIDATE_LIMITATIONS.md) first — it
states plainly what has and has not been validated on this candidate.

A correctness release. 0.2.0 was about capability and speed; this one is about being wrong
less often, and refusing loudly where the compiler cannot prove it is right. 86 commits in
`182c182..1ff716ae` plus a reviewed overlay, the majority of them fixes.

### Reported by users

- **GitHub issue #4 (trifast / AlphaFold triangle attention on M1)** — eight verified defects,
  three of them silent wrongs. Seven are fixed, four of which moved from *refuse* to *works*:
  reduction-extent argument naming, loop-carried pointers, the MEPT 2-D accumulator, the
  constant QK-result score scale, multi-output copy-back, loop-carry layout propagation, and
  the over-31-buffer-argument case. The eighth, shared key/value attention, was censused on
  GPU across eight pointer spellings, both dtypes and three head dimensions, plus the exact
  parameter-count signature the reporter described, against an independent double-precision
  oracle. No wrong answer appeared; where the value role cannot be proved, the compiler
  refuses. We do not have the reporter's original kernel, so their exact source is not claimed
  covered.
- The trifast kernels used as regression fixtures are attributed in
  [docs/TRIFAST_PROVENANCE.md](docs/TRIFAST_PROVENANCE.md), with the upstream MIT notice
  preserved at `third_party/trifast/LICENSE` and named in `project.license-files`.

### Correctness

- **Upstream conformance: 5,780 passed / 0 failed / 3,562 skipped** across all 9,342 nodes
  (2026-09-10), every status identical to the recorded baseline. At 0.2.0 (2026-08-18) the same
  suite's totals were 5,559 passed / 3,783 skipped. (Totals as recorded on those dates; no
  node-by-node historical comparison between the two is attached.)
- **Rank-3 and higher `tt.dot` recovery** closed the last conformance family: the upstream
  ratchet went 151 remaining failures to 0 via batch-separable lowering.
- **Host round-trip marshalling** is now storage-faithful for aliases, before-base offsets and
  past-view strides — a P0 that could silently miscompute on any 2-D-grid kernel launched
  through the host path.
- Numerous silent wrongs closed across the dot, reduce, scan, atomic and FlashAttention
  paths, each replaced by either a proven-correct lowering or an explicit refusal. Reductions
  now preserve the sign of zero: a sum whose source produces `-0.0` no longer returns `+0.0`.
- Per-result metadata replaced printed-type inspection throughout the lowerer, so lowering
  decisions rest on ownership-proven facts rather than on text that may describe an alias.

### Validation

- Correctness and performance now run as **separate jobs**. `pytest tests/` selects
  correctness only; the four hardware throughput floors run under
  `python scripts/run_project_tests.py`. The floors themselves are unchanged. See
  [docs/VALIDATION_LANES.md](docs/VALIDATION_LANES.md).
- Project suite: **4,488 passed / 0 failed / 12 skipped**.

### Known limits

Performance numbers in this repository are **not qualified on this candidate**, the optional
C++ route remains **unaudited and off by default**, and the absolute throughput floors target
the development M4 Max rather than any M-series Mac. See the limitations document.


## Unreleased

### Compiler failure handling

- Nonzero Metal compiler and linker exits now surface immediately on both binary
  paths, including internal errors and failures without source-location diagnostics.
  They are no longer silently retried as presumed transient failures. Bounded
  recovery for explicitly observed missing-artifact races remains in place.

### Build requirements

- Source builds require `setuptools>=77.0.3` for PEP 639 license-file metadata.
  Both the project license and the TriFast notice are included in distributions.
  Build environments using an older setuptools must upgrade; the build must not
  proceed while silently omitting the notices.

## 0.2.0 - 2026-08-18

Major update. FlashAttention now **beats PyTorch SDPA** and is competitive with Apple's
MLX, plus weight-only **int8/int4 quantized inference**, latent (**MLA**) attention, **KDA**
linear/delta attention, a **trainable Metal FlashAttention backward**, and byte-identical
output verified on a **third GPU vendor (AMD)**. Performance was re-verified cold on an M4
Max for this release: FlashAttention fp16 head_dim=128 runs ~1.65-1.99x PyTorch SDPA (full)
and up to ~4x (causal), at ~0.88x Apple MLX at the largest sizes; weight-only int8 decode
runs ~3.7x the fp32 decode, at the memory roofline. Details below.

### Performance

- **FlashAttention parallel online-softmax — same accuracy, up to 1.8× faster on
  causal.** The simdgroup FA did its per-row online-softmax
  with only 32 of 256 threads, each serially scanning all BN=64 score columns (measured
  ~14% of a full block, and much more of a causal block where work-skipping leaves fewer
  MMAs). It now runs on all 256 threads — 8 per row, each scanning BN/8 columns, reduced
  via small threadgroup partials — with **identical numerics** (masked columns become
  `-INFINITY` in the max pass, so `exp()`=0 in the sum pass, preserving the causal/tail
  mask without a second guard). Same-accuracy A/B vs the prior kernel: **full 1.11–1.13×,
  causal 1.78–1.84×**. Absolute (fp16 vs SDPA / MLX): full 2.0–2.5× SDPA (0.79–0.88×
  MLX), **causal 4.1–4.6× SDPA**; about even with MLX's own `mask="causal"` kernel at
  moderate N and ~0.88× at the largest (the earlier "1.65–1.68× MLX" reading compared
  against MLX's non-skipping mask-array path, not its causal kernel). Correct across
  hd64/hd128/MLA, full+causal, fp16+fp32.

- **FlashAttention now beats PyTorch SDPA (was ~0.3× — a dispatch bug, not a slow
  kernel).** The simdgroup FA kernel (head_dim=128, BM=32/BN=64) was always fast, but
  its **2-D threadgroup grid** `(n_q_blocks, Z·H)` disqualified it from the zero-copy
  `compile_shader` fast-path (which hard-required a 1-D grid), so every launch fell to
  the **host-roundtrip metallib path — ~2.5–4.2× slower**. Attribution was decisive: the
  *same* kernel dispatched via `compile_shader` with its **native** 2-D grid vs a
  1-D-linearized grid measured identically (6.32 vs 6.31 TF), so the entire gap is
  dispatch overhead, not the kernel or the grid shape. A new `flash_attention` dispatch
  descriptor routes the FA kernel through `compile_shader` with its native grid
  (**fail-OPEN**: unlike the quantized path, the host path is equally correct, so any
  miss simply falls through — never wrong, never refused). Result, end-to-end through the
  shipping `@triton.jit` path (head_dim=128, cold A/B vs `F.scaled_dot_product_attention`):
  **fp16 full 1.65–1.99×**, fp32 full 1.27–1.53× (causal below, with the skip). A split-KV /
  flash-decoding variant was prototyped and **rejected** — once the dispatch is fixed, the
  online-softmax combine overhead makes it lose to the plain kernel at every size.

- **Causal FlashAttention skips the masked upper triangle.** The causal kernel ran
  *every* KV block and masked per-element, doing ~2× the necessary work (the fully-masked
  upper-triangle blocks). A q-block covering rows `[q_start, q_start+BM)` can stop its KV
  loop once `kv_start > q_start+BM-1` — every later block is entirely masked. This prunes
  only provably-all-masked blocks, so the per-element mask still handles the diagonal block
  and the numerics are **byte-identical**. On the zero-copy path it is a strict
  **1.03–1.69× over the un-skipped causal kernel** (the same loop-clamp was tried and
  reverted earlier, but on the old dispatch-bound path halving compute was invisible).
  Causal vs SDPA now: **fp16 1.42–1.95×**, fp32 1.01–1.29× for N≥1024 (only tiny N=512
  stays ~0.77–0.95×).

- **head_dim=64 FlashAttention now uses the simd MMA kernel too (and fp16 hd64 runs on
  the GPU at all for the first time).** The simd kernel is parameterized on head_dim
  (`TPG = (head_dim//8)//n_groups = head_dim//64`), so it is correct for any multiple of
  64 — but the router only ever sent head_dim=128 to it. head_dim=64 fell through to the
  generic path, where it was slow in fp32 (~0.19–0.33× SDPA) and, in **fp16, silently
  CPU-fell-back**: the generic K^T is a 32×64 `tt.trans` that exceeds the 1024-thread
  cap, so it refused and ran on CPU. A guarded hd64 branch now routes contiguous
  32×32 fp16/fp32 hd64 to the simd template (inheriting the zero-copy dispatch and causal
  skip for free); **any** detection refusal or ineligibility (e.g. non-contiguous)
  falls through to the unchanged generic path, so nothing regresses. hd64 vs SDPA:
  **fp16 full 2.00–2.89×, fp16 causal 1.55–2.31×, fp32 full 1.80–2.27×, fp32 causal
  1.47–1.93×** — and fp16 hd64 attention runs on the device instead of the CPU.

- **Asymmetric-head_dim FlashAttention (the MLA / DeepSeek-style attention core).** The
  2026-frontier models (DeepSeek-V3, Kimi) use Multi-head Latent Attention, whose reference
  form has a *different* head_dim for the QK score (192 = 128 nope + 64 rope) than for V/output
  (128). The simd FA kernel now takes an optional `v_head_dim`: the QK contraction runs over
  `head_dim` while the output/V tiling runs over `v_head_dim`. Because registers scale with
  the *output* width only, MLA's qk=192 / v=128 has the **same register footprint as hd128** —
  the 192-wide QK just adds loop iterations (a *symmetric* head_dim=192 overflows the register
  file and won't even build). `v_head_dim=None` is the default and is byte-identical to before.
  The MLA-core kernel is correct vs SDPA (err ≤2e-3 fp16) and **beats it 1.32–1.67× (full),
  1.26–1.81× (causal)**. fp32 qk=192 exceeds the 32 KB threadgroup budget (192-wide Q
  staging), so this targets fp16/bf16 — the dtype MLA runs in anyway.

- **`@triton.jit` auto-routing for asymmetric (MLA-shaped) attention.** `_detect_flash_attention`
  now reads a *separate* `v_head_dim` from the P@V output (it required `[block_m, head_dim]`
  before, refusing anything asymmetric), and the FA router accepts `_fa_maxdim` up to 256,
  routing an asymmetric shape to the simd template when it's simd-eligible (contiguous, v ∈
  {64,128}, fp16 for qk>128) and falling through to a loud refuse otherwise (the scalar/tiled
  fallback is symmetric-only). Symmetric hd64/hd128 are byte-identical (the FA suite is
  unchanged). A canonical asymmetric `@triton.jit` attention (qk=128/v=64, fp16) now
  auto-routes and matches SDPA. A gen-side budget guard turns an over-budget head_dim (e.g.
  qk=256 fp16 → 33 KB) into a clear error instead of a cryptic pipeline-state failure.
  Note: `tl.arange` needs power-of-2 bounds and qk=256 overflows the budget, so expressing
  real MLA (qk=192) from `@jit` uses the nope/rope split (see below).

- **Real MLA (nope/rope) attention auto-routes from `@triton.jit` — DeepSeek/Kimi-style
  attention now runs on Metal.** A real MLA kernel computes the QK score as two chained
  dots, `q_nope@k_nope^T + q_rope@k_rope^T` (128 + 64), over *separate* tensors — three
  `tt.dot`s total (Triton fuses the sum into the dot accumulator chain). The FA detector
  now recognizes this 3-dot shape (guarded, disjoint from the 2-dot symmetric path so it's
  byte-identical): it identifies the two QK dots (whose A operand is a load) vs the PV dot
  (A is the softmax `exp`, no load), extracts all four QK pointer chains, and sets
  `qk_head_dim = nope + rope = 192`. Routing is a **fail-closed cat-dispatch**: it
  concatenates `[q_nope|q_rope]` / `[k_nope|k_rope]` into contiguous `[.,192]` (a ~0.01 ms
  copy) and runs the validated qk=192 / v=128 kernel; the host path can't run the
  different-ABI kernel, so a miss (non-MPS, compile_shader off) refuses. A canonical
  pre-scaled MLA `@jit` kernel auto-routes, runs on GPU, matches SDPA (err ≤1e-3 fp16,
  full + causal), and **beats it 1.19–1.54× (full), 1.11–1.59× (causal)** end-to-end. The
  post-scaled form (scale on the summed dot result) still refuses cleanly (fused-scale
  gate) — correct-or-refuse preserved. fp16/bf16 only (fp32 qk=192 exceeds the 32 KB budget).

- **Split-K for skinny/deep fp32 matmul.** Small-M/N, deep-K matmuls (e.g.
  M=64,N=64,K=8192) were ~0.1× torch — occupancy-starved: a handful of output tiles
  means a handful of threadgroups, each running the whole K-loop serially. A new
  **deterministic two-pass** split-K path (each of G threadgroups computes 1/G of K
  into a partials buffer, then a reduce kernel sums over G — **no atomics**, so it is
  byte-identical run-to-run) fires for fp32 shapes with `n_tiles < 64` and
  `K >= 2048`: M=64,N=64,K=8192 **0.16 → 0.67 TF** (now > torch), M=128,N=128,K=8192
  **0.58 → 1.43 TF** (1.34× torch). Isolated + additive — any non-fit (moderate tile
  counts, shallow K, fp16/bf16) falls through to the unchanged single-pass fast
  kernel, so no shape regresses (the threshold was tuned by measuring the *shipped*
  path, where the partials-alloc + second-dispatch overhead made n_tiles=128 regress).

### Attention (frontier ops + training)

- **KDA (Kimi Delta Attention / gated DeltaNet) now runs on Metal.** Linear/delta-rule
  attention with a per-key-dimension forget gate — not softmax attention, and its chunked
  prefill needs a UT-transform triangular solve, so it ships as a **direct op**
  (`triton_msl.kda`) rather than a `@triton.jit` route. Chunked MMA prefill (C=8) +
  recurrent decode, fp16/fp32, validated against a recurrent reference
  (`tests/test_kda.py`). See `docs/attention_ops.md`.

- **FlashAttention backward pass — attention is now trainable on Metal.** The forward FA
  was inference-only; `triton_msl.fa_backward.flash_attention` is a
  `torch.autograd.Function` whose dQ/dK/dV run on Metal (tiled FA-2 backward, MMA, causal +
  full, fp16/fp32; the logsumexp is computed on-device by a flash kernel, so the backward
  never materializes the N×N score matrix). Cross-checked against torch's own SDPA gradients
  (`tests/test_fa_backward.py`). The two suites add 11 tests.

### Portability

- **Verified byte-identical on AMD, not just NVIDIA.** The example `@triton.jit` kernels,
  unmodified, ran on a rented AMD Instinct MI300X (ROCm 7.2.4, gfx942, Triton 3.6.0)
  alongside the earlier NVIDIA A40 and the Apple M4 Max. `vector_add` and the `ieee` matmul
  produced the **same SHA-256** across all three vendors (Metal == CUDA == ROCm); softmax
  matched to fp rounding (~7.5e-9); tf32 is refused on Metal and absent on AMD. The central
  claim — develop on the Mac you own, run unchanged on datacenter silicon — now holds across
  three ISAs (Metal / PTX / CDNA3) and three Triton versions (3.0.0 / 3.6.0 / 3.7.0).
  PORTABILITY.md + README updated.

### Platform

- **Validated on macOS 26.6 (Tahoe), build 25G72** — full project suite 1971
  passed / 0 failed on real hardware. The Metal-version probe already runtime-checks
  each `-std=metalX.Y` for actual loadability rather than guessing from SDK version
  numbers, so it was immune to the 15.x→26.x SDK-version jump that broke PyTorch MPS
  version parsing. No code change was needed — the design anticipated it.
- **Metal Toolchain detection (macOS 26 / Xcode 26)** — on Tahoe the Metal shader
  compiler is a separate on-demand component; when absent (fresh setup or an Xcode
  update), `xcrun metal` fails permanently. triton-msl now detects this specific
  failure and raises an actionable error naming the fix
  (`sudo xcodebuild -downloadComponent MetalToolchain`) instead of retrying it as a
  transient flake and reporting a misleading "transient, all 3 attempts failed". The
  Requirements doc calls out the component too.
- **C++ path AIR version now dynamic (macOS 26)** — the opt-in C++/LLVM-IR lowering
  path hardcoded `air.version = {2,7,0}` / `Metal 3.2`, which macOS 26's toolchain
  rejects (it wants AIR 2.8 / Metal 4.0), silently degrading the C++ path to CPU
  execution. device_detect now runtime-probes the toolchain's actual AIR /
  Metal-language version and threads it through `run_to_llvm` into the emitted
  metadata, so the C++ path runs on GPU on macOS 26 without breaking older macOS
  (the probe falls back to 2.7 / 3.2). Requires rebuilding `_triton_msl_cpp`.

### Gluon

- **Basic Gluon kernels now run on Metal.** Implemented `MetalBackend.get_target_name`
  (a hole in Triton's out-of-tree `BaseBackend` contract — the Gluon runtime calls it
  generically, so any backend without it raised `AttributeError`) and populated
  `metadata["shared"]` in the Gluon lowering path (`gluon_to_ttgir`, which the Gluon
  path runs instead of `make_ttgir`). Gluon copy / elementwise kernels with an explicit
  `BlockedLayout` (size_per_thread 1–8 × 4–8 warps) compile and run correctly.
  NVIDIA-specific Gluon (mma, warp specialization, mbarrier, TMA) remains out of scope.
  A companion upstream change — a default `get_target_name` on `BaseBackend` — would
  let any out-of-tree backend get Gluon for free (it is a hole in the base contract).

### Correctness

- **Weight-only int4 (GPTQ/AWQ per-group) decode GEMV now auto-routes** (was refused —
  the generic lowerer can't emit the bitwise nibble unpack). A canonical int4 decode
  GEMV written as a `@triton.jit` — packed weight `[N,K/2]` uchar (2 nibbles/byte),
  per-group fp32 scale/zero, `nib = (extui(w) >> ((k%2)*4)) & 0xF; acc += tl.sum(x *
  (sitofp(nib) - zero[g]) * scale[g])` — is recognized at compile time and dispatched
  to `make_int4_gemv` (~1.47× over int8, half the weight bytes). The recognizer
  (`_maybe_quant_gemv_int4_descriptor`) is **correct-or-refuse**: it verifies the
  nibble-unpack chain and **every packing constant** (byte = k//2, shift = (k%2)*4,
  mask 0xF) against the kernel's assumption, extracts and matches the per-group size,
  and binds each leg to its arg — so a mismatched packing refuses rather than
  mis-computes; int8 GEMVs (no nibble unpack) are not misrouted.
- **Weight-only int8 decode GEMV now RUNS on Metal** (was refused). A canonical
  weight-only int8 decode GEMV written as an in-loop row reduce —
  `acc += tl.sum(x[None,:] * (w_i8.to(f32) - zero[:,None]), axis=1)` over K, then
  `out = acc * scale` — with weight `[N,K]` contiguous (GPTQ/AWQ), per-N float
  scale/zero, fp32 in/out, is recognized at compile time and dispatched to the
  dedicated `make_int8_gemv` kernel (one simdgroup per output column, char4/float4
  coalesced, `simd_sum`) via `compile_shader`. Correct (~1e-5 err) and near the
  memory roofline (~426 GB/s at N=11008, K=4096 — the dominant LLM decode shape). It
  reuses the quantized-matmul dispatch (tagged `"gemv"`) and its fail-closed driver
  path; the recognizer (`_maybe_quant_gemv_descriptor`) is **correct-or-refuse** —
  it verifies the sum-reduce, the exact `mulf(bcast(x), subf(sitofp(w), bcast(zero)))`
  input, and the `acc*scale` epilogue, tracing every leg to its arg. A plain fp32
  GEMV (no dequant) is **not** misrouted — it refuses via the reduce-layout guard.
- **In-loop 2-D axis reduce (GEMV via `tl.sum`) now refuses instead of silently
  collapsing to row 0.** A `tl.sum` (or max/…) over one axis of a 2-D tile produces
  its per-row result in the row-broadcast layout (thread `lid` holds row `lid/N`).
  When that result is accumulated across a K-loop (`acc += tl.sum(x[None,:]*w,
  axis=1)` inside `for k in range(0, K, BK)`), the row-broadcast layout is not
  propagated across the `scf.for` loop-carry, so the loop-carried accumulator + 1-D
  store instead assume one-row-per-thread and every output row silently collapsed to
  the first (a plain or quantized GEMV written this way returned `o[n] = o[0]`). The
  existing 2-D-reduce coverage guard did not catch this (the tile fits the
  threadgroup) — it is a distinct *layout* bug. Detected structurally (a 2-D→1-D
  reduce whose result feeds an `scf.for` carried value) and refused loudly; the
  single-tile form (no K-loop) is correct and unaffected. Full fix (propagating the
  broadcast layout across the loop-carry) is a follow-up. Found via the quantized
  GEMV/decode investigation.
- **Weight-only int8 quantized matmul now RUNS on Metal** (was refused). A canonical
  weight-only int8 matmul — `out = a @ ((w_i8.to(f32) - zero) * scale)` with the
  standard `(input, weight, output, scale, zero, M, N, K, ...strides)` signature,
  weight stored `[K,N]` contiguous, per-N float scale/zero, fp32 in/out — is
  recognized at compile time and dispatched to a dedicated dequant kernel
  (`make_int8_matmul_fast`, layout `kn`) via `compile_shader`: the int8 tile is staged
  to a float threadgroup tile with the zero folded in, a float simdgroup MMA runs on
  it, and the epilogue applies the per-N scale. Near-bit-exact vs the float reference
  (max abs err ~1e-4 at fp32 accumulate). The recognizer
  (`_maybe_quant_matmul_descriptor`) is **correct-or-refuse**: it verifies the exact
  dequant tree, traces `scale`→arg3 / `zero`→arg4, confirms the layout via
  address-traced strides, and anchors the K-loop bound to the K arg — any deviation
  refuses. **Both weight layouts are routed**: `[K,N]` (natural `tl.dot(a, w)`, layout
  `kn`) and `[N,K]` (GPTQ/AWQ prefill, `out = a @ tl.trans(dequant(w))` — the
  transpose lowers to a `ttg.memdesc_trans`, detected and routed to layout `nk`, so
  prefill and decode share the `[N,K]` weight layout). The path is **fail-closed** end
  to end: a launch that can't use the fast kernel (non-MPS, `compile_shader` off, or
  dims not `M%32 / N%16 / K%32`) is refused, never mis-run. Not yet routed (each
  refuses cleanly): symmetric (no-zero) quant, fp16 in/out, and int4.
- **Matmul role-resolution fixed for kernels with extra ptr args.** In a K-loop matmul
  the A/B pointers are loop-carried (advanced each iteration) so they don't trace back
  to a func-arg — only C (the store target) traces reliably. The role-resolution
  fallback then picked the *last* ptr arg as C: correct by luck for a 3-pointer (A,B,C)
  matmul, but wrong for one with **extra ptr args** (a quantized matmul's scale/zero, or
  a bias), which mis-inferred strides and over-refused it as a "batched matmul". Now the
  reliably-traced C is kept and A/B fall back per-leg.
- **MEPT chained-addptr fix** — an array-of-offsets base pointer (`x_ptr + offs*K`)
  advanced by a *scalar* loop variable (`+ k`, inside a runtime loop) fell through to
  the scalar-offset lowering and emitted an invalid double subscript
  (`base[arr[0]][k]`) that failed to compile. This was always a **loud** failure
  (`MetalCompilationError`), never a silent-wrong. The scalar is now folded into every
  register-array slot. Surfaced by the macOS-26 C++-backend suite; guarded by a new
  default-path regression test with a full red-green cycle.

## 0.1.0a2 (2026-07-01)

The project suite grew **877 → 1,968 passed / 0 failed** over this cycle.

### Portability — verified on NVIDIA silicon

- **The central claim is now measured, not argued**: the same unmodified `@triton.jit`
  kernels were run on an Apple M4 Max (Metal, Triton 3.7.0) and a rented NVIDIA A40
  (CUDA, Triton 3.0.0), each checked against the same NumPy reference. vector-add and
  the fp32/`ieee` matmul produced **bit-identical** outputs across vendors (Δ = 0);
  softmax matched to fp rounding (~1e-9). The one divergence is NVIDIA's **tf32**
  default for fp32 `tl.dot` (6.1e-2 vs an fp64 reference; `input_precision="ieee"` →
  bit-identical). Metal has no tf32, and triton-msl refuses it rather than silently
  approximating. See the new [`PORTABILITY.md`](PORTABILITY.md) and the reproduce
  harness `benchmarks/cross_backend_verify.py`.
- **Runnable local-dev example** — `examples/local_triton_dev.py`: the three canonical
  Triton tutorial kernels (vector-add, fused softmax, tiled matmul) on deliberately
  non-multiple shapes, each verified against NumPy on the Metal GPU, with a regression
  test pinning them.

### Correctness — the anti-silent-wrong campaign

Systematic adversarial audits (multiple independent rounds, plus three new fuzzers)
closed **~75 silent-wrong bugs** across the dot/reduce/store surfaces; the dot and
reduce surfaces are now correct-or-refuse by construction:

- **General address-traced matmul stride inference** — transposed / sliced /
  column-major / pre-transposed-staged operands now compute correctly or refuse
  loudly, independent of variable naming. The three legacy stride mechanisms were
  deleted in favor of the one address-traced inference; twin dot-lowering paths
  (simple-dot vs K-loop epilogue, fragment selection, bias-init detection) were
  de-duplicated so a fix can no longer land in one copy and miss the other.
  Chain-dot / 3-D / batched matmuls and ambiguous tile-vs-batch shapes refuse.
- **Reduce-combine classifier rewritten structurally** — reduce lowering now
  exact-matches the combine region (sum/max/min/and/or/xor, argmax/argmin, Welford)
  instead of substring-sniffing, and refuses anything it cannot prove. Closed the
  unsigned (u8/16/32/64) max/min/argminmax-computed-signed class, i64 coverage
  holes, NaN-propagation handling, and shared-memory races (missing barriers) on
  broadcast reads in 2-D/3-D/N-D reduce and fused argminmax.
- **Masked-store correctness** — matmul/FA templates compute the full tile and clip
  only at the tile boundary, which silently dropped tighter user store-masks. Now:
  FlashAttention head_dim ≤ 64 non-tile-boundary output masks **compute + clip**
  (the honoring generic path); head_dim=128 FA and matmul templates **refuse**; a
  constant mask bound on a multi-block (`pid*BLOCK + arange`) index is correctly
  treated as non-trivial (it clips later blocks) instead of being dropped.
- **fp8 round-to-nearest-even** — the fp32→fp8 downcast rounded half-away-from-zero;
  now RTNE, matching hardware casts.
- **Three systemic gates added** — a 97-cell differential routing-boundary sweep
  (invariant: correct OR loud-refuse, never silent/cryptic, with an OOB canary) and
  matmul/reduce/combination fuzzers run in the suite.

### torch.compile

- **Persistent MSL stash** — the zero-copy dispatch path now survives Inductor
  cache restores: warm GPT-2 small went 50.7 ms → **2.1 ms (~24×)**, faster than
  PyTorch's own native MPS Inductor backend (4.6 ms) and eager (7.3 ms) on M4 Max.
- **NaN-propagating max/min** lower correctly (was refused → broke compiled softmax
  and training). CNN/BatchNorm compile via under-filling persistent-reduction
  configs; product reductions and a 2-D reduce fused with a 2-D scan in one kernel
  are supported; genuinely-impossible tiles (>1024 threads) refuse loudly.

### Performance

- **bf16 fast matmul** — bf16 is now a fast-matmul input via the M-series
  `simdgroup_bfloat8x8` matrix unit (bf16 in + float32 accumulate, fp32/bf16 out),
  ~12 TFLOP/s vs the ~2.4 TFLOP/s generic float-compute fallback (~4.9×). bf16 is
  the dominant training dtype. FlashAttention bf16 stays refused (FA kernel is
  fp16/fp32 only); the fused matmul+softmax bf16 path now uses the bfloat MMA unit.
- **Deterministic occupancy-gated matmul tile selection** — extends the fast path
  to unaligned M (`M%32≠0`): ~3.7–4.8× for large unaligned-M matmuls vs the generic
  path, no-op aligned, never-regress small (`TRITON_MSL_MATMUL_AUTOTUNE=0` opts out).
- **N%32 / N%16 / N%8 fast path** — the tuner's strip-width gate was relaxed so
  unaligned-N shapes keep simdgroup speed (N%16 ~11 TF, N%8 ~5.7 TF, byte-exact),
  closing the N-alignment perf cliff symmetrically with the M-side rescue.
- **simdgroup-MMA FlashAttention** at head_dim=128 (fp32 + fp16, causal + non-causal).
- **`num_stages`** is a documented, honest no-op (pipelining measured not to help on
  Apple — no `cp.async`; the fast paths already overlap load/compute).

### Reporting honesty

- **3 silent-wrongs fixed** (2026-06-21 dual-lens audit): the fp16/bf16 simple-dot
  epilogue raced on a shared threadgroup slot; bf16 FlashAttention at head_dim 32/64
  dispatched wrong (now refused via a dtype gate); a 3D reduce with a pre-reduce op
  (`tl.sum(a*s)`) silently dropped the op (now refused, since both reduce paths
  mis-handle it). Each is regression-tested.
- **Reporting-honesty pass** — fixed a skip-count parser undercount (3,634 → true
  3,782), refreshed the stale conformance ratchet baseline (4,280 → 5,560), corrected
  the fp16 matmul headline label (fp16-in/fp32-out), and stale doc counts.

### Tooling / CI

- Hosted CI gates on **lint + format** (`ruff check` + `ruff format --check`; a
  repo-wide `ruff format` pass landed, verified behavior-preserving by the full
  suite). GPU/Metal correctness is validated locally (hosted macOS runners cannot
  build Triton within their time limit) — documented in CONTRIBUTING.
- sdist no longer ships a broken half-copy of the test suite (tests and repo-only
  dirs are pruned; the wheel was always clean).

## 0.1.0a1 — first PyPI release as `triton-msl` (2026-06-19)

First public release on PyPI: `pip install triton-msl`, `import triton_msl`.

### Rebrand → `triton-msl` (2026-06-19)

- Renamed the project/import package `triton_metal` → `triton_msl` and the PyPI
  distribution `triton-metal` → **`triton-msl`**. The obvious name is taken on PyPI by
  an unrelated project and blocked by PEP 541 confusability; `triton-msl` is a distinct
  stem (MSL = Metal Shading Language, which this backend emits). The `metal`
  backend/device id and every Apple-Metal API term (`Metal*` classes, `metal::`,
  `.metal`, `xcrun metal`) are unchanged. Env vars hard-renamed `TRITON_METAL_*` →
  `TRITON_MSL_*`. Verified regression-free: project suite 787/0, `test_core`
  5,560/0/3,782.

### torch.compile + training via the inductor backend (2026-06-18)

- `torch.compile(model, backend="inductor")` (and `backend="metal"`) routes through
  triton-msl on Python 3.10–3.14, for **inference and training** — AOTAutograd's backward
  graph lowers to ordinary Triton kernels — static and `dynamic=True`. 32/32 torch.compile
  model tests plus the training suite pass. Fixed four latent silent-wrong bugs exposed
  once torch.compile actually ran: a native-MPS device-op-override registration clobber,
  Metal fork-unsafe compile subprocesses corrupting the cache, a cross-graph MSL
  cache-key collision (re-keyed by content hash), and a softmax template arg mis-map
  (now refuses to the generic path). The inductor autotuner is disabled on Metal (noisy
  timing selected miscompiled tiles); an under-filling persistent-reduction config that
  produced corrupt gradients is also structurally filtered out.

### 2D `tt.gather` (2026-06-18)

- 2D `tt.gather` lowers correctly — axis=0 (including ragged row counts) and same-shape
  axis=1 — via full-tile shared staging; the upstream `test_gather[[4,4]->[8,4],0]` case
  now passes. A previously *silent-wrong* 2D path was first closed with a loud refusal,
  then implemented. Oversized tiles (>1024 threads), ragged axis=1, and register-array
  operands are refused loudly, never guessed.

### FlashAttention — large head_dim + integrity hardening (2026-06-17)

- **head_dim 128 @ `BLOCK_M=BLOCK_N=32`** now supported (fp32 + fp16, causal + non-causal).
  A real `@triton.jit` FlashAttention-2 kernel is routed to a new head-dim-tiled FA2 MSL
  template (`make_flash_attention_kernel_tiled`) that chunks the head dimension to fit
  Metal's 32 KB threadgroup budget (the un-tiled lowering hit `OutOfResources` at 128).
  Routing is a prescan detector with **refuse-on-any-ambiguity** — an FA-shaped kernel whose
  pointers/strides/scale can't be resolved unambiguously is refused, never guessed; the
  detector is robust to Triton's `equal_to_1` arg specialization. Stays FA2 (FA3/FA4 are
  Hopper/Blackwell async-hardware co-designs with no Apple analog).
- **Closed a small-block silent-wrong hole:** FlashAttention at `BLOCK_M`/`BLOCK_N` < 32
  silently mis-computed (rows past the first → garbage) for *any* head_dim — including the
  otherwise-supported 32/64 — which the previous head_dim>64-only guard missed. The prescan
  now refuses min-dot-tile-dim < 32. head_dim > 128, other block sizes, and bf16 matmul
  inputs are refused loudly.
- **Skip-list reclaim:** +28 upstream `test_core` passes recovered from over-broad skips
  (a Gluon-tool base-name collision wrongly skipping core `test_cat`/`test_split`, plus
  `test_tl_range_num_stages`, a uint16/fp16 modulus, `test_pointer_arguments[cpu_pinned]`) —
  each verified a real pass, not a loose-assertion false-pass.
- **Tooling:** `scripts/run_upstream_tests.py` now loads `-p conftest_metal`, so the cited
  source-of-truth command reproduces the documented conformance number.

### Phase 4 — zero-copy execution + fast matmul + Phase-5 readiness audit (2026-06-16)

- **Zero-copy MPS execution** via `torch.mps.compile_shader`: routes emitted MSL through
  PyTorch's compiler so kernels run against MPS tensors without the per-launch host
  round-trip. ~10× on memory-bound kernels (vector_add 28 → ~347 GB/s ≈ 64% of the M4 Max
  546 GB/s roof). Flag `TRITON_MSL_COMPILE_SHADER` (default-on, `=0` escape hatch).
- **Fast simdgroup matmul** (`make_simdgroup_matmul_kernel_fast`) dispatched zero-copy for
  aligned MPS matmuls. Measured at 2048³: fp32 ~9.6–11.5 TFLOP/s (~55–62% of the 18.4
  fp32 peak — competitive with MLX/MPS GEMM), fp16 ~7.8–12, fp16-output ~12.3 — vs the
  ~2.8 generic fallback. Float accumulation (precision); fp16 output via a cast epilogue.
  Flag `TRITON_MSL_FAST_MATMUL`; correctness-gated (test_core dot/matmul on==off identical).
  This is **not** MLX-parity (fp16 runs at ~fp32 rate to keep float accumulation); the
  earlier "~13.8 TFLOP/s MLX parity" docstring claim was an overstatement and is corrected.
- **MEPT** multi-element-per-thread register-array model is the default lowering path.
- **Test suite (Triton 3.7.0):** upstream `test_core.py` **5,560 passed / 0 failed /
  3,782 feature-gap skips** (each a loud refusal or HW-impossible); the single source of
  truth is `scripts/run_upstream_tests.py` (`--device cpu`, which loads the `conftest_metal`
  skip plugin), not hand-maintained counts.
  Project suite **754 passed / 0 failed**. FlashAttention causal + non-causal at HEAD_DIM
  32 / 64 / 128 via the **Python/MSL** lowering — the C++ MLIR→LLVM path named in the
  2026-05-30 snapshot below was shelved (AGX compiler blocker; Python/MSL is primary).
- **Phase-5 readiness audit** (dual NVIDIA/Triton + MLX/Apple lens) recorded in
  `docs/audits/2026-06-16-phase5-readiness-audit.md`; remaining pre-1.0 items tracked there.

### Integrity prescan (silent-wrong → loud refusal)

Added a structural integrity contract: when the compiler recognizes a kernel
but cannot lower it correctly, it raises `MetalNonRecoverableError` (surfaced
as a clear error to the user) instead of emitting wrong numbers. The catalog
lives in `GenericLowerer._refuse_unsafe_unsupported_ops`; the underlying
audit method was *classifying* the skip-listed feature-gap tests as
loud-failure-safe vs silent-wrong, and migrating the latter to refusals.

Cases now refused (each was a silent-wrong producer before the guard):

- `tt.dot_scaled` — microscaling matmul; no Apple hardware (`test_scaled_dot`).
- pid-tiled matmul with constexpr-baked M/N (`test_dot_mulbroadcasted`).
- rank ≥ 3 `tt.trans` with a non-identity permutation (`test_trans_4d`).
- rank ≥ 2 `tt.cat` / `tt.join` (`test_cat_nd`).
- `tt.dot` inside a noinline device function (`test_noinline[shared]`).
- `tt.join` result feeding `tt.dot` (`test_join_with_mma`).
- unstructured kernel-level control flow / `cf.cond_br` (`test_nested_if_else_return`).

### False-pass exposed by the `cf.cond_br` refusal

`test_constexpr_if_return` shares the void-early-return shape of
`test_nested_if_else_return`; before the guard the legacy parser had been
emitting `Out = pid + 0` for it — dropping the `atomic_add`, dropping the
early return, writing out of bounds — but the test only asserts `out >= 0`,
which `pid + 0` happens to satisfy. The garbage went undetected. Now
correctly refused; skip-listed with that rationale. *A passing test is not
the same as a correct kernel.*

### GPU-hang root cause

`test_dot_max_num_imprecise_acc` was investigated for a per-config hang and
*ruled out* as a per-kernel defect: each config either runs correctly or
raises `OutOfResources` cleanly. The apparent hang was reproduced as **GPU
driver-state accumulation** from many back-to-back fp8 dispatches in a
single process — the same environmental class as concurrent test sweeps.
Documented; not a code fix.

### Test suite (as of 2026-05-30, fresh cache, Python 3.14)

- `test_core.py` (upstream Triton): **4,326 passed / 5,016 skipped / 0 failed**.
- Project suite (codegen / GPU correctness / integration / FlashAttention /
  MLX, excluding torch.compile suites blocked on Py 3.14): **507 passed /
  0 failed**.
- FlashAttention: 11/11 at HEAD_DIM=32 (via the C++ MLIR→LLVM path).
- MLX backend: 15/15.
- `torch.compile` suites (32 + 9 tests) are environment-blocked on Python
  3.14 — PyTorch's own platform guard refuses `torch.compile` on 3.14;
  honest skips with `skipif(py>=3.14)`, will auto-lift when PyTorch ships
  3.14 Dynamo support.

### Hardware profiling harness (WS0/C6)

- Added `benchmarks/hw_harness.py` + `triton_msl/profiling/roofline.py` +
  `triton_msl/profiling/disasm.py`: per-kernel GPU-timestamp timing →
  roofline classification (% of the M4 Max 546 GB/s memory roof / estimated
  compute roof, memory- vs compute-bound), pipeline-reflection occupancy,
  best-effort native-AGX disassembly, and an MLX comparison ratio. Emits
  per-kernel JSON + summary.md + baseline.json. This is the empirical
  backbone for the WS1 perf work ("optimal bounds = saturate the limiting
  counter"). First run surfaced a concrete target: reduce_sum at ~16% of the
  bandwidth roof / 1.3x slower than MLX, vs vector_add (72% of roof, 0.89x
  MLX) and silu (48%, 0.59x MLX).
- Vendored `applegpu` (dougallj — REFERENCES.md [11]) under
  `third_party/applegpu/` for native-AGX disassembly. Honest scope: live GPU
  counters (ALU%/occupancy/registers) are NOT programmatically available on
  Apple Silicon (the device vends only the `timestamp` counter set), and
  applegpu is M1-era so M4/AGX2 disassembly is partial (the harness reports a
  decode-coverage %). `docs/INSTRUMENTS.md` documents the Xcode-capture /
  Instruments path for the counters the programmatic API can't provide. No
  Swift counter-helper was built — it would hit the same Metal API wall.
- Tests: `tests/test_roofline.py` (9), `tests/test_disasm.py` (6, incl.
  fat-header parsing for the 0xCBFEBABE GPU-archive magic + graceful
  degradation).

### Documentation & roadmap

- Added `REFERENCES.md` and `CITING.md` (citations for Triton [1],
  FlashAttention v1/v2 [4,5], online softmax [6], MLX [7], Asahi/`applegpu`
  [10,11], MSL spec [8], PyTorch Inductor [12], M4 Max hardware [13]).
- Added `docs/superpowers/specs/2026-05-30-triton-msl-roadmap.md`
  (umbrella roadmap: WS0 foundation, WS1 the register-array spine,
  WS2 orthogonal-refusal cleanup, WS3 experimental sub-AIR AGX).
- Added `docs/superpowers/specs/2026-05-30-ws0-foundation-design.md`
  (the first workstream's full design — documentation truth, citations,
  test hygiene, C++ build hardening, integrity single source of truth,
  hardware profiling + disassembly harness).
- `docs/ARCHITECTURE.md` reconciled: corrected stale upstream-test numbers,
  added integrity-model section, refusal catalog, MEPT experimental
  charter, scoped FlashAttention claim to HEAD_DIM=32.

## 0.1.0-alpha (2026-03-10)

First public alpha release of triton-msl.

### Milestone 1: First Kernel on Metal
- `@triton.jit` vector add running on Apple GPU via Metal Shading Language

### Milestone 2: Kernel Coverage
- 28 `@triton.jit` tests: sum, max, min, softmax, matmul, SiLU, sigmoid, GELU, SwiGLU, RMS norm, layer norm, fused add+ReLU, leaky ReLU, clamp, FMA, FP16, negation, exp+log

### Milestone 3: Real Compiler
- Replaced pattern-matching parser with proper MLIR walker + op-by-op generic lowerer
- All kernels route through new pipeline (`mlir_walker.py` + `generic_lowerer.py`)
- Legacy parser (`ttgir_parser.py` + `msl_emitter.py`) kept as safety fallback only

### Milestone 4: Upstream Compatibility
- 4,279 / 9,334 upstream `test_core.py` tests passing (0 failures)
- Completed: atomics, while loops, `tt.dot` (strided matmul + all epilogues), 2D/3D reduce, argmax/argmin, `tt.histogram`, `tt.gather`, `tl.cat`, `tl.join`/`tl.split`, reshape, permute, transpose, `scf.for`/`scf.if`, NaN propagation, floor div, shift ops
- Triton tutorials 01 (vector add), 02 (softmax), 03 (matmul), 05 (layer norm) all passing
- `@triton.autotune` working end-to-end

### Milestone 5: torch.compile
- 32/32 torch.compile tests passing
- Models: Identity, ReLU, GELU, SiLU, Sigmoid, Tanh, ELU, LeakyReLU, Dropout, Linear, LayerNorm, BatchNorm2d, GroupNorm, InstanceNorm, Embedding, Conv2d, AvgPool, MaxPool, Softmax, LogSoftmax, MLP, LargeMLP, ResBlock, DepthwiseSeparable, ConvNet, TransformerBlock, MHA, SmallGPT, GPT, MiniViT, LSTM, EmbeddingBag
- `torch.compile(model, backend="metal")` integration via Triton inductor

### Milestone 6: MLX Backend
- 15/15 MLX backend tests passing
- Zero-copy dispatch via `mx.fast.metal_kernel()`
- API: `triton_msl.mlx.triton_call(kernel_fn, *args, grid=(...), **constexpr_kwargs)`

### Performance (M4 Max)
- Vector add (16M): 137.5 GB/s
- Softmax (8192x1024): 109.4 GB/s (1.26x vs CPU)
- Matmul (512x512): 826 GFLOP/s
- Layer norm (4096x1024): 77.5 GB/s
- MLX dispatch: ~0.12ms (zero-copy, comparable to native MLX)

### Known Limitations
- ~0.15ms buffer copy overhead per kernel launch (MPS tensors)
- No FP64, FP8, or TF32 (Metal hardware limitation)
- No backward pass / training support
- 32x32 matmul tile size (larger tiles would improve throughput)
