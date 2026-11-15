# Compiler cache contracts

The current source/stash and Metal-binary caches validate versioned, content-bound
products. An old, incomplete or corrupt entry is a miss, not a source of guessed
launch metadata. Binary records bind the actual private linker output bytes;
a concurrent replacement of the public cache file cannot be certified as that output.

Compiler and linker subprocess failures are not cache races: every nonzero exit
raises `MetalCompilationError` immediately on both MSL and LLVM-IR paths. Diagnostic
wording, missing source coordinates or empty stderr cannot make that failure
retriable or autotuner-prunable. The missing-toolchain diagnostic only supplies
installation guidance; it does not change this failure disposition. Bounded retries
remain for specifically observed missing-artifact conditions, including an absent
AIR file after a successful compiler exit and a vanished cache file. A later
successful attempt must not erase an earlier failed compiler or linker exit.

Source records use an explicit native `GPUTarget` codec; a generic object-to-dict
fallback is not allowed. Unknown native metadata makes that product non-cacheable
without dropping fields. Source keys bind the complete incoming compiler metadata
as well as TTGIR/options/policy, so a hit cannot overwrite a different current
target or outer compilation hash. Ordinary option tuples have Triton's normal
JSON array representation; the target is reconstructed as a native `GPUTarget`.

Triton's persistent backend key includes package implementation contents, current
effective compilation policy, complete target, and the selected Metal toolchain
and SDK identity. Inner binary keys additionally distinguish MSL from LLVM IR,
all compilation options and the actual emitted compiler flags, including the
effective Metal language standard resolved before lookup.

The source contract also binds the selected Triton, Torch, NumPy and optional
MLX package contents, plus the Darwin PyObjC/Metal/Cocoa/MPS binding packages.
Headers and other runtime resources are included: Torch's compile_shader reads
headers from its package. Native binaries are content-hashed, not represented
only by version strings. Generated Python bytecode/cache directories are excluded
because their paths/build artifacts are not package source identity. Native-only
modules and ordered namespace-package providers are handled explicitly; unknown
or incomplete required layouts refuse.

The runtime inventory also covers the actual Python executable reported by dyld
(which can differ from `sys.executable`), its standard library, and the backend's
native extension. The standard-library inventory excludes `site-packages` and
`dist-packages`: selected framework roots are inventoried separately, not replaced
by an uncontrolled census of every installed package.

The optional C++ child's actual import spec is selected independently of the
top-level backend package. Editable-install finders can resolve that child from
another build/worktree: its real bytes and dependencies are then inventoried,
and switching its selected provider within a process refuses. This is not an
exception in the native guard and does not certify the C++ lowering's correctness
or source/build compatibility; that route has a separate audit/opt-in contract.

On Darwin a checked Mach-O parser follows native Python entry points and their
transitive providers. It preserves dependency edges, CPU slice, weak absence,
loader/executable-relative paths and inherited RPATH context. Ambiguous providers,
missing strong dependencies, malformed commands, unproved loader commands and
unknown search tokens refuse. It does not search by basename. Non-Darwin ELF
resources shipped by Triton remain content-hashed but cannot be dyld entry points.
Package dylibs inherit their importer's context; they are not guessed standalone
entry points. External providers are hashed in full; Apple system images use the
separate OS-build contract. Each root/edge association survives in the graph.

The first snapshot additionally inventories actual already-loaded non-system
images outside the named framework graph. This covers application imports such
as tokenizer extensions without a library-name allowlist. Their bytes, complete
dependency graph and initial loaded order participate in identity. Unresolvable
ambient dependencies still refuse; no directory or arbitrary future load is
certified. The observed image list must remain stable throughout the inventory.

Before reusing a framework snapshot, the actual ordered image selection is
checked, including same-count replacements, removals and already-known providers
loaded later. A changed selection triggers a complete new, stable native inventory
and a new product identity; unresolved or malformed providers still refuse. This
supports ordinary lazy imports such as Inductor's Z3 solver without a library-name
exemption. Failed or interrupted inventory construction restores both the old
snapshot and its old guard, and cannot certify the new selection with an old key.

On the verified Apple-arm64 dyld ABI, an unchanged loader generation can reuse a
completed name scan. The reader uses public own-process `TASK_DYLD_INFO` and the
versioned `dyld_all_image_infos` prefix, not a private symbol or loader callback.
Any changed generation requires another full scan; the token never replaces the
content fingerprint or enters persistent keys. NULL/in-progress or malformed
warm states refuse. Unsupported ABIs retain full scanning. SDK-offset and real
same-count load/unload tests cover this boundary. See Apple's
[structure definition](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/dyld_images.h)
and [add/remove generation updates](https://github.com/apple-oss-distributions/dyld/blob/main/dyld/ExternallyViewableState.cpp).

Old resident compiled handles refuse before dispatch under the new identity.
The owning JIT function's pre-run hook clears its stale device cache so a fresh JIT
call recompiles; unrelated JIT caches are not globally cleared. This is not
permission to edit an already-loaded installation or hot-reload package code.
The backend initializes its required compiler module map before capturing cache
keys or producer stamps, so its own first import does not change native selection
halfway through an otherwise ordinary cold compilation.
Do not concurrently mutate native loader state during a compile/launch. This does not authenticate hostile
in-process interposition, arbitrary foreign pipelines, or edits to installations
already loaded in memory; installation changes require restart.

## Installation and selection lifetime

Package code, native artifacts, toolchain and SDK contents are immutable within a
Python interpreter. Restart it after upgrading or editing an installation. This
is not a hot-reload protocol. The first toolchain lookup hashes actual contents
of the selected `.xctoolchain` bundle(s) and SDK, including native resources and
directory-symlink contents/topology, unless a persisted identity record applies
(see "Persisted identity records" below). Later lookups reuse that process snapshot.
The small `xcrun --find metal` selector is not mistaken for the actual compiler.

Changing PATH, DEVELOPER_DIR, SDKROOT or TOOLCHAINS after that snapshot refuses
with a restart instruction, because the existing device/AIR probes also have
process lifetime. Missing/unreadable tools or dependencies do not share an
`unknown` identity. Unsupported nonempty CPATH, C_INCLUDE_PATH,
CPLUS_INCLUDE_PATH, LIBRARY_PATH, COMPILER_PATH, GCC_EXEC_PREFIX,
DYLD_LIBRARY_PATH or DYLD_INSERT_LIBRARIES refuse rather than add untracked inputs.
Standard Apple `.xctoolchain` layouts are supported; other layouts require an
explicit dependency-discovery implementation, not a guessed identity.

Framework package installations are also immutable until restart. Current module
specs and import search selection are rechecked, with filesystem resolution only
when import selection changes; a changed selected provider refuses rather than
hot-reloading it. Identical relocation after restart keeps its content identity.
PYTORCH_MPS_FAST_MATH, PYTORCH_MPS_PREFER_METAL and PYTORCH_ENABLE_MPS_FALLBACK are
process-selection settings: changing them after the snapshot requires restart.
Additional DYLD framework/fallback/versioned/root/image-suffix search overrides
are unsupported and refused alongside library-path/injection overrides.
All other nonempty `DYLD_*` settings also refuse, including unknown future keys:
flat-namespace or binding policy cannot hide outside a fixed list of names.

The inventory can take seconds on a cold process; no low-latency claim is made.
No dependency is omitted to meet a timing target. Absolute installation paths
and mtimes are not persistent identities; logical names and content hashes are.
OS/compiler/SDK build metadata supplements the content inventory.

The toolchain contract additionally walks native dependencies of the selected
resolver, compiler/linker selectors, the compiler's actual InstalledDir executable,
and Mach-O executable helpers throughout the selected bundles. A selector's small
binary is not used as a proxy for an independently loaded external provider.
Each subprocess has its own executable-relative context; unproved dependencies
refuse. Static archives and dylib resources remain in the full content inventory,
but are not misclassified as independently executed programs. This is a contract
for the selected standard toolchain, not arbitrary injected compiler plugins.

## Persisted identity records

The toolchain and framework identities above are full-byte inventories. To avoid
repeating them in every new process, a successful strict inventory of each
component publishes a per-user identity record. A later process that observes the
same cheap evidence returns the recorded digest instead of inventorying again.
The record stores the finished strict digest, so every cache key is identical to
the key strict mode computes unless a change listed under "What strict mode
guarantees that a record does not" goes undetected. Any doubt (a missing, unreadable, corrupt or
mismatching record, or any error while checking it) is a miss: that component's
strict inventory runs unchanged and, where records can be used, publishes a new
record. A record never causes an error, a refusal or a digest that strict mode
did not produce when the record was published.

When no attempt is made. In these cases no record is read or written and the
strict inventory runs exactly as in strict mode:

- `triton_msl.backend._cache_contract` is not yet imported, for example a direct
  call of the framework identity before anything else in the backend. The record
  code never imports it, because importing it can load native images and so
  change the loaded-image list that the framework identity binds. The backend
  path and the command line import it first.
- `TRITON_MSL_IDENTITY_DIR` is set to a relative path.
- Any path whose stat state the record would trust is not on APFS (below).

Location and format. Records live in `TRITON_MSL_IDENTITY_DIR` if it is set (it
must be absolute), else in `~/.cache/triton_msl/identity-v1-uid<euid>`. The
directory is deliberately not derived from `TRITON_MSL_CACHE_DIR`, which may be
shared. It is created with mode 0700 and is used only if it is a directory (not a
symlink) owned by the effective user with no group or other permissions and no
extended ACL; otherwise no record is read or written. A directory that inherits
an ACL entry from its parent is therefore never used. Only the record directory
itself is checked, not the permissions of its parent directories. Each component has its own
file, `toolchain-<locator>.json` or `framework-<locator>.json`. The locator is a
SHA-256 over the record schema, the effective user, the running package's
implementation identity, the Python version and ABI tag, the machine architecture
and the executable dyld started, plus, for the toolchain, the selection
variables, the resolver and the resolved bundle and SDK paths, and for the
framework, the selected package paths, the runtime selection settings and the
ordered list of loaded native images. A different loaded-image list is therefore
a different record. Files are canonical JSON of at most 1 MiB with mode 0600, one
link and no extended ACL, read without following symlinks and decoded strictly:
duplicate keys, extra or missing keys, wrong types and malformed digests are
misses. A record of another schema, package build or locator is a miss. Its
`record_sha256` detects corruption only; it is not authentication. Records are
written to a temporary file, synced and renamed into place, so a reader sees a
complete old or new record; a temporary file that carries an ACL is never
renamed into place. The newest 64 records of each kind are kept and older ones
are removed; a temporary file with the writer's own name pattern that is more
than an hour old (left by a killed writer) is removed too.

Toolchain evidence. The live selection always runs first, exactly as in strict
mode: the external-search and `DYLD_*` refusals, the selection variables, the
`xcrun` queries for the compiler, linker, version and SDK, the OS build and the
full SHA-256 of `xcrun`. The record is used only if the compiler version, SDK
build, OS build and `xcrun` digest equal the recorded ones, and then:

- for every bundle: a stat signature of the whole tree (below), which includes
  whatever a symlink inside the bundle resolves to, the same root directory
  status, and the same bytes of the small version files (`ToolchainInfo.plist` at
  the bundle root and `usr/metal/*/ToolchainInfo.json` and `.plist`, including
  which of them exist);
- for a bundle on a read-only mount, in addition: the same mount (file-system ID,
  read-only flag, block and inode totals);
- for the SDK: the same root directory status, the same bytes of
  `SDKSettings.json` and `System/Library/CoreServices/SystemVersion.plist`, and,
  inside an Xcode bundle, the same bytes of `Contents/version.plist` and the same
  status of `Contents/_CodeSignature/CodeResources`;
- every native provider image the strict inventory hashed still has the device,
  inode, mode, owner, size and modification and change times it had when it was
  hashed, and every path the native dependency graph resolved, or found absent,
  gives the same answer. A versioned framework directory's listing records each
  child's name and, for each `.framework` child, whether it is a symlink and
  whether it is a directory, because both decide whether it is a candidate.

Framework evidence. Discovery and the existing snapshot and loaded-image guard
checks always run live. The record is used only if the selection and OS build are
equal, every root has the same stat signature (with the strict inventory's
exclusions and options), the executable has the same status, the live ordered
loaded-image list equals the recorded one, every hashed native image has the same
status and every native graph path answer is the same. The loaded-image guard is
then built from the recorded provider set and verified, discovery is repeated,
and the process snapshot and guard are set exactly as a strict inventory sets
them; only then is the outcome a hit. A failed attempt leaves the previous
snapshot and guard in place before the strict inventory runs.

A stat signature walks a tree exactly as the strict inventory does (the same
order, exclusions, symlink and directory back-edge handling, refusals and empty
tree rule) without opening any file. It covers each entry's relative name, kind,
size, modification and change times, inode, device and mode, each symlink's own
status and target text, and the status of whatever the symlink resolves to. The
small anchor and version files are read only if they are regular files, and only
the file that was checked is read.

Filesystems. Stat evidence is trusted only on APFS, whose timestamps have
nanosecond resolution. Before an attempt is offered, the bundles, the SDK, the
framework roots and the executable must be on APFS; every device that a stat
signature, the executable status, the Xcode seal status or a native provider's
status relies on is checked again (by `statfs(2)`, once per device) when the
evidence is collected, both for a lookup and before a publication. On any other
filesystem, where a same-size rewrite within one timestamp tick can leave every
stat field unchanged, strict mode runs and no record is read or written.

A record is published only after a strict inventory succeeded and only if the
evidence observed before that inventory equals the evidence observed after it, and
each provider's status after the inventory equals its status when it was hashed.
A tree that changed during the inventory is therefore not recorded.

What strict mode guarantees that a record does not. Strict mode derives the
identity from bytes the process read itself. A record relies on metadata
evidence, and the following changes are not detected:

- a write through a shared memory mapping that stays open across the change, on a
  writable framework root (or a writable toolchain bundle), can leave the file's
  size and times unchanged until the mapping is closed;
- with the SDK anchor policy, edits to SDK files other than the anchor files; a
  non-anchor SDK entry that would make strict mode refuse (an unsupported file
  kind, an unreadable file) is likewise not seen by a record hit;
- on a read-only toolchain mount, byte changes that leave every stat field of
  every entry unchanged;
- privileged changes to file times or raw devices;
- a record written by the same user with a chosen digest (the same user can
  already replace any of their own cache entries).

An undetected change reuses products keyed to the earlier identity. Strict mode,
`refresh` and `verify` restore the full-byte guarantee on demand, within the
scope given below.

Modes. `TRITON_MSL_IDENTITY_RECORD` unset or `1` uses records. `0` is strict mode:
no record is read or written and the inventories run exactly as without this
feature. `refresh` skips the lookup, runs the strict inventories and publishes new
records. Any other value behaves as `0`.

Command line. `python -m triton_msl.tools.identity` takes one command, imports
`triton_msl.backend._cache_contract` before computing anything (as the backend
path does) and creates no GPU device:

- `status` prints the mode, the directory and each record's kind, creation time,
  producer and whether it decodes for the running package;
- `refresh` runs the strict toolchain and framework inventories and publishes;
- `clear` deletes the record files (`toolchain-*.json`, `framework-*.json`) in the
  directory, and nothing else; a directory that does not exist holds none
  (`removed 0 record(s)`);
- `verify` computes each identity once with records and once in strict mode,
  prints which component differs, deletes a differing record and exits with
  status 1 when anything differs. When the record-mode run used no earlier record
  (none existed or it no longer matched, so that run was itself a strict
  inventory and published a new record), the line says that no earlier record
  was compared.

Exit status 0 means success (for `verify`, every digest agreed), 1 means `verify`
found and removed a differing record, and 2 means an error, for example an
untrusted record directory.

Because the framework record depends on the ordered loaded native images,
`refresh` and `verify` cover only the framework record of the command's own
process shape, not the record an application with other imports uses. For an
application's record, run `clear`, or run that application once with
`TRITON_MSL_IDENTITY_RECORD=refresh`.

## Resident and restored handles

A successful final compile records an immutable execution-contract stamp under
an unchanged compilation policy. Restoring metadata, packing it or constructing
a launcher cannot invent today's stamp for an old artifact. Both metadata packing
and launcher construction validate it; every launch validates again before
launch-enter/exit hooks or driver runtime access. Missing/legacy/stale stamps
refuse with a recompile/outer-cache-clear instruction.

For ordinary Triton JIT functions, the launcher's first construction attaches one
public per-function pre-run hook. If the policy changes, it clears only that
function's per-device kernel/binder caches before Triton's next lookup, allowing
recompilation. Existing user hooks are preserved. The guard holds only a weak
reference to its owner. Warmup-only handles that never constructed a launcher,
direct restored handles and unsupported JIT hook APIs retain the unconditional
fail-closed check; transparent invalidation is not claimed for those paths.
Do not mutate process compilation policy concurrently with a compile/launch.

## Packed dispatch records

Fresh compilation also writes a versioned record of the kernel name, execution
stamp and all eleven packed dispatch fields. Optional defaults are made explicit
ONLY there. Restoration/packing rejects missing fields or any mismatch with that
record; it cannot invent defaults for old metadata. Launcher construction retains
an immutable serialized snapshot. Each launch checks exact arity and values
before hooks or runtime use, then executes from an unaliased copy so a hook cannot
change a checked nested descriptor through the caller's mutable references.

This is descriptor integrity relative to the producer, not proof that every
producer descriptor is semantically valid for its dynamic tensor arguments.
It also does not authenticate an independently substituted pipeline binary.
Those distinct ABI/artifact boundaries still require validation and measurement.

## Source argument positions and scalar representation

AST parameter positions follow the declared function order restricted to entries
in its signature: an omitted compile-time parameter is not inserted into runtime
positions. IRSource positional signatures must be dense integers starting at
zero. Native `f16`/`f32` aliases are normalized to the same binding types as JIT
signatures. Unknown position maps refuse rather than bind by Python value order.

Before hooks, argument count and tuple shape must match the source signature;
nested compile-time leaves are removed from the flattened list, just as they are
from TTGIR. Output-copyback indices refer to that constexpr-free list. Scalars
are packed with their DECLARED byte width, not the size of their current Python
value. Unknown scalar declarations or unrepresentable conversions refuse. In
particular fp64 scalars are not silently treated as fp32. The fast route also
checks Python representation: an int/bool supplied to a declared-fp32 slot uses
typed host packing, not raw integer bits in a Metal float parameter.

These checks do not authenticate a separately supplied pipeline, prove every
runtime pointer's declared dtype/storage contract, or implement arbitrary
foreign descriptor/argument adapters. No full dynamic-ABI closure is claimed.

### Already-loaded install-name providers

The native graph now binds `@rpath` references to an already-loaded dylib with
the exact full `LC_ID_DYLIB` name before filesystem RPATH resolution, matching
dyld's loaded-image lookup. This supports wheels whose build-time RPATH no
longer exists but which correctly link against a library the process already
loaded (including torchvision using Torch's libc10). It is not a basename match
or an extra search directory. The selected image's full bytes and dependency
edges are included; reading its install name alone does not certify those edges.
Conflicting loaded install names, malformed images, unknown providers and
selection changes during inventory still refuse. The framework identity schema
is bumped so earlier identities do not silently inherit this broader proof.

## Boundaries still being implemented

This is not a claim that all outer caches are protected. Triton's in-memory JIT
lookup and Inductor-restored handles can precede `backend.hash()`. The execution
stamp checks their source/toolchain policy, and the packed record checks descriptor
contents, but neither proves the full dynamic argument/pipeline ABI.
Runtime/tool-native closure is implemented but its full project/release gates and
overhead measurements remain required. Outer Inductor/MLX adapters, a complete
no-cache contract, diagnostic evidence on warm
hits, wheel/relocation tests and final exact-tree release gates remain unfinished.
An inner cache miss must not be presented as proof that an old resident executable
was invalidated. These changes are uncommitted development work, not a release claim.
