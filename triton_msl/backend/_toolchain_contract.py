"""Content identity for the selected Metal toolchain and SDK.

Installed package/toolchain/SDK contents are immutable within an interpreter:
upgrades require restarting it. Selector-environment changes are refused, not
silently combined with cached device/AIR probes. No persistent stat-only memo is
trusted. The first lookup reads actual bytes; subsequent lookups reuse that
process snapshot. Whole-tree inventory is intentionally conservative.

A toolchain executable's embedded DYLD_VERSIONED_FRAMEWORK_PATH can make dyld
open a relative comparison name (``@rpath/...``) literally. Compiler and linker
processes, including device probes, run in the selected Metal toolchain root.
Relative comparison lookups must stay inside that inventoried installation and
be absent there. Installation changes require restart, as above; changing the
application's working directory cannot affect these lookups. No filesystem
check is added to the warm execution-contract path.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import threading

from triton_msl.errors import MetalToolchainSelectionError


_SELECTION = ("PATH", "DEVELOPER_DIR", "SDKROOT", "TOOLCHAINS")
_EXTERNAL_SEARCH = (
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "DYLD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "DYLD_ROOT_PATH",
    "DYLD_IMAGE_SUFFIX",
    "DYLD_VERSIONED_LIBRARY_PATH",
    "DYLD_VERSIONED_FRAMEWORK_PATH",
)
_lock = threading.RLock()
_snapshot = None
_compiler_context = None
_environment_inputs = None


def _read_only_mount(path_or_fd):
    """Unknown mount status must retain byte reads, never authorize reuse."""
    try:
        status = os.fstatvfs(path_or_fd) if isinstance(path_or_fd, int) else os.statvfs(path_or_fd)
        return bool(status.f_flag & os.ST_RDONLY)
    except (OSError, AttributeError, NotImplementedError):
        return False


def _file_state(status):
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


_INVENTORY_WORKER_LIMIT = 8
_INVENTORY_POOL_MIN_BYTES = 256 * 1024


def _inventory_workers(workers):
    """Bounded hashing concurrency for large files. 1 is the exact serial reference."""
    if workers is None:
        return max(1, min(_INVENTORY_WORKER_LIMIT, (os.cpu_count() or 1) // 2))
    if type(workers) is not int or workers < 1:
        raise ValueError("inventory workers must be a positive integer")
    return workers


def _tree_manifest(
    root,
    *,
    ignore_bytecode=False,
    exclude_root_names=(),
    allow_empty=False,
    reuse_readonly_aliases=False,
    provider_digests=None,
    workers=None,
):
    """Logical names + bytes, including directory-symlink topology and contents.

    Traversal, ordering, exclusions, back-edge records and every file below
    ``_INVENTORY_POOL_MIN_BYTES`` are handled serially in the calling thread,
    exactly as before. Files at or above that size are hashed by a bounded
    thread pool while the traversal continues: SHA-256 and the read/stat system
    calls release the interpreter lock. Every file keeps its own open, full read
    to EOF, descriptor/path change checks and refusals; records are assembled in
    traversal order, so the manifest is identical for any worker count when the
    inputs are stable. The first failure in traversal order raises, no partial
    manifest is returned and no worker outlives the call.
    """
    root = Path(root).resolve(strict=True)
    count = _inventory_workers(workers)
    # Only toolchain callers opt in. SDK/package inventories retain every read.
    # Metadata alone is insufficient: an open shared mmap can change bytes with
    # unchanged times. Read-only mounts supply the premise for alias reuse.
    reuse_readonly_aliases = reuse_readonly_aliases and _read_only_mount(root)
    # Seed only from full native reads in this identity construction. Copy so
    # inventories cannot export their reads to other stages or later calls.
    file_digests = dict(provider_digests or {}) if reuse_readonly_aliases else {}
    digest_lock = threading.Lock()
    # Aliases of one physical file may be in flight concurrently. The first to
    # reach a state hashes it; later aliases open and check their own descriptor
    # as always, then wait for that digest instead of re-reading. If the first
    # fails, waiters read the bytes themselves. Aliases share a size, so the
    # calling thread and pool workers never wait on each other.
    pending = {}
    # Reuse storage, never content: every cache miss still reads through EOF.
    # One buffer per thread, local to this invocation, so concurrent or
    # reentrant inventories cannot alias it.
    storage = threading.local()

    def buffers():
        view = getattr(storage, "view", None)
        if view is None:
            storage.buffer = bytearray(256 * 1024)
            storage.view = view = memoryview(storage.buffer)
        return storage.buffer, view

    def hash_file(path, rel, before):
        state, owner = None, None
        try:
            with path.open("rb", buffering=0) as handle:
                # A symlink may escape a read-only root to a writable mount.
                # Check the opened file, and still open/check every alias.
                if reuse_readonly_aliases and _read_only_mount(handle.fileno()):
                    state = _file_state(before)
                    if state != _file_state(os.fstat(handle.fileno())):
                        raise RuntimeError(f"toolchain input changed before hashing: {path}")
                digest = None
                if state is not None:
                    with digest_lock:
                        digest = file_digests.get(state)
                        if digest is None:
                            waiting = pending.get(state)
                            if waiting is None:
                                owner = pending[state] = threading.Event()
                    if digest is None and owner is None:
                        waiting.wait()
                        with digest_lock:
                            digest = file_digests.get(state)
                if digest is None:
                    read_buffer, read_view = buffers()
                    hasher = hashlib.sha256()
                    while True:
                        filled = handle.readinto(read_buffer)
                        if filled is None:
                            raise BlockingIOError(f"toolchain input read would block: {path}")
                        if filled == 0:
                            break
                        hasher.update(read_view[:filled])
                    digest = hasher.hexdigest()
                if state is not None and state != _file_state(os.fstat(handle.fileno())):
                    raise RuntimeError(f"toolchain input changed while hashing: {path}")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise RuntimeError(f"toolchain input changed while hashing: {path}")
            if state is not None:
                if state != _file_state(after):
                    raise RuntimeError(f"toolchain input changed while hashing: {path}")
                with digest_lock:
                    file_digests[state] = digest
            return (rel, before.st_size, digest, "file")
        finally:
            if owner is not None:
                with digest_lock:
                    pending.pop(state, None)
                owner.set()

    pool = None
    if count > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="triton-msl-inventory")
    plan = []  # traversal order: ready record tuples, or futures for pooled large files

    def visit(path, rel, ancestors):
        before = path.stat()
        inode = before.st_dev, before.st_ino
        if stat.S_ISDIR(before.st_mode):
            if inode in ancestors:
                plan.append((rel, 0, ancestors[inode], "directory-back-edge"))
                return
            for child in sorted(path.iterdir(), key=lambda p: p.name):
                if not rel and child.name in exclude_root_names:
                    continue
                if ignore_bytecode and (child.name == "__pycache__" or child.suffix in (".pyc", ".pyo")):
                    continue
                visit(child, f"{rel}/{child.name}" if rel else child.name, ancestors | {inode: rel})
        elif stat.S_ISREG(before.st_mode):
            if pool is not None and before.st_size >= _INVENTORY_POOL_MIN_BYTES:
                plan.append(pool.submit(hash_file, path, rel, before))
            else:
                plan.append(hash_file(path, rel, before))
        else:
            raise RuntimeError(f"unsupported toolchain input kind: {path}")

    def cancel_pending():
        for item in plan:
            if type(item) is not tuple:
                item.cancel()

    try:
        try:
            visit(root, "", {})
        except Exception as walk_error:
            # An ordinary failure in the walk or an inline hash stops traversal.
            # Every pooled file that precedes it in traversal order was already
            # submitted; expose the earliest failure in that order, as the serial
            # walk would, with the exact exception object. Interrupts (not
            # Exception subclasses) are re-raised as they are, without waiting.
            for item in plan:
                if type(item) is not tuple:
                    try:
                        item.result()
                    except Exception as earlier:
                        cancel_pending()
                        raise earlier
            raise walk_error
        # A later future that failed first in time is never exposed ahead of an
        # earlier one: results are consumed in traversal order.
        records = [item if type(item) is tuple else item.result() for item in plan]
    except BaseException:
        cancel_pending()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if not records and not allow_empty:
        raise RuntimeError(f"empty toolchain input tree: {root}")
    return records


def _toolchain_native_dependencies(roots, entries, *, provider_digests=None):
    """Bind subprocess entry points and executable helpers, not only their bundle."""
    from ._native_dependencies import dependency_manifest, is_native, _image_header, _read_image
    import platform

    cpu = {"arm64": 0x0100000C, "x86_64": 0x01000007}.get(platform.machine())
    if cpu is None:
        raise RuntimeError("unproved native toolchain CPU architecture")
    readonly_roots = tuple(
        root for root in {Path(roots[role]).resolve(strict=True) for role in ("metal", "metallib")}
        if provider_digests is not None and _read_only_mount(root)
    )

    def observe_digest(path, fd, before, opened, after, digest):
        # Graphs can reach outside the selected bundle. Both the containing
        # toolchain root and the actual opened file must still be read-only.
        if not any(path.is_relative_to(root) and _read_only_mount(root) for root in readonly_roots):
            return
        if not _read_only_mount(fd):
            return
        state = _file_state(before)
        if state == _file_state(opened) == _file_state(after) == _file_state(os.fstat(fd)):
            provider_digests[state] = digest

    def read_provider(path, cpu_type):
        # Never bypass parsing or hashing a provider: only the subsequent tree
        # inventory consumes these digests. Graph contexts remain independent.
        return _read_image(path, cpu_type, digest=True, digest_observer=observe_digest)

    reader_options = {"image_reader": read_provider} if readonly_roots else {}
    entries = dict(entries)
    visited = set()
    for role in ("metal", "metallib"):
        root = roots[role]
        pending = [(root, "")]
        while pending:
            path, rel = pending.pop()
            status = path.stat()
            inode = status.st_dev, status.st_ino
            if inode in visited:
                continue
            visited.add(inode)
            if stat.S_ISDIR(status.st_mode):
                pending.extend(
                    (child, f"{rel}/{child.name}" if rel else child.name) for child in reversed(sorted(path.iterdir()))
                )
            elif stat.S_ISREG(status.st_mode) and path.suffix != ".a" and is_native(path):
                # Classification discards the digest. Keep the complete parser
                # and change checks, then hash selected providers in their graph.
                # The later tree inventory also binds every unselected file.
                if _image_header(path, cpu).file_type == 2:  # MH_EXECUTE, not standalone dylib
                    entries[f"{role}-helper:{rel}"] = path
    directory = roots["metal"].resolve(strict=True)
    manifests = {
        role: dependency_manifest(
            [path], cpu_type=cpu, executable=path, working_directory=directory, **reader_options
        )
        for role, path in sorted(entries.items())
    }
    for name in _working_directory_lookups({"native_providers": manifests}):
        if not (directory / name).resolve().is_relative_to(directory):
            raise ValueError(f"relative compiler comparison lookup escapes the inventoried toolchain: {name}")
    return manifests


def compiler_process_context():
    """Resolve once, then pin the resolver and compiler CWD to selected installation.

    This is selection only, not a substitute for the full toolchain identity.
    It is also usable by early device probes without importing the compiler or
    recursively requesting its content fingerprint. Only subprocess creation
    calls this helper; resident launches do not.
    """
    global _compiler_context
    selection = tuple(os.environ.get(name) for name in _SELECTION)
    with _lock:
        if _compiler_context is not None:
            if selection != _compiler_context[0]:
                raise MetalToolchainSelectionError("Metal toolchain selection changed in this process; restart before compiling")
            return _compiler_context[1]
        resolver = shutil.which("xcrun")
        if resolver is None:
            raise FileNotFoundError("xcrun is unavailable; install/select the Metal toolchain")
        resolver = str(Path(resolver).resolve(strict=True))
        selected = subprocess.check_output(
            [resolver, "-sdk", "macosx", "--find", "metal"], text=True, stderr=subprocess.PIPE
        ).strip()
        if not selected:
            raise MetalToolchainSelectionError("empty Metal compiler selection")
        selector = Path(selected).resolve(strict=True)
        directory = next((p for p in selector.parents if p.name.endswith(".xctoolchain")), None)
        if directory is None:
            raise MetalToolchainSelectionError(f"unproved metal toolchain layout: {selector}")
        if selection != tuple(os.environ.get(name) for name in _SELECTION):
            raise MetalToolchainSelectionError("Metal toolchain selection changed during compiler context construction")
        context = resolver, str(directory)
        _compiler_context = selection, context
        return context


def run_metal_tool(command, **kwargs):
    """Run a compiler/linker command with no dependence on application CWD."""
    if not command or command[0] != "xcrun" or "cwd" in kwargs:
        raise ValueError("Metal command must use the selected resolver and fixed working directory")
    resolver, directory = compiler_process_context()
    return subprocess.run([resolver, *command[1:]], cwd=directory, **kwargs)


def _working_directory_lookups(metadata):
    from ._native_dependencies import working_directory_lookups

    names = set()
    for manifest in metadata.get("native_providers", {}).values():
        names.update(working_directory_lookups(manifest))
    return tuple(sorted(names))


def _resolve_toolchain(*, provider_digests=None):
    try:
        resolver, directory = compiler_process_context()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    def query(*args):
        value = subprocess.check_output(
            [resolver, "-sdk", "macosx", *args], text=True, stderr=subprocess.PIPE, cwd=directory
        ).strip()
        if not value:
            raise RuntimeError(f"empty toolchain query: {args}")
        return value

    roots, entries = {}, {"resolver": Path(resolver).resolve(strict=True)}
    for name in ("metal", "metallib"):
        selector = Path(query("--find", name)).resolve(strict=True)
        bundle = next((p for p in selector.parents if p.name.endswith(".xctoolchain")), None)
        if bundle is None:
            raise RuntimeError(f"unproved {name} toolchain layout: {selector}")
        roots[name] = bundle
        entries[name + "-selector"] = selector
    if roots["metal"] != Path(directory):
        raise RuntimeError("Metal compiler working directory differs from inventoried toolchain")
    version = query("metal", "--version")
    installed = [
        line.removeprefix("InstalledDir: ").strip()
        for line in version.splitlines()
        if line.startswith("InstalledDir: ")
    ]
    if len(installed) != 1:
        raise RuntimeError("Metal compiler does not identify its actual InstalledDir")
    actual = (Path(installed[0]) / "metal").resolve(strict=True)
    if not actual.is_file() or not actual.is_relative_to(roots["metal"]):
        raise RuntimeError("actual Metal compiler is outside the inventoried toolchain")
    entries["actual-metal"] = actual
    roots["sdk"] = Path(query("--show-sdk-path")).resolve(strict=True)
    # InstalledDir is diagnostic, not identity; its bytes are included by logical
    # bundle name. Relocating identical SDK/toolchain contents must retain identity.
    metadata = {
        "schema": 2,
        "compiler_cwd_policy": "inventoried-metal-root-v1",
        "compiler_version": "\n".join(line for line in version.splitlines() if not line.startswith("InstalledDir:")),
        "sdk_build": query("--show-sdk-build-version"),
        "os_build": subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.osversion"], text=True).strip(),
        "xcrun_sha256": hashlib.sha256(Path(resolver).read_bytes()).hexdigest(),
    }
    if not metadata["os_build"]:
        raise RuntimeError("OS build identity is unavailable")
    options = {} if provider_digests is None else {"provider_digests": provider_digests}
    metadata["native_providers"] = _toolchain_native_dependencies(roots, entries, **options)
    return roots, metadata


def toolchain_identity():
    """Fail closed on incomplete identity or live selector changes, never 'unknown'."""
    from ._environment_snapshot import environment_snapshot

    return _identity_for_environment(environment_snapshot())


def _identity_for_environment(environment):
    """Shared implementation; internal callers capture the live environment first."""
    global _snapshot, _environment_inputs
    from triton_msl.errors import MetalNonRecoverableError
    from ._environment_snapshot import is_snapshot

    cached = _environment_inputs
    if cached is not None and cached[0] is environment and cached[1:3] == (_SELECTION, _EXTERNAL_SEARCH):
        selection, external = cached[3:]
    else:
        selection = tuple(environment.get(key) for key in _SELECTION)
        external = sorted(
            {key for key in _EXTERNAL_SEARCH if environment.get(key)}
            | {key for key in environment if key.startswith("DYLD_") and environment.get(key)}
        )
        if is_snapshot(environment):
            _environment_inputs = environment, _SELECTION, _EXTERNAL_SEARCH, selection, external
    if external:
        raise MetalNonRecoverableError(
            "untracked compiler search/injection environment: "
            + ", ".join(external)
            + "; clear it before compiling with the Metal cache contract"
        )
    with _lock:
        if _snapshot is not None:
            if _snapshot[0] != selection:
                raise MetalNonRecoverableError(
                    "Metal toolchain selection changed in this process; restart before compiling"
                )
            return _snapshot[1]
        try:
            provider_digests = {}  # Discarded on success or failure; never a process cache.
            roots, metadata = _resolve_toolchain(provider_digests=provider_digests)
            # Deduplicate physical roots only as a read-cost optimization; each
            # named compiler role retains the complete same manifest in the key.
            manifests = {}
            by_root = {}
            for name, path in sorted(roots.items()):
                resolved = Path(path).resolve(strict=True)
                reuse = name in ("metal", "metallib") and _read_only_mount(resolved)
                inventory = resolved, reuse
                if inventory not in by_root:
                    options = {"provider_digests": provider_digests} if reuse and provider_digests else {}
                    by_root[inventory] = (
                        _tree_manifest(resolved, reuse_readonly_aliases=True, **options)
                        if reuse else _tree_manifest(resolved)
                    )
                manifests[name] = by_root[inventory]
            if tuple(os.environ.get(key) for key in _SELECTION) != selection:
                raise RuntimeError("toolchain selection changed during fingerprint construction")
            payload = json.dumps({"metadata": metadata, "inputs": manifests}, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).hexdigest()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            raise MetalNonRecoverableError(f"cannot establish Metal compiler/SDK identity: {exc}") from exc
        _snapshot = selection, digest
        return digest


_standard_toolchain_identity = toolchain_identity
_standard_identity_for_environment = _identity_for_environment
