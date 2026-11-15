"""Content-bound runtime/compiler package selection, immutable per interpreter.

Includes headers consumed by torch.mps.compile_shader, not only Python/version
strings. System framework build identity is separately in the Metal toolchain
contract. This inventory does not purport to authenticate arbitrary injected
libraries or a foreign pipeline; those are separate boundaries.

Across processes, a per-user identity record (``_identity_record``) may supply
the digest an earlier strict inventory produced for the same selection and the
same ordered loaded-image list. Discovery and the snapshot/guard checks always
run live. No record is consulted for a call made before ``_cache_contract`` is
imported (a direct call), whose loaded-image list a record lookup could change.
The record is used only if every root's stat signature, the executable, the OS
build, every native provider's state and every path answer of the native graph
(versioned framework listings bind each child's kind) still equal what was
recorded, all on APFS; the guard is then rebuilt and verified exactly as the
strict inventory leaves it, and only then is the outcome a hit. Stat signatures
cannot see a same-size write through a mapping held open across the change;
strict mode (``TRITON_MSL_IDENTITY_RECORD=0``) and ``python -m
triton_msl.tools.identity verify`` re-read bytes.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import MappingProxyType

from . import _identity_record
from ._toolchain_contract import _tree_manifest


_snapshot = None
_discovery_snapshot = None
_owned_selection = None
_native_guard = None
_lock = threading.RLock()
_RUNTIME_SELECTION_ENV = ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL", "PYTORCH_ENABLE_MPS_FALLBACK")
_REQUIRED_NAMES = ("triton",) + (
    ("objc", "Metal", "Foundation", "CoreFoundation", "AppKit", "MetalPerformanceShaders")
    if sys.platform == "darwin"
    else ()
)
# Editable finders may resolve an optional native CHILD from a different
# installation even when the top-level package came from this worktree.
# Bind its actual selected spec/bytes; never infer it from __init__.__file__.
_SELECTION_NAMES = (*_REQUIRED_NAMES, "torch", "numpy", "mlx", "triton_msl", "triton_msl._triton_msl_cpp")


def _python_roots():
    """The executable dyld actually started, plus the standard-library providers."""
    import sysconfig

    roots = {}
    for name in ("stdlib", "platstdlib"):
        roots["python-" + name] = (Path(sysconfig.get_path(name)).resolve(strict=True),)
    roots["python-executable"] = (_native_executable(),)
    return roots


def _native_executable():
    import ctypes

    if sys.platform == "darwin":
        function = ctypes.CDLL(None)._NSGetExecutablePath
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        function.restype = ctypes.c_int
        size = ctypes.c_uint32(0)
        function(None, ctypes.byref(size))
        if not 0 < size.value <= 1024 * 1024:
            raise RuntimeError("unproved native executable path length")
        buffer = ctypes.create_string_buffer(size.value)
        if function(buffer, ctypes.byref(size)) != 0:
            raise RuntimeError("cannot discover actual native executable")
        return Path(os.fsdecode(buffer.value)).resolve(strict=True)
    return Path(sys.executable).resolve(strict=True)


def _native_entrypoints(roots):
    import stat
    from ._native_dependencies import is_native

    seeds, visited = [], set()
    # torch/__init__.py:_load_global_deps performs this explicit ctypes load;
    # it need not be an LC_LOAD_DYLIB edge of torch._C. No directory search.
    explicit = {path / "lib/libtorch_global_deps.dylib" for path in roots.get("torch", ())}

    def visit(path, excludes=(), *, canonical_parent=False):
        # Only ordinary non-entry files can avoid re-walking the already
        # resolved parent. Roots, directories, symlinks and every potential
        # native entry retain full strict resolution. Never use cached dirent
        # metadata: a missing/inaccessible child must still refuse discovery.
        if not canonical_parent:
            path = path.resolve(strict=True)
        else:
            mode = path.lstat().st_mode
            if (
                not stat.S_ISREG(mode)
                or path.suffix == ".so"
                or path == roots["python-executable"][0]
                or path in explicit
            ):
                path = path.resolve(strict=True)
        if path in visited:
            return
        visited.add(path)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.name in excludes or child.name == "__pycache__" or child.suffix in (".pyc", ".pyo"):
                    continue
                visit(child, canonical_parent=True)
        elif path.is_file() and (path.suffix == ".so" or path == roots["python-executable"][0] or path in explicit):
            if not is_native(path):
                with path.open("rb") as handle:
                    if handle.read(4) == b"\x7fELF" and path.suffix == ".so":
                        # Triton ships Linux CUDA/ROCm resources on Darwin too.
                        # Their bytes are inventoried but dyld cannot load ELF.
                        return
                raise RuntimeError(f"unproved native Python entry point: {path}")
            seeds.append(path)

    for name, paths in sorted(roots.items()):
        for path in paths:
            visit(path, ("site-packages", "dist-packages") if name in ("python-stdlib", "python-platstdlib") else ())
    return seeds


def _native_dependencies(roots, observer=None):
    global _native_guard
    from ._native_dependencies import dependency_graph, parse_image, _read_image, _system
    import platform

    if sys.platform != "darwin":
        raise RuntimeError("native runtime provider closure requires Darwin")
    cpu = {"arm64": 0x0100000C, "x86_64": 0x01000007}.get(platform.machine())
    if cpu is None:
        raise RuntimeError("unproved native runtime CPU architecture")
    seeds = _native_entrypoints(roots)
    executable = roots["python-executable"][0]
    loaded = _loaded_native_images()
    read, options = parse_image, {}
    if observer is not None:
        # The identity record needs each image's state when its bytes were
        # hashed and every path answer the graph used; neither changes them.
        def read(path, cpu_type):
            return _read_image(path, cpu_type, digest=True, digest_observer=observer.image)

        options = {"image_reader": read, "probe_observer": observer.probe}
    # Dylibs inherit an importing extension's RPATH stack. Treating every dylib
    # as a separately dlopened entry loses that context (e.g. MLX libjaccl).
    # All package bytes are still inventoried. An application can also have
    # imported native extensions outside these named frameworks BEFORE our
    # first snapshot (e.g. a model tokenizer). Those are real cache inputs, not
    # a reason to reject every subsequent compilation or a basename exemption.
    loaded_paths = tuple(Path(os.fsdecode(raw)).resolve(strict=True) for raw in loaded if not _system(os.fsdecode(raw)))
    if observer is not None:
        names = (os.fsdecode(raw) for raw in loaded if not _system(os.fsdecode(raw)))
        for name, path in zip(names, loaded_paths):
            observer.probe("present", name, path)
    rows, providers = dependency_graph(seeds, cpu_type=cpu, executable=executable, loaded_paths=loaded_paths, **options)
    ambient = []
    order = []
    for raw in loaded:
        name = os.fsdecode(raw)
        if _system(name):
            order.append(("system-build", os.path.normpath(name)))
            continue
        path = Path(name).resolve(strict=True)
        # Binding loaded order prevents different provider/interposition orders
        # from accidentally sharing a persistent product key. Paths remain a
        # live guard, not persistent relocation identity; contents are hashed.
        order.append(("native", read(path, cpu).digest))
        if path not in providers and path not in ambient:
            ambient.append(path)
    if ambient:
        # Use the existing closed resolver, including its RPATH/ambiguity and
        # malformed-image refusals. Never auto-certify a directory or all .so's.
        rows, providers = dependency_graph(
            [*seeds, *ambient], cpu_type=cpu, executable=executable, loaded_paths=loaded_paths, **options
        )
    guard = _LoadedNativeGuard(providers)
    guard.verify()
    if _loaded_native_images() != loaded:
        raise RuntimeError("native image selection changed during content inventory")
    _native_guard = guard  # publish only a complete, stable cold inventory
    return {"graph": rows, "loaded_order": tuple(order)}


def _loaded_native_images():
    """Read actual dyld selection in order, without importing optional packages."""
    import ctypes

    loader = ctypes.CDLL(None)
    count = loader._dyld_image_count
    count.restype = ctypes.c_uint32
    name = loader._dyld_get_image_name
    name.argtypes = [ctypes.c_uint32]
    name.restype = ctypes.c_char_p
    size = count()
    result = tuple(name(index) for index in range(size))
    if count() != size or any(item is None for item in result):
        raise RuntimeError("native image selection changed during inspection")
    return result


class _NativeSelectionChanged(RuntimeError):
    """A new provider requires a new identity, not certification of old handles."""


class _LoadedNativeGuard:
    """A newly loaded image must already belong to the frozen provider graph."""

    def __init__(self, providers):
        import ctypes

        loader = ctypes.CDLL(None)
        self.count = loader._dyld_image_count
        self.count.restype = ctypes.c_uint32
        self.name = loader._dyld_get_image_name
        self.name.argtypes = [ctypes.c_uint32]
        self.name.restype = ctypes.c_char_p
        self.providers = providers
        self.checked_count = None
        self.approved_names = set()
        self.checked_names = None
        from ._native_generation import create_generation_reader

        self._generation_reader = create_generation_reader()
        self._generation_checked = None

    def verify(self, *, allow_scan=True):
        from ._native_dependencies import _system

        reader = getattr(self, "_generation_reader", None)
        generation = reader() if reader is not None else None
        if generation is not None and generation == getattr(self, "_generation_checked", None):
            return
        if not allow_scan:
            # A speculative contract hit may check the live generation, but may
            # not invoke the scan's callbacks and then replay them on fallback.
            # The complete evaluator owns scans and inventory reconstruction.
            raise _NativeSelectionChanged("native generation requires complete evaluation")
        count = self.count()
        # A dlclose followed by dlopen may leave the image count unchanged.
        # Recheck names, not merely the count; unchanged names avoid filesystem
        # resolution. Installation-byte mutation still requires process restart.
        names = []
        for index in range(count):
            raw = self.name(index)
            if raw is None:
                raise RuntimeError("native image selection changed during inspection")
            names.append(raw)
            if raw in self.approved_names:
                continue
            name = os.fsdecode(raw)
            if not _system(name) and Path(name).resolve(strict=True) not in self.providers:
                raise _NativeSelectionChanged(
                    f"loaded native provider is outside the proved runtime graph: {name}; a new native inventory is required"
                )
        if self.count() != count:
            raise RuntimeError("native image selection changed during inspection")
        if reader is not None and reader() != generation:
            raise RuntimeError("native image selection changed during generation-guarded scan")
        previous = getattr(self, "checked_names", None)
        if previous is not None and tuple(names) != previous:
            # Membership alone misses a removed provider, a changed load order,
            # or the first actual load of a provider already in the static graph.
            # Each changes the loaded-order input bound into the product key.
            raise _NativeSelectionChanged("loaded native selection changed; a new native inventory is required")
        self.checked_names = tuple(names)
        self.checked_count = count
        self.approved_names.update(names)
        self._generation_checked = generation


def _discover_selection():
    global _discovery_snapshot, _owned_selection
    required, names = _REQUIRED_NAMES, _SELECTION_NAMES
    # Resolve filesystem locations only when the Python import selection can
    # have changed. Loaded modules' current spec values are inspected, not just
    # their version labels. Code/installation mutation still requires restart.
    module_tokens = []
    for name in names:
        module = sys.modules.get(name)
        spec = getattr(module, "__spec__", None)
        module_tokens.append(
            (
                name,
                id(module),
                id(spec),
                getattr(spec, "origin", None),
                tuple(getattr(spec, "submodule_search_locations", ()) or ()),
            )
        )
    runtime_env = tuple((key, os.environ.get(key)) for key in _RUNTIME_SELECTION_ENV)
    # The token carries id() values of the import machinery. An id names an
    # object only while that object is alive, so the machinery is captured by
    # an owning tuple at the moment its ids are read and retained with the memo
    # built from it. Every name below is selected exactly where the original
    # expression selected it; the capture adds one `tuple` selection and call
    # per list before the attribute read. The selected constructors are kept so
    # the store can prove the capture is an owning snapshot (a replaced
    # constructor may hand back the live list); an unproved capture is never
    # memoized.
    token = (
        tuple(sys.path),
        tuple(map(id, (finders := (finder_constructor := tuple)(sys.meta_path)))),
        tuple(map(id, (hooks := (hook_constructor := tuple)(sys.path_hooks)))),
        tuple(module_tokens),
        id(resolver := importlib.util.find_spec),
        sys.version,
        sys.implementation.cache_tag,
        sys.byteorder,
        runtime_env,
    )
    if _discovery_snapshot is not None and _discovery_snapshot[0] == token:
        return _discovery_snapshot[1:3]
    roots = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None:
            if name in required:
                raise RuntimeError(f"required framework package {name} is unavailable")
            roots[name] = ()  # proved absence, not an unknown lookup/error
            continue
        locations = spec.submodule_search_locations
        if locations:
            # Namespace packages (mlx) may have multiple ordered providers. Each
            # provider is included; changing the order is a selection change.
            roots[name] = tuple(Path(path).resolve(strict=True) for path in locations)
        elif spec.origin not in (None, "built-in", "frozen"):
            roots[name] = (Path(spec.origin).resolve(strict=True),)
        else:
            raise RuntimeError(f"unproved framework package layout for {name}")
    roots.update(_python_roots())
    metadata = {
        "python_abi": sys.implementation.cache_tag,
        "python_version": sys.version,
        "byteorder": sys.byteorder,
        "runtime_environment": runtime_env,
    }
    # These dictionaries are private construction products, not caller-owned
    # proxies. Freeze their entire normal value graph before reusing its text.
    # Foreign/nonstandard metadata keeps the original live serialization path.
    if (
        all(type(metadata[key]) is str for key in ("python_abi", "python_version", "byteorder"))
        and all(type(key) is str and (value is None or type(value) is str) for key, value in runtime_env)
        and all(
            type(name) is str and type(paths) is tuple and all(type(path) is type(Path()) for path in paths)
            for name, paths in roots.items()
        )
    ):
        selection = _format_selection(roots, metadata)
        roots = MappingProxyType(roots)
        metadata = MappingProxyType(metadata)
        _owned_selection = roots, metadata, selection
    # Memoize only a capture proved to own the observed machinery, and only if
    # the machinery live now IS that machinery, object for object. The exact
    # tuple type is taken from a literal, not from a name, so this certificate
    # performs no live builtin lookups; it reads sys.meta_path, sys.path_hooks
    # and importlib.util.find_spec once more (miss path only). A capture made
    # by a replaced constructor, or machinery replaced during resolution, is
    # returned but not memoized; the next call observes again.
    owning = ().__class__
    same = finder_constructor is owning and hook_constructor is owning and importlib.util.find_spec is resolver
    if same:
        live_finders = owning(sys.meta_path)
        live_hooks = owning(sys.path_hooks)
        same = live_finders.__len__() == finders.__len__() and live_hooks.__len__() == hooks.__len__()
        index = 0
        while same and index < live_finders.__len__():
            same = live_finders[index] is finders[index]
            index += 1
        index = 0
        while same and index < live_hooks.__len__():
            same = live_hooks[index] is hooks[index]
            index += 1
    if same:
        _discovery_snapshot = token, roots, metadata, (finders, hooks, resolver)
    return roots, metadata


def _format_selection(roots, metadata):
    return (
        tuple((name, tuple(map(str, paths))) for name, paths in sorted(roots.items())),
        json.dumps(metadata, sort_keys=True),
    )


def _selection_identity(roots, metadata):
    owned = _owned_selection
    if owned is not None and roots is owned[0] and metadata is owned[1]:
        return owned[2]
    return _format_selection(roots, metadata)


def _inventory_options(name):
    """Exclusions and empty-tree allowance of one logical root's inventory."""
    excludes = ("site-packages", "dist-packages") if name in ("python-stdlib", "python-platstdlib") else ()
    # A venv's platstdlib may contain only site-packages;
    # its real stdlib/dynload provider is inventoried through
    # python-stdlib. Record the empty supplemental inventory,
    # never omit the root or waive an empty primary stdlib.
    return excludes, name == "python-platstdlib"


def _adopt_record(roots, metadata, record):
    """Finish from a validated identity record exactly as the strict inventory finishes."""
    global _native_guard
    evidence = record["evidence"]
    guard = _LoadedNativeGuard(frozenset(Path(path) for path in evidence["guard"]))
    guard.verify()
    if [os.fsdecode(raw) for raw in _loaded_native_images()] != evidence["base"]["loaded"]:
        raise RuntimeError("native image selection changed during identity record use")
    again, again_metadata = _discover_selection()
    if again != roots or again_metadata != metadata:
        raise RuntimeError("framework selection changed during identity record use")
    _native_guard = guard
    return record["digest"]


def _json_metadata(metadata):
    owned = _owned_selection
    if owned is not None and metadata is owned[1]:
        return dict(metadata)
    return metadata


def framework_identity():
    """Hash package contents once; recheck actual module selection every time."""
    global _snapshot, _native_guard
    from triton_msl.errors import MetalNonRecoverableError

    with _lock:
        previous_snapshot, previous_guard = _snapshot, _native_guard
        try:
            roots, metadata = _discover_selection()
            # Paths are only a live-process selection guard, NEVER persistent
            # identity. Identical relocations after restart retain their hash.
            selection = _selection_identity(roots, metadata)
            if _snapshot is not None:
                if _snapshot[0] != selection:
                    raise MetalNonRecoverableError(
                        "framework selection changed in this process; restart before compiling"
                    )
                if _native_guard is None:
                    return _snapshot[1]
                try:
                    _native_guard.verify()
                except _NativeSelectionChanged:
                    # Libraries loaded lazily by the compiler (e.g. Inductor's
                    # solver) are a new runtime input. Rebuild the closed graph
                    # and rekey every product; NEVER stamp an old executable
                    # with this new identity. Its resident stamp must mismatch.
                    pass
                else:
                    return _snapshot[1]
            attempt = None
            policy = _identity_record.mode(os.environ)
            if policy != "strict":
                digest = None
                try:
                    attempt = _identity_record.framework_attempt(os.environ, policy, roots, selection)
                    recorded = None if attempt is None else attempt.lookup()
                    if recorded is not None:
                        digest = _adopt_record(roots, metadata, recorded)
                        attempt.used()
                except Exception:
                    # A failed record use leaves no trace; the strict inventory follows.
                    _snapshot, _native_guard = previous_snapshot, previous_guard
                    if attempt is not None:
                        attempt.rejected()
                if digest is not None:
                    _snapshot = selection, digest
                    return digest
            physical, manifests = {}, {}
            for name, paths in sorted(roots.items()):
                parts = []
                for path in paths:
                    excludes, allow_empty = _inventory_options(name)
                    physical_key = path, excludes, allow_empty
                    if physical_key not in physical:
                        physical[physical_key] = _tree_manifest(
                            path,
                            ignore_bytecode=True,
                            exclude_root_names=excludes,
                            allow_empty=allow_empty,
                        )
                    parts.append(physical[physical_key])
                manifests[name] = parts
            native = _native_dependencies(roots) if attempt is None else _native_dependencies(roots, attempt)
            again, again_metadata = _discover_selection()
            if again != roots or again_metadata != metadata:
                raise RuntimeError("framework selection changed during content inventory")
            payload = json.dumps(
                {"schema": 5, "metadata": _json_metadata(metadata), "packages": manifests, "native": native},
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()
            _snapshot = selection, digest
            if attempt is not None and _native_guard is not None:
                attempt.publish(digest, _native_guard.providers)
            return digest
        except BaseException as exc:
            # _native_dependencies constructs a new guard before the final
            # selection/payload checks. Roll back BOTH pieces even on an
            # interrupt; an old digest + a newer guard would certify stale code.
            _snapshot, _native_guard = previous_snapshot, previous_guard
            if isinstance(exc, (OSError, ValueError, RuntimeError, TypeError)) and not isinstance(
                exc, MetalNonRecoverableError
            ):
                raise MetalNonRecoverableError(f"cannot establish framework package identity: {exc}") from exc
            raise
