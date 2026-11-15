"""Closed Mach-O provider graph for the selected runtime installation.

This is dependency identity, not executable authentication. OS-shipped images
are bound by the separate OS-build contract. Third-party images are hashed in
full. Unsupported loader semantics fail closed; no basename/search-path guess
is permitted. Constants/layouts follow the installed mach-o/loader.h and fat.h.

LC_DYLD_ENVIRONMENT is proved only for a main executable's
DYLD_VERSIONED_FRAMEWORK_PATH. dyld (ProcessConfig::PathOverrides) expands
@executable_path/@loader_path against the main executable, scans each listed
directory for ``Name.framework/Name``, keys every readable dylib by its install
name and compares its current_version with the library found AT that install
name: opened literally (so a relative spelling such as @rpath/... is a lookup in
the process working directory), else in the dyld shared cache, else the
candidate wins outright. The on-disk comparison library is a selection input
even when nothing loads it, so it is bound as a comparison-only node and its OWN
install name becomes the override key; a cached comparison image is OS-build
bound and its canonical name (dyld's real path in the cache) becomes the key,
with the version comparison unreadable and therefore taken as selectable; a
relative lookup is proved absent in the working directory the compiler will
inherit and refused when something shadows it. Candidates and comparison
libraries must prove macOS loadability. The AMFI gate is per-process and
unobservable, so BOTH selectable providers are bound on a keyed request: the
override bytes with their transitive edges under the requesting context and
the ordinary resolution. Any other embedded variable, or one outside an
executable, refuses.
"""

from dataclasses import dataclass
import hashlib
import mmap
import os
from pathlib import Path
import struct


_THIN = {
    b"\xce\xfa\xed\xfe": ("<", 28),
    b"\xcf\xfa\xed\xfe": ("<", 32),
    b"\xfe\xed\xfa\xce": (">", 28),
    b"\xfe\xed\xfa\xcf": (">", 32),
}
_FAT = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbf\xba\xfe\xca": ("<", True),
}
_DEPENDENCIES = {0xC, 0x80000018, 0x8000001F, 0x80000023, 0x20}
_ENVIRONMENT = 0x27  # LC_DYLD_ENVIRONMENT, dylinker_command layout
_VERSIONED_FRAMEWORK_PATH = "DYLD_VERSIONED_FRAMEWORK_PATH"
_BUILD_VERSION, _VERSION_MIN_MACOSX, _PLATFORM_MACOS = 0x32, 0x24, 1
# Obsolete FVM/prebound/sub-framework loading, embedded environment variables
# other than the one proved above, and the new lazy-load-info trie are not
# proved by this parser. Ordinary segment/symbol/
# signing/build-version commands do not select a different provider.
_NON_PROVIDER = {
    0x1,
    0x2,
    0x3,
    0x4,
    0x5,
    0x8,
    0xA,
    0xB,
    0x11,
    0x16,
    0x17,
    0x19,
    0x1A,
    0x1B,
    0x1D,
    0x1E,
    0x21,
    0x22,
    0x80000022,
    0x24,
    0x25,
    0x26,
    0x80000028,
    0x29,
    0x2A,
    0x2B,
    0x2C,
    0x2D,
    0x2E,
    0x2F,
    0x30,
    0x31,
    0x32,
    0x80000033,
    0x80000034,
    0x36,
    0x37,
    0x38,
    0x39,
}
_STRINGS = _DEPENDENCIES | {0xD, 0xE, 0xF, 0x8000001C, _ENVIRONMENT}


@dataclass(frozen=True)
class NativeImage:
    digest: str
    dependencies: tuple
    rpaths: tuple
    file_type: int
    install_name: str | None
    environment: tuple = ()  # (variable, value) pairs proved for a main executable
    current_version: int | None = None  # LC_ID_DYLIB packed X.Y.Z, dyld's comparison operand
    platforms: frozenset = frozenset()  # LC_BUILD_VERSION / LC_VERSION_MIN_MACOSX platforms


def is_native(path):
    with Path(path).open("rb") as handle:
        magic = handle.read(4)
    return magic in _THIN or magic in _FAT


def _parse(data, cpu_type):
    length = len(data)
    if length < 4:
        raise ValueError("short native Mach-O header")
    offset, size = 0, length
    magic = data[:4]
    if magic in _FAT:
        endian, wide = _FAT[magic]
        if length < 8:
            raise ValueError("short native fat header")
        count = struct.unpack_from(endian + "I", data, 4)[0]
        entry_size = 32 if wide else 20
        table_end = 8 + count * entry_size
        if not count or table_end > length:
            raise ValueError("invalid native fat architecture table")
        matches, intervals = [], []
        for i in range(count):
            fields = struct.unpack_from(endian + ("IIQQII" if wide else "IIIII"), data, 8 + i * entry_size)
            cpu, _, start, span, align = fields[:5]
            if align > 63 or start < table_end or not span or start + span > length or start % (1 << align):
                raise ValueError("invalid native fat slice bounds/alignment")
            if any(start < end and begin < start + span for begin, end in intervals):
                raise ValueError("overlapping native fat slices")
            intervals.append((start, start + span))
            if cpu == cpu_type:
                matches.append((start, span))
        if len(matches) != 1:
            raise ValueError("missing or ambiguous native CPU slice")
        offset, size = matches[0]
        magic = data[offset : offset + 4]
    if magic not in _THIN:
        raise ValueError("unproved native Mach-O magic")
    endian, header = _THIN[magic]
    if size < header:
        raise ValueError("short native Mach-O header")
    _, cpu, _, file_type, count, command_bytes, _ = struct.unpack_from(endian + "7I", data, offset)
    if cpu != cpu_type:
        raise ValueError("native Mach-O CPU does not match selected architecture")
    end = offset + header + command_bytes
    if end > offset + size or count > command_bytes // 8:
        raise ValueError("native load command bounds exceed image")
    cursor, dependencies, rpaths, environment, platforms = offset + header, [], [], [], set()
    install_name = current_version = None
    for _ in range(count):
        if cursor + 8 > end:
            raise ValueError("short native load command")
        cmd, command_size = struct.unpack_from(endian + "II", data, cursor)
        if command_size < 8 or command_size % (8 if header == 32 else 4) or cursor + command_size > end:
            raise ValueError("invalid native load command size")
        if cmd not in _NON_PROVIDER and cmd not in _STRINGS:
            raise ValueError(f"unsupported native load command {cmd:#x}")
        if cmd == _BUILD_VERSION:
            if command_size < 24:
                raise ValueError("short native build-version command")
            platforms.add(struct.unpack_from(endian + "I", data, cursor + 8)[0])
        elif cmd == _VERSION_MIN_MACOSX:
            platforms.add(_PLATFORM_MACOS)
        if cmd in _STRINGS:
            fixed = 12 if cmd in (0xE, 0xF, 0x8000001C, _ENVIRONMENT) else 24
            if command_size < fixed:
                raise ValueError("short native dependency command")
            string_offset = struct.unpack_from(endian + "I", data, cursor + 8)[0]
            weak = cmd == 0x80000018
            if fixed == 24 and struct.unpack_from(endian + "I", data, cursor + 12)[0] == 0x1A741800:
                fixed = 28  # dylib_use_command, macOS 15+
                if command_size < fixed:
                    raise ValueError("short native dylib-use command")
                flags = struct.unpack_from(endian + "I", data, cursor + 24)[0]
                if flags & ~0xF:
                    raise ValueError("unknown native dylib-use flags")
                weak = weak or bool(flags & 1)
            if not fixed <= string_offset < command_size:
                raise ValueError("invalid native command string offset")
            raw = data[cursor + string_offset : cursor + command_size]
            stop = raw.find(b"\0")
            if stop <= 0:
                raise ValueError("unterminated/empty native command string")
            try:
                value = raw[:stop].decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ValueError("unproved native command string encoding") from exc
            if cmd == 0x8000001C:
                rpaths.append(value)
            elif cmd == 0xD:
                if file_type != 6 or install_name is not None:
                    raise ValueError("invalid or duplicate native dylib install name")
                install_name = value
                current_version = struct.unpack_from(endian + "I", data, cursor + 16)[0]
            elif cmd == _ENVIRONMENT:
                # dyld honours embedded variables only from the process's main
                # executable (MH_EXECUTE). Elsewhere the command is unproved.
                if file_type != 2:
                    raise ValueError("native dyld environment command outside a main executable")
                key, separator, setting = value.partition("=")
                if not separator or key != _VERSIONED_FRAMEWORK_PATH:
                    raise ValueError(f"unsupported native dyld environment variable: {key}")
                environment.append((key, setting))
            elif cmd not in (0xD, 0xF):
                dependencies.append((value, weak))
        cursor += command_size
    if cursor != end:
        raise ValueError("native load command count/size mismatch")
    return (
        tuple(dependencies),
        tuple(rpaths),
        file_type,
        install_name,
        tuple(environment),
        current_version,
        frozenset(platforms),
    )


def parse_image(path, cpu_type):
    return _read_image(path, cpu_type, digest=True)


def _image_header(path, cpu_type):
    """Parse without hashing; caller must bind bytes separately or by OS build.

    Toolchain discovery only classifies file types here. Provider graphs and the
    complete tree inventory subsequently hash their inputs before publication.
    """
    return _read_image(path, cpu_type, digest=False)


def _read_image(path, cpu_type, *, digest, digest_observer=None):
    path = Path(path)
    before = path.stat()
    if not before.st_size:
        raise ValueError("empty native Mach-O image")
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        opened = os.fstat(handle.fileno()) if digest_observer is not None else None
        dependencies, rpaths, file_type, install_name, environment, version, platforms = _parse(data, cpu_type)
        digest = hashlib.sha256(data).hexdigest() if digest else None
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"native provider changed while reading: {path}")
        # Observe only a completed full-file digest, with the actual descriptor
        # still open. Header classification cannot supply a reusable digest.
        if digest is not None and digest_observer is not None:
            digest_observer(path, handle.fileno(), before, opened, after, digest)
    return NativeImage(digest, dependencies, rpaths, file_type, install_name, environment, version, platforms)


def _system(path):
    # Normalize '..' before granting the system-build classification.
    value = os.path.normpath(str(path))
    return value.startswith(("/System/Library/", "/usr/lib/"))


def _shared_cache_contains(path):
    """dyld's own answer for an OS image that is no longer a file on disk.

    Since macOS 11 the OS dylibs live only in the dyld shared cache; the same
    cache serves every arm64 process on this OS build (a DYLD_SHARED_REGION
    override is refused by the environment contract). None means the query is
    unavailable, so callers keep the conservative by-prefix classification.
    """
    global _shared_cache_query
    if _shared_cache_query is None:
        try:
            import ctypes

            function = ctypes.CDLL("/usr/lib/libSystem.B.dylib")._dyld_shared_cache_contains_path
            function.restype, function.argtypes = ctypes.c_bool, [ctypes.c_char_p]
            _shared_cache_query = function
        except (OSError, AttributeError):
            _shared_cache_query = False
    if _shared_cache_query is False:
        return None
    return bool(_shared_cache_query(os.fsencode(str(path))))


_shared_cache_query = None


def _shared_cache_canonical(path):
    """dyld's real path of a cached image: its install name, the versioned-override key.

    ``_dyld_shared_cache_real_path`` resolves cache aliases (libc -> libSystem.B,
    Foo.framework/Foo -> Versions/A/Foo, '..' segments) to the image's canonical
    path, which is what PathOverrides::checkVersionedPath reads from the cache
    header as targetInstallName. None: the query is unavailable. False: not in
    the cache. Otherwise the canonical path string.
    """
    global _shared_cache_path_query
    if _shared_cache_path_query is None:
        try:
            import ctypes

            function = ctypes.CDLL("/usr/lib/libSystem.B.dylib")._dyld_shared_cache_real_path
            function.restype, function.argtypes = ctypes.c_char_p, [ctypes.c_char_p]
            _shared_cache_path_query = function
        except (OSError, AttributeError):
            _shared_cache_path_query = False
    if _shared_cache_path_query is False:
        return None
    value = _shared_cache_path_query(os.fsencode(str(path)))
    return False if value is None else os.fsdecode(value)


_shared_cache_path_query = None


def _macos_loadable(image):
    return _PLATFORM_MACOS in image.platforms


def working_directory_lookups(rows):
    """Relative comparison lookups a manifest proved absent in the working directory.

    dyld opens these strings literally when the compiler starts. The identity
    holder must either re-prove their absence before reuse or fix the compiler's
    directory inside an inventoried, process-immutable installation. The Metal
    toolchain contract uses the latter policy and rejects escaping lookups.
    """
    names = []
    for _digest, edges in rows:
        for _reference, target in edges:
            if isinstance(target, (tuple, list)) and target and target[0] == "versioned-overrides":
                for _index, _key, comparison in target[1]:
                    if isinstance(comparison, (tuple, list)) and comparison[0] == "absent-relative":
                        names.append(comparison[1])
    return tuple(dict.fromkeys(names))


def _unobserved(kind, path, result):
    """Default probe observer: the graph's own output never depends on observation."""


def _installed_system_image(candidate):
    """Whether dyld's search would find this OS-prefixed candidate: on disk or in the cache."""
    if candidate.exists():
        return True
    return _shared_cache_contains(candidate)


def _expand(reference, loader, executable):
    for token, root in (("@loader_path", loader.parent), ("@executable_path", executable.parent)):
        if reference == token or reference.startswith(token + "/"):
            return root / reference[len(token) :].lstrip("/")
    if reference.startswith("/"):
        return Path(reference)
    raise ValueError(f"unproved native dependency provider: {reference}")


def _versioned_framework_candidates(setting, executable, observe=_unobserved):
    """The ``Name.framework/Name`` images dyld would consider for one setting.

    Mirrors ProcessConfig::PathOverrides::addEnvVar + processVersionedPaths:
    colon-separated directories, @executable_path/@loader_path both relative to
    the main executable, one directory level, entries ending in .framework. A
    missing directory contributes nothing (dyld's directory scan fails silently)
    and so does a framework directory without its binary (getDylibInfo fails).
    SyscallDelegate::forEachInDirectory(dirsOnly=true) keeps only entries whose
    own object type is a directory, so a symlinked Name.framework is skipped.
    Anything dyld would take verbatim but we cannot prove (relative or foreign
    @-prefixed entries, empty entries) refuses.
    """
    candidates = []
    for entry in setting.split(":"):
        if not entry:
            raise ValueError("empty native versioned framework directory entry")
        directory = _expand(entry, executable, executable)
        if not directory.is_dir():
            observe("listing", directory, None)
            continue
        children = sorted(directory.iterdir(), key=lambda p: p.name)
        kinds = None
        if observe is not _unobserved:
            # A ``.framework`` child's own kind decides candidacy as much as its
            # name does, so the observed answer binds both and the decision below
            # uses that answer. Other children are never queried, as without an
            # observer; their rows carry (False, False).
            kinds = {
                child.name: (child.is_symlink(), child.is_dir()) if child.suffix == ".framework" else (False, False)
                for child in children
            }
            observe("listing", directory, tuple((child.name, *kinds[child.name]) for child in children))
        for child in children:
            if child.suffix != ".framework":
                continue
            if kinds is None:
                if child.is_symlink() or not child.is_dir():
                    continue
            elif kinds[child.name] != (False, True):
                continue
            candidate = child / child.stem
            if candidate.exists():
                if observe is not _unobserved:
                    # The graph interns this path later; its resolution is the observed answer.
                    observe("present", candidate, candidate.resolve(strict=True))
                candidates.append(candidate)
            elif candidate.is_symlink():
                raise ValueError(f"broken native versioned framework candidate: {candidate}")
            else:
                observe("absent", candidate, None)
    return candidates


def _override_key(candidate, cpu_type, bind_comparison, working_directory, observe=_unobserved, ambient_cwd=False):
    """dyld's checkVersionedPath for one candidate: (effective override key, comparison input).

    The candidate's install name is opened literally (SyscallDelegate::getDylibInfo
    -> withReadOnlyMappedFile -> open), so:

    * relative spelling (``@rpath/...``, ``@loader_path/...``, ``lib.dylib``): a
      lookup in the process working directory. Proved absent there -> selectable
      unconditionally for that string, recorded ``("absent-relative", name)`` so
      the holder can certify the launch directory; anything present -> refuse.
    * absolute path on disk: that library is the comparison input (bound as a
      comparison-only node, or header-read under the OS-build stamp); selectable
      only when the candidate's current_version is strictly greater, and then
      for requests equal to the comparison library's OWN install name
      (targetInstallName). Older or equal -> key None = proved not selectable.
    * absolute path only in the dyld shared cache: the key is the cached image's
      canonical path (``_dyld_shared_cache_real_path``, dyld's targetInstallName);
      its version is unreadable here, so the candidate is taken as selectable for
      that key, recorded ``("shared-cache", canonical)``.
    * nowhere: selectable unconditionally for its own install name
      (rdar://53215116), recorded ``"absent"``.
    * an unloadable or non-dylib comparison file, a dangling symlink, or an
      unavailable cache query: refuse.

    A key of None means exactly "proved not selectable through this mechanism";
    the candidate is still a bound node in every branch.

    ``observe`` sees each filesystem answer this decision rests on. When the
    working directory is the process's own (``ambient_cwd``), a relative lookup
    is reported by name so that an observer re-checks it where it is next used.
    """
    target = candidate.install_name
    path = Path(target)
    if not path.is_absolute():
        literal = working_directory / target
        if literal.exists() or literal.is_symlink():
            raise ValueError(f"working directory shadows native versioned comparison lookup: {target}")
        if ambient_cwd:
            observe("absent-cwd", target, None)
        else:
            observe("absent", literal, None)
        return target, ("absent-relative", target)
    if path.exists():
        if _system(path):
            header, comparison = _image_header(path, cpu_type), None
        else:
            actual = path.resolve(strict=True)
            observe("present", path, actual)
            index, header = bind_comparison(actual)
            comparison = ("comparison", index)
        if header.file_type != 6 or header.install_name is None or not _macos_loadable(header):
            raise ValueError(f"unproved native versioned comparison library: {path}")
        if comparison is None:
            comparison = ("system-build", header.install_name, header.current_version)
        key = header.install_name if candidate.current_version > header.current_version else None
        return key, comparison
    if path.is_symlink():
        raise ValueError(f"broken native versioned comparison library: {path}")
    observe("absent", path, None)
    canonical = _shared_cache_canonical(path)
    if canonical is None:
        raise ValueError(f"cannot prove native versioned comparison library: {path}")
    if canonical is False:
        return target, "absent"
    return canonical, ("shared-cache", canonical)


def dependency_graph(
    seed_paths,
    *,
    cpu_type,
    executable,
    loaded_paths=(),
    working_directory=None,
    image_reader=None,
    probe_observer=None,
):
    """Ordered provider graph, with indices preserving each root/edge association.

    Logical root order is supplied by the package inventory. Absolute resolved paths
    are traversal guards only, not persistent identity. Separate RPATH contexts are
    separate graph nodes; identical bytes in two directories cannot erase their
    different dependency providers. Cycles terminate by path+deduplicated context.

    For @rpath references dyld first checks already-loaded images by their full
    install name (Loader::getLoader / JustInTimeLoader::matchesPath). The caller
    must supply an actual stable dyld snapshot, not a package inventory or search
    directory. Bind the selected image's bytes and transitive edges. Conflicting
    loaded install names remain unproved; basename matches never qualify.

    A main executable's DYLD_VERSIONED_FRAMEWORK_PATH makes every framework in
    the listed directories a candidate provider. Each candidate is a graph node
    (bytes plus transitive edges, under the executable's RPATH context) listed
    on the executable's row as ``(index, effective key, comparison input)``
    per ``_override_key``; an on-disk comparison library is bound as a
    comparison-only node (bytes, no edges: dyld reads its header, never loads
    it), a cached one is keyed by its canonical cache path, a relative one is a
    working-directory lookup. An edge whose reference string equals an effective key records
    ``("versioned", candidate indices, ordinary target)`` with the candidate
    re-interned under the requesting node's RPATH context, which is the search
    stack dyld gives it when it replaces that edge; both alternatives stay bound
    because the AMFI embedded-variable gate is not observable here.

    ``probe_observer(kind, path, result)`` receives every non-system path answer
    the graph used: "present" (resolved target), "absent", "absent-cwd", "resolve"
    (an RPATH directory's resolution) and "listing" (a versioned framework
    directory's ``(name, is_symlink, is_dir)`` rows, the kind queried only for
    ``.framework`` children, or None). It is optional and never changes the result.
    """
    read_image = parse_image if image_reader is None else image_reader
    observe = _unobserved if probe_observer is None else probe_observer
    executable = Path(executable).resolve(strict=True)
    # The directory dyld will open relative comparison lookups in: the compiler
    # subprocess inherits the process working directory unless launched elsewhere.
    ambient_cwd = working_directory is None
    working_directory = Path(os.getcwd() if working_directory is None else working_directory)
    parsed, indices, pending, rows = {}, {}, [], []
    loaded_names = {}
    for raw in loaded_paths:
        path = Path(raw).resolve(strict=True)
        if path not in parsed:
            parsed[path] = read_image(path, cpu_type)
        name = parsed[path].install_name
        if name is not None and name.startswith("@rpath/"):
            previous = loaded_names.setdefault(name, path)
            if previous != path:
                raise ValueError(f"ambiguous loaded native install name: {name}")

    def intern(path, inherited):
        path = Path(path).resolve(strict=True)
        if path not in parsed:
            parsed[path] = read_image(path, cpu_type)
        image = parsed[path]
        own = tuple(_resolved_rpath(_expand(ref, path, executable), observe) for ref in image.rpaths)
        context = tuple(dict.fromkeys((*own, *inherited)))
        key = path, context
        if key not in indices:
            if len(indices) >= 20000:
                raise ValueError("native dependency graph exceeds proved traversal bound")
            indices[key] = len(pending)
            pending.append(key)
        return indices[key]

    def comparison_node(path):
        # Header-read selection input, bound by bytes but never traversed: a
        # distinct node kind (context None) so an ordinary load of the same file
        # under a real RPATH context keeps its own edges.
        if path not in parsed:
            parsed[path] = read_image(path, cpu_type)
        key = path, None
        if key not in indices:
            indices[key] = len(pending)
            pending.append(key)
        return indices[key], parsed[path]

    for path in seed_paths:
        intern(path, ())
    # Embedded environment of the process's main executable (dyld reads no
    # other image's). Selectable override frameworks are bound whether or not a
    # static edge names them: dlopen by install name selects them just the same.
    if executable not in parsed:
        parsed[executable] = read_image(executable, cpu_type)
    main = parsed[executable]
    overrides, environment_edges = {}, []
    if main.environment:
        main_context = tuple(
            dict.fromkeys(_resolved_rpath(_expand(ref, executable, executable), observe) for ref in main.rpaths)
        )
        for variable, setting in main.environment:
            bound = []
            for candidate in _versioned_framework_candidates(setting, executable, observe):
                index = intern(candidate, main_context)
                image = parsed[pending[index][0]]
                if image.file_type != 6 or image.install_name is None or not _macos_loadable(image):
                    raise ValueError(f"unproved native versioned framework candidate: {candidate}")
                key, comparison = _override_key(
                    image, cpu_type, comparison_node, working_directory, observe, ambient_cwd
                )
                if key is not None:
                    overrides.setdefault(key, []).append(pending[index][0])
                bound.append((index, key, comparison))
            environment_edges.append((f"{variable}={setting}", ("versioned-overrides", tuple(bound))))
    cursor = 0
    while cursor < len(pending):
        path, context = pending[cursor]
        image, edges = parsed[path], []
        if context is None:
            rows.append((image.digest, (("comparison-only", "not-loaded"),)))
            cursor += 1
            continue
        for reference, weak in image.dependencies:
            selectable = overrides.get(reference)  # dyld: exact strcmp on the requested path
            if reference.startswith("@rpath/") and reference in loaded_names:
                target = ("provider", intern(loaded_names[reference], context))
            else:
                if reference.startswith("@rpath/"):
                    candidates = [p / reference[len("@rpath/") :] for p in context]
                    if not candidates and not selectable:
                        raise ValueError(f"unproved native rpath provider: {reference}")
                else:
                    candidates = [_expand(reference, path, executable)]
                resolved = {}
                for candidate in candidates:
                    if _system(candidate):
                        # An RPATH search tries each directory in turn and dyld skips
                        # an OS-prefixed path that is neither on disk nor in the shared
                        # cache; a direct absolute reference is bound by OS build as-is.
                        if reference.startswith("@rpath/") and _installed_system_image(candidate) is False:
                            continue
                        resolved[str(candidate)] = "system-build"
                    elif candidate.exists():
                        actual = candidate.resolve(strict=True)
                        observe("present", candidate, actual)
                        resolved[str(actual)] = actual
                    elif candidate.is_symlink():
                        raise ValueError(f"broken native dependency provider: {candidate}")
                    else:
                        observe("absent", candidate, None)
                if len(resolved) > 1:
                    raise ValueError(f"ambiguous native rpath dependency provider: {reference}")
                if not resolved:
                    if not weak and not selectable:
                        raise ValueError(f"missing native dependency provider: {reference}")
                    target = "weak-absent" if weak else "absent"
                else:
                    provider = next(iter(resolved.values()))
                    target = "system-build" if provider == "system-build" else ("provider", intern(provider, context))
            if selectable:
                target = ("versioned", tuple(intern(p, context) for p in selectable), target)
            edges.append((reference, target))
        if path == executable:
            edges.extend(environment_edges)
        rows.append((image.digest, tuple(edges)))
        cursor += 1
    # Reading a loaded image's install name is NOT proving its dependencies.
    # Only traversed nodes belong to the closure; other loaded images must still
    # be added as ambient roots by the caller and have every edge checked.
    return tuple(rows), frozenset(path for path, _context in pending)


def _resolved_rpath(path, observe):
    resolved = path.resolve()
    observe("resolve", path, resolved)
    return resolved


def dependency_manifest(
    seed_paths, *, cpu_type, executable, working_directory=None, image_reader=None, probe_observer=None
):
    options = {} if image_reader is None else {"image_reader": image_reader}
    if probe_observer is not None:
        options["probe_observer"] = probe_observer
    return dependency_graph(
        seed_paths, cpu_type=cpu_type, executable=executable, working_directory=working_directory, **options
    )[0]
