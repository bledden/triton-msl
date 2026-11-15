"""Persisted per-user identity records: a finished strict digest plus cheap evidence.

The toolchain and framework identities are full-byte inventories. After one
strict inventory of a component succeeds, a record holding its finished digest
and cheap evidence (metadata, mount identity, small anchor files, stat
signatures, provider states and the native graph's path answers) is published
in a private per-user directory. A later process whose live evidence equals the
recorded evidence returns that digest, so cache keys are bit-identical to strict
mode. Any doubt is a miss: the caller runs its strict inventory unchanged. A
record never raises, refuses, or yields a digest that strict mode did not
produce when the record was published.

No attempt is made, and strict mode runs, unless ``_cache_contract`` is already
imported (this module never imports it for a computation), the record directory
is absolute, and the bundles, SDK, framework roots and executable are on APFS.
Every other device the stat evidence relies on (symlink targets, providers) is
checked when the evidence is collected; off APFS nothing is used or published.
The directory and each record file must be owned by the user and carry no group
or other mode bits and no extended ACL.

What a record cannot see is listed in docs/CACHE_CONTRACTS.md: a same-size write
through a shared mapping held open across the change on a writable framework
root, edits to SDK files other than the anchor files, byte changes that leave
every stat field unchanged on a read-only mount, privileged time or raw-device
changes, and a record forged by the same user. ``record_sha256`` detects
corruption only; it is not authentication, because the directory belongs to
that user anyway.

``TRITON_MSL_IDENTITY_RECORD``: unset or ``1`` uses records, ``refresh`` runs
strict and republishes, anything else (``0``) is strict mode, in which no record
is read or written.
"""

import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import threading
import time


SCHEMA = 1
KINDS = ("toolchain", "framework")
SWITCH = "TRITON_MSL_IDENTITY_RECORD"
DIRECTORY = "TRITON_MSL_IDENTITY_DIR"
_MAX_BYTES = 1024 * 1024
_KEEP = 64  # records per kind
_NAME = re.compile(r"(toolchain|framework)-[0-9a-f]{64}\.json\Z")
# Exactly the temporary names _write_record creates.
_TEMPORARY = re.compile(r"\.(toolchain|framework)-[0-9a-f]{64}\.json\.[0-9]+\.[0-9]+\.[0-9a-f]{12}\.tmp\Z")
_TEMPORARY_AGE = 3600  # seconds; a writer finishes or removes its own file long before
_SDK_ANCHORS = ("SDKSettings.json", "System/Library/CoreServices/SystemVersion.plist")
_CACHE_CONTRACT = "triton_msl.backend._cache_contract"
# Last outcome per kind, for status and verify output only. Never an input:
# absent, stale, invalid, validated (then hit or rejected), refresh, published,
# changed or unavailable.
_outcomes = {}


def mode(environment):
    value = environment.get(SWITCH)
    if value is None or value == "1":
        return "record"
    if value == "refresh":
        return "refresh"
    return "strict"


def directory(environment):
    # Deliberately not under TRITON_MSL_CACHE_DIR, which may be shared.
    explicit = environment.get(DIRECTORY)
    if explicit:
        # A relative directory would follow the working directory; no attempt then.
        if not os.path.isabs(explicit):
            raise RuntimeError(f"{DIRECTORY} must be an absolute path")
        return os.path.normpath(explicit)
    home = os.path.expanduser("~")
    if not os.path.isabs(home):
        raise RuntimeError("no home directory for identity records")
    return os.path.join(home, ".cache", "triton_msl", f"identity-v1-uid{os.geteuid()}")


def last_outcome(kind):
    return _outcomes.get(kind)


# ---------------------------------------------------------------- record format


class _Optional:
    __slots__ = ("item",)

    def __init__(self, item):
        self.item = item


class _List:
    __slots__ = ("item",)

    def __init__(self, item):
        self.item = item


class _Row:
    __slots__ = ("items",)

    def __init__(self, *items):
        self.items = items


_HEX = "sha256-hex"


def _conforms(value, spec):
    """Exact key sets and exact primitive types: a bool is never an int."""
    if spec is int or spec is str or spec is bool:
        return type(value) is spec
    if spec is _HEX:
        return type(value) is str and len(value) == 64 and not value.strip("0123456789abcdef")
    if type(spec) is dict:
        return (
            type(value) is dict
            and value.keys() == spec.keys()
            and all(_conforms(value[key], item) for key, item in spec.items())
        )
    if type(spec) is _Optional:
        return value is None or _conforms(value, spec.item)
    if type(spec) is _List:
        return type(value) is list and all(_conforms(item, spec.item) for item in value)
    if type(spec) is _Row:
        return (
            type(value) is list
            and len(value) == len(spec.items)
            and all(_conforms(item, kind) for item, kind in zip(value, spec.items))
        )
    raise TypeError("unknown identity record template")


_STAT = _Row(int, int, int, int, int, int, int)
_FILES = _List(_Row(str, _Optional(_HEX)))
_EVIDENCE = {
    "toolchain": {
        "base": {
            "selection": _List(_Optional(str)),
            "resolver": str,
            "roots": {"metal": str, "metallib": str, "sdk": str},
            "metadata": str,
            "toolchain": _List(
                {
                    "path": str,
                    "read_only": bool,
                    "mount": _Optional(_Row(int, bool, int, int)),
                    "root": _STAT,
                    "version_files": _FILES,
                    "tree": _HEX,
                }
            ),
            "sdk": {
                "path": str,
                "root": _STAT,
                "anchors": _FILES,
                "xcode": _Optional(
                    {"bundle": str, "version_plist": _Optional(_HEX), "code_resources": _Optional(_STAT)}
                ),
            },
        },
        "images": _List(_Row(str, _STAT)),
        "probes": _List(_Row(str, str, _Optional(str))),
    },
    "framework": {
        "base": {
            "selection": _List(_Row(str, _List(str))),
            "metadata": str,
            "os_build": str,
            "roots": _List(_Row(str, _List(str), bool, _HEX)),
            "executable": _STAT,
            "loaded": _List(str),
        },
        "images": _List(_Row(str, _STAT)),
        "probes": _List(_Row(str, str, _Optional(str))),
        "guard": _List(str),
    },
}


def _template(kind):
    return {
        "schema": int,
        "kind": str,
        "locator": _HEX,
        "producer": {"version": str, "implementation": _HEX},
        "created": int,
        "evidence": _EVIDENCE[kind],
        "digest": _HEX,
        "record_sha256": _HEX,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalized(value):
    # Evidence is compared with decoded JSON, so tuples become lists here too.
    return json.loads(_canonical(value))


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate identity record key")
        result[key] = value
    return result


def _no_constant(name):
    raise ValueError(f"non-finite identity record number: {name}")


def _imported_cache_contract():
    # Importing it here would load modules (and images) the strict inventory
    # of this process never loaded; attempts are only made once it is present.
    module = sys.modules.get(_CACHE_CONTRACT)
    if module is None:
        raise RuntimeError("identity records need triton_msl.backend._cache_contract imported first")
    return module


def producer():
    import triton_msl

    return {
        "version": str(triton_msl.__version__),
        "implementation": _imported_cache_contract().implementation_identity(),
    }


def _checksum(body):
    unsigned = {key: value for key, value in body.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical(unsigned).encode()).hexdigest()


def _decode_body(data, kind):
    """Structure and corruption check only; None for anything else."""
    try:
        body = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not _conforms(body, _template(kind)) or body["record_sha256"] != _checksum(body):
        return None
    return body


def decode(data, kind, locator):
    """The record body if it is a well-formed record of this package for this locator."""
    body = _decode_body(data, kind)
    if body is None or body["schema"] != SCHEMA or body["kind"] != kind or body["locator"] != locator:
        return None
    return body if body["producer"] == producer() else None


def encode(kind, locator, evidence, digest):
    body = {
        "schema": SCHEMA,
        "kind": kind,
        "locator": locator,
        "producer": producer(),
        "created": int(time.time()),  # diagnostic only, never validated
        "evidence": evidence,
        "digest": digest,
    }
    body["record_sha256"] = _checksum(body)
    data = _canonical(body).encode()
    if len(data) > _MAX_BYTES:
        raise ValueError("identity record exceeds its size limit")
    return data


# ------------------------------------------------------------ directory and I/O


# ctypes is imported lazily: strict mode, which never reaches these, must load
# exactly the extension modules it loaded before records existed.
_ACL_TYPE_EXTENDED = 0x00000100
_ACL_FIRST_ENTRY = 0
_acl_functions = None


def _has_extended_acl(fd):
    """Whether the object behind ``fd`` carries an extended ACL; any doubt counts as one.

    The mode-bit checks below say nothing about ACL entries, which can grant
    other users access, and a directory's inheritable entries reach every file
    created in it.
    """
    global _acl_functions
    import ctypes

    if _acl_functions is None:
        library = ctypes.CDLL(None, use_errno=True)
        get, entry, free = library.acl_get_fd_np, library.acl_get_entry, library.acl_free
        get.argtypes, get.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_void_p
        entry.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        entry.restype = ctypes.c_int
        free.argtypes, free.restype = [ctypes.c_void_p], ctypes.c_int
        _acl_functions = get, entry, free
    get, entry, free = _acl_functions
    ctypes.set_errno(0)
    acl = get(fd, _ACL_TYPE_EXTENDED)
    if not acl:
        return ctypes.get_errno() != errno.ENOENT  # ENOENT is exactly "no ACL"
    try:
        ctypes.set_errno(0)
        if entry(acl, _ACL_FIRST_ENTRY, ctypes.byref(ctypes.c_void_p())) == 0:
            return True
        return ctypes.get_errno() != errno.EINVAL  # EINVAL: an ACL without entries
    finally:
        free(acl)


def _open_directory(path, *, create=False):
    """A descriptor for the leaf, which must be ours alone; every use goes through it."""
    if create:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        status = os.fstat(fd)
        if (
            not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.geteuid()
            or status.st_mode & 0o077
            or _has_extended_acl(fd)
        ):
            raise PermissionError(f"untrusted identity record directory: {path}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_record(dir_fd, name):
    """Bytes of one private regular file, or None. O_NONBLOCK keeps a FIFO from blocking."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    try:
        status = os.fstat(fd)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or status.st_mode & 0o077
            or status.st_nlink != 1
            or not 0 < status.st_size <= _MAX_BYTES
            or _has_extended_acl(fd)
        ):
            return None
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, _MAX_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_BYTES:
                return None
        data = b"".join(chunks)
        return data if len(data) == status.st_size else None
    finally:
        os.close(fd)


def _write_record(dir_fd, name, data):
    # The temporary name never matches a record name, so readers ignore it;
    # _TEMPORARY matches it, so pruning can remove one a killed writer left.
    temporary = f".{name}.{os.getpid()}.{threading.get_ident()}.{os.urandom(6).hex()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dir_fd)
    try:
        try:
            os.fchmod(fd, 0o600)
            # Readers refuse a file with an ACL; never publish one (an entry
            # inherited from a directory that gained one after it was checked).
            if _has_extended_acl(fd):
                raise PermissionError(f"identity record would carry an ACL: {name}")
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=dir_fd)
        except OSError:
            pass
        raise


def _record_names(dir_fd):
    for name in os.listdir(dir_fd):
        match = _NAME.match(name)
        if match is None:
            continue
        try:
            status = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(status.st_mode):
            yield match.group(1), name, status


def _prune(dir_fd):
    """Keep the newest ``_KEEP`` records of each kind; drop stale temporary files."""
    by_kind = {}
    for kind, name, status in _record_names(dir_fd):
        by_kind.setdefault(kind, []).append((status.st_mtime_ns, name))
    for entries in by_kind.values():
        entries.sort()
        for _mtime, name in entries[: max(0, len(entries) - _KEEP)]:
            try:
                os.unlink(name, dir_fd=dir_fd)
            except FileNotFoundError:
                pass
    cutoff = time.time() - _TEMPORARY_AGE
    for name in os.listdir(dir_fd):
        if _TEMPORARY.match(name) is None:
            continue
        try:
            status = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if stat.S_ISREG(status.st_mode) and status.st_mtime < cutoff:
                os.unlink(name, dir_fd=dir_fd)
        except FileNotFoundError:
            pass


# -------------------------------------------------------------------- evidence


def _stat_row(status):
    return [
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_uid,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    ]


def _optional_stat(path):
    try:
        return _stat_row(os.stat(path))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _small_file_digest(path):
    """SHA-256 of a small regular anchor file, None if it does not exist.

    Only a regular file is opened, and only the file that was checked is read:
    anything else (a FIFO, a device, a replacement between the two) raises,
    which is a failed evidence collection and so a miss.
    """
    try:
        expected = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if not stat.S_ISREG(expected.st_mode) or expected.st_size > _MAX_BYTES:
        raise ValueError(f"unexpected identity anchor: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        status = os.fstat(fd)
        if (
            not stat.S_ISREG(status.st_mode)
            or (status.st_dev, status.st_ino) != (expected.st_dev, expected.st_ino)
            or status.st_size > _MAX_BYTES
        ):
            raise ValueError(f"identity anchor changed while opening: {path}")
        chunks, total = [], 0
        while total <= _MAX_BYTES:
            chunk = os.read(fd, _MAX_BYTES + 1 - total)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        data = b"".join(chunks)
        if len(data) != status.st_size:
            raise ValueError(f"identity anchor changed while reading: {path}")
        return hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


_statfs = None


def _filesystem_type(path):
    """``f_fstypename`` of the filesystem holding ``path`` (symlinks followed), from statfs(2)."""
    global _statfs
    import ctypes

    if _statfs is None:

        class FsId(ctypes.Structure):
            _fields_ = [("val", ctypes.c_int32 * 2)]

        class StatFs(ctypes.Structure):
            # <sys/mount.h> struct statfs with 64-bit inodes (the only layout on arm64).
            _fields_ = [
                ("f_bsize", ctypes.c_uint32),
                ("f_iosize", ctypes.c_int32),
                ("f_blocks", ctypes.c_uint64),
                ("f_bfree", ctypes.c_uint64),
                ("f_bavail", ctypes.c_uint64),
                ("f_files", ctypes.c_uint64),
                ("f_ffree", ctypes.c_uint64),
                ("f_fsid", FsId),
                ("f_owner", ctypes.c_uint32),
                ("f_type", ctypes.c_uint32),
                ("f_flags", ctypes.c_uint32),
                ("f_fssubtype", ctypes.c_uint32),
                ("f_fstypename", ctypes.c_char * 16),
                ("f_mntonname", ctypes.c_char * 1024),
                ("f_mntfromname", ctypes.c_char * 1024),
                ("f_flags_ext", ctypes.c_uint32),
                ("f_reserved", ctypes.c_uint32 * 7),
            ]

        if ctypes.sizeof(StatFs) != 2168:
            raise OSError("unexpected struct statfs layout")
        # x86_64 keeps the 32-bit-inode layout under the plain name.
        name = "statfs" if platform.machine() == "arm64" else "statfs$INODE64"
        function = getattr(ctypes.CDLL(None, use_errno=True), name)
        function.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFs)]
        function.restype = ctypes.c_int
        _statfs = StatFs, function
    structure, function = _statfs
    result = structure()
    if function(os.fsencode(path), ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "statfs failed", os.fsdecode(path))
    return result.f_fstypename.decode("ascii", "replace")


def _require_apfs(entries, filesystems):
    """Raise unless each ``(st_dev, path)`` is on APFS; stat evidence is trusted nowhere else.

    Another filesystem may keep coarse timestamps, under which a same-size
    rewrite within one tick leaves every stat field unchanged. ``filesystems``
    caches the answer per device for one attempt.
    """
    for device, path in entries:
        kind = filesystems.get(device)
        if kind is None:
            kind = filesystems[device] = _filesystem_type(path)
        if kind != "apfs":
            raise RuntimeError(f"identity record stat evidence needs APFS, not {kind!r}: {path}")


def _stat_on_apfs(path, filesystems):
    """``os.stat`` of ``path`` (symlinks followed), refused off APFS."""
    status = os.stat(path)
    _require_apfs([(status.st_dev, path)], filesystems)
    return status


def _mount_identity(path):
    status = os.statvfs(path)
    return [status.f_fsid, bool(status.f_flag & os.ST_RDONLY), status.f_blocks, status.f_files]


def _os_build():
    import ctypes

    function = ctypes.CDLL(None).sysctlbyname
    function.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    size = ctypes.c_size_t(0)
    if function(b"kern.osversion", None, ctypes.byref(size), None, 0) != 0 or not 0 < size.value <= 256:
        raise OSError("OS build identity is unavailable")
    buffer = ctypes.create_string_buffer(size.value)
    if function(b"kern.osversion", buffer, ctypes.byref(size), None, 0) != 0 or not buffer.value:
        raise OSError("OS build identity is unavailable")
    return buffer.value.decode("ascii")


def _version_files(root):
    rows = [["ToolchainInfo.plist", _small_file_digest(root / "ToolchainInfo.plist")]]
    try:
        names = sorted(os.listdir(root / "usr" / "metal"))
    except (FileNotFoundError, NotADirectoryError):
        names = []
    for name in names:
        for leaf in ("ToolchainInfo.json", "ToolchainInfo.plist"):
            relative = f"usr/metal/{name}/{leaf}"
            rows.append([relative, _small_file_digest(root / relative)])
    return rows


def _toolchain_bundles(selected):
    roots = selected[0]
    return sorted({Path(roots[role]).resolve(strict=True) for role in ("metal", "metallib")}, key=str)


def _toolchain_base(selection, selected, filesystems):
    from . import _toolchain_contract as toolchain

    roots, entries, metadata = selected
    bundles, devices = [], {}
    for path in _toolchain_bundles(selected):
        read_only = bool(toolchain._read_only_mount(path))
        root = os.lstat(path)
        devices.setdefault(root.st_dev, path)
        bundles.append(
            {
                "path": str(path),
                "read_only": read_only,
                "mount": _mount_identity(path) if read_only else None,
                "root": _stat_row(root),
                "version_files": _version_files(path),
                # Always the whole tree: a symlink inside a read-only bundle may
                # resolve to a writable location, which the mount does not cover.
                "tree": toolchain._stat_signature(path, devices=devices),
            }
        )
    sdk = Path(roots["sdk"]).resolve(strict=True)
    sdk_root = os.lstat(sdk)
    devices.setdefault(sdk_root.st_dev, sdk)
    bundle = next((parent for parent in sdk.parents if parent.suffix == ".app"), None)
    xcode = None
    if bundle is not None:
        seal = bundle / "Contents" / "_CodeSignature" / "CodeResources"
        xcode = {
            "bundle": str(bundle),
            "version_plist": _small_file_digest(bundle / "Contents" / "version.plist"),
            "code_resources": _optional_stat(seal),
        }
        if xcode["code_resources"] is not None:
            devices.setdefault(xcode["code_resources"][0], seal)
    _require_apfs(devices.items(), filesystems)
    return _normalized(
        {
            "selection": list(selection),
            "resolver": str(entries["resolver"]),
            "roots": {role: str(roots[role]) for role in ("metal", "metallib", "sdk")},
            "metadata": _canonical(metadata),
            "toolchain": bundles,
            "sdk": {
                "path": str(sdk),
                "root": _stat_row(sdk_root),
                "anchors": [[name, _small_file_digest(sdk / name)] for name in _SDK_ANCHORS],
                "xcode": xcode,
            },
        }
    )


def _framework_base(roots, selection, filesystems):
    from . import _framework_contract as framework
    from . import _toolchain_contract as toolchain

    physical, seen, devices = [], set(), {}
    for name, paths in sorted(roots.items()):
        excludes, allow_empty = framework._inventory_options(name)
        for path in paths:
            key = path, excludes, allow_empty
            if key in seen:
                continue
            seen.add(key)
            # The same traversal, exclusions and options as the strict inventory of this root.
            signature = toolchain._stat_signature(
                path, ignore_bytecode=True, exclude_root_names=excludes, allow_empty=allow_empty, devices=devices
            )
            physical.append([str(path), list(excludes), allow_empty, signature])
    executable = os.stat(roots["python-executable"][0])
    devices.setdefault(executable.st_dev, roots["python-executable"][0])
    _require_apfs(devices.items(), filesystems)
    return _normalized(
        {
            "selection": [[name, list(paths)] for name, paths in selection[0]],
            "metadata": selection[1],
            "os_build": _os_build(),
            "roots": physical,
            "executable": _stat_row(executable),
            "loaded": [os.fsdecode(raw) for raw in framework._loaded_native_images()],
        }
    )


def _probe_text(result):
    if result is None or type(result) is str:
        return result
    if type(result) in (tuple, list):
        # A listing: one (name, is_symlink, is_dir) row per child, in name order.
        return _canonical([[str(name), bool(link), bool(directory)] for name, link, directory in result])
    return str(result)


def _probe_value(kind, path):
    """The live answer to a recorded native-graph probe, in the recorded form."""
    target = Path(path)
    if kind == "present":
        return str(target.resolve(strict=True)) if target.exists() else None
    if kind in ("absent", "absent-cwd"):
        if kind == "absent-cwd":
            target = Path(os.getcwd()) / path
        return None if not (target.exists() or target.is_symlink()) else "present"
    if kind == "resolve":
        return str(target.resolve())
    if kind == "listing":
        if not target.is_dir():
            return None
        # As observed: the kind is queried only for the children it can decide.
        rows = sorted(
            [child.name, child.is_symlink(), child.is_dir()]
            if child.suffix == ".framework"
            else [child.name, False, False]
            for child in target.iterdir()
        )
        return _canonical(rows)
    raise ValueError(f"unknown identity record probe: {kind}")


# --------------------------------------------------------------------- attempts


class _Attempt:
    """Record use for one component within one identity construction.

    ``lookup`` returns a validated record body, or None after capturing the
    evidence that precedes the strict inventory. The caller reports ``used``
    once it has finished from that body, or ``rejected`` if it could not, so
    the outcome is "hit" only for a digest that was returned. ``image`` and
    ``probe`` observe the strict native graph without affecting it.
    ``publish`` writes a record only if nothing observed changed. None of these
    raise ``Exception``.
    """

    def __init__(self, kind, policy, path, locator, collect, filesystems):
        self.kind, self.policy, self.path, self.locator = kind, policy, path, locator
        self._collect = collect
        self._filesystems = filesystems
        self.base = None
        self._usable = True
        self._images = {}
        self._probes = {}

    def _outcome(self, outcome):
        _outcomes[self.kind] = outcome, self.locator

    def used(self):
        self._outcome("hit")

    def rejected(self):
        self._outcome("rejected")

    def _image_state(self, path):
        return _stat_row(_stat_on_apfs(path, self._filesystems))

    def image(self, path, fd, before, opened, after, digest):
        """``digest_observer`` of a strict provider read: the state its bytes were hashed in."""
        try:
            self._images.setdefault(str(path), set()).add(tuple(_stat_row(before)))
        except Exception:
            self._usable = False

    def probe(self, kind, path, result):
        try:
            value = _probe_text(result)
            if self._probes.setdefault((kind, str(path)), value) != value:
                self._usable = False
        except Exception:
            self._usable = False

    def _load(self):
        try:
            fd = _open_directory(self.path)
        except FileNotFoundError:
            return None
        except Exception:
            # An untrusted directory is never written either.
            self._usable = False
            raise
        try:
            data = _read_record(fd, f"{self.kind}-{self.locator}.json")
        except Exception:
            data = None  # a symlink, unreadable or vanished record is only a miss
        finally:
            os.close(fd)
        return None if data is None else decode(data, self.kind, self.locator)

    def lookup(self):
        try:
            body = self._load() if self.policy == "record" else None
            self.base = self._collect(self._filesystems)
            if body is None:
                self._outcome("absent" if self.policy == "record" else self.policy)
                return None
            evidence = body["evidence"]
            if evidence["base"] != self.base:
                self._outcome("stale")
                return None
            for path, row in evidence["images"]:
                if self._image_state(path) != row:
                    self._outcome("stale")
                    return None
            for kind, path, result in evidence["probes"]:
                if _probe_value(kind, path) != result:
                    self._outcome("stale")
                    return None
            if "guard" in evidence and not set(evidence["guard"]) <= {path for path, _row in evidence["images"]}:
                self._outcome("invalid")
                return None
            self._outcome("validated")  # "hit" only once the caller has used it
            return body
        except Exception:
            if self.base is None:
                self._usable = False  # nothing to compare a publication against
            self._outcome("stale" if self._usable else "unavailable")
            return None

    def publish(self, digest, guard=None):
        try:
            if not self._usable or self.base is None:
                return
            # Evidence observed after the inventory must equal what preceded it;
            # a tree that changed during the inventory is never recorded.
            if self._collect(self._filesystems) != self.base:
                self._outcome("changed")
                return
            images = []
            for path, states in sorted(self._images.items()):
                if len(states) != 1 or self._image_state(path) != list(next(iter(states))):
                    self._outcome("changed")
                    return
                images.append([path, list(next(iter(states)))])
            probes = []
            for (kind, path), result in sorted(self._probes.items()):
                if _probe_value(kind, path) != result:
                    self._outcome("changed")
                    return
                probes.append([kind, path, result])
            evidence = {"base": self.base, "images": images, "probes": probes}
            if self.kind == "framework":
                if guard is None:
                    return
                evidence["guard"] = sorted(str(path) for path in guard)
            data = encode(self.kind, self.locator, evidence, digest)
            if decode(data, self.kind, self.locator) is None:
                return  # never write what could not be read back
            fd = _open_directory(self.path, create=True)
            try:
                _write_record(fd, f"{self.kind}-{self.locator}.json", data)
                self._outcome("published")
                try:
                    _prune(fd)
                except Exception:
                    pass
            finally:
                os.close(fd)
        except Exception:
            self._outcome("unavailable")


def _common_locator():
    from ._framework_contract import _native_executable

    return [
        SCHEMA,
        os.geteuid(),
        _imported_cache_contract().implementation_identity(),
        sys.version,
        sys.implementation.cache_tag,
        platform.machine(),
        str(_native_executable()),
    ]


def _locator(kind, parts, common=None):
    common = _common_locator() if common is None else common
    return hashlib.sha256(_canonical([common, kind, parts]).encode()).hexdigest()


def _load_like_strict(kind):
    """Import what the strict inventory of ``kind`` imports, in the same order.

    The framework identity binds the ordered list of loaded images, and these
    imports load extension modules. A process that uses a record must reach the
    list that a process running the strict inventory reaches, or the two would
    look up and publish different framework records.
    """
    from . import _toolchain_contract as toolchain

    def graph():
        from . import _native_dependencies  # noqa: F401

    def pool():
        if toolchain._inventory_workers(None) > 1:
            from concurrent.futures import ThreadPoolExecutor  # noqa: F401

    # Toolchain: native providers, then tree inventories. Framework: the reverse.
    for step in (graph, pool) if kind == "toolchain" else (pool, graph):
        step()


def _offered(environment):
    """The record directory, if an attempt may be made at all; raises otherwise.

    Nothing here imports or loads anything: a direct identity call made before
    ``_cache_contract`` is imported runs strict, as does a relative directory.
    """
    if _CACHE_CONTRACT not in sys.modules:
        raise RuntimeError("triton_msl.backend._cache_contract is not imported")
    return directory(environment)


def _roots_on_apfs(paths, filesystems):
    _require_apfs([(os.stat(path).st_dev, path) for path in paths], filesystems)


def toolchain_attempt(environment, policy, selection, selected):
    """None unless records are in use and the locator can be formed."""
    try:
        path = _offered(environment)
        _load_like_strict("toolchain")
        roots, entries, _metadata = selected
        parts = [
            list(selection),
            str(entries["resolver"]),
            {role: str(roots[role]) for role in ("metal", "metallib", "sdk")},
        ]
        locator = _locator("toolchain", _normalized(parts))
        # Every tree the record trusts by stat state; the full set is checked again with the evidence.
        filesystems = {}
        _roots_on_apfs([*_toolchain_bundles(selected), Path(roots["sdk"]).resolve(strict=True)], filesystems)
        return _Attempt(
            "toolchain",
            policy,
            path,
            locator,
            lambda filesystems: _toolchain_base(selection, selected, filesystems),
            filesystems,
        )
    except Exception:
        _outcomes["toolchain"] = "unavailable", None
        return None


def framework_attempt(environment, policy, roots, selection):
    try:
        path = _offered(environment)
        from ._framework_contract import _loaded_native_images

        _load_like_strict("framework")
        # Its imports may load images too, so the list is read after them.
        common = _common_locator()
        loaded = [os.fsdecode(raw) for raw in _loaded_native_images()]
        parts = [
            [[name, list(paths)] for name, paths in selection[0]],
            selection[1],
            hashlib.sha256(_canonical(loaded).encode()).hexdigest(),
        ]
        locator = _locator("framework", _normalized(parts), common)
        filesystems = {}
        _roots_on_apfs([root for paths in roots.values() for root in paths], filesystems)

        def collect(filesystems):
            base = _framework_base(roots, selection, filesystems)
            if base["loaded"] != loaded:
                raise RuntimeError("loaded native images changed during identity record lookup")
            return base

        return _Attempt("framework", policy, path, locator, collect, filesystems)
    except Exception:
        _outcomes["framework"] = "unavailable", None
        return None


# ------------------------------------------------------------------ maintenance


def status(environment=None):
    """Plain data about the record directory; never creates it."""
    environment = os.environ if environment is None else environment
    result = {
        "mode": mode(environment),
        "directory": None,
        "exists": True,
        "trusted": False,
        "error": None,
        "records": [],
    }
    try:
        result["directory"] = path = directory(environment)
        fd = _open_directory(path)
    except FileNotFoundError:
        # The state before the first publication, which creates it; not an error.
        result["exists"] = False
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        result["trusted"] = True
        try:
            current = producer()
        except Exception:
            current = None  # no record can be current without the cache contract
        for kind, name, _status in sorted(_record_names(fd), key=lambda item: item[1]):
            entry = {"name": name, "kind": kind, "created": None, "producer": None, "decodes": False, "current": False}
            try:
                data = _read_record(fd, name)
                body = None if data is None else _decode_body(data, kind)
            except Exception:
                body = None
            if body is not None and body["schema"] == SCHEMA and body["kind"] == kind:
                entry.update(
                    created=body["created"],
                    producer=body["producer"],
                    decodes=True,
                    current=current is not None
                    and body["producer"] == current
                    and name == f"{kind}-{body['locator']}.json",
                )
            result["records"].append(entry)
    finally:
        os.close(fd)
    return result


def clear(environment=None):
    """Delete record files only: regular toolchain-*.json / framework-*.json, never recursive.

    A directory that does not exist holds no records: 0, and nothing is created.
    """
    environment = os.environ if environment is None else environment
    try:
        fd = _open_directory(directory(environment))
    except FileNotFoundError:
        return 0
    removed = 0
    try:
        for name in os.listdir(fd):
            if not (name.startswith(("toolchain-", "framework-")) and name.endswith(".json")):
                continue
            if stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                os.unlink(name, dir_fd=fd)
                removed += 1
    finally:
        os.close(fd)
    return removed


def remove(environment, kind, locator):
    fd = _open_directory(directory(environment))
    try:
        os.unlink(f"{kind}-{locator}.json", dir_fd=fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(fd)
