"""Source-cache contract primitives (the outer-cache migration is separate).

Implementation identity is immutable for a Python process: editing installed
sources is an upgrade requiring restart, not a supported hot-reload operation.
Policy is revalidated on every call. Cache keys never use checkout paths/mtimes.
"""

import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import weakref


def _select_validation_native(import_module, find_spec, modules):
    """Only proved extension absence permits Python fallback; errors stay loud."""
    name = "triton_msl.backend._validation_native"
    if name not in modules and find_spec(name) is None:
        return None
    native = import_module(name)
    for entry in (
        "implementation_unchanged",
        "same_items",
        "identity_probes",
        "type_version",
        "mro_absent",
        "dict_keys_exact_str",
        "dict_keys_exact_bytes",
        "bytes_dict_items_are",
        "function_code_is",
        "function_state",
        "function_state_is",
        "identity_fast_path_disabled",
    ):
        if not callable(getattr(native, entry, None)):
            raise ImportError(f"validation extension has no callable {entry} entry point")
    return native


# Before framework/provider inventory, not a lazy import halfway through policy
# observation. Both persistent and resident identities bind the actual image.
from importlib import import_module as _import_module
from importlib.util import find_spec as _find_spec
import sys as _sys

_validation_native = _select_validation_native(_import_module, _find_spec, _sys.modules)
# The optional extent capability is frozen only after the helper load-time test.
_HOST_EXTENT_READY = (
    _validation_native is not None
    and getattr(_validation_native, "host_extent_ready", False) is True
    and callable(getattr(_validation_native, "host_extent", None))
)
# Optional packed copying must be selected before the first implementation or
# framework/provider stamp. Broken installed helpers propagate; absence alone
# retains the original Python copier. No launch can introduce a late native image.
from triton_msl.backend import _launch_contract as _packed_launch_contract

# Binding can also select a native image. Complete that selection before any
# provider stamp, including pointer-only launches whose binding plan stays Python.
# Do not catch broken-helper imports or refresh a previously captured stamp.
from triton_msl.backend import _launch_signature as _launch_binding_contract


SOURCE_SCHEMA = 3
_jit_guard_lock = threading.RLock()
_policy_snapshot = None


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _package_content(root):
    """Stable logical filenames plus bytes of runtime source/native inputs.

    Bind shipped C sources and actual native image bytes; neither substitutes for the other.
    Symlink targets are read for their bytes, but absolute paths never enter the
    result. Missing/unreadable files propagate instead of hashing as 'unknown'.
    """
    suffixes = {".py", ".c", ".so", ".dylib", ".a", ".bc", ".metal", ".metallib"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)
    if not files or not (root / "__init__.py").is_file():
        raise RuntimeError("cannot fingerprint an incomplete triton_msl package")
    return [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]


@functools.lru_cache(maxsize=1)
def implementation_identity():
    """Package implementation ONLY; not a claim to fingerprint the toolchain.

    The persistent compiler/SDK/dependency and packed-descriptor boundaries are
    explicitly still owed by the full B migration. This primitive is separated
    so callers cannot confuse a code identity with a complete binary identity.
    """
    return _digest(_package_content(Path(__file__).resolve().parents[1]))


def effective_policy():
    """Current settings used by codegen/eligibility, with existing truth rules.

    FORCE_PYTHON and USE_CPP are retained separately: the family predicate reads
    USE_CPP itself, so collapsing them to one boolean would lose that dependency.
    """
    from ._environment_snapshot import environment_snapshot

    return _policy_for_environment(environment_snapshot())


def _policy_for_environment(environment):
    from ._environment_snapshot import is_snapshot

    global _policy_snapshot
    cached = _policy_snapshot
    if cached is not None and cached[0] is environment:
        return cached[1].copy()
    result = {}
    for name in ("MEPT", "QUANT_MATMUL", "FAST_MATMUL", "COMPILE_SHADER", "FA_FAST"):
        result[name] = environment.get("TRITON_MSL_" + name, "1") != "0"
    for name in ("INFER_LAYOUT", "LEGACY", "USE_CPP", "FORCE_PYTHON", "CPP_SCALAR_CFG"):
        result[name] = environment.get("TRITON_MSL_" + name) == "1"
    result["HOST_EXTENT"] = (
        (True if _HOST_EXTENT_READY else "unavailable") if environment.get("TRITON_MSL_HOST_EXTENT") == "1" else False
    )
    result["FA_HALF_ACCUM"] = environment.get("TRITON_MSL_FA_HALF_ACCUM", "0") in ("1", "true", "True")
    # Preserve exact skip spelling until the caller's split/strip contract is
    # centralized; this can over-invalidate but cannot merge distinct policies.
    result["CPP_SKIP"] = environment.get("TRITON_MSL_CPP_SKIP", "")
    if is_snapshot(environment):
        _policy_snapshot = environment, result.copy()
    return result


def source_contract():
    from triton_msl import CODEGEN_VERSION

    return {
        "schema": SOURCE_SCHEMA,
        "implementation": implementation_identity(),
        "frameworks": framework_identity(),
        "policy": effective_policy(),
        "label": CODEGEN_VERSION,
    }


def source_key(mod_text, options_hash, input_metadata=None):
    # Structured fields avoid concatenation ambiguity between text and options.
    return _digest(
        {
            "source": mod_text,
            "options": options_hash,
            "contract": source_contract(),
            "input_metadata": _metadata_payload(input_metadata) if input_metadata is not None else None,
        }
    )


def _source_digest(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _target_fields(value):
    if (
        type(value) is not dict
        or set(value) != {"backend", "arch", "warp_size"}
        or type(value["backend"]) is not str
        or not value["backend"]
        or type(value["arch"]) not in (str, int)
        or type(value["warp_size"]) is not int
        or value["warp_size"] <= 0
    ):
        raise TypeError("unrepresentable native GPUTarget")
    return value


def _metadata_payload(metadata):
    """Closed codec for the one native object in Triton's compiler metadata.

    No general default=vars: unknown objects must disable caching, not silently
    lose their type. The complete initial metadata also participates in the key,
    so a hit cannot replace this compilation's target/hash/options with another's.
    """
    from triton.backends.compiler import GPUTarget

    result = dict(metadata)
    if "target" in result:
        target = result["target"]
        if type(target) is not GPUTarget:
            raise TypeError("source metadata target is not a native GPUTarget")
        result["target"] = _target_fields(
            {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size}
        )
    return result


def metadata_record(source, metadata, key, *, kind="msl-source-product"):
    # A non-JSON descriptor makes the WHOLE cache product non-cacheable. Dropping
    # individual fields (especially dictionary descriptors) loses semantics.
    payload = _metadata_payload(metadata)
    record = {
        "schema": SOURCE_SCHEMA,
        "kind": kind,
        "key": key,
        "contract": source_contract(),
        "source_sha256": _source_digest(source),
        "metadata": payload,
        "metadata_sha256": _digest(payload),
    }
    return json.dumps(record, sort_keys=True, allow_nan=False)


def decode_metadata(source, record, key, *, kind="msl-source-product"):
    """Validate an already-loaded record at memory AND disk boundaries.

    No guessed launch defaults. A missing/partial/legacy/corrupt partner is not
    proof that a cached shader has the current dispatch or output-copy contract.
    """
    try:
        if not isinstance(record, dict) or type(record.get("schema")) is not int:
            return None
        if (
            record["schema"] != SOURCE_SCHEMA
            or record.get("kind") != kind
            or record.get("key") != key
            or record.get("contract") != source_contract()
            or record.get("source_sha256") != _source_digest(source)
        ):
            return None
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or record.get("metadata_sha256") != _digest(metadata):
            return None
        metadata = dict(metadata)
        if "target" in metadata:
            from triton.backends.compiler import GPUTarget

            metadata["target"] = GPUTarget(**_target_fields(metadata["target"]))
        return metadata
    except (OSError, ValueError, TypeError):
        return None


def read_metadata(source, path, key):
    """Read a source-product envelope; stash records cannot impersonate it."""
    try:
        return decode_metadata(source, json.loads(Path(path).read_text()), key)
    except (OSError, ValueError, TypeError):
        return None


def toolchain_identity():
    from triton_msl.backend._toolchain_contract import toolchain_identity as implementation

    return implementation()


def framework_identity():
    from triton_msl.backend._framework_contract import framework_identity as implementation

    return implementation()


_STAMP_FIELDS = frozenset(("schema", "implementation", "frameworks", "policy", "label"))
_STAMP_FLAGS = (
    "MEPT",
    "QUANT_MATMUL",
    "FAST_MATMUL",
    "COMPILE_SHADER",
    "FA_FAST",
    "INFER_LAYOUT",
    "LEGACY",
    "USE_CPP",
    "FORCE_PYTHON",
    "CPP_SCALAR_CFG",
    "FA_HALF_ACCUM",
)
_STAMP_POLICY_FIELDS = frozenset((*_STAMP_FLAGS, "CPP_SKIP", "HOST_EXTENT"))


@functools.lru_cache(maxsize=8, typed=True)
def _encode_known_stamp(schema, implementation, frameworks, label, toolchain, *policy):
    # Only exact immutable primitives reach this cache. All live source/native/
    # toolchain/policy checks have ALREADY executed before its key is assembled.
    source = {
        "schema": schema,
        "implementation": implementation,
        "frameworks": frameworks,
        "label": label,
        "policy": dict(zip((*_STAMP_FLAGS, "CPP_SKIP", "HOST_EXTENT"), policy)),
    }
    return json.dumps({"schema": 1, "source": source, "toolchain": toolchain}, sort_keys=True, separators=(",", ":"))


def _encode_execution(source, toolchain):
    if (
        type(source) is dict
        and all(type(key) is str for key in source)
        and source.keys() == _STAMP_FIELDS
        and type(source["schema"]) is int
        and all(type(source[key]) is str for key in ("implementation", "frameworks", "label"))
        and type(toolchain) is str
    ):
        policy = source["policy"]
        if (
            type(policy) is dict
            and all(type(key) is str for key in policy)
            and policy.keys() == _STAMP_POLICY_FIELDS
            and all(type(policy[key]) is bool for key in _STAMP_FLAGS)
            and type(policy["CPP_SKIP"]) is str
            and (
                type(policy["HOST_EXTENT"]) is bool
                or (type(policy["HOST_EXTENT"]) is str and policy["HOST_EXTENT"] == "unavailable")
            )
        ):
            return _encode_known_stamp(
                source["schema"],
                source["implementation"],
                source["frameworks"],
                source["label"],
                toolchain,
                *(policy[key] for key in _STAMP_FLAGS),
                policy["CPP_SKIP"],
                policy["HOST_EXTENT"],
            )
    # Unknown schemas/values retain the original encoder, including its errors.
    # Never use Python's bool/int or custom equality as JSON type equivalence.
    return json.dumps({"schema": 1, "source": source, "toolchain": toolchain}, sort_keys=True, separators=(",", ":"))


_standard_source_contract = source_contract
_standard_effective_policy = effective_policy
_standard_policy_for_environment = _policy_for_environment
_standard_toolchain_identity = toolchain_identity
_execution_snapshot = None


def execution_contract():
    """Recheck live identities; reuse only our own immutable derived stamp.

    One environment capture is shared by policy and toolchain validation within
    THIS evaluation. The JIT guard and launcher still evaluate independently.
    Capture follows framework discovery/native verification, which may import
    modules or run custom finders. Never cache across those live checks.
    """
    global _execution_snapshot
    saved = _identity_fast_path.check()
    if saved is not None:
        return saved
    # Only a MISS can release obsolete certificate ownership. Finalizers may
    # re-enter or change policy: drain before selecting ANY complete-evaluator
    # input, never in the hot probe loop or after building its replacement stamp.
    # Older helper images keep their existing bounded-cache behavior; the pure
    # Python fallback owns no native program cache.
    if _validation_native is not None:
        collect = getattr(_validation_native, "probe_cache_collect", None)
        if collect is not None:
            collect()
    if source_contract is not _standard_source_contract:
        return _encode_execution(source_contract(), toolchain_identity())
    from triton_msl import CODEGEN_VERSION

    schema = SOURCE_SCHEMA
    implementation = implementation_identity()
    frameworks = framework_identity()
    # Capture framework inputs at their original phase, before later callbacks.
    _identity_fast_path.build(frameworks)

    # Select providers at their ORIGINAL observation points. A framework
    # callback can replace policy; policy can replace the toolchain provider.
    # Never restart source_contract after either callback or bypass its successor.
    policy_provider = effective_policy
    owned = False
    if policy_provider is _standard_effective_policy:
        from . import _environment_snapshot as snapshots

        capture = snapshots.environment_snapshot
        policy_helper = _policy_for_environment  # selected before argument evaluation
        environment = capture()
        policy = policy_helper(environment)
        owned = (
            capture is snapshots._standard_environment_snapshot
            and policy_helper is _standard_policy_for_environment
            and snapshots.is_snapshot(environment)
        )
    else:
        policy = policy_provider()

    compiler_provider = toolchain_identity
    shared = False
    if owned and compiler_provider is _standard_toolchain_identity:
        from . import _toolchain_contract as toolchain

        if (
            toolchain.toolchain_identity is toolchain._standard_toolchain_identity
            and toolchain._identity_for_environment is toolchain._standard_identity_for_environment
        ):
            compiler = toolchain._identity_for_environment(environment)
            shared = True
        else:
            compiler = compiler_provider()
    else:
        compiler = compiler_provider()

    if shared:
        cached = _execution_snapshot
        # Strong references, not id() values: identity reuse after GC cannot
        # turn changed inputs into a hit. Every identity/check above stays live.
        if (
            cached is not None
            and cached[0] is environment
            and cached[1] is schema
            and cached[2] is implementation
            and cached[3] is frameworks
            and cached[4] is CODEGEN_VERSION
            and cached[5] is compiler
        ):
            _identity_fast_path.finalize(
                cached[6],
                (
                    schema,
                    implementation,
                    frameworks,
                    policy_provider,
                    capture,
                    policy_helper,
                    environment,
                    compiler_provider,
                    compiler,
                    CODEGEN_VERSION,
                ),
            )
            return cached[6]
    source = {
        "schema": schema,
        "implementation": implementation,
        "frameworks": frameworks,
        "policy": policy,
        "label": CODEGEN_VERSION,
    }
    stamp = _encode_execution(source, compiler)
    if (
        shared
        and type(schema) is int
        and type(implementation) is str
        and type(frameworks) is str
        and type(CODEGEN_VERSION) is str
        and type(compiler) is str
    ):
        _execution_snapshot = environment, schema, implementation, frameworks, CODEGEN_VERSION, compiler, stamp
        _identity_fast_path.finalize(
            stamp,
            (
                schema,
                implementation,
                frameworks,
                policy_provider,
                capture,
                policy_helper,
                environment,
                compiler_provider,
                compiler,
                CODEGEN_VERSION,
            ),
        )
    return stamp


def validate_execution_contract(stamp):
    """Old/unknown/current-policy-mismatched handles refuse before runtime effects.

    Canonical serialization equality also rejects JSON type aliases such as
    schema=true vs schema=1. This stamp is not invented at cache restoration.
    """
    from triton_msl.errors import MetalNonRecoverableError

    if not isinstance(stamp, str) or not stamp or stamp != execution_contract():
        raise MetalNonRecoverableError(
            "compiled kernel execution contract is missing, stale or incompatible; "
            "recompile/clear the outer kernel cache, and restart after installation or toolchain changes"
        )
    return stamp


class _JitPolicyGuard:
    """Use Triton's per-function public pre-run hook, never a global monkeypatch."""

    def __init__(self, fn, stamp):
        self.fn = weakref.ref(fn)
        self.stamp = stamp

    def __call__(self, *args, **kwargs):
        fn = self.fn()
        if fn is None:
            return
        current = execution_contract()
        with _jit_guard_lock:
            if current == self.stamp:
                return
            # JITFunction.run executes pre_run_hooks BEFORE retrieving these
            # per-device kernel/binder caches. Clear only this owning function.
            # Already-restored/direct handles still validate at launch itself.
            fn.device_caches.clear()
            self.stamp = current


def install_jit_policy_guard(fn, stamp):
    """Best available transparent recompile, with an unconditional launcher backstop.

    Warmup-only compiled handles may not have constructed a launcher yet, and
    non-JIT restored handles need not have this public hook. Those cases safely
    refuse a stale execution contract rather than claim transparent invalidation.
    """
    if (
        fn is None
        or not callable(getattr(fn, "add_pre_run_hook", None))
        or not isinstance(getattr(fn, "device_caches", None), dict)
    ):
        return False
    with _jit_guard_lock:
        existing = getattr(fn, "_triton_msl_policy_guard", None)
        if existing is not None:
            from triton_msl.errors import MetalNonRecoverableError

            if not isinstance(existing, _JitPolicyGuard) or existing.fn() is not fn:
                raise MetalNonRecoverableError("unrecognized per-JIT Metal execution contract guard; restart")
            return True
        try:
            guard = _JitPolicyGuard(fn, stamp)
        except TypeError:
            return False  # no weakref support: the launch backstop still applies
        fn.add_pre_run_hook(guard)
        fn._triton_msl_policy_guard = guard
        return True


def binary_key(source, options_hash, route, flags):
    if route not in ("msl", "llir"):
        raise ValueError("unknown Metal compiler input route")
    return _digest(
        {
            "kind": "metal-compiled-binary",
            "source": source,
            "options": options_hash,
            "route": route,
            "flags": list(flags),
            "contract": source_contract(),
            "toolchain": toolchain_identity(),
        }
    )


def read_binary_product(path, source, key):
    """Partial, legacy, or content-mismatched binary/record pairs miss."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        record = json.loads(Path(str(path) + ".meta.json").read_text())
        metadata = decode_metadata(source, record, key, kind="metallib-product")
        if not data or metadata != {"binary_sha256": hashlib.sha256(data).hexdigest()}:
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def publish_binary_record(path, source, key, data):
    """Best-effort commit record, bound to the PRIVATE compiler output bytes.

    Publishing a record can fail without invalidating the freshly compiled
    result. It must never hash a possibly replaced public cache file and certify
    that as our output. A concurrent replacement then simply causes a cache miss.
    """
    tmp = None
    try:
        if not isinstance(data, bytes) or not data:
            raise ValueError("empty or invalid compiled binary")
        record = metadata_record(
            source, {"binary_sha256": hashlib.sha256(data).hexdigest()}, key, kind="metallib-product"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=Path(path).parent, suffix=".metallib-record.tmp", delete=False
        ) as handle:
            tmp = handle.name
            handle.write(record)
        os.replace(tmp, str(path) + ".meta.json")
        return True
    except OSError:
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# Bound last: the fast path observes the providers defined above by identity.
from . import _identity_fast_path  # noqa: E402
