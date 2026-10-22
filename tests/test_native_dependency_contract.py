"""Checked Mach-O dependency closure; fixtures are data, never executed."""

from pathlib import Path
import struct
import sys

import pytest

ARM64 = 0x0100000C
X86_64 = 0x01000007


MH_EXECUTE, MH_DYLIB = 2, 6
LC_DYLD_ENVIRONMENT = 0x27


def _command(cmd, text, version=0):
    # LC_RPATH, LC_LOAD_DYLINKER and LC_DYLD_ENVIRONMENT share the 12-byte lc_str layout;
    # dylib commands carry timestamp, current_version (packed X.Y.Z) and compatibility_version.
    fixed = 12 if cmd in (0x8000001C, 0xE, LC_DYLD_ENVIRONMENT) else 24
    body = text.encode() + b"\0"
    size = (fixed + len(body) + 7) & ~7
    tail = struct.pack("<III", 0, version, 0) if fixed == 24 else b""
    return (struct.pack("<III", cmd, size, fixed) + tail + body).ljust(size, b"\0")


def _version(major, minor=0, patch=0):
    return (major << 16) | (minor << 8) | patch


LC_BUILD_VERSION, PLATFORM_MACOS = 0x32, 1


def _image(*commands, cpu=ARM64, file_type=MH_DYLIB, platform=PLATFORM_MACOS):
    # LC_BUILD_VERSION last, so byte-offset damage fixtures keep hitting the first command.
    if platform is not None:
        commands = (*commands, struct.pack("<6I", LC_BUILD_VERSION, 24, platform, 0, 0, 0))
    body = b"".join(commands)
    return struct.pack("<8I", 0xFEEDFACF, cpu, 0, file_type, len(commands), len(body), 0, 0) + body


def _framework(root, name, *commands, install_name=None, payload=b"", version=0):
    """A real framework bundle layout: Versions/A binary, Current and top-level symlinks."""
    versions = root / f"{name}.framework" / "Versions"
    (versions / "A").mkdir(parents=True)
    binary = versions / "A" / name
    identity = _command(0xD, install_name or f"@rpath/{name}.framework/Versions/A/{name}", version)
    binary.write_bytes(_image(identity, *commands) + payload)
    (versions / "Current").symlink_to("A")
    (root / f"{name}.framework" / name).symlink_to(f"Versions/Current/{name}")
    return binary


def _tool(root, *commands, versioned="@executable_path/../Frameworks", file_type=MH_EXECUTE):
    """An executable whose LC_DYLD_ENVIRONMENT names the versioned framework directory."""
    (root / "bin").mkdir(exist_ok=True)
    tool = root / "bin" / "tool"
    environment = (
        () if versioned is None else (_command(LC_DYLD_ENVIRONMENT, f"DYLD_VERSIONED_FRAMEWORK_PATH={versioned}"),)
    )
    tool.write_bytes(_image(*commands, *environment, file_type=file_type))
    return tool


def test_parser_keeps_dependency_kinds_and_rpaths(tmp_path):
    from triton_msl.backend._native_dependencies import parse_image

    path = tmp_path / "image.dylib"
    path.write_bytes(
        _image(
            _command(0xC, "@rpath/one.dylib"),
            _command(0x80000018, "/usr/lib/two.dylib"),
            _command(0x8000001F, "@loader_path/three.dylib"),
            _command(0x8000001C, "@loader_path/lib"),
            _command(0xD, "@rpath/this.dylib"),
        )
    )
    image = parse_image(path, ARM64)
    assert image.dependencies == (
        ("@rpath/one.dylib", False),
        ("/usr/lib/two.dylib", True),
        ("@loader_path/three.dylib", False),
    )
    assert image.rpaths == ("@loader_path/lib",)
    assert image.install_name == "@rpath/this.dylib"
    assert image.platforms == frozenset({PLATFORM_MACOS})
    path.write_bytes(_image(platform=None))
    assert parse_image(path, ARM64).platforms == frozenset()
    # There is no well-defined full-name binding with two LC_ID_DYLIBs.
    path.write_bytes(_image(_command(0xD, "@rpath/one.dylib"), _command(0xD, "@rpath/two.dylib")))
    with pytest.raises(ValueError, match="duplicate.*install name"):
        parse_image(path, ARM64)
    data = bytearray(_image(_command(0xD, "@rpath/not-a-dylib")))
    struct.pack_into("<I", data, 12, 8)  # MH_BUNDLE is not MH_DYLIB
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid.*install name"):
        parse_image(path, ARM64)


def test_fat_parser_selects_the_actual_cpu_slice(tmp_path):
    from triton_msl.backend._native_dependencies import parse_image

    a, b = _image(_command(0xC, "/usr/lib/arm.dylib")), _image(_command(0xC, "/usr/lib/x86.dylib"), cpu=X86_64)
    start = 8 + 40
    data = struct.pack(">II", 0xCAFEBABE, 2)
    data += struct.pack(">IIIII", X86_64, 0, start, len(b), 0)
    data += struct.pack(">IIIII", ARM64, 0, start + len(b), len(a), 0)
    path = tmp_path / "fat.so"
    path.write_bytes(data + b + a)
    assert parse_image(path, ARM64).dependencies == (("/usr/lib/arm.dylib", False),)


@pytest.mark.parametrize(
    "damage",
    ["short_header", "short_commands", "zero_command", "bad_string_offset", "no_nul", "dyld_environment", "lazy_info"],
)
def test_unproved_native_load_commands_refuse(tmp_path, damage):
    from triton_msl.backend._native_dependencies import parse_image

    command = _command(0xC, "/usr/lib/lib.dylib")
    data = _image(command)
    if damage == "short_header":
        data = data[:12]
    elif damage == "short_commands":
        data = data[:-1]
    elif damage == "zero_command":
        data = data[:36] + bytes(4) + data[40:]
    elif damage == "bad_string_offset":
        data = data[:40] + struct.pack("<I", 10000) + data[44:]
    elif damage == "no_nul":
        data = data[:56] + b"x" * (len(data) - 56)
    elif damage == "dyld_environment":
        data = _image(_command(0x27, "DYLD_LIBRARY_PATH=/foreign"))
    else:
        data = _image(struct.pack("<4I", 0x3A, 16, 0, 0))
    path = tmp_path / "bad.so"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="native|Mach-O|load|command|string"):
        parse_image(path, ARM64)


def test_ambiguous_rpath_is_not_a_basename_guess(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest

    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "lib.dylib").write_bytes(_image() + name.encode())
    root = tmp_path / "root.so"
    root.write_bytes(
        _image(
            _command(0xC, "@rpath/lib.dylib"),
            _command(0x8000001C, "@loader_path/a"),
            _command(0x8000001C, "@loader_path/b"),
        )
    )
    with pytest.raises(ValueError, match="ambiguous"):
        dependency_manifest([root], cpu_type=ARM64, executable=root)


def test_absolute_external_provider_is_content_bound(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest

    external = tmp_path / "external.dylib"
    external.write_bytes(_image() + b"first implementation")
    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, str(external))))
    first = dependency_manifest([root], cpu_type=ARM64, executable=root)
    external.write_bytes(_image() + b"second implementation")
    assert dependency_manifest([root], cpu_type=ARM64, executable=root) != first


def test_loader_relative_dependency_and_cycle_are_bounded(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest

    a, b = tmp_path / "a.so", tmp_path / "b.dylib"
    a.write_bytes(_image(_command(0xC, "@loader_path/b.dylib")))
    b.write_bytes(_image(_command(0xC, "@loader_path/a.so")))
    rows = dependency_manifest([a], cpu_type=ARM64, executable=a)
    assert len(rows) == 2


@pytest.mark.parametrize(
    "reference", ["relative.dylib", "@unknown/lib.dylib", "@rpath/missing.dylib", "@loader_path/missing.dylib"]
)
def test_unknown_or_missing_strong_provider_refuses(tmp_path, reference):
    from triton_msl.backend._native_dependencies import dependency_manifest

    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, reference)))
    with pytest.raises((ValueError, FileNotFoundError), match="native|dependency|rpath|provider"):
        dependency_manifest([root], cpu_type=ARM64, executable=root)


def test_system_framework_uses_the_separate_os_build_contract(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest

    root = tmp_path / "root.so"
    root.write_bytes(_image(_command(0xC, "/System/Library/Frameworks/Metal.framework/Metal")))
    rows = dependency_manifest([root], cpu_type=ARM64, executable=root)
    assert rows[0][1] == (("/System/Library/Frameworks/Metal.framework/Metal", "system-build"),)


def test_framework_identity_includes_external_native_bytes(tmp_path, monkeypatch):
    from triton_msl.backend import _framework_contract as framework

    root = tmp_path / "package"
    root.mkdir()
    (root / "__init__.py").write_text("VERSION = 'unchanged'\n")
    external = tmp_path / "external.dylib"
    external.write_bytes(_image() + b"first")
    image = root / "plugin.so"
    image.write_bytes(_image(_command(0xC, str(external))))
    roots = {"triton": (root,)}
    monkeypatch.setattr(framework, "_snapshot", None)
    monkeypatch.setattr(framework, "_native_guard", None)
    monkeypatch.setattr(framework, "_discover_selection", lambda: (roots, {}))

    def native(_roots):
        from triton_msl.backend._native_dependencies import dependency_manifest

        return dependency_manifest([image], cpu_type=ARM64, executable=image)

    monkeypatch.setattr(framework, "_native_dependencies", native, raising=False)
    first = framework.framework_identity()
    external.write_bytes(_image() + b"second")
    monkeypatch.setattr(framework, "_snapshot", None)  # model the required restart
    assert framework.framework_identity() != first


def test_extension_supplies_inherited_rpath_to_its_dylib(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_manifest

    libs = tmp_path / "lib"
    libs.mkdir()
    entry, first, second = tmp_path / "core.so", libs / "first.dylib", libs / "second.dylib"
    entry.write_bytes(_image(_command(0xC, "@rpath/first.dylib"), _command(0x8000001C, "@loader_path/lib")))
    first.write_bytes(_image(_command(0xC, "@rpath/second.dylib")))
    second.write_bytes(_image())
    rows = dependency_manifest([entry], cpu_type=ARM64, executable=entry)
    assert len(rows) == 3
    assert rows[0][1] == (("@rpath/first.dylib", ("provider", 1)),)
    assert rows[1][1] == (("@rpath/second.dylib", ("provider", 2)),)
    # In isolation the same dylib is NOT given a guessed adjacent-directory path.
    with pytest.raises(ValueError, match="rpath"):
        dependency_manifest([first], cpu_type=ARM64, executable=entry)


def test_new_loaded_external_native_image_refuses_before_reusing_snapshot(tmp_path):
    from triton_msl.backend._framework_contract import _LoadedNativeGuard

    known, foreign = tmp_path / "known.so", tmp_path / "foreign.so"
    known.write_bytes(_image())
    foreign.write_bytes(_image() + b"different")
    names = [str(known).encode()]
    guard = object.__new__(_LoadedNativeGuard)
    guard.count = lambda: len(names)
    guard.name = lambda i: names[i]
    guard.providers = frozenset((known.resolve(),))
    guard.checked_count = None
    guard.approved_names = set()
    guard.verify()
    names.append(str(foreign).encode())
    with pytest.raises(RuntimeError, match="outside.*proved"):
        guard.verify()
    assert guard.checked_count == 1  # failure never certifies the new selection


def test_same_count_native_replacement_is_not_a_warm_identity_hit(tmp_path):
    from triton_msl.backend._framework_contract import _LoadedNativeGuard

    known, foreign = tmp_path / "known.so", tmp_path / "foreign.so"
    known.write_bytes(_image())
    foreign.write_bytes(_image() + b"other")
    names = [str(known).encode()]
    guard = object.__new__(_LoadedNativeGuard)
    guard.count, guard.name = lambda: len(names), lambda i: names[i]
    guard.providers = frozenset((known.resolve(),))
    guard.checked_count, guard.approved_names = None, set()
    guard.verify()
    names[0] = str(foreign).encode()
    with pytest.raises(RuntimeError, match="outside.*proved"):
        guard.verify()
    assert names[0] not in guard.approved_names


def test_python_entrypoints_are_not_all_packaged_dylibs_or_linux_resources(tmp_path):
    from triton_msl.backend._framework_contract import _native_entrypoints

    executable, extension = tmp_path / "Python", tmp_path / "core.so"
    executable.write_bytes(_image())
    extension.write_bytes(_image())
    (tmp_path / "dependency.dylib").write_bytes(_image())
    (tmp_path / "linux.so").write_bytes(b"\x7fELF" + b"Linux-only packaged resource")
    roots = {"python-executable": (executable,), "mlx": (tmp_path,)}
    assert set(_native_entrypoints(roots)) == {executable, extension}
    (tmp_path / "unknown.so").write_bytes(b"unproved native entry point")
    with pytest.raises(RuntimeError, match="unproved.*entry"):
        _native_entrypoints(roots)


def test_torch_explicit_ctypes_global_dependency_is_a_named_entrypoint(tmp_path):
    from triton_msl.backend._framework_contract import _native_entrypoints

    executable = tmp_path / "Python"
    executable.write_bytes(_image())
    package = tmp_path / "torch"
    (package / "lib").mkdir(parents=True)
    library = package / "lib/libtorch_global_deps.dylib"
    library.write_bytes(_image())
    (package / "lib/not_an_entry.dylib").write_bytes(_image())
    roots = {"python-executable": (executable,), "torch": (package,)}
    assert set(_native_entrypoints(roots)) == {executable, library}


def test_executable_versioned_framework_path_binds_selectable_providers(tmp_path, monkeypatch):
    """Xcode 27's coreai-build layout: LC_DYLD_ENVIRONMENT=DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/..."""
    from triton_msl.backend import _native_dependencies as native
    from triton_msl.backend._native_dependencies import dependency_graph, parse_image

    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: False)  # the host cache is not a fixture

    frameworks = tmp_path / "Frameworks"
    os_name = "/System/Library/SubFrameworks/Alpha.framework/Versions/A/Alpha"
    _framework(frameworks, "Alpha", _command(0xC, "@rpath/Beta.framework/Versions/A/Beta"), install_name=os_name)
    _framework(frameworks, "Beta", payload=b"beta")
    _framework(frameworks, "Gamma", payload=b"never referenced statically")
    tool = _tool(tmp_path, _command(0xC, os_name), _command(0x8000001C, "@executable_path/../Frameworks"))
    image = parse_image(tool, ARM64)
    assert image.file_type == MH_EXECUTE
    assert image.environment == (("DYLD_VERSIONED_FRAMEWORK_PATH", "@executable_path/../Frameworks"),)

    rows, providers = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    alpha, beta, gamma = (
        parse_image(frameworks / f"{n}.framework/{n}", ARM64).digest for n in ("Alpha", "Beta", "Gamma")
    )
    assert [row[0] for row in rows] == [image.digest, alpha, beta, gamma]
    # The OS copy is bound by the OS-build contract, the override by its bytes: dyld's
    # current_version comparison against the shared cache picks one at launch.
    assert rows[0][1] == (
        (os_name, ("versioned", (1,), "system-build")),
        (
            "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Frameworks",
            (
                "versioned-overrides",
                (
                    (1, os_name, "absent"),
                    (
                        2,
                        "@rpath/Beta.framework/Versions/A/Beta",
                        ("absent-relative", "@rpath/Beta.framework/Versions/A/Beta"),
                    ),
                    (
                        3,
                        "@rpath/Gamma.framework/Versions/A/Gamma",
                        ("absent-relative", "@rpath/Gamma.framework/Versions/A/Gamma"),
                    ),
                ),
            ),
        ),
    )
    # Transitive edges of an override are proved under the requesting RPATH context;
    # an @rpath install name is also an override key (dyld strcmp on the request).
    assert rows[1][1] == (("@rpath/Beta.framework/Versions/A/Beta", ("versioned", (2,), ("provider", 2))),)
    assert rows[2][1] == () and rows[3][1] == ()
    # Every candidate is a provider node, including one no static edge names.
    assert {p.name for p in providers} == {"tool", "Alpha", "Beta", "Gamma"}
    # Content binding: any candidate's bytes change the identity.
    for name in ("Alpha", "Beta", "Gamma"):
        binary = frameworks / f"{name}.framework/Versions/A/{name}"
        binary.write_bytes(binary.read_bytes() + b"rebuilt")
        assert dependency_graph([tool], cpu_type=ARM64, executable=tool)[0] != rows


def test_versioned_override_serves_an_otherwise_missing_request(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_graph

    frameworks = tmp_path / "Frameworks"
    _framework(frameworks, "Delta", install_name="@rpath/Delta.framework/Versions/A/Delta")
    # No LC_RPATH at all: without the override this @rpath request is unproved.
    tool = _tool(tmp_path, _command(0xC, "@rpath/Delta.framework/Versions/A/Delta"))
    rows, _ = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert rows[0][1][0] == ("@rpath/Delta.framework/Versions/A/Delta", ("versioned", (1,), "absent"))
    plain = _tool(tmp_path, _command(0xC, "@rpath/Delta.framework/Versions/A/Delta"), versioned=None)
    with pytest.raises(ValueError, match="rpath"):
        dependency_graph([plain], cpu_type=ARM64, executable=plain)


def test_versioned_directory_scan_matches_dyld(tmp_path, monkeypatch):
    """Missing directory and framework-without-binary contribute nothing; two directories both bind."""
    from triton_msl.backend import _native_dependencies as native
    from triton_msl.backend._native_dependencies import dependency_graph

    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: False)

    tool = _tool(tmp_path, versioned="@executable_path/../Missing")
    rows, providers = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert rows[0][1] == (("DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Missing", ("versioned-overrides", ())),)
    assert providers == {tool.resolve()}

    first, second = tmp_path / "First", tmp_path / "Second"
    _framework(first, "Same", install_name="/System/Library/Frameworks/Same.framework/Versions/A/Same", payload=b"1")
    _framework(second, "Same", install_name="/System/Library/Frameworks/Same.framework/Versions/A/Same", payload=b"2")
    (first / "Empty.framework").mkdir()  # dyld: getDylibInfo fails, no override
    (first / "loose.dylib").write_bytes(_image())  # not a framework entry
    _framework(
        tmp_path / "Elsewhere", "Linked", install_name="/System/Library/Frameworks/Linked.framework/Versions/A/Linked"
    )
    (first / "Linked.framework").symlink_to(
        tmp_path / "Elsewhere" / "Linked.framework"
    )  # dyld: VLNK, not VDIR: skipped
    tool = _tool(
        tmp_path,
        _command(0xC, "/System/Library/Frameworks/Same.framework/Versions/A/Same"),
        versioned=f"@loader_path/../First:{second}",
    )
    rows, _ = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert rows[0][1][0] == (
        "/System/Library/Frameworks/Same.framework/Versions/A/Same",
        ("versioned", (1, 2), "system-build"),
    )
    assert rows[0][1][1][1] == (
        "versioned-overrides",
        (
            (1, "/System/Library/Frameworks/Same.framework/Versions/A/Same", "absent"),
            (2, "/System/Library/Frameworks/Same.framework/Versions/A/Same", "absent"),
        ),
    )
    assert len(rows) == 3


@pytest.mark.parametrize(
    "variable",
    [
        "DYLD_LIBRARY_PATH=@executable_path/../lib",
        "DYLD_FRAMEWORK_PATH=@executable_path/../Frameworks",
        "DYLD_FALLBACK_LIBRARY_PATH=/opt/lib",
        "DYLD_INSERT_LIBRARIES=@executable_path/../lib/hook.dylib",
        "DYLD_VERSIONED_LIBRARY_PATH=@executable_path/../lib",
        "DYLD_IMAGE_SUFFIX=_debug",
        "DYLD_VERSIONED_FRAMEWORK_PATH",
        "PATH=/usr/bin",
    ],
)
def test_other_embedded_environment_variables_refuse(tmp_path, variable):
    from triton_msl.backend._native_dependencies import parse_image

    path = tmp_path / "tool"
    path.write_bytes(_image(_command(LC_DYLD_ENVIRONMENT, variable), file_type=MH_EXECUTE))
    with pytest.raises(ValueError, match="unsupported native dyld environment"):
        parse_image(path, ARM64)


def test_embedded_environment_outside_an_executable_refuses(tmp_path):
    from triton_msl.backend._native_dependencies import parse_image

    for file_type in (MH_DYLIB, 8):  # dyld reads only the main executable's command
        path = tmp_path / "image"
        path.write_bytes(
            _image(
                _command(LC_DYLD_ENVIRONMENT, "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../F"),
                file_type=file_type,
            )
        )
        with pytest.raises(ValueError, match="outside a main executable"):
            parse_image(path, ARM64)


@pytest.mark.parametrize(
    "setting", ["Frameworks", "@unknown/Frameworks", "", "@executable_path/../A:", "/abs:@rpath/x"]
)
def test_unproved_versioned_directory_entries_refuse(tmp_path, setting):
    from triton_msl.backend._native_dependencies import dependency_graph

    tool = _tool(tmp_path, versioned=setting)
    with pytest.raises(ValueError, match="native|provider|entry"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)


def test_unproved_versioned_candidates_refuse(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_graph

    frameworks = tmp_path / "Frameworks"
    tool = _tool(tmp_path)
    (frameworks / "Text.framework").mkdir(parents=True)
    (frameworks / "Text.framework" / "Text").write_bytes(b"not a Mach-O image")
    with pytest.raises(ValueError, match="Mach-O"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)
    (frameworks / "Text.framework" / "Text").write_bytes(_image(file_type=MH_EXECUTE))  # no LC_ID_DYLIB
    with pytest.raises(ValueError, match="unproved native versioned framework candidate"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)
    (frameworks / "Text.framework" / "Text").unlink()
    (frameworks / "Text.framework" / "Text").symlink_to("Versions/Current/Text")  # dangling
    with pytest.raises(ValueError, match="broken native versioned framework candidate"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)


def test_rpath_search_skips_an_os_directory_without_the_image(tmp_path, monkeypatch):
    """coreai-build's LC_RPATH order is /usr/lib/swift then its own directory (dyld: try each, skip absent)."""
    from triton_msl.backend import _native_dependencies as native

    libs = tmp_path / "lib"
    libs.mkdir()
    (libs / "own.dylib").write_bytes(_image())
    root = tmp_path / "root.so"
    root.write_bytes(
        _image(
            _command(0xC, "@rpath/own.dylib"),
            _command(0x80000018, "@rpath/weak.dylib"),
            _command(0x8000001C, "/usr/lib/swift"),
            _command(0x8000001C, "@loader_path/lib"),
        )
    )
    asked = []

    def cache(path):
        asked.append(str(path))
        return False

    monkeypatch.setattr(native, "_shared_cache_contains", cache)
    rows = native.dependency_manifest([root], cpu_type=ARM64, executable=root)
    assert rows[0][1] == (("@rpath/own.dylib", ("provider", 1)), ("@rpath/weak.dylib", "weak-absent"))
    assert asked == ["/usr/lib/swift/own.dylib", "/usr/lib/swift/weak.dylib"]
    # An OS image the shared cache does hold keeps the ambiguity refusal.
    monkeypatch.setattr(native, "_shared_cache_contains", lambda path: True)
    with pytest.raises(ValueError, match="ambiguous"):
        native.dependency_manifest([root], cpu_type=ARM64, executable=root)
    # Without the cache query (None) the conservative by-prefix classification stands.
    monkeypatch.setattr(native, "_shared_cache_contains", lambda path: None)
    with pytest.raises(ValueError, match="ambiguous"):
        native.dependency_manifest([root], cpu_type=ARM64, executable=root)
    # A strong dependency whose only RPATH directory is an absent OS path refuses.
    root.write_bytes(_image(_command(0xC, "@rpath/gone.dylib"), _command(0x8000001C, "/usr/lib/swift")))
    monkeypatch.setattr(native, "_shared_cache_contains", lambda path: False)
    with pytest.raises(ValueError, match="missing native dependency provider"):
        native.dependency_manifest([root], cpu_type=ARM64, executable=root)
    # A direct absolute OS reference is never existence-checked: it is OS-build bound.
    root.write_bytes(_image(_command(0xC, "/usr/lib/swift/gone.dylib")))
    assert native.dependency_manifest([root], cpu_type=ARM64, executable=root)[0][1] == (
        ("/usr/lib/swift/gone.dylib", "system-build"),
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="dyld shared cache query is Darwin-only")
def test_shared_cache_query_answers_for_real_os_images():
    from triton_msl.backend._native_dependencies import _shared_cache_contains

    assert _shared_cache_contains("/usr/lib/libSystem.B.dylib") is True
    assert _shared_cache_contains("/usr/lib/swift/definitely-not-shipped.dylib") is False


def _witness(tmp_path, comparison_version):
    """The primary's executable witness in fixture form (primary_xcode27_alias_witness.py)."""
    ordinary, comparison = tmp_path / "ordinary.dylib", tmp_path / "comparison.dylib"
    ordinary.write_bytes(_image(_command(0xD, str(ordinary), _version(1))) + b"returns 1")
    comparison.write_bytes(_image(_command(0xD, str(ordinary), _version(comparison_version))) + b"header only")
    frameworks = tmp_path / "Frameworks"
    if not frameworks.exists():
        _framework(frameworks, "Override", install_name=str(comparison), version=_version(2), payload=b"returns 2")
    tool = _tool(tmp_path, _command(0xC, str(ordinary)), versioned="@executable_path/../Frameworks")
    return tool, ordinary, comparison


def test_versioned_comparison_library_keys_the_override_and_is_bound(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_graph, parse_image

    tool, ordinary, comparison = _witness(tmp_path, comparison_version=1)
    assert parse_image(comparison, ARM64).current_version == _version(1)
    rows, providers = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    override = parse_image(tmp_path / "Frameworks/Override.framework/Override", ARM64).digest
    # Override (v2) > comparison (v1): dyld keys the override by the comparison library's
    # OWN install name, i.e. the ordinary library the executable actually requests.
    assert [row[0] for row in rows] == [
        parse_image(tool, ARM64).digest,
        override,
        parse_image(comparison, ARM64).digest,
        parse_image(ordinary, ARM64).digest,
    ]
    assert rows[0][1] == (
        (str(ordinary), ("versioned", (1,), ("provider", 3))),
        (
            "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Frameworks",
            ("versioned-overrides", ((1, str(ordinary), ("comparison", 2)),)),
        ),
    )
    assert rows[2] == (parse_image(comparison, ARM64).digest, (("comparison-only", "not-loaded"),))
    assert comparison.resolve() in providers
    # Rebuilding only the comparison library at v3 flips dyld's choice (the witness prints
    # 2 then 1); the identity must differ and the ordinary edge loses its override.
    tool, ordinary, comparison = _witness(tmp_path, comparison_version=3)
    later, _ = dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert later != rows
    assert later[0][1][0] == (str(ordinary), ("provider", 3))
    assert later[0][1][1][1][1] == ((1, None, ("comparison", 2)),)
    # Equal versions are not "greater": dyld registers nothing.
    tool, ordinary, comparison = _witness(tmp_path, comparison_version=2)
    assert dependency_graph([tool], cpu_type=ARM64, executable=tool)[0][0][1][1][1][1][0][1] is None


def test_versioned_comparison_against_an_os_image_on_disk(tmp_path, monkeypatch):
    from triton_msl.backend import _native_dependencies as native

    system = tmp_path / "os"
    system.mkdir()
    shipped = system / "libshipped.dylib"
    canonical = "/usr/lib/libshipped.dylib"
    shipped.write_bytes(_image(_command(0xD, canonical, _version(1, 5))))
    real_system = native._system
    monkeypatch.setattr(native, "_system", lambda path: str(path).startswith(str(system)) or real_system(path))
    frameworks = tmp_path / "Frameworks"
    _framework(frameworks, "Newer", install_name=str(shipped), version=_version(2))
    tool = _tool(tmp_path, _command(0xC, canonical))
    rows, providers = native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    # The OS image is OS-build bound: header read, no node, but its name and version are recorded
    # and its own install name is the key the executable's request matches.
    assert rows[0][1] == (
        (canonical, ("versioned", (1,), "system-build")),
        (
            "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Frameworks",
            ("versioned-overrides", ((1, canonical, ("system-build", canonical, _version(1, 5))),)),
        ),
    )
    assert shipped.resolve() not in providers and len(rows) == 2


def test_cached_comparison_is_keyed_by_its_canonical_cache_path(tmp_path, monkeypatch):
    """The primary's cached-context witness: the override's transitive edge under the requester."""
    from triton_msl.backend import _native_dependencies as native

    cached = "/System/Library/Frameworks/ExampleCached.framework/Versions/A/ExampleCached"
    base, caller_dir = tmp_path / "main-libs", tmp_path / "caller-libs"
    base.mkdir()
    caller_dir.mkdir()
    (base / "leaf.dylib").write_bytes(_image(_command(0xD, "@rpath/leaf.dylib")) + b"main provider")
    _framework(
        tmp_path / "Frameworks",
        "Override",
        _command(0xC, "@rpath/leaf.dylib"),
        install_name=cached,
        version=_version(999),
    )
    caller = tmp_path / "caller.dylib"
    caller.write_bytes(_image(_command(0xD, str(caller)), _command(0xC, cached), _command(0x8000001C, str(caller_dir))))
    tool = _tool(tmp_path, _command(0xC, str(caller)), _command(0x8000001C, str(base)))
    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: cached if str(path) == cached else False)
    rows, providers = native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    # The key is the cached image's canonical path, the version comparison is unreadable so the
    # candidate is taken as selectable, and the override is re-interned under the CALLER's
    # RPATH context (caller-libs then main-libs), which is where its leaf request resolves.
    assert rows[0][1][-1] == (
        "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Frameworks",
        ("versioned-overrides", ((1, cached, ("shared-cache", cached)),)),
    )
    caller_row = next(row for row in rows if row[0] == native.parse_image(caller, ARM64).digest)
    # Node order: tool, Override@main, caller, main-libs leaf@main, Override@caller, leaf@caller.
    assert caller_row[1][0] == (cached, ("versioned", (4,), "system-build"))
    assert rows[4][0] == rows[1][0] and rows[4][1] == (("@rpath/leaf.dylib", ("provider", 5)),)
    assert rows[5][0] == rows[3][0]  # same bytes, distinct requesting context
    assert (base / "leaf.dylib").resolve() in providers
    # A second leaf provider in the caller's directory is the ambiguity the closed resolver refuses.
    (caller_dir / "leaf.dylib").write_bytes(_image(_command(0xD, "@rpath/leaf.dylib")) + b"requester provider")
    with pytest.raises(ValueError, match="ambiguous native rpath dependency provider: @rpath/leaf.dylib"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    # Every candidate is also traversed under the main executable's context (dlopen by name from
    # the executable resolves there), so a leaf only the caller's directory provides refuses
    # rather than guessing; with the executable searching the caller's directory as well, the
    # caller-side provider is bound and its change is visible.
    (base / "leaf.dylib").unlink()
    with pytest.raises(ValueError, match="missing native dependency provider: @rpath/leaf.dylib"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    tool = _tool(tmp_path, _command(0xC, str(caller)), _command(0x8000001C, str(caller_dir)))
    only_caller, providers = native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert (caller_dir / "leaf.dylib").resolve() in providers
    (caller_dir / "leaf.dylib").write_bytes(_image(_command(0xD, "@rpath/leaf.dylib")) + b"changed")
    assert native.dependency_graph([tool], cpu_type=ARM64, executable=tool)[0] != only_caller
    tool = _tool(tmp_path, _command(0xC, str(caller)), _command(0x8000001C, str(base)))
    (base / "leaf.dylib").write_bytes(_image(_command(0xD, "@rpath/leaf.dylib")) + b"main provider")
    (caller_dir / "leaf.dylib").unlink()
    # An alias request is not the key: dyld's strcmp is against the canonical name only.
    alias = "/System/Library/Frameworks/ExampleCached.framework/ExampleCached"
    monkeypatch.setattr(
        native, "_shared_cache_canonical", lambda path: cached if str(path) in (cached, alias) else False
    )
    caller.write_bytes(_image(_command(0xD, str(caller)), _command(0xC, alias), _command(0x8000001C, str(caller_dir))))
    rows, _ = native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    caller_row = next(row for row in rows if row[0] == native.parse_image(caller, ARM64).digest)
    assert caller_row[1][0] == (alias, "system-build")


def test_cached_comparison_query_states_are_distinct(tmp_path, monkeypatch):
    from triton_msl.backend import _native_dependencies as native

    name = "/System/Library/Frameworks/Cached.framework/Versions/A/Cached"
    _framework(tmp_path / "Frameworks", "Cached", install_name=name, version=_version(9))
    tool = _tool(tmp_path, _command(0xC, name))
    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: None)  # query unavailable
    with pytest.raises(ValueError, match="cannot prove native versioned comparison"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: False)  # proved absent
    rows, _ = native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    assert rows[0][1] == (
        (name, ("versioned", (1,), "system-build")),
        (
            "DYLD_VERSIONED_FRAMEWORK_PATH=@executable_path/../Frameworks",
            ("versioned-overrides", ((1, name, "absent"),)),
        ),
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="dyld shared cache query is Darwin-only")
def test_shared_cache_canonical_resolves_real_aliases():
    from triton_msl.backend._native_dependencies import _shared_cache_canonical

    assert _shared_cache_canonical("/usr/lib/libSystem.B.dylib") == "/usr/lib/libSystem.B.dylib"
    assert _shared_cache_canonical("/usr/lib/../lib/libSystem.B.dylib") == "/usr/lib/libSystem.B.dylib"
    assert _shared_cache_canonical("/usr/lib/swift/definitely-not-shipped.dylib") is False
    assert _shared_cache_canonical("@rpath/x.dylib") is False


def test_relative_comparison_lookup_is_a_working_directory_input(tmp_path, monkeypatch):
    """The primary's relative witness: dyld opens '@rpath/comparison.dylib' literally in the CWD."""
    from triton_msl.backend import _native_dependencies as native

    ordinary = tmp_path / "ordinary.dylib"
    ordinary.write_bytes(_image(_command(0xD, str(ordinary), _version(1))))
    _framework(tmp_path / "Frameworks", "Override", install_name="@rpath/comparison.dylib", version=_version(2))
    tool = _tool(tmp_path, _command(0xC, str(ordinary)))
    work = tmp_path / "work"
    work.mkdir()
    rows, _ = native.dependency_graph([tool], cpu_type=ARM64, executable=tool, working_directory=work)
    entry = (1, "@rpath/comparison.dylib", ("absent-relative", "@rpath/comparison.dylib"))
    assert rows[0][1][-1][1] == ("versioned-overrides", (entry,))
    assert native.working_directory_lookups(rows) == ("@rpath/comparison.dylib",)
    monkeypatch.chdir(work)
    assert native.dependency_graph([tool], cpu_type=ARM64, executable=tool)[0] == rows  # default = os.getcwd()
    (work / "@rpath").mkdir()
    (work / "@rpath" / "comparison.dylib").write_bytes(_image(_command(0xD, str(ordinary), _version(1))))
    for directory in (work, None):
        with pytest.raises(ValueError, match="working directory shadows native versioned comparison lookup"):
            native.dependency_graph([tool], cpu_type=ARM64, executable=tool, working_directory=directory)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert native.dependency_graph([tool], cpu_type=ARM64, executable=tool, working_directory=elsewhere)[0] == rows
    (work / "@rpath" / "comparison.dylib").unlink()
    (work / "@rpath" / "comparison.dylib").symlink_to(tmp_path / "gone")  # dangling: still an open() input
    with pytest.raises(ValueError, match="working directory shadows"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool, working_directory=work)


def test_unproved_platform_refuses(tmp_path, monkeypatch):
    from triton_msl.backend import _native_dependencies as native

    monkeypatch.setattr(native, "_shared_cache_canonical", lambda path: False)
    frameworks = tmp_path / "Frameworks"
    ios = frameworks / "Mobile.framework"
    ios.mkdir(parents=True)
    (ios / "Mobile").write_bytes(_image(_command(0xD, "@rpath/Mobile.framework/Mobile"), platform=2))  # iOS only
    tool = _tool(tmp_path)
    with pytest.raises(ValueError, match="unproved native versioned framework candidate"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool)
    (ios / "Mobile").unlink()
    ios.rmdir()
    target = tmp_path / "target.dylib"
    target.write_bytes(_image(_command(0xD, str(target), _version(1)), platform=None))  # no platform proof
    _framework(frameworks, "Over", install_name=str(target), version=_version(2))
    with pytest.raises(ValueError, match="unproved native versioned comparison library"):
        native.dependency_graph([tool], cpu_type=ARM64, executable=tool)


def test_unproved_versioned_comparison_library_refuses(tmp_path):
    from triton_msl.backend._native_dependencies import dependency_graph

    target = tmp_path / "target.dylib"
    _framework(tmp_path / "Frameworks", "Over", install_name=str(target), version=_version(2))
    tool = _tool(tmp_path)
    target.write_bytes(_image(file_type=MH_EXECUTE))  # not a dylib: getDylibInfo would fail
    with pytest.raises(ValueError, match="unproved native versioned comparison library"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)
    target.unlink()
    target.symlink_to(tmp_path / "gone.dylib")
    with pytest.raises(ValueError, match="broken native versioned comparison library"):
        dependency_graph([tool], cpu_type=ARM64, executable=tool)
