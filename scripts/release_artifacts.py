"""Build/check the explicit release bundle and smoke an off-checkout installation.

Repository-only tooling. No Triton, Torch, GPU, upload or publication operations.
"""
import argparse
import email.parser
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import zipfile

HELPERS = ('_validation_native', '_packed_native', '_binder_native')
NOTICES = ('LICENSE', 'third_party/trifast/LICENSE')
BINARY_SUFFIXES = ('.so', '.dylib', '.dll', '.pyd', '.a', '.o')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def native_platform():
    require(sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 14)
            and sys.platform == 'darwin' and platform.machine() == 'arm64'
            and not sysconfig.get_config_var('Py_GIL_DISABLED')
            and int(platform.mac_ver()[0].split('.')[0] or 0) >= 15,
            'Release native wheel requires GIL-enabled CPython 3.14 on macOS 15+ arm64')


def wheel_payload(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel members')
        return {name: archive.read(name) for name in names if not name.endswith('/')}


def inspect_bundle(source, dist):
    import tomllib  # Build/check require Python 3.11+; installed smoke also runs on 3.10.

    version = tomllib.loads((source / 'pyproject.toml').read_text())['project']['version']
    stem = 'triton_msl-' + version
    expected = {'native': stem + '-cp314-cp314-macosx_15_0_arm64.whl',
                'pure': stem + '-py3-none-any.whl', 'sdist': stem + '.tar.gz'}
    require({p.name for p in dist.iterdir()} == set(expected.values()),
            'Release bundle must contain exactly one cp314 macOS15 arm64 wheel, one pure wheel and one sdist')
    with tarfile.open(dist / expected['sdist']) as archive:
        members = archive.getmembers()
        files = [m for m in members if m.isfile()]
        require(all(m.isfile() or m.isdir() for m in members), 'sdist links or special files')
        require(len({m.name for m in members}) == len(members), 'Duplicate sdist members')
        require(all((m.name == stem and m.isdir()) or (m.name.startswith(stem + '/') and '..' not in Path(m.name).parts)
                    for m in members),
                'Unexpected sdist path')
        sdist = {m.name[len(stem) + 1:]: archive.extractfile(m).read() for m in files}
    require(not any(n.startswith('tests/') or n.endswith(BINARY_SUFFIXES) for n in sdist),
            'sdist contains tests or binaries')
    for name in NOTICES:
        require(sdist.get(name) == (source / name).read_bytes(), 'sdist notice missing/changed: ' + name)
    for helper in HELPERS:
        name = 'triton_msl/backend/' + helper + '.c'
        require(sdist.get(name) == (source / name).read_bytes(), 'sdist C source missing/changed: ' + name)
    package = {n: b for n, b in sdist.items() if n.startswith('triton_msl/')}
    require('triton_msl/__init__.py' in package, 'Missing package')
    source_package = {p.relative_to(source).as_posix(): p.read_bytes()
                      for p in (source / 'triton_msl').rglob('*')
                      if p.is_file() and p.suffix in ('.py', '.c')}
    require({n: b for n, b in package.items() if n.endswith(('.py', '.c'))} == source_package,
            'sdist source payload differs from checkout')
    for mode in ('native', 'pure'):
        payload = wheel_payload(dist / expected[mode])
        info = stem + '.dist-info/'
        metadata = email.parser.BytesParser().parsebytes(payload[info + 'WHEEL'])
        tag = 'cp314-cp314-macosx_15_0_arm64' if mode == 'native' else 'py3-none-any'
        require(metadata.get_all('Tag') == [tag], 'Wrong wheel metadata tag')
        require(metadata['Root-Is-Purelib'] == str(mode == 'pure').lower(), 'Wrong purelib declaration')
        package_metadata = email.parser.BytesParser().parsebytes(payload[info + 'METADATA'])
        require(package_metadata['Version'] == version and package_metadata['Name'] == 'triton-msl',
                'Wrong wheel project/version')
        require(not any(n.startswith('tests/') for n in payload), 'Wheel includes tests')
        for name in NOTICES:
            require(payload.get(info + 'licenses/' + name) == (source / name).read_bytes(),
                    'Wheel notice missing/changed: ' + name)
        binaries = {n: b for n, b in payload.items() if n.endswith(BINARY_SUFFIXES)}
        names = {'triton_msl/backend/' + h + '.cpython-314-darwin.so' for h in HELPERS}
        require(set(binaries) == (names if mode == 'native' else set()), 'Wrong native helper membership')
        for name, blob in binaries.items():
            require(len(blob) >= 8 and struct.unpack('<II', blob[:8]) == (0xfeedfacf, 0x0100000c),
                    'Helper is not a thin arm64 Mach-O: ' + name)
        wheel_package = {n: b for n, b in payload.items() if n.startswith('triton_msl/') and n not in binaries}
        require(wheel_package == package, 'Wheel package payload differs from sdist')
    return {mode: str((dist / name).resolve()) for mode, name in expected.items()}


def build_bundle(source, dist):
    native_platform()  # Fail before any build; requested native cannot become pure.
    require(not dist.exists(), 'Output directory already exists')
    dist.mkdir(parents=True)
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET='15.0',
               TRITON_MSL_BUILD_VALIDATION_NATIVE='0', TRITON_MSL_BUILD_PACKED_NATIVE='0',
               TRITON_MSL_BUILD_BINDER_NATIVE='0')
    env.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='triton-release-') as temporary:
        staging = Path(temporary)
        # Create the sole source archive once. MANIFEST excludes local native products.
        subprocess.run([sys.executable, '-m', 'build', '--sdist', '--outdir', str(dist), str(source)],
                       check=True, env=env)
        sdist, = dist.glob('*.tar.gz')
        for mode in ('native', 'pure'):
            tree = staging / mode
            tree.mkdir()
            with tarfile.open(sdist) as archive:
                archive.extractall(tree, filter='data')
            root, = tree.iterdir()
            require(not any(p.suffix in BINARY_SUFFIXES for p in root.rglob('*')), 'Fresh source has a binary')
            setting = '1' if mode == 'native' else '0'
            build_env = dict(env, TRITON_MSL_BUILD_VALIDATION_NATIVE=setting,
                             TRITON_MSL_BUILD_PACKED_NATIVE=setting, TRITON_MSL_BUILD_BINDER_NATIVE=setting)
            subprocess.run([sys.executable, '-m', 'build', '--wheel', '--outdir', str(dist), str(root)],
                           check=True, env=build_env)
    return inspect_bundle(source, dist)


def installed_smoke(wheel, mode):
    if mode == 'native':
        native_platform()
    payload = wheel_payload(wheel)
    import triton_msl
    root = Path(triton_msl.__file__).resolve().parent
    require(root.is_relative_to(Path(sys.prefix).resolve()), 'Package did not load from this virtual environment')
    require(not Path.cwd().resolve().is_relative_to(root.parent), 'Smoke must run outside installed package')
    for name, blob in payload.items():
        if name.startswith('triton_msl/'):
            require((root.parent / name).read_bytes() == blob, 'Installed payload mismatch: ' + name)
    from triton_msl.backend import _cache_contract as cache, _launch_contract as launch, _launch_signature as binding
    for helper in HELPERS:
        name = 'triton_msl.backend.' + helper
        spec = importlib.util.find_spec(name)
        if mode == 'pure':
            require(spec is None, 'Pure install contains a native helper')
        else:
            module = importlib.import_module(name)
            require(Path(module.__file__).resolve().parent == root / 'backend', 'Native origin outside package')
    if mode == 'native':
        require(cache._validation_native is not None and launch._packed_native is not None
                and binding._binder_native is not None, 'Native helper not selected')
        key, value = object(), object()
        require(cache._validation_native.same_items({key: value}, {key: value}) is True, 'Native identity positive failed')
        require(cache._validation_native.same_items({key: value}, {key: object()}) is False, 'Native identity negative failed')
        require(cache._validation_native.identity_probes(()) is True, 'Native probe call failed')
    else:
        require(cache._validation_native is None and launch._packed_native is None
                and binding._binder_native is None, 'Pure fallback not selected')
    names, signature, values = ['pointer', 'size'], {'pointer': '*fp32', 'size': 'i32'}, (None, 17)
    binding_plan = binding.make_binding_plan(names, signature)
    before_binding = None if mode == 'pure' else binding._binder_native.statistics()
    bound = binding.bind_arguments_with_plan(values, names, signature, binding_plan)
    repeated = binding.bind_arguments_with_plan(values, names, signature, binding_plan)
    expected = binding.bind_arguments(values, names, signature, _plan=binding_plan)
    require(bound == repeated == expected and bound[3] == [None, struct.pack('<i', 17)], 'Binder payload smoke failed')
    require(all(a is not b for a, b in zip(bound, repeated)), 'Binder reused mutable output containers')
    if mode == 'native':
        after_binding = binding._binder_native.statistics()
        require({key: after_binding[key] - before_binding[key] for key in before_binding}
                == {'hits': 2, 'misses': 0, 'errors': 0}, 'Installed native binder was not used')
    # The bound Struct owner is mutable; a changed format must retain original
    # Python payload semantics instead of returning hard-coded four-byte data.
    binding_plan.packers['i32'][1].__self__.__init__('<q')
    changed_binding = binding.bind_arguments_with_plan(values, names, signature, binding_plan)
    require(changed_binding[3] == [None, struct.pack('<q', 17)]
            and changed_binding == binding.bind_arguments(values, names, signature, _plan=binding_plan),
            'Binder accepted stale scalar packing format')
    data = ['shader' * 256, {'value': [7]}]
    plan = launch._packed_snapshot_plan(launch._canonical(data))
    require(plan is not None, 'Packed smoke did not select a plan')
    copied = launch._checked_packed_copy(data, plan)
    require(copied == data and copied is not data and copied[1] is not data[1], 'Packed copy smoke failed')
    data[1]['value'][0] = 8
    from triton_msl.errors import MetalNonRecoverableError
    try:
        launch._checked_packed_copy(data, plan)
    except MetalNonRecoverableError:
        pass
    else:
        raise RuntimeError('Packed copy accepted changed live payload')
    for name, module in tuple(sys.modules.items()):
        if name == 'triton_msl' or name.startswith('triton_msl.'):
            file = getattr(module, '__file__', None)
            require(file is not None and Path(file).resolve().is_relative_to(root), 'Foreign package module: ' + name)
    return {'mode': mode, 'origin': str(root), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'native_helpers': mode == 'native', 'gpu_dispatches': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'check', 'select'):
        p = sub.add_parser(command)
        p.add_argument('--source', type=Path, required=True)
        p.add_argument('--dist', type=Path, required=True)
        if command == 'select':
            p.add_argument('--mode', choices=('native', 'pure'), required=True)
    p = sub.add_parser('smoke')
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--mode', choices=('native', 'pure'), required=True)
    args = parser.parse_args()
    if args.command == 'smoke':
        result = installed_smoke(args.wheel.resolve(), args.mode)
    else:
        source, dist = args.source.resolve(), args.dist.resolve()
        result = build_bundle(source, dist) if args.command == 'build' else inspect_bundle(source, dist)
        if args.command == 'select':
            print(result[args.mode])
            return
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
