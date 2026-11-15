"""CPU-only malformed-artifact controls; runnable directly without project conftest."""

import importlib.util
import io
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SPEC = importlib.util.spec_from_file_location(
    "release_artifacts", Path(__file__).resolve().parents[1] / "scripts/release_artifacts.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.dist = Path(self.tmp.name) / "dist"
        self.source.mkdir()
        self.dist.mkdir()
        self.stem = "triton_msl-0.3.0rc4"
        self.files = {
            "pyproject.toml": b'[project]\nversion="0.3.0rc4"\n',
            "LICENSE": b"MIT notice",
            "third_party/trifast/LICENSE": b"trifast notice",
            "triton_msl/__init__.py": b'"""Fixture package."""\n',
            **{"triton_msl/backend/" + h + ".c": b"/* fixture source */\n" for h in release.HELPERS},
        }
        for name, data in self.files.items():
            p = self.source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        self.sd = dict(self.files)
        self.wheels = {}
        for mode, tag in [("native", "cp314-cp314-macosx_15_0_arm64"), ("pure", "py3-none-any")]:
            info = self.stem + ".dist-info/"
            payload = {n: b for n, b in self.files.items() if n.startswith("triton_msl/")}
            payload.update({info + "licenses/" + n: self.files[n] for n in release.NOTICES})
            payload[info + "WHEEL"] = (
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: {str(mode == 'pure').lower()}\nTag: {tag}\n".encode()
            )
            payload[info + "METADATA"] = b"Name: triton-msl\nVersion: 0.3.0rc4\n"
            if mode == "native":
                for helper in release.HELPERS:
                    payload["triton_msl/backend/" + helper + ".cpython-314-darwin.so"] = struct.pack(
                        "<II", 0xFEEDFACF, 0x0100000C
                    )
            self.wheels[mode] = (self.dist / (self.stem + "-" + tag + ".whl"), payload)
        self.write()

    def write(self):
        with tarfile.open(self.dist / (self.stem + ".tar.gz"), "w:gz") as archive:
            top = tarfile.TarInfo(self.stem)
            top.type = tarfile.DIRTYPE
            archive.addfile(top)
            for name, data in self.sd.items():
                info = tarfile.TarInfo(self.stem + "/" + name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        for path, payload in self.wheels.values():
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in payload.items():
                    archive.writestr(name, data)

    def rejects(self, reason):
        self.write()
        with self.assertRaisesRegex(RuntimeError, reason):
            release.inspect_bundle(self.source, self.dist)

    def test_complete_bundle(self):
        self.assertEqual(release.HELPERS, ("_validation_native", "_packed_native", "_binder_native"))
        self.assertEqual(set(release.inspect_bundle(self.source, self.dist)), {"native", "pure", "sdist"})

    def test_missing_binder(self):
        del self.wheels["native"][1]["triton_msl/backend/_binder_native.cpython-314-darwin.so"]
        self.rejects("helper membership")

    def test_binder_binary_leaks_into_pure(self):
        self.wheels["pure"][1]["triton_msl/backend/_binder_native.cpython-314-darwin.so"] = b"stale"
        self.rejects("helper membership")

    def test_sdist_missing_binder_source(self):
        del self.sd["triton_msl/backend/_binder_native.c"]
        self.rejects("C source missing")

    def test_missing_helper(self):
        del self.wheels["native"][1]["triton_msl/backend/_packed_native.cpython-314-darwin.so"]
        self.rejects("helper membership")

    def test_binary_leaks_into_pure(self):
        self.wheels["pure"][1]["triton_msl/backend/_packed_native.cpython-314-darwin.so"] = b"stale"
        self.rejects("helper membership")

    def test_wrong_binary_architecture(self):
        self.wheels["native"][1]["triton_msl/backend/_packed_native.cpython-314-darwin.so"] = struct.pack(
            "<II", 0xFEEDFACF, 0x01000007
        )
        self.rejects("arm64 Mach-O")

    def test_wrong_wheel_tag(self):
        self.wheels["native"][1][self.stem + ".dist-info/WHEEL"] = b"Tag: cp313-cp313-macosx_15_0_arm64\n"
        self.rejects("metadata tag")

    def test_extra_artifact(self):
        (self.dist / "unexpected.whl").write_bytes(b"extra")
        self.rejects("exactly one")

    def test_missing_artifact(self):
        self.wheels["pure"][0].unlink()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            release.inspect_bundle(self.source, self.dist)

    def test_sdist_binary(self):
        self.sd["triton_msl/backend/stale.so"] = b"stale"
        self.rejects("tests or binaries")

    def test_sdist_tests(self):
        self.sd["tests/test_example.py"] = b""
        self.rejects("tests or binaries")

    def test_sdist_missing_c_source(self):
        del self.sd["triton_msl/backend/_packed_native.c"]
        self.rejects("C source missing")

    def test_missing_notice(self):
        del self.wheels["pure"][1][self.stem + ".dist-info/licenses/third_party/trifast/LICENSE"]
        self.rejects("notice missing")

    def test_changed_package_payload(self):
        self.wheels["native"][1]["triton_msl/__init__.py"] = b"changed"
        self.rejects("payload differs")

    def test_sdist_missing_python_source(self):
        (self.source / "triton_msl/missing.py").write_text("expected = True\n")
        self.rejects("source payload differs")

    def test_wrong_platform_fails_before_build(self):
        with patch.object(release.sys, "platform", "linux"), patch.object(release.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "requires GIL-enabled"):
                release.build_bundle(self.source, self.dist / "new")
            run.assert_not_called()

    def test_fresh_sources_and_explicit_native_modes(self):
        output = Path(self.tmp.name) / "built"
        calls = []

        def build(command, *, check, env):
            self.assertTrue(check)
            calls.append((command, env.copy()))
            if "--sdist" in command:
                self.assertEqual(env["TRITON_MSL_BUILD_BINDER_NATIVE"], "0")
                output.joinpath(self.stem + ".tar.gz").write_bytes((self.dist / (self.stem + ".tar.gz")).read_bytes())
            else:
                root = Path(command[-1])
                self.assertFalse(any(root.rglob("*.so")))
                mode = "native" if env["TRITON_MSL_BUILD_PACKED_NATIVE"] == "1" else "pure"
                self.assertEqual(env["TRITON_MSL_BUILD_VALIDATION_NATIVE"], env["TRITON_MSL_BUILD_PACKED_NATIVE"])
                self.assertEqual(env["TRITON_MSL_BUILD_BINDER_NATIVE"], env["TRITON_MSL_BUILD_PACKED_NATIVE"])
                self.assertEqual(env["MACOSX_DEPLOYMENT_TARGET"], "15.0")
                self.assertEqual(env["ARCHFLAGS"], "-arch arm64")
                self.assertEqual(env["_PYTHON_HOST_PLATFORM"], "macosx-15.0-arm64")
                wheel = self.wheels[mode][0]
                output.joinpath(wheel.name).write_bytes(wheel.read_bytes())
                (root / "contamination.so").write_bytes(b"cannot enter the other build")

        with (
            patch.object(release, "native_platform"),
            patch.object(release.subprocess, "run", side_effect=build),
            patch.dict(release.os.environ, {"TRITON_MSL_BUILD_BINDER_NATIVE": "1"}),
        ):
            release.build_bundle(self.source, output)
        self.assertEqual(len(calls), 3)
        self.assertNotEqual(calls[1][0][-1], calls[2][0][-1])


if __name__ == "__main__":
    unittest.main()
