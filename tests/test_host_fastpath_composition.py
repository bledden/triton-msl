"""Run one-shot host admission controls in a fresh selected-package process.

The parent suite may already have imported driver and configured scalar_gate.
Neither that state nor the root conftest's device fixtures enter this child.
The normal subprocess API preserves CPU-controller Popen wrapping and environment.
"""

from pathlib import Path
import subprocess

from tests.python_child import package_child_command


def test_host_fastpath_composition_in_fresh_cpu_child(tmp_path):
    controls = Path(__file__).with_name("host_fastpath_controls.py").resolve()
    program = """
import os
from pathlib import Path
import sys
assert 'triton_msl.backend.driver' not in sys.modules
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
import pytest
controls = Path(sys.argv[1]).resolve()
exit_code = pytest.main([
    '--noconftest', '-p', 'no:cacheprovider', '-o', 'addopts=', '-q', '-s',
    '--basetemp=' + sys.argv[2], str(controls),
])
assert 'triton_msl.backend.driver' not in sys.modules
raise SystemExit(exit_code)
"""
    command = package_child_command(program, str(controls), str(tmp_path / "child-tmp"))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout, result.stdout
