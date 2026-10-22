"""Device-free discriminators for the candidate test helper."""
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tests import source_identity as helper


def module(root, name, package=False):
    path = root.joinpath(*name.split('.'))
    path = path / '__init__.py' if package else path.with_suffix('.py')
    value = ModuleType(name)
    value.__file__ = str(path)
    value.__spec__ = importlib.util.spec_from_file_location(name, path)
    return value


@pytest.fixture
def setup(tmp_path, monkeypatch):
    checkout, installed, foreign = [tmp_path / n for n in ('checkout', 'installed', 'foreign')]
    for root in (checkout, installed, foreign):
        for name in ('triton_msl/__init__.py', 'triton_msl/codegen/generic_lowerer.py'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'# bound candidate source\n')
    monkeypatch.setattr(helper.sysconfig, 'get_path', lambda kind: str(installed))
    return checkout, installed, foreign


@pytest.mark.parametrize('location', ('checkout', 'installed'))
def test_exact_sources_pass(setup, location):
    checkout, installed, _ = setup
    root = checkout if location == 'checkout' else installed
    rows = helper.assert_candidate_source(checkout, module(root, 'triton_msl', True),
                                         module(root, 'triton_msl.codegen.generic_lowerer'))
    assert len(rows) == 2 and rows[0]['path'] != rows[1]['path']


@pytest.mark.parametrize('fault', ('foreign', 'mixed', 'changed_package', 'changed_child', 'missing',
                                  'name', 'spec_name', 'spec_origin', 'no_spec', 'empty', 'wrong_first'))
def test_adverse_sources_refuse(setup, fault):
    checkout, installed, foreign = setup
    package = module(installed, 'triton_msl', True)
    child = module(installed, 'triton_msl.codegen.generic_lowerer')
    modules = [package, child]
    if fault == 'foreign': modules = [module(foreign, 'triton_msl', True)]
    elif fault == 'mixed': modules[1] = module(checkout, 'triton_msl.codegen.generic_lowerer')
    elif fault.startswith('changed_'):
        Path((package if fault == 'changed_package' else child).__file__).write_bytes(b'# stale source\n')
    elif fault == 'missing': Path(child.__file__).unlink()
    elif fault == 'name': child.__name__ = 'other.codegen.generic_lowerer'
    elif fault == 'spec_name': child.__spec__.name = 'triton_msl.codegen.other'
    elif fault == 'spec_origin': child.__spec__.origin = str(checkout / 'triton_msl/codegen/generic_lowerer.py')
    elif fault == 'no_spec': child.__spec__ = None
    elif fault == 'empty': modules = []
    elif fault == 'wrong_first': modules = [child, package]
    with pytest.raises((AssertionError, FileNotFoundError)):
        helper.assert_candidate_source(checkout, *modules)
