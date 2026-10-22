"""CPU-only build guard: the optional extension is not a Python embedding host."""

from pathlib import Path
import re


CMAKE_FILE = Path(__file__).resolve().parents[1] / "triton_msl" / "csrc" / "CMakeLists.txt"


def test_cpp_extension_uses_module_not_embedding_linkage():
    text = re.sub(r"#[^\n]*", "", CMAKE_FILE.read_text())
    declarations = re.findall(r"pybind11_add_module\s*\(\s*_triton_msl_cpp\s+([^)]*)\)", text, re.S)
    assert len(declarations) == 1
    words = declarations[0].split()
    assert "MODULE" in words and "SHARED" not in words and "STATIC" not in words
    links = re.findall(r"target_link_libraries\s*\(\s*_triton_msl_cpp\s+([^)]*)\)", text, re.S)
    assert len(links) == 1 and "${LIBTRITON_PATH}" in links[0]
    assert not any(word in links[0] for word in ("Python::Python", "Python3::Python", "pybind11::embed"))
    options = re.findall(r"target_link_options\s*\(\s*_triton_msl_cpp\s+([^)]*)\)", text, re.S)
    assert any("-undefined,dynamic_lookup" in option for option in options)
