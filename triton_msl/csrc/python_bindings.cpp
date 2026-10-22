// pybind11 requires RTTI, but LLVM/MLIR headers are built without RTTI.
// To avoid ABI conflicts, this file does NOT include any MLIR headers.
// Instead it calls through a thin C-linkage wrapper defined in
// python_bindings_bridge.cpp (compiled with -fno-rtti alongside the
// rest of the MLIR code).

#include <pybind11/pybind11.h>
#include <string>
#include <stdexcept>

namespace py = pybind11;

// Defined in python_bindings_bridge.cpp (compiled with -fno-rtti).
extern "C" void triton_msl_register_passes();
extern "C" const char* triton_msl_run_to_llvm(const char* mlir_text,
                                                 int air_major, int air_minor,
                                                 int lang_major, int lang_minor,
                                                 int optimize_scalar_cfg, int* out_success);

PYBIND11_MODULE(_triton_msl_cpp, m) {
    m.doc() = "C++ MLIR passes for triton-msl";
    m.def("register_metal_passes", []() {
        triton_msl_register_passes();
    }, "Register all Metal conversion passes with MLIR's pass infrastructure");

    m.def("run_to_llvm", [](const std::string &mlir_text,
                            int air_major, int air_minor,
                            int lang_major, int lang_minor, bool optimize_scalar_cfg) -> std::string {
        int success = 0;
        const char* result = triton_msl_run_to_llvm(mlir_text.c_str(),
                                                     air_major, air_minor,
                                                     lang_major, lang_minor, optimize_scalar_cfg, &success);
        if (!success) {
            throw std::runtime_error(std::string("Metal-to-LLVM pass failed: ") + result);
        }
        return std::string(result);
    }, py::arg("mlir_text"),
       // Defaults preserve the pre-macOS-26 behavior (AIR 2.7 / Metal 3.2) if a
       // caller does not pass a probed version.
       py::arg("air_major") = 2, py::arg("air_minor") = 7,
       py::arg("lang_major") = 3, py::arg("lang_minor") = 2,
       py::arg("optimize_scalar_cfg") = false,
    "Run the Metal-to-LLVM pass pipeline on MLIR text (with target AIR / Metal language version), returning lowered LLVM dialect text");
}
