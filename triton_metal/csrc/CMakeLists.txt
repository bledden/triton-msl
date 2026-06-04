cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Triton source/build path resolution.
#
# The C++ MLIR->LLVM path links Triton's dialect OBJECT files from a
# FROM-SOURCE Triton build (a pip wheel lacks them — libtriton.so does not
# export the TTG symbols on macOS). See the evaluation in
# docs/superpowers/specs/2026-05-30-ws0-c4-cpp-build-evaluation.md.
#
# Resolution order (so this builds on any machine with such a build, not just
# the original developer's): env var -> -D override -> sensible default/GLOB.
# ---------------------------------------------------------------------------
if(DEFINED ENV{TRITON_ROOT})
  set(_triton_src_default "$ENV{TRITON_ROOT}")
else()
  set(_triton_src_default "$ENV{HOME}/Documents/triton")
endif()
set(TRITON_SRCPATH "${_triton_src_default}"
    CACHE PATH "Path to Triton source tree (env: TRITON_ROOT)")

if(DEFINED ENV{TRITON_BUILD})
  set(_triton_build_default "$ENV{TRITON_BUILD}")
else()
  # Triton's build dir is cmake.<platform>-<arch>-cpython-<pyver>; the suffix
  # varies by macOS/Python version, so GLOB it rather than hardcode a version.
  file(GLOB _triton_build_candidates
       "${TRITON_SRCPATH}/build/cmake.*-arm64-cpython-*")
  if(_triton_build_candidates)
    list(SORT _triton_build_candidates)
    list(GET _triton_build_candidates -1 _triton_build_default)  # newest
  else()
    set(_triton_build_default "${TRITON_SRCPATH}/build/cmake-build-not-found")
  endif()
endif()
set(TRITON_BUILDPATH "${_triton_build_default}"
    CACHE PATH "Path to Triton build tree for headers+objects (env: TRITON_BUILD)")

if(NOT EXISTS "${TRITON_BUILDPATH}")
  message(FATAL_ERROR
    "Triton build tree not found at '${TRITON_BUILDPATH}'.\n"
    "The C++ MLIR->LLVM path requires a FROM-SOURCE Triton build (a pip wheel "
    "lacks the dialect object files this extension links). The Python MSL "
    "path has no such requirement.\n"
    "Point the build at your Triton checkout via either:\n"
    "  -DTRITON_SRCPATH=/path/to/triton  (or env TRITON_ROOT), and/or\n"
    "  -DTRITON_BUILDPATH=/path/to/triton/build/cmake.*  (or env TRITON_BUILD)\n"
    "See docs/superpowers/specs/2026-05-30-ws0-c4-cpp-build-evaluation.md.")
endif()

# Triton commit the dialect ABI linkage is aligned to. Bump as a tested step
# (the linked .o files must come from a compatible build). Soft-checked below.
set(TRITON_PINNED_COMMIT "4da2e268dcccf090370e7bb7b59940e7567bf0ce"
    CACHE STRING "Triton commit the C++ dialect linkage is aligned to")
find_program(_git_exe git)
if(_git_exe AND EXISTS "${TRITON_SRCPATH}/.git")
  execute_process(
    COMMAND "${_git_exe}" -C "${TRITON_SRCPATH}" rev-parse HEAD
    OUTPUT_VARIABLE _triton_head OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_triton_head AND NOT _triton_head STREQUAL "${TRITON_PINNED_COMMIT}")
    message(WARNING
      "Triton source at '${TRITON_SRCPATH}' is on ${_triton_head}, but the "
      "C++ linkage is pinned to ${TRITON_PINNED_COMMIT}. ABI drift may cause "
      "link/load errors; align the checkout or bump TRITON_PINNED_COMMIT after "
      "rebuilding Triton.")
  endif()
endif()

# Match the macOS deployment target Triton uses for its build, so that
# linking against pre-built triton .o files doesn't trigger version
# mismatch warnings or load failures. Read it from triton's CMakeCache.
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(_triton_cache "${TRITON_BUILDPATH}/CMakeCache.txt")
  if(EXISTS "${_triton_cache}")
    file(STRINGS "${_triton_cache}" _osx_target_line
         REGEX "^CMAKE_OSX_DEPLOYMENT_TARGET:STRING=.+$")
    if(_osx_target_line)
      string(REGEX REPLACE "^CMAKE_OSX_DEPLOYMENT_TARGET:STRING=" "" _osx_target
             "${_osx_target_line}")
      set(CMAKE_OSX_DEPLOYMENT_TARGET "${_osx_target}" CACHE STRING
          "macOS deployment target (inherited from triton's CMakeCache)" FORCE)
      message(STATUS "Inheriting OSX deployment target from triton: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
  endif()
endif()

project(triton_metal_backend CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# LLVM path. TRITON_SRCPATH / TRITON_BUILDPATH are resolved at the top of this
# file (env-overridable + GLOB); only LLVM remains to set here.
# ---------------------------------------------------------------------------
# Use the version-agnostic symlink so this tracks the LLVM hash from
# triton/cmake/llvm-hash.txt without needing edits on every bump.
set(LLVM_SYSPATH "$ENV{HOME}/.triton/llvm/llvm-macos-arm64"
    CACHE PATH "Path to LLVM/MLIR install")

# ---------------------------------------------------------------------------
# Find LLVM & MLIR (for include paths and cmake modules only)
# ---------------------------------------------------------------------------
set(LLVM_DIR "${LLVM_SYSPATH}/lib/cmake/llvm" CACHE PATH "")
set(MLIR_DIR "${LLVM_SYSPATH}/lib/cmake/mlir" CACHE PATH "")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
# LLVM/MLIR headers (needed for compilation, NOT for linking)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# Triton source headers (Dialect definitions, etc.)
include_directories(SYSTEM "${TRITON_SRCPATH}/include")

# Triton build-generated headers (TableGen outputs)
include_directories(SYSTEM "${TRITON_BUILDPATH}/include")

# Our own headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Compiler flags — match Triton conventions
# ---------------------------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fPIC")
endif()

# Default to -fno-rtti (LLVM/MLIR convention); pybind11 source overrides below
if(NOT LLVM_ENABLE_RTTI)
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
  endif()
endif()

# ---------------------------------------------------------------------------
# Triton dialect object files — Triton builds as OBJECT libraries so we
# collect the pre-built .o files for the Triton IR dialect definitions.
# These are needed because the Triton ops (GetProgramIdOp, LoadOp, etc.)
# have non-inline methods defined in these compilation units.
# ---------------------------------------------------------------------------
set(TRITON_IR_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir")

file(GLOB TRITON_IR_OBJS "${TRITON_IR_OBJS_DIR}/*.o")

# TritonGPU dialect object files — needed for type-safe matching on
# TTG ops (LocalAllocOp, LocalLoadOp, etc.). TTG symbols are not
# exported from libtriton.so on macOS, so we link the .o files directly.
set(TRITON_GPU_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir")

file(GLOB TRITON_GPU_OBJS "${TRITON_GPU_OBJS_DIR}/*.o")

if(NOT TRITON_GPU_OBJS)
  message(FATAL_ERROR "TritonGPU dialect objects not found at ${TRITON_GPU_OBJS_DIR}")
endif()
list(LENGTH TRITON_GPU_OBJS _ttg_obj_count)
message(STATUS "Found ${_ttg_obj_count} TritonGPU objects")

# TritonNvidiaGPU dialect objects — TTG Ops.cpp references TMEM ops
# (TMEMStoreOp, TensorMemoryEncodingAttr, etc.) in canonicalize patterns,
# so the TypeID symbols for those types must be present in the final
# module. We don't register the dialect — just provide the symbols.
set(TRITON_NVIDIA_GPU_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir")

file(GLOB TRITON_NVIDIA_GPU_OBJS "${TRITON_NVIDIA_GPU_OBJS_DIR}/*.o")

if(NOT TRITON_NVIDIA_GPU_OBJS)
  message(FATAL_ERROR "TritonNvidiaGPU dialect objects not found at ${TRITON_NVIDIA_GPU_OBJS_DIR}")
endif()
list(LENGTH TRITON_NVIDIA_GPU_OBJS _tng_obj_count)
message(STATUS "Found ${_tng_obj_count} TritonNvidiaGPU objects")

# Triton support object files — LinearLayout, LayoutUtils, supportMMA,
# inferDstEncoding, etc. These helpers are referenced by TTG Dialect.cpp
# / Ops.cpp in verify/canonicalize paths. On macOS they live in libtriton.so
# as local symbols (not exported), so we must pull in the specific .o files.
set(TRITON_SUPPORT_OBJS
    "${TRITON_BUILDPATH}/lib/Tools/CMakeFiles/TritonTools.dir/LinearLayout.cpp.o"
    "${TRITON_BUILDPATH}/lib/Tools/CMakeFiles/TritonTools.dir/LayoutUtils.cpp.o"
    "${TRITON_BUILDPATH}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Utility.cpp.o"
    "${TRITON_BUILDPATH}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Utility.cpp.o"
    "${TRITON_BUILDPATH}/third_party/f2reduce/CMakeFiles/f2reduce.dir/f2reduce.cpp.o"
)
foreach(_obj ${TRITON_SUPPORT_OBJS})
  if(NOT EXISTS "${_obj}")
    message(FATAL_ERROR "Triton support object not found: ${_obj}")
  endif()
endforeach()
list(LENGTH TRITON_SUPPORT_OBJS _ts_obj_count)
message(STATUS "Using ${_ts_obj_count} Triton support objects")

# ---------------------------------------------------------------------------
# libtriton.so path — Triton's shared library that contains all MLIR symbols.
# By linking against this instead of standalone MLIR libraries, we avoid
# duplicate MLIR dialect registrations when both are loaded in the same
# Python process.
# ---------------------------------------------------------------------------
set(LIBTRITON_PATH "${TRITON_SRCPATH}/python/triton/_C/libtriton.so")
if(NOT EXISTS "${LIBTRITON_PATH}")
  message(FATAL_ERROR "libtriton.so not found at ${LIBTRITON_PATH}")
endif()
message(STATUS "Using libtriton.so at ${LIBTRITON_PATH}")

# ---------------------------------------------------------------------------
# Standalone shared library — aggregates all conversion objects.
# This links against individual MLIR libraries and is used for standalone
# testing only (NOT loaded alongside libtriton.so in the same process).
# ---------------------------------------------------------------------------
add_subdirectory(lib/Conversion)

add_library(triton_metal_backend SHARED
    $<TARGET_OBJECTS:TritonMetalToLLVM>
    ${TRITON_IR_OBJS}
    python_bindings_bridge.cpp
)

target_link_libraries(triton_metal_backend
    PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRTensorDialect
    LLVMSupport
    LLVMCore
)

install(TARGETS triton_metal_backend LIBRARY DESTINATION lib)

# ---------------------------------------------------------------------------
# pybind11 Python extension — exposes C++ MLIR passes to Python
#
# KEY CHANGE: This module links against Triton's libtriton.so for MLIR
# symbols instead of standalone MLIR libraries. This prevents the duplicate
# MLIR dialect registration crash when both our .so and libtriton.so are
# loaded in the same Python process.
#
# All conversion source files are compiled directly into this module (no
# intermediate library), avoiding the two-library symbol resolution problem.
# ---------------------------------------------------------------------------
# Discover pybind11 via the active Python (robust across Python versions and
# checkout layouts — e.g. git worktrees that don't carry a .venv). Honor a
# -DPython_EXECUTABLE override (the build driver passes the venv python);
# fall back to the in-tree .venv path if discovery fails.
if(NOT DEFINED Python_EXECUTABLE)
  find_package(Python COMPONENTS Interpreter QUIET)
endif()
set(_pybind11_hint
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11")
if(Python_EXECUTABLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _pb_rc)
  if(_pb_rc EQUAL 0 AND _pybind11_cmakedir)
    set(_pybind11_hint "${_pybind11_cmakedir}")
  endif()
endif()
find_package(pybind11 REQUIRED PATHS "${_pybind11_hint}")

pybind11_add_module(_triton_metal_cpp NO_EXTRAS SHARED
    python_bindings.cpp
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    lib/Conversion/SharedMemoryOpToLLVM.cpp
    lib/Conversion/SharedMemoryAliasingPass.cpp
    lib/Conversion/DotOpToLLVM.cpp
)

# NO_EXTRAS disables pybind11's automatic -flto/IPO injection. LTO requires
# all symbols be resolvable at link time, which defeats -undefined
# dynamic_lookup (the mechanism we use to defer MLIR/LLVM symbol resolution
# until libtriton.so is loaded into the same Python process).
#
# SHARED (not pybind11's default MODULE/-bundle) is required on macOS so
# the linker accepts undefined references that resolve via libtriton.so at
# runtime. -bundle's stricter symbol resolution rejects them. Python loads
# .so files via dlopen regardless of whether they're -bundle or -dynamiclib.
# Derive the extension suffix from the building Python's ABI rather than
# hardcoding a version (was ".cpython-314-darwin.so", which breaks on any
# other Python). Falls back to the 3.14 tag if discovery fails.
set(_ext_suffix ".cpython-314-darwin.so")
if(Python_EXECUTABLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE _py_ext_suffix OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _es_rc)
  if(_es_rc EQUAL 0 AND _py_ext_suffix)
    set(_ext_suffix "${_py_ext_suffix}")
  endif()
endif()
set_target_properties(_triton_metal_cpp PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
    PREFIX ""
    SUFFIX "${_ext_suffix}"
)

# Include our headers for the conversion sources compiled into this module
target_include_directories(_triton_metal_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against libtriton.so for ALL MLIR/LLVM symbols. This is the same
# copy of MLIR that Triton uses, so no duplicate dialect registrations.
#
# REQUIRES libtriton built with TRITON_EXT_ENABLED=1 (upstream's flag for
# plugin/extension builds). Default libtriton hides all symbols except
# _PyInit_libtriton; with TRITON_EXT_ENABLED visibility is "default" so
# MLIR/LLVM symbols are exported and our module can dynamic_lookup them
# at runtime. See triton commits 6cb844590 (introduce flag) and
# fdfded1d8 (minimize default exports). The runtime guard in
# lib/Tools/PluginUtils.cpp refuses to load us if libtriton was built
# without TRITON_EXT_ENABLED, so the failure mode is loud, not silent.
target_link_libraries(_triton_metal_cpp PRIVATE ${LIBTRITON_PATH})

# macOS: defer MLIR/LLVM symbol resolution to runtime when libtriton.so is
# loaded into the same process. Same pattern triton_metal_plugin uses.
if(APPLE)
  target_link_options(_triton_metal_cpp PRIVATE
      "LINKER:-undefined,dynamic_lookup")
endif()

# NOTE: pre-TRITON_EXT_ENABLED, we needed to pull in TritonIR/TritonGPUIR/
# TritonNvidiaGPUIR/Tools .o files directly because their symbols (TypeID
# resolvers, getShape, dialect ctors) lived in libtriton as private-local
# symbols. With TRITON_EXT_ENABLED libtriton exports all of these with
# default visibility, so dynamic_lookup at runtime resolves them — no need
# to embed our own copies. Embedding them would also resurrect the
# duplicate dialect registration crash that was the original reason this
# build was structured around libtriton.so in the first place.

# Per-file RTTI settings:
# - python_bindings.cpp needs RTTI for pybind11 (typeid)
# - Everything else needs -fno-rtti to match LLVM/MLIR conventions
set_source_files_properties(
    python_bindings.cpp
    PROPERTIES COMPILE_FLAGS "-frtti"
)
set_source_files_properties(
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    lib/Conversion/SharedMemoryOpToLLVM.cpp
    lib/Conversion/SharedMemoryAliasingPass.cpp
    lib/Conversion/DotOpToLLVM.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

# ---------------------------------------------------------------------------
# triton-ext compatible plugin shared library
# Loads via TRITON_PASS_PLUGIN_PATH without pybind11 dependency.
# ---------------------------------------------------------------------------
add_library(triton_metal_plugin SHARED
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    lib/Conversion/SharedMemoryOpToLLVM.cpp
    lib/Conversion/SharedMemoryAliasingPass.cpp
    lib/Conversion/DotOpToLLVM.cpp
    ${TRITON_IR_OBJS}
    ${TRITON_GPU_OBJS}
    ${TRITON_NVIDIA_GPU_OBJS}
    ${TRITON_SUPPORT_OBJS}
)

target_link_libraries(triton_metal_plugin PRIVATE ${LIBTRITON_PATH})
target_include_directories(triton_metal_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# TritonGPU/LinearLayout methods live in libtriton.so as "local" symbols
# (internal linkage on macOS). Allow dynamic lookup so they resolve at
# runtime when loaded alongside libtriton.so, matching the pybind11
# module's bundle semantics.
if(APPLE)
  target_link_options(triton_metal_plugin PRIVATE
      "LINKER:-undefined,dynamic_lookup")
endif()
set_source_files_properties(
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    lib/Conversion/SharedMemoryOpToLLVM.cpp
    lib/Conversion/SharedMemoryAliasingPass.cpp
    lib/Conversion/DotOpToLLVM.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti"
)
