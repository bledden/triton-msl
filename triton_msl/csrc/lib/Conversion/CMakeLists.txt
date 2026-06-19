# Build conversion sources as an OBJECT library so the top-level shared lib
# can aggregate them without creating an intermediate static archive.
add_library(TritonMSLToLLVM OBJECT
    TritonMSLToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    SharedMemoryOpToLLVM.cpp
    SharedMemoryAliasingPass.cpp
    DotOpToLLVM.cpp
)

target_include_directories(TritonMSLToLLVM
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
