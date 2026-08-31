cmake_minimum_required(VERSION 3.21)
project(qingming_gfx1100_gemv LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1100)
endif()

set_source_files_properties(benchmark.cpp gemv.cpp PROPERTIES LANGUAGE HIP)
add_executable(benchmark benchmark.cpp gemv.cpp)
target_compile_options(benchmark PRIVATE -O3)
target_include_directories(benchmark PRIVATE /opt/rocm/include)
target_link_directories(benchmark PRIVATE /opt/rocm/lib /opt/rocm/lib64)
target_link_libraries(benchmark PRIVATE rocblas hipblaslt rocprofiler-sdk-roctx)
set_target_properties(benchmark PROPERTIES
  BUILD_RPATH "/opt/rocm/lib;/opt/rocm/lib64")

set_source_files_properties(numerical/fp32_accuracy.cpp PROPERTIES LANGUAGE HIP)
add_executable(fp32_accuracy numerical/fp32_accuracy.cpp gemv.cpp)
target_compile_options(fp32_accuracy PRIVATE -O3)
target_include_directories(fp32_accuracy PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  /opt/rocm/include)
target_link_directories(fp32_accuracy PRIVATE /opt/rocm/lib /opt/rocm/lib64)
target_link_libraries(fp32_accuracy PRIVATE rocblas hipblaslt)
set_target_properties(fp32_accuracy PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED YES
  BUILD_RPATH "/opt/rocm/lib;/opt/rocm/lib64")
