cmake_minimum_required(VERSION 3.15)
project(rl2048_alphazero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)
find_package(OpenMP REQUIRED)

pybind11_add_module(az_engine cpp/az_engine.cpp)
target_compile_options(az_engine PRIVATE -O3 -march=native)
target_link_libraries(az_engine PRIVATE OpenMP::OpenMP_CXX)

# Place az_engine*.so at the repo root so `import az_engine` works from here
set_target_properties(az_engine PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
