cmake_minimum_required(VERSION 3.12)
project(Game_2048_Pybind11)

# Ensure C++14
set(CMAKE_CXX_STANDARD 14)

# Add your include directory
include_directories(include)

# Find Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Add pybind11
add_subdirectory(pybind11)

# Build the Python module
pybind11_add_module(game env/bindings.cpp env/2048_game.cpp)

# Link Python libraries to your module
target_link_libraries(game PRIVATE ${PYTHON_LIBRARIES})