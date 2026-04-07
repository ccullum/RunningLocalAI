cmake_minimum_required(VERSION 3.15...3.27)
project(llama_engine)

# Use standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRITICAL FIX 1: Force static linking so we don't get "Missing DLL" errors in Python!
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static libraries" FORCE)

# Configure llama.cpp
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Disable llama tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Disable llama examples")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Disable llama server")

# Tell CMake to compile the llama.cpp submodule
add_subdirectory(llama.cpp)

# Configure PyBind11
find_package(pybind11 REQUIRED)

# Build our Python Module
pybind11_add_module(llama_engine embedding_engine.cpp)
target_link_libraries(llama_engine PRIVATE llama)

# CRITICAL FIX 2: Tell CMake to actually put our compiled engine into the Python wheel!
install(TARGETS llama_engine DESTINATION .)