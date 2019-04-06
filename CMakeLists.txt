
cmake_minimum_required (VERSION 3.3)
project(multigrid)

enable_testing()

include_directories(./include ./xtensor/include ./xtl/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

add_library(krylov_int ./src/mesh.cpp)
target_compile_options(krylov_int PUBLIC -fPIC)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(krylov ./src/bindpy.cpp)
target_link_libraries(krylov PRIVATE krylov_int)
set_target_properties(krylov PROPERTIES
                      CXX_STANDARD 17
		      CXX_STANDARD_REQUIRED ON
		      CXX_EXTENSIONS OFF)

add_executable(krylov_tester ./src/test.cpp)
target_link_libraries(krylov_tester krylov_int gtest pthread)
add_test(krylov_test krylov_tester)

configure_file(./scripts/prob1.py prob1.py COPYONLY)
