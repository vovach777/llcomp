cmake_minimum_required(VERSION 3.21)
project(image_codec CXX)
add_compile_definitions(_USE_MATH_DEFINES NOMINMAX )
set(CMAKE_CXX_STANDARD 17)
add_executable(llcompc llcompc.cpp llcomp.hpp netpbm.hpp)
add_executable(llcompd llcompd.cpp llcomp.hpp netpbm.hpp)

add_executable(llcompc-lm llcompc.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompc-lm PRIVATE SELECT_MODEL=ModelSizeLarge)

add_executable(llcompc-tm llcompc.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompc-tm PRIVATE SELECT_MODEL=ModelSizeTiny)

add_executable(llcompd-lm llcompd.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompd-lm PRIVATE SELECT_MODEL=ModelSizeLarge)

add_executable(llcompd-tm llcompd.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompd-tm PRIVATE SELECT_MODEL=ModelSizeTiny)
