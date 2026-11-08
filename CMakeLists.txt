cmake_minimum_required(VERSION 3.21)
project(image_codec CXX)
add_compile_definitions(_USE_MATH_DEFINES NOMINMAX )
set(CMAKE_CXX_STANDARD 17)
add_executable(llcompc llcompc.cpp llcomp.hpp netpbm.hpp)
add_executable(llcompd llcompd.cpp llcomp.hpp netpbm.hpp)

add_executable(llcompc-xl llcompc.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompc-xl PRIVATE SELECT_MODEL=ModelSizeLarge)
add_executable(llcompd-xl llcompd.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompd-xl PRIVATE SELECT_MODEL=ModelSizeLarge)

add_executable(llcompc-s llcompc.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompc-s PRIVATE SELECT_MODEL=ModelSizeSmall)
add_executable(llcompd-s llcompd.cpp llcomp.hpp netpbm.hpp)
target_compile_definitions(llcompd-s PRIVATE SELECT_MODEL=ModelSizeSmall)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME roundtrip
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/roundtrip.py
                         --build ${CMAKE_BINARY_DIR})
    endif()
endif()
