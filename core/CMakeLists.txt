cmake_minimum_required(VERSION 3.16)
project(human_core C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

set(HUMAN_SOURCES
    hmath.c
    skeleton.c
    morph.c
    skin.c
    proto.c
    builder.c
    io.c
    human.c
)

add_library(human STATIC ${HUMAN_SOURCES})
target_include_directories(human PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(human PUBLIC m)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine.c)
    add_executable(test_engine tests/test_engine.c)
    target_link_libraries(test_engine PRIVATE human)
    enable_testing()
    add_test(NAME engine COMMAND test_engine)
endif()
