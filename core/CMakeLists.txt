cmake_minimum_required(VERSION 3.16)
project(human_core C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

add_library(human STATIC human.c)
target_include_directories(human PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(human PUBLIC m)
