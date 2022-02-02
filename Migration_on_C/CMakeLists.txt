cmake_minimum_required(VERSION 3.21)
project(Migration_on_C)

set(pybind11_ROOT "C:\\Users\\GLEB2001\\anaconda3\\envs\\distributed-kirchhoff-migration\\Lib\\site-packages\\pybind11\\share\\cmake\\pybind11")

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11)
find_package(Python3)

add_executable(Migration_on_C test_main.cpp)

message("PYTHON_INCLUDE_PATH == ${PYTHON_INCLUDE_PATH}")

target_include_directories(Migration_on_C PRIVATE "C:\\Users\\GLEB2001\\anaconda3\\envs\\distributed-kirchhoff-migration\\include")
target_include_directories(Migration_on_C PRIVATE "C:\\Users\\GLEB2001\\anaconda3\\envs\\distributed-kirchhoff-migration\\Lib\\site-packages\\pybind11\\include")