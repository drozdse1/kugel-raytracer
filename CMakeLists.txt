cmake_minimum_required(VERSION 3.16)
project(kugel-raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kugel-raytracer src/main.cpp)
target_include_directories(kugel-raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(kugel-raytracer SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_options(kugel-raytracer PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(kugel-raytracer PRIVATE
    /usr/lib/x86_64-linux-gnu/libvulkan.so.1
    /usr/lib/x86_64-linux-gnu/libX11.so.6
    /usr/lib/x86_64-linux-gnu/libXext.so.6)
