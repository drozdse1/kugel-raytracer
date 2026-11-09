cmake_minimum_required(VERSION 3.16)
project(kugel-raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(KUGEL_SOURCES
    src/main.cpp
    src/scene.cpp
    src/gpu.cpp
    src/image.cpp)

# Linux bleibt beim X11-Fenster. Das läuft auch unter Wayland, sobald XWayland da ist.
# macOS hat kein X11. Dort zeigt SDL dasselbe Bild, MoltenVK spricht die GPU an.
if(APPLE)
    list(APPEND KUGEL_SOURCES src/live_view_mac.cpp)
else()
    list(APPEND KUGEL_SOURCES src/live_view.cpp)
endif()

add_executable(kugel-raytracer ${KUGEL_SOURCES})
target_include_directories(kugel-raytracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(kugel-raytracer SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_compile_options(kugel-raytracer PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED IMPORTED_TARGET sdl2)
    find_library(MOLTENVK_LIBRARY MoltenVK PATHS /opt/homebrew/lib /usr/local/lib)
    if(NOT MOLTENVK_LIBRARY)
        message(FATAL_ERROR "MoltenVK nicht gefunden. Installation: brew install molten-vk")
    endif()
    target_link_libraries(kugel-raytracer PRIVATE
        PkgConfig::SDL2
        ${MOLTENVK_LIBRARY}
        "-framework CoreText"
        "-framework CoreGraphics"
        "-framework CoreFoundation")
else()
    target_link_libraries(kugel-raytracer PRIVATE
        /usr/lib/x86_64-linux-gnu/libvulkan.so.1
        /usr/lib/x86_64-linux-gnu/libX11.so.6
        /usr/lib/x86_64-linux-gnu/libXext.so.6)
endif()
