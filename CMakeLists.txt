cmake_minimum_required(VERSION 3.18)

# Define project and enable C, C++, and CUDA languages
project(CUDA_OpenGL_interop LANGUAGES C CXX CUDA)

# Explicitly locate the CUDA Toolkit on the system
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Language & Compiler Options
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    # Static runtime library linkage (matching premake "staticruntime on")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
    # Restrict MSVC flags (/MP and /W0) ONLY to C and C++ files so NVCC ignores them
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP> $<$<COMPILE_LANGUAGE:CXX>:/W0>)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/MP> $<$<COMPILE_LANGUAGE:C>:/W0>)
endif()

# ------------------------------------------------------------------------------
# Dependency: GLAD (StaticLib)
# ------------------------------------------------------------------------------
add_library(glad STATIC 
    vendor/glad/src/glad.c
)
target_include_directories(glad PUBLIC vendor/glad/include)

# ------------------------------------------------------------------------------
# Dependency: GLFW (StaticLib)
# ------------------------------------------------------------------------------
set(GLFW_SOURCES
    vendor/glfw/include/GLFW/glfw3.h
    vendor/glfw/include/GLFW/glfw3native.h
    vendor/glfw/src/internal.h
    vendor/glfw/src/platform.h
    vendor/glfw/src/mappings.h
    vendor/glfw/src/context.c
    vendor/glfw/src/init.c
    vendor/glfw/src/input.c
    vendor/glfw/src/monitor.c
    vendor/glfw/src/platform.c
    vendor/glfw/src/vulkan.c
    vendor/glfw/src/window.c
    vendor/glfw/src/egl_context.c
    vendor/glfw/src/osmesa_context.c
    vendor/glfw/src/null_platform.h
    vendor/glfw/src/null_joystick.h
    vendor/glfw/src/null_init.c
    vendor/glfw/src/null_monitor.c
    vendor/glfw/src/null_window.c
    vendor/glfw/src/null_joystick.c
)

if(WIN32)
    list(APPEND GLFW_SOURCES 
        vendor/glfw/src/win32_init.c
        vendor/glfw/src/win32_module.c
        vendor/glfw/src/win32_joystick.c
        vendor/glfw/src/win32_monitor.c
        vendor/glfw/src/win32_time.h
        vendor/glfw/src/win32_time.c
        vendor/glfw/src/win32_thread.h
        vendor/glfw/src/win32_thread.c
        vendor/glfw/src/win32_window.c
        vendor/glfw/src/wgl_context.c
    )
elseif(APPLE)
    list(APPEND GLFW_SOURCES
        vendor/glfw/src/cocoa_init.m
        vendor/glfw/src/cocoa_joystick.m
        vendor/glfw/src/cocoa_joystick.h
        vendor/glfw/src/cocoa_monitor.m
        vendor/glfw/src/cocoa_platform.h
        vendor/glfw/src/cocoa_time.c
        vendor/glfw/src/cocoa_window.m
        vendor/glfw/src/nsgl_context.m
        vendor/glfw/src/nsgl_context.h
        vendor/glfw/src/posix_thread.c
        vendor/glfw/src/posix_thread.h
    )
elseif(UNIX AND NOT APPLE)
    list(APPEND GLFW_SOURCES
        vendor/glfw/src/glx_context.c
        vendor/glfw/src/glx_context.h
        vendor/glfw/src/linux_joystick.c
        vendor/glfw/src/linux_joystick.h
        vendor/glfw/src/posix_time.c
        vendor/glfw/src/posix_time.h
        vendor/glfw/src/posix_thread.c
        vendor/glfw/src/posix_thread.h
        vendor/glfw/src/x11_init.c
        vendor/glfw/src/x11_monitor.c
        vendor/glfw/src/x11_platform.h
        vendor/glfw/src/x11_window.c
        vendor/glfw/src/xkb_unicode.c
        vendor/glfw/src/xkb_unicode.h
    )
endif()

add_library(glfw STATIC ${GLFW_SOURCES})
target_include_directories(glfw PUBLIC vendor/glfw/include)

if(WIN32)
    target_compile_definitions(glfw PRIVATE _GLFW_WIN32 _CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    target_compile_definitions(glfw PRIVATE _GLFW_COCOA)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(glfw PRIVATE _GLFW_X11)
endif()

# ------------------------------------------------------------------------------
# Main Application: CUDA_OpenGL_interop
# ------------------------------------------------------------------------------
file(GLOB_RECURSE APP_SOURCES 
    "CUDA_OpenGL_interop/src/*.cpp"
    "CUDA_OpenGL_interop/src/*.h"
    "CUDA_OpenGL_interop/src/*.hpp"
    "CUDA_OpenGL_interop/src/*.cu"
    "CUDA_OpenGL_interop/src/*.cuh"
)

add_executable(CUDA_OpenGL_interop ${APP_SOURCES})

# Include directories (explicitly passing CUDA Toolkit includes to MSVC for .cpp files)
target_include_directories(CUDA_OpenGL_interop PRIVATE
    vendor/glm
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Macro definitions
target_compile_definitions(CUDA_OpenGL_interop PRIVATE 
    GLFW_INCLUDE_NONE
    $<$<CONFIG:Debug>:DEBUG>
)

# Link libraries (including CUDA runtime target)
target_link_libraries(CUDA_OpenGL_interop PRIVATE 
    glad 
    glfw 
    CUDA::cudart
)

if(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    target_link_libraries(CUDA_OpenGL_interop PRIVATE 
        ${COREFOUNDATION_LIBRARY} 
        ${COCOA_LIBRARY} 
        ${IOKIT_LIBRARY} 
        ${COREVIDEO_LIBRARY}
    )
endif()

# ------------------------------------------------------------------------------
# CUDA Properties & NVCC Flags
# ------------------------------------------------------------------------------
set_target_properties(CUDA_OpenGL_interop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(CUDA_OpenGL_interop PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math 
        --ptxas-options=-v 
        -maxrregcount=32
    >
)