# ===============================
# CMakeLists.txt for DDAR Python module (pybind11)
# ===============================

cmake_minimum_required(VERSION 3.14)
project(DDAR)

# -------------------------------
# 1. 设置 C++ 标准
# -------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # -fPIC

# -------------------------------
# 2. 设置 Release 构建类型和优化选项
# -------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -march=native -flto -DNDEBUG)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(MSVC)
    add_compile_options(/O2 /GL /DNDEBUG /W3)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# -------------------------------
# 3. 查找 pybind11
# -------------------------------
find_package(pybind11 REQUIRED)

# -------------------------------
# 4. 搜索项目源文件
# -------------------------------
file(GLOB_RECURSE SOURCES
    RELATIVE ${CMAKE_SOURCE_DIR}
    "*.cpp"
)

# 排除 build 目录（可选）
list(FILTER SOURCES EXCLUDE REGEX "build/CMakeFiles/.*")

# -------------------------------
# 5. 添加 Python 模块库
# -------------------------------
pybind11_add_module(${PROJECT_NAME} ${SOURCES})

# Windows 上使用 .pyd
if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
endif()

# 可选：添加本地 include 路径
target_include_directories(${PROJECT_NAME} PRIVATE .)


set(DDAR_PGO "OFF" CACHE STRING "DDAR PGO mode: OFF, GENERATE, or USE")
if(NOT DDAR_PGO STREQUAL "OFF")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "DDAR PGO requires GCC")
    endif()
    set(profile_dir "${CMAKE_BINARY_DIR}/profile")
    if(DDAR_PGO STREQUAL "GENERATE")
        set(profile_flag "-fprofile-generate=${profile_dir}")
    elseif(DDAR_PGO STREQUAL "USE")
        set(profile_flag "-fprofile-use=${profile_dir}")
        target_compile_options(DDAR PRIVATE -fprofile-correction
            -Werror=missing-profile -Werror=coverage-mismatch)
        target_link_options(DDAR PRIVATE -fprofile-correction)
    else()
        message(FATAL_ERROR "Unknown DDAR_PGO mode: ${DDAR_PGO}")
    endif()
    target_compile_options(DDAR PRIVATE "${profile_flag}")
    target_link_options(DDAR PRIVATE "${profile_flag}")
    target_compile_definitions(DDAR PRIVATE DDAR_PGO_BUILD)
endif()
