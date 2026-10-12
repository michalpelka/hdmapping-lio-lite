cmake_minimum_required(VERSION 3.16)
project(mini_lio LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# No version in find_package: Eigen 5's config only accepts its own major version.
find_package(Eigen3 REQUIRED NO_MODULE)
if(Eigen3_VERSION VERSION_LESS 3.3)
  message(FATAL_ERROR "Eigen >= 3.3 required, found ${Eigen3_VERSION}")
endif()
find_package(OpenMP) # optional: parallel Hessian / map update

# Some LASzip builds (e.g. `make install` on macOS) use an install name of
# @executable_path/../lib/liblaszip.*.dylib, which only resolves from a bin/ dir.
# Rewrite that reference in the linked binary to the library's real path.
function(mini_lio_fix_laszip_install_name target)
  if(NOT APPLE)
    return()
  endif()
  get_filename_component(_real "${LASZIP_LIBRARY}" REALPATH)
  execute_process(COMMAND otool -D "${_real}" OUTPUT_VARIABLE _out OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "^[^\n]*\n" "" _id "${_out}")
  if(_id MATCHES "^@executable_path")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND install_name_tool -change "${_id}" "${_real}" $<TARGET_FILE:${target}>
      COMMAND codesign --force -s - $<TARGET_FILE:${target}>
      VERBATIM)
  endif()
endfunction()

# --- vendored, tiny, dependency free ---
add_library(mini_lio_vqf STATIC ext/vqf/vqf.cpp)
target_include_directories(mini_lio_vqf PUBLIC ext/vqf)

add_library(mini_lio_fusion STATIC
  ext/fusion/FusionAhrs.c ext/fusion/FusionCompass.c ext/fusion/FusionOffset.c)
target_include_directories(mini_lio_fusion PUBLIC ext/fusion)

# --- the library (header only) ---
add_library(mini_lio INTERFACE)
target_include_directories(mini_lio INTERFACE
  include ext/unordered_dense ext/obs_eq)
target_link_libraries(mini_lio INTERFACE Eigen3::Eigen mini_lio_vqf mini_lio_fusion)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mini_lio INTERFACE OpenMP::OpenMP_CXX)
endif()

option(MINI_LIO_BUILD_EXAMPLES "Build the synthetic-data demo" ON)
if(MINI_LIO_BUILD_EXAMPLES)
  add_executable(sim_demo examples/sim_demo.cpp)
  target_link_libraries(sim_demo PRIVATE mini_lio)
  target_compile_options(sim_demo PRIVATE -Wall -Wextra -Wno-unused-parameter)
  find_path(LASZIP_INCLUDE_DIR laszip/laszip_api.h)
  find_library(LASZIP_LIBRARY NAMES laszip laszip_api)
  if(LASZIP_INCLUDE_DIR AND LASZIP_LIBRARY)  # optional: sim_demo --export-session
    target_compile_definitions(sim_demo PRIVATE MINI_LIO_HAVE_LASZIP)
    target_include_directories(sim_demo PRIVATE ${LASZIP_INCLUDE_DIR})
    target_link_libraries(sim_demo PRIVATE ${LASZIP_LIBRARY})
    mini_lio_fix_laszip_install_name(sim_demo)
  endif()
endif()

# --- Mandeye session runner (LAZ + IMU csv in, HDMapping-style result folder out); needs LASzip (liblaszip-dev) ---
option(MINI_LIO_BUILD_TOOLS "Build mini_lio_run (needs LASzip)" ON)
if(MINI_LIO_BUILD_TOOLS)
  find_path(LASZIP_INCLUDE_DIR laszip/laszip_api.h)
  find_library(LASZIP_LIBRARY NAMES laszip laszip_api)
  if(LASZIP_INCLUDE_DIR AND LASZIP_LIBRARY)
    add_executable(mini_lio_run tools/mini_lio_run.cpp)
    target_include_directories(mini_lio_run PRIVATE ${LASZIP_INCLUDE_DIR})
    target_link_libraries(mini_lio_run PRIVATE mini_lio ${LASZIP_LIBRARY})
    mini_lio_fix_laszip_install_name(mini_lio_run)
    target_compile_options(mini_lio_run PRIVATE -Wall -Wextra -Wno-unused-parameter)
  else()
    message(STATUS "LASzip not found: mini_lio_run not built (apt install liblaszip-dev)")
  endif()
endif()
