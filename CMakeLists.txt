# NOTE: C++17 supported since CMake 3.8.0:
# https://cmake.org/cmake/help/v3.8/prop_tgt/CXX_STANDARD.html
cmake_minimum_required(VERSION 3.8.0)

# NOTE: policy for using the CheckIPOSupported module:
# https://cmake.org/cmake/help/latest/policy/CMP0069.html
cmake_policy(SET CMP0069 NEW)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(heyoka VERSION 0.1.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "heyoka version: ${heyoka_VERSION}")

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# The build options.
option(HEYOKA_BUILD_TESTS "Build unit tests." OFF)
option(HEYOKA_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(HEYOKA_ENABLE_IPO "Enable IPO (requires CMake >= 3.9 and compiler support)." OFF)
mark_as_advanced(HEYOKA_ENABLE_IPO)

# Assemble the flags.
set(HEYOKA_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG})
set(HEYOKA_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS})
if(YACMA_COMPILER_IS_MSVC)
  # On both cl and clang-cl, disable the idiotic minmax macros and enable the bigobj option.
  # Also, enable the WIN32_LEAN_AND_MEAN definition:
  # https://stackoverflow.com/questions/11040133/what-does-defining-win32-lean-and-mean-exclude-exactly
  list(APPEND HEYOKA_CXX_FLAGS_DEBUG "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  list(APPEND HEYOKA_CXX_FLAGS_RELEASE "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  if(YACMA_COMPILER_IS_CLANGXX)
    # clang-cl emits various warnings, let's just silence them.
    # NOTE: at one point in the recent past, MSVC added an options similar to GCC's isystem:
    # https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
    # We probably just need to wait for this to be picked up by CMake/clang-cl. Let's
    # revisit the issue in the future.
    list(APPEND _HEYOKA_CLANG_CL_DISABLED_WARNINGS
        "-Wno-unused-variable"
        "-Wno-inconsistent-dllimport"
        "-Wno-unknown-pragmas"
        "-Wno-unused-parameter"
        "-Wno-sign-compare"
        "-Wno-deprecated-declarations"
        "-Wno-deprecated-dynamic-exception-spec"
        "-Wno-old-style-cast"
        "-Wno-sign-conversion"
        "-Wno-non-virtual-dtor"
        "-Wno-deprecated"
        "-Wno-shadow"
        "-Wno-shorten-64-to-32"
        "-Wno-reserved-id-macro"
        "-Wno-undef"
        "-Wno-c++98-compat-pedantic"
        "-Wno-documentation-unknown-command"
        "-Wno-zero-as-null-pointer-constant"
        "-Wno-language-extension-token"
        "-Wno-gnu-anonymous-struct"
        "-Wno-nested-anon-types"
        "-Wno-documentation"
        "-Wno-comma"
        "-Wno-nonportable-system-include-path"
        "-Wno-global-constructors"
        "-Wno-redundant-parens"
        "-Wno-exit-time-destructors"
        "-Wno-missing-noreturn"
        "-Wno-switch-enum"
        "-Wno-covered-switch-default"
        "-Wno-float-equal"
        "-Wno-double-promotion"
        "-Wno-microsoft-enum-value"
        "-Wno-missing-prototypes"
        "-Wno-implicit-fallthrough"
        "-Wno-format-nonliteral"
        "-Wno-cast-qual"
        "-Wno-disabled-macro-expansion"
        "-Wno-unused-private-field"
        "-Wno-unused-template"
        "-Wno-unused-macros"
        "-Wno-extra-semi-stmt"
        "-Wno-c++98-compat")
    list(APPEND HEYOKA_CXX_FLAGS_DEBUG ${_HEYOKA_CLANG_CL_DISABLED_WARNINGS})
    list(APPEND HEYOKA_CXX_FLAGS_RELEASE ${_HEYOKA_CLANG_CL_DISABLED_WARNINGS})
    unset(_HEYOKA_CLANG_CL_DISABLED_WARNINGS)
  else()
    # Same as above, disable some cl warnings.
    list(APPEND HEYOKA_CXX_FLAGS_DEBUG "/wd4459" "/wd4127")
    list(APPEND HEYOKA_CXX_FLAGS_RELEASE "/wd4459" "/wd4127")
  endif()
  # Enable strict conformance mode, if supported.
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_compiler_flag("/permissive-" _HEYOKA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
  unset(CMAKE_REQUIRED_QUIET)
  if(_HEYOKA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    message(STATUS "The '/permissive-' flag is supported, enabling it.")
    list(APPEND HEYOKA_CXX_FLAGS_DEBUG "/permissive-")
    list(APPEND HEYOKA_CXX_FLAGS_RELEASE "/permissive-")
  endif()
  unset(_HEYOKA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
endif()

# List of source files.
set(HEYOKA_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llvm_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/number.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/binary_operator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/variable.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp"
    #"${CMAKE_CURRENT_SOURCE_DIR}/src/gp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math_functions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/string_conv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/llvm_helpers.cpp"
)

# Setup of the heyoka shared library.
add_library(heyoka SHARED "${HEYOKA_SRC_FILES}")
set_property(TARGET heyoka PROPERTY VERSION "1.0")
set_property(TARGET heyoka PROPERTY SOVERSION 1)
set_target_properties(heyoka PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(heyoka PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)

# Setup common to both the shared and static variants.
target_compile_options(heyoka PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
)
# Let's setup the target C++ standard, but only if the user did not provide it manually.
if(NOT CMAKE_CXX_STANDARD)
    set_property(TARGET heyoka PROPERTY CXX_STANDARD 17)
endif()
set_property(TARGET heyoka PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET heyoka PROPERTY CXX_EXTENSIONS NO)
target_include_directories(heyoka PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# IPO setup.
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0")
    if(HEYOKA_ENABLE_IPO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT _HEYOKA_IPO_RESULT OUTPUT _HEYOKA_IPO_OUTPUT)
        if (_HEYOKA_IPO_RESULT)
            message(STATUS "IPO requested and supported, enabling.")
            set_property(TARGET heyoka PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        else()
            message(STATUS "IPO requested, but it is not supported by the compiler:\n${_HEYOKA_IPO_OUTPUT}")
        endif()
        unset(_HEYOKA_IPO_RESULT)
        unset(_HEYOKA_IPO_OUTPUT)
    endif()
endif()

find_package(LLVM REQUIRED)

if(${LLVM_VERSION_MAJOR} LESS 10)
  message(FATAL_ERROR "LLVM >= 10 is required.")
endif()

add_library(heyoka::llvm_headers INTERFACE IMPORTED)
set_target_properties(heyoka::llvm_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS}")
target_link_libraries(heyoka PUBLIC heyoka::llvm_headers LLVM)

if(HEYOKA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(HEYOKA_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
