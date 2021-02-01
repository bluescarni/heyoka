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

project(heyoka VERSION 0.3.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "heyoka version: ${heyoka_VERSION}")

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# The build options.
option(HEYOKA_BUILD_TESTS "Build unit tests." OFF)
option(HEYOKA_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(HEYOKA_BUILD_TUTORIALS "Build tutorials." OFF)
option(HEYOKA_WITH_MPPP "Enable features relying on mp++." OFF)
option(HEYOKA_WITH_SLEEF "Enable features relying on SLEEF." OFF)
option(HEYOKA_BUILD_STATIC_LIBRARY "Build heyoka as a static library, instead of dynamic." OFF)
option(HEYOKA_ENABLE_IPO "Enable IPO (requires CMake >= 3.9 and compiler support)." OFF)
mark_as_advanced(HEYOKA_ENABLE_IPO)

# NOTE: on Unix systems, the correct library installation path
# could be something other than just "lib", such as "lib64",
# "lib32", etc., depending on platform/configuration. Apparently,
# CMake provides this information via the GNUInstallDirs module.
# Let's enable this for now on all Unixes except OSX.
# NOTE: potentially, this could be applicable to Cygwin as well.
#
# https://cmake.org/cmake/help/v3.15/module/GNUInstallDirs.html
# https://cmake.org/pipermail/cmake/2013-July/055375.html
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_HEYOKA_INSTALL_LIBDIR_DEFAULT "${CMAKE_INSTALL_LIBDIR}")
else()
    set(_HEYOKA_INSTALL_LIBDIR_DEFAULT "lib")
endif()
if(NOT HEYOKA_INSTALL_LIBDIR)
    set(HEYOKA_INSTALL_LIBDIR "${_HEYOKA_INSTALL_LIBDIR_DEFAULT}" CACHE STRING
        "Library installation directory." FORCE)
endif()
mark_as_advanced(HEYOKA_INSTALL_LIBDIR)
message(STATUS "Library installation directory: ${HEYOKA_INSTALL_LIBDIR}")

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
    list(APPEND HEYOKA_CXX_FLAGS_DEBUG "/wd4459" "/wd4127" "/wd4251")
    list(APPEND HEYOKA_CXX_FLAGS_RELEASE "/wd4459" "/wd4127" "/wd4251")
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

# Explanation: on MSVC, when building static libraries, it is good practice to link
# to the static runtime. CMake, however, is hard-coded to link to the dynamic runtime.
# Hence we hackishly replace the /MD flag with /MT. This is the approach suggested
# in the CMake FAQ:
#
# https://gitlab.kitware.com/cmake/community/wikis/FAQ#how-can-i-build-my-msvc-application-with-a-static-runtime
#
# Note that at one point CMake added the possiblity to set this as a target property,
# so in the future we should definitely migrate to that approach:
#
# https://cmake.org/cmake/help/git-master/prop_tgt/MSVC_RUNTIME_LIBRARY.html
if(YACMA_COMPILER_IS_MSVC AND HEYOKA_BUILD_STATIC_LIBRARY)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

# List of source files.
set(HEYOKA_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llvm_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/binary_operator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/func.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/number.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/variable.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/param.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nbody.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mascon.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/cos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/pow.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sqrt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/square.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/tan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/time.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/string_conv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/math_wrappers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/llvm_helpers.cpp"
    # NOTE: sleef.cpp needs to be compiled even if we are not
    # building with sleef support on.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/sleef.cpp"
)

if(HEYOKA_BUILD_STATIC_LIBRARY)
    # Setup of the heyoka static library.
    message(STATUS "heyoka will be built as a static library.")
    add_library(heyoka STATIC "${HEYOKA_SRC_FILES}")
else()
    # Setup of the heyoka shared library.
    add_library(heyoka SHARED "${HEYOKA_SRC_FILES}")
    set_property(TARGET heyoka PROPERTY VERSION "3.0")
    set_property(TARGET heyoka PROPERTY SOVERSION 3)
    set_target_properties(heyoka PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(heyoka PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
endif()

# Setup common to both the shared and static variants.
target_compile_options(heyoka PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
)

# Ensure that C++17 is employed when both compiling and consuming heyoka.
target_compile_features(heyoka PUBLIC cxx_std_17)
# Enforce vanilla C++17 when compiling heyoka.
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

# Mandatory dependency on LLVM.
find_package(LLVM REQUIRED CONFIG)

if(${LLVM_VERSION_MAJOR} LESS 10)
  message(FATAL_ERROR "LLVM >= 10 is required.")
endif()

message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

# Add the definitions required by LLVM.
target_compile_definitions(heyoka PRIVATE ${LLVM_DEFINITIONS})
if(MSVC)
    # On MSVC, silence std::iterator warnings coming from the LLVM headers.
    target_compile_definitions(heyoka PRIVATE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
endif()

# Add the LLVM include dirs.
target_include_directories(heyoka SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

if(TARGET LLVM)
    # On some platforms, LLVM is built as a single library,
    # which is exported by LLVM's CMake config-file package as
    # a target called LLVM.
    target_link_libraries(heyoka PRIVATE LLVM)
else()
    # Otherwise, LLVM is built as a collection of
    # separate libraries/components.

    # NOTE: this is a nasty hack originating from the fact
    # that currently the CMake files installed by LLVM contain a direct
    # link to an entity called "z" representing the zlib library.
    # If the file z.lib is not in a "standard" location (as it is
    # often the case), the linker will not find it, resulting in
    # an error at build time. Thus, we fetch the location
    # of z.lib from CMake's FindZLIB.cmake and add it
    # to the linker path.
    find_package(ZLIB REQUIRED)
    get_target_property(_HEYOKA_ZLIB_LOCATION ZLIB::ZLIB LOCATION)
    message(STATUS "ZLIB::ZLIB location : ${_HEYOKA_ZLIB_LOCATION}")
    get_filename_component(_HEYOKA_ZLIB_LOCATION_DIR "${_HEYOKA_ZLIB_LOCATION}" DIRECTORY)
    message(STATUS "ZLIB::ZLIB directory : ${_HEYOKA_ZLIB_LOCATION_DIR}")
    target_link_directories(heyoka PRIVATE "${_HEYOKA_ZLIB_LOCATION_DIR}")

    unset(_HEYOKA_ZLIB_LOCATION)
    unset(_HEYOKA_ZLIB_LOCATION_DIR)

    # NOTE: these components have been determined heuristically.
    set(_HEYOKA_LLVM_COMPONENTS native orcjit)
    # NOTE: not sure what these two do, I copied from symengine's CMakeLists.txt.
    llvm_map_components_to_libnames(_HEYOKA_LLVM_LIBS_DIRECT ${_HEYOKA_LLVM_COMPONENTS})
    llvm_expand_dependencies(_HEYOKA_LLVM_LIBS ${_HEYOKA_LLVM_LIBS_DIRECT})
    target_link_libraries(heyoka PRIVATE ${_HEYOKA_LLVM_LIBS})

    unset(_HEYOKA_LLVM_COMPONENTS)
    unset(_HEYOKA_LLVM_LIBS_DIRECT)
    unset(_HEYOKA_LLVM_LIBS)
endif()

# Mandatory dependency on fmt.
find_package(fmt REQUIRED CONFIG)
target_link_libraries(heyoka PRIVATE fmt::fmt)

# Mandatory dependency on Boost.
find_package(Boost 1.60 REQUIRED COMPONENTS filesystem)

# Public dependency on the headers, private dependency
# on filesystem.
target_link_libraries(heyoka PUBLIC Boost::boost)
target_link_libraries(heyoka PRIVATE Boost::filesystem)
# NOTE: quench warnings from Boost when building the library.
target_compile_definitions(heyoka PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)

# Optional dependency on mp++.
# NOTE: put this into a separate variable for reuse later.
set(_HEYOKA_MIN_MPPP_VERSION "0.22")
if(HEYOKA_WITH_MPPP)
    find_package(mp++ REQUIRED CONFIG)
    if(${mp++_VERSION} VERSION_LESS ${_HEYOKA_MIN_MPPP_VERSION})
        message(FATAL_ERROR "mp++ >= ${_HEYOKA_MIN_MPPP_VERSION} is required.")
    endif()
    target_link_libraries(heyoka PUBLIC mp++::mp++)
endif()

# Optional dependency on SLEEF.
if(HEYOKA_WITH_SLEEF)
    find_package(heyoka_SLEEF REQUIRED)
    target_link_libraries(heyoka PRIVATE heyoka::SLEEF)
endif()

# Configure config.hpp.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/heyoka/config.hpp" @ONLY)

# Configure the doc files.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/doc/conf.py.in" "${CMAKE_CURRENT_SOURCE_DIR}/doc/conf.py" @ONLY)

# Installation of the header files.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/heyoka" DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/heyoka/config.hpp" DESTINATION include/heyoka)

# Installation of the library.
install(TARGETS heyoka
    EXPORT heyoka_export
    LIBRARY DESTINATION "${HEYOKA_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${HEYOKA_INSTALL_LIBDIR}"
    RUNTIME DESTINATION bin
)

# Setup of the CMake config file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/heyoka-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/heyoka-config.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/heyoka-config.cmake"
    DESTINATION "${HEYOKA_INSTALL_LIBDIR}/cmake/heyoka")
install(EXPORT heyoka_export NAMESPACE heyoka:: DESTINATION "${HEYOKA_INSTALL_LIBDIR}/cmake/heyoka")
# Take care of versioning.
include(CMakePackageConfigHelpers)
# NOTE: SameMinorVersion available only
# since CMake 3.11.
if(${CMAKE_VERSION} VERSION_LESS "3.11.0")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/heyoka-config-version.cmake" COMPATIBILITY SameMajorVersion)
else()
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/heyoka-config-version.cmake" COMPATIBILITY SameMinorVersion)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/heyoka-config-version.cmake" DESTINATION "${HEYOKA_INSTALL_LIBDIR}/cmake/heyoka")

# Cleanup.
unset(_HEYOKA_MIN_MPPP_VERSION)

if(HEYOKA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(HEYOKA_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(HEYOKA_BUILD_TUTORIALS)
    add_subdirectory(tutorial)
endif()
