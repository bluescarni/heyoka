# NOTE: the 3.18 requirement comes from here:
# https://cmake.org/cmake/help/latest/module/CheckLinkerFlag.html
cmake_minimum_required(VERSION 3.18.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(heyoka VERSION 5.1.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "heyoka version: ${heyoka_VERSION}")

# Print some info about the long double type.
include(CheckTypeSize)
CHECK_TYPE_SIZE("long double" _HEYOKA_LONG_DOUBLE_SIZE LANGUAGE CXX)
message(STATUS "Size of the long double type: ${_HEYOKA_LONG_DOUBLE_SIZE}")
unset(_HEYOKA_LONG_DOUBLE_SIZE)

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# The build options.
option(HEYOKA_BUILD_TESTS "Build unit tests." OFF)
option(HEYOKA_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(HEYOKA_BUILD_TUTORIALS "Build tutorials." OFF)
option(HEYOKA_WITH_MPPP "Enable features relying on mp++." OFF)
option(HEYOKA_WITH_SLEEF "Enable features relying on SLEEF." OFF)
option(HEYOKA_BUILD_STATIC_LIBRARY "Build heyoka as a static library, instead of dynamic." OFF)
option(HEYOKA_ENABLE_IPO "Enable IPO (requires compiler support)." OFF)
option(HEYOKA_FORCE_STATIC_LLVM "Force linking to the static version of the LLVM libraries." OFF)
mark_as_advanced(HEYOKA_FORCE_STATIC_LLVM)
option(HEYOKA_HIDE_LLVM_SYMBOLS "Try to hide LLVM symbols when linking statically to LLVM." OFF)
mark_as_advanced(HEYOKA_HIDE_LLVM_SYMBOLS)

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
    list(APPEND HEYOKA_CXX_FLAGS_DEBUG "/wd4459" "/wd4127" "/wd4251" "/wd4661")
    list(APPEND HEYOKA_CXX_FLAGS_RELEASE "/wd4459" "/wd4127" "/wd4251" "/wd4661")
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

# Mandatory dependency on LLVM.
#
# NOTE: do it early so that we can set up
# compiler flags based on the LLVM version
# if needed.
#
# NOTE: forcibly include GNUInstallDirs (even
# though we might have already included it
# earlier) in order to work around an apparent issue
# in Findzstd.cmake triggered by the LLVM config-file packages.
# See:
#
# https://github.com/llvm/llvm-project/issues/58558
include(GNUInstallDirs)
find_package(LLVM REQUIRED CONFIG)

if(${LLVM_VERSION_MAJOR} LESS 13 OR ${LLVM_VERSION_MAJOR} GREATER 18)
  message(FATAL_ERROR "LLVM >= 13 and <= 18 is required.")
endif()

# List of source files.
set(HEYOKA_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_00.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_01.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_02.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_adaptive.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_adaptive_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/taylor_stream_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/i_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nt_event.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/t_event.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dtens.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cfunc_class.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/continuous_output.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/var_ode_sys.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/cm_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/event_detection.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/llvm_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/llvm_helpers_celmec.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/dfloat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/num_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/num_identity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/sum_sq.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/div.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/sub.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vector_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/empty_callable_s11n.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/setup_variational_ics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/tm_data.cpp"
    # NOTE: this will be an empty file in case we are not
    # building with support for real.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/real_helpers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/validate_ode_sys.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llvm_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ensemble_propagate.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/func.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/number.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/variable.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/param.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression_basic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression_diff.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression_ops.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/expression_cfunc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lagrangian.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hamiltonian.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logging.cpp"
    # VSOP2013 details.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_1_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_2_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_3_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_4_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_5_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_6_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_7_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_8_6.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_5.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/vsop2013/vsop2013_9_6.cpp"
    # ELP2000 details.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/elp2000/elp2000_1_3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/elp2000/elp2000_4_9.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/elp2000/elp2000_10_15.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/elp2000/elp2000_16_21.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/elp2000/elp2000_22_36.cpp"
    # Models.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/pendulum.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/nbody.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/fixed_centres.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/rotating.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/mascon.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/vsop2013.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/elp2000.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/cr3bp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/ffnn.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/cart2geo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/nrlmsise00_tn.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model/jb08_tn.cpp"
    # Callbacks.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/callback/angle_reducer.cpp"
    # Math functions.
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/kepE.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/kepF.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/kepDE.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/cos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/exp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/log.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/pow.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sigmoid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/relu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sqrt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/tan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/asin.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/acos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/atan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/atan2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/time.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/cosh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sinh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/tanh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/asinh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/acosh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/atanh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/erf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/sum.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/prod.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/constants.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/dfun.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/relational.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/math/logical.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/string_conv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/logging_impl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/step_callback.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/llvm_state_mem_cache.cpp"
)

if(HEYOKA_WITH_SLEEF)
    # NOTE: the sleef dummy file needs to be compiled only
    # if we are building with sleef support on.
    set(HEYOKA_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/sleef_dummy.cpp"
        "${HEYOKA_SRC_FILES}"
    )
endif()

# Setup the heyoka ABI version number.
set(HEYOKA_ABI_VERSION 29)

if(HEYOKA_BUILD_STATIC_LIBRARY)
    # Setup of the heyoka static library.
    message(STATUS "heyoka will be built as a static library.")
    add_library(heyoka STATIC "${HEYOKA_SRC_FILES}")
else()
    # Setup of the heyoka shared library.
    add_library(heyoka SHARED "${HEYOKA_SRC_FILES}")
    set_property(TARGET heyoka PROPERTY VERSION "${HEYOKA_ABI_VERSION}.0")
    set_property(TARGET heyoka PROPERTY SOVERSION ${HEYOKA_ABI_VERSION})
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

# Ensure that C++20 is employed when both compiling and consuming heyoka.
target_compile_features(heyoka PUBLIC cxx_std_20)
# Enforce vanilla C++17 when compiling heyoka.
set_property(TARGET heyoka PROPERTY CXX_EXTENSIONS NO)

target_include_directories(heyoka PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# IPO setup.
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

# Add a define to signal that we are building the library.
target_compile_definitions(heyoka PRIVATE HEYOKA_BUILD_LIBRARY)

message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")
message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

# Define an interface target for linking to the LLVM bits.
add_library(heyoka_llvm_internal INTERFACE)
# Add the definitions required by LLVM.
target_compile_definitions(heyoka_llvm_internal INTERFACE ${LLVM_DEFINITIONS})
# On MSVC, silence std::iterator warnings coming from the LLVM headers.
if(MSVC)
    target_compile_definitions(heyoka_llvm_internal INTERFACE _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
endif()

# Add the LLVM include dirs.
target_include_directories(heyoka_llvm_internal SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})

if(HEYOKA_FORCE_STATIC_LLVM OR NOT TARGET LLVM)
    # NOTE: static link to llvm if explicitly requested or if the
    # LLVM target does *not* exist. The LLVM target, if
    # it exists, consists of a single shared library.

    # NOTE: this is necessary because LLVM's config-file packages
    # do not look for zlib, despite linking to the ZLIB::ZLIB
    # target. I think this is not necessary in shared library builds
    # because zlib might be a private dependency and thus it is not
    # necessary to bring it into the link chain explicitly (as the
    # LLVM shared object already brings it in).
    find_package(ZLIB REQUIRED)

    # NOTE: these components have been determined heuristically.
    set(_HEYOKA_LLVM_COMPONENTS native orcjit)
    # NOTE: not sure what these two do, I copied from symengine's CMakeLists.txt.
    llvm_map_components_to_libnames(_HEYOKA_LLVM_LIBS_DIRECT ${_HEYOKA_LLVM_COMPONENTS})
    llvm_expand_dependencies(_HEYOKA_LLVM_LIBS ${_HEYOKA_LLVM_LIBS_DIRECT})
    target_link_libraries(heyoka_llvm_internal INTERFACE ${_HEYOKA_LLVM_LIBS})

    unset(_HEYOKA_LLVM_COMPONENTS)
    unset(_HEYOKA_LLVM_LIBS_DIRECT)
    unset(_HEYOKA_LLVM_LIBS)

    set(_HEYOKA_LLVM_STATIC_LINK TRUE)
else()
    # NOTE: shared linking is preferred over static if both are available and
    # if HEYOKA_FORCE_STATIC_LLVM is *not* set.
    target_link_libraries(heyoka_llvm_internal INTERFACE LLVM)

    set(_HEYOKA_LLVM_STATIC_LINK FALSE)
endif()

# Link to LLVM privately.
# NOTE: the BUILD_INTERFACE bit is a workaround needed when building
# heyoka as a static library: in this configuration, CMake would complain
# that heyoka_llvm_internal needs to be in the export set, even if it is just
# an interface library. See the discussion here:
# https://gitlab.kitware.com/cmake/cmake/-/issues/15415
target_link_libraries(heyoka PRIVATE "$<BUILD_INTERFACE:heyoka_llvm_internal>")

if(_HEYOKA_LLVM_STATIC_LINK AND HEYOKA_HIDE_LLVM_SYMBOLS)
    # NOTE: LLVM symbol hiding is done in a heuristic fashion
    # by checking if the linker supports certain flags:
    #
    # - first we check for "-Wl,--exclude-libs,ALL", which is supported
    #   by the GNU linker; if this fails,
    # - we check for "-Wl,-unexported_symbol", which is supported by
    #   OSX's ld64 linker; otherwise,
    # - we don't do anything.
    #
    # This is not a great solution, but it works for common use cases.
    # For the GNU linker, it would be better to use "-Wl,--exclude-libs"
    # on all the individual LLVM libraries we are linking to, rather than
    # on ALL the static libraries, but getting such a list out of CMake
    # seems hard: we link directly to the _HEYOKA_LLVM_LIBS libraries,
    # but these in turn bring in indirectly several other LLVM components.

    include(CheckLinkerFlag)

    set(CMAKE_REQUIRED_QUIET TRUE)

    check_linker_flag(CXX "-Wl,--exclude-libs,ALL" _HEYOKA_LINKER_SUPPORTS_EXCLUDE_LIBS)
    if(_HEYOKA_LINKER_SUPPORTS_EXCLUDE_LIBS)
        message(STATUS "Adding the linker flag \"-Wl,--exclude-libs,ALL\" to hide LLVM symbols.")
        target_link_options(heyoka PRIVATE "-Wl,--exclude-libs,ALL")
    else()
        check_linker_flag(CXX "-Wl,-unexported_symbol,dummy" _HEYOKA_LINKER_SUPPORTS_UNEXPORTED_SYMBOL)
        if(_HEYOKA_LINKER_SUPPORTS_UNEXPORTED_SYMBOL)
            message(STATUS "Adding the linker flag \"-Wl,-unexported_symbol\" to hide LLVM symbols.")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,__ZTVN4llvm*")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,__ZNK4llvm*")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,__ZN4llvm*")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,__ZTIN4llvm*")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,_LLVM*")
            target_link_options(heyoka PRIVATE "-Wl,-unexported_symbol,__ZTSN4llvm*")
        endif()
    endif()

    unset(CMAKE_REQUIRED_QUIET)
endif()

# Mandatory dependency on fmt.
set(_HEYOKA_MIN_SUPPORTED_FMT_VERSION 9)
set(_HEYOKA_MAX_SUPPORTED_FMT_VERSION 10)
find_package(fmt REQUIRED CONFIG)
if(${fmt_VERSION_MAJOR} VERSION_LESS ${_HEYOKA_MIN_SUPPORTED_FMT_VERSION} OR
   ${fmt_VERSION_MAJOR} VERSION_GREATER ${_HEYOKA_MAX_SUPPORTED_FMT_VERSION})
   message(FATAL_ERROR "The supported fmt versions are in the range [${_HEYOKA_MIN_SUPPORTED_FMT_VERSION}, ${_HEYOKA_MAX_SUPPORTED_FMT_VERSION}], but version ${fmt_VERSION_MAJOR} was found instead.")
endif()
message(STATUS "fmt version: ${fmt_VERSION}")
target_link_libraries(heyoka PUBLIC fmt::fmt)

# Mandatory dependency on spdlog.
find_package(spdlog REQUIRED CONFIG)
target_link_libraries(heyoka PRIVATE spdlog::spdlog)

# Mandatory dependency on TBB.
find_package(TBB REQUIRED CONFIG)
target_link_libraries(heyoka PRIVATE TBB::tbb)

# Mandatory dependency on Boost.
# NOTE: need 1.69 for safe numerics.
set(_HEYOKA_MIN_BOOST_VERSION "1.69")
find_package(Boost ${_HEYOKA_MIN_BOOST_VERSION} REQUIRED serialization)
target_link_libraries(heyoka PUBLIC Boost::boost Boost::serialization)
# NOTE: quench warnings from Boost when building the library.
target_compile_definitions(heyoka PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)

# NOTE: recent versions of libcxx remove std::unary_function/std::binary_function
# (which have been deprecated since long) in C++17 mode. This breaks, e.g., hashing
# functionality on older Boost versions. See:
# https://github.com/boostorg/container_hash/issues/22
# Note that the better approach here would be to detect libcxx rather than
# just Apple, but at this time it seems like there's no easy way to detect
# the stdlib implementation/version from CMake.
if(APPLE AND Boost_VERSION VERSION_LESS "1.81.0")
    # NOTE: make this a PUBLIC definition just to err on the side
    # of caution.
    target_compile_definitions(heyoka PUBLIC BOOST_NO_CXX98_FUNCTION_BASE)
endif()

# Optional dependency on mp++.
set(_HEYOKA_MPPP_VERSION "1.0.0")
if(HEYOKA_WITH_MPPP)
    find_package(mp++ ${_HEYOKA_MPPP_VERSION} REQUIRED CONFIG)
    message(STATUS "mp++ version: ${mp++_VERSION}")

    # NOTE: this is necessary for the serialisation of several classes.
    if(NOT mp++_WITH_BOOST_S11N)
        message(FATAL_ERROR "mp++ must be installed with support for Boost.serialization.")
    endif()

    # NOTE: needed for formatting numbers.
    if(NOT mp++_WITH_FMT)
        message(FATAL_ERROR "mp++ must be installed with support for fmt.")
    endif()

    target_link_libraries(heyoka PUBLIC mp++::mp++)

    # NOTE: _HEYOKA_WITH_REAL128 is used to signal the support
    # for mppp::real128 in the config-file package.
    set(_HEYOKA_WITH_REAL128 ${mp++_WITH_QUADMATH})
    # Similarly for mppp::real.
    set(_HEYOKA_WITH_REAL ${mp++_WITH_MPFR})
else()
    set(_HEYOKA_WITH_REAL128 NO)
    set(_HEYOKA_WITH_REAL NO)
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
# NOTE: since we use semantic versioning, the correct setting here is SameMajorVersion: it requires equality
# in the major version, but higher minor versions will be considered compatible. So, if heyoka 2.0.0 is requested
# and 2.1.0 is found, then all is good. However, the reverse will lead to a failure.
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/heyoka-config-version.cmake" COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/heyoka-config-version.cmake" DESTINATION "${HEYOKA_INSTALL_LIBDIR}/cmake/heyoka")

# Cleanup.
unset(_HEYOKA_MIN_BOOST_VERSION)
unset(_HEYOKA_MPPP_VERSION)
unset(_HEYOKA_WITH_REAL128)
unset(_HEYOKA_WITH_REAL)
unset(_HEYOKA_LLVM_STATIC_LINK)
unset(_HEYOKA_MIN_SUPPORTED_FMT_VERSION)
unset(_HEYOKA_MAX_SUPPORTED_FMT_VERSION)

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
