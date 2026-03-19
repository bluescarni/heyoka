include(CheckCXXCompilerFlag)

# Helper macro: check a list of compiler flag candidates for a given ISA level.
#
# The first candidate that succeeds is used. Each candidate gets a unique cache variable
# (via string(MAKE_C_IDENTIFIER)) to avoid the CMake cache trap where a failed check for
# one candidate (e.g., -msse2 on MSVC) would prevent subsequent candidates (e.g., /arch:SSE2)
# from being tested.
macro(_heyoka_check_simd_flag _isa _suffixes _flags)
    set(_${_isa}_found FALSE)
    foreach(_candidate ${ARGN})
        string(MAKE_C_IDENTIFIER "${_candidate}" _candidate_id)
        check_cxx_compiler_flag("${_candidate}" _HEYOKA_HAS_${_isa}_${_candidate_id})
        if(_HEYOKA_HAS_${_isa}_${_candidate_id})
            list(APPEND ${_suffixes} "${_isa}")
            list(APPEND ${_flags} "${_candidate}")
            set(_${_isa}_found TRUE)
            break()
        endif()
    endforeach()
endmacro()

function(heyoka_get_simd_configs OUT_SUFFIXES OUT_FLAGS)
    set(_suffixes "")
    set(_flags "")

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)

    # x86.
    if(_proc MATCHES "x86|amd64")
        _heyoka_check_simd_flag(sse2 _suffixes _flags "-msse2" "/arch:SSE2")
        _heyoka_check_simd_flag(avx _suffixes _flags "-mavx" "/arch:AVX")
        _heyoka_check_simd_flag(avx2 _suffixes _flags "-mavx2" "/arch:AVX2")
        _heyoka_check_simd_flag(avx512 _suffixes _flags "-mavx512f" "/arch:AVX512")

        # Sanity checks: higher ISA levels imply lower ones.
        if(_avx_found AND NOT _sse2_found)
            message(FATAL_ERROR "AVX detected but SSE2 is not available - this should not happen.")
        endif()
        if(_avx2_found AND NOT _avx_found)
            message(FATAL_ERROR "AVX2 detected but AVX is not available - this should not happen.")
        endif()
        if(_avx512_found AND NOT _avx2_found)
            message(FATAL_ERROR "AVX-512 detected but AVX2 is not available - this should not happen.")
        endif()
    # AArch64: AdvSIMD is always available as baseline.
    elseif(_proc MATCHES "aarch64|arm64")
        list(APPEND _suffixes "advsimd")
        list(APPEND _flags "")
    # PPC64: VSX is baseline on POWER7+.
    elseif(_proc MATCHES "ppc64")
        list(APPEND _suffixes "vsx")
        list(APPEND _flags "")
    endif()

    set(${OUT_SUFFIXES} "${_suffixes}" PARENT_SCOPE)
    set(${OUT_FLAGS} "${_flags}" PARENT_SCOPE)

    message(STATUS "List of SIMD suffixes: ${_suffixes}")
    message(STATUS "List of SIMD compiler flags: ${_flags}")
endfunction()
