include(CheckCXXCompilerFlag)

function(heyoka_get_simd_configs OUT_SUFFIXES OUT_FLAGS)
    set(_suffixes "")
    set(_flags "")

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _proc)

    # x86.
    if(_proc MATCHES "x86|amd64")
        # SSE2.
        foreach(_candidate "-msse2" "/arch:SSE2")
            check_cxx_compiler_flag("${_candidate}" _HEYOKA_HAS_SSE2)
            if(_HEYOKA_HAS_SSE2)
                list(APPEND _suffixes "sse2")
                list(APPEND _flags "${_candidate}")
                break()
            endif()
        endforeach()

        # AVX.
        foreach(_candidate "-mavx" "/arch:AVX")
            check_cxx_compiler_flag("${_candidate}" _HEYOKA_HAS_AVX)
            if(_HEYOKA_HAS_AVX)
                list(APPEND _suffixes "avx")
                list(APPEND _flags "${_candidate}")
                break()
            endif()
        endforeach()

        # AVX2.
        foreach(_candidate "-mavx2" "/arch:AVX2")
            check_cxx_compiler_flag("${_candidate}" _HEYOKA_HAS_AVX2)
            if(_HEYOKA_HAS_AVX2)
                list(APPEND _suffixes "avx2")
                list(APPEND _flags "${_candidate}")
                break()
            endif()
        endforeach()

        # AVX-512.
        foreach(_candidate "-mavx512f" "/arch:AVX512")
            check_cxx_compiler_flag("${_candidate}" _HEYOKA_HAS_AVX512)
            if(_HEYOKA_HAS_AVX512)
                list(APPEND _suffixes "avx512")
                list(APPEND _flags "${_candidate}")
                break()
            endif()
        endforeach()

        # Sanity checks: higher ISA levels imply lower ones.
        if(_HEYOKA_HAS_AVX AND NOT _HEYOKA_HAS_SSE2)
            message(FATAL_ERROR "AVX detected but SSE2 is not available - this should not happen.")
        endif()
        if(_HEYOKA_HAS_AVX2 AND NOT _HEYOKA_HAS_AVX)
            message(FATAL_ERROR "AVX2 detected but AVX is not available - this should not happen.")
        endif()
        if(_HEYOKA_HAS_AVX512 AND NOT _HEYOKA_HAS_AVX2)
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
