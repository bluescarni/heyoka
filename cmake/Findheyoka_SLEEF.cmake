if(HEYOKA_SLEEF_INCLUDE_DIR AND HEYOKA_SLEEF_LIBRARY)
    # Already in cache, be silent
    set(heyoka_SLEEF_FIND_QUIETLY TRUE)
endif()

find_path(HEYOKA_SLEEF_INCLUDE_DIR NAMES sleef.h)
find_library(HEYOKA_SLEEF_LIBRARY NAMES sleef)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(heyoka_SLEEF DEFAULT_MSG HEYOKA_SLEEF_INCLUDE_DIR HEYOKA_SLEEF_LIBRARY)

mark_as_advanced(HEYOKA_SLEEF_INCLUDE_DIR HEYOKA_SLEEF_LIBRARY)

# NOTE: this has been adapted from CMake's FindPNG.cmake.
if(heyoka_SLEEF_FOUND AND NOT TARGET heyoka::SLEEF)
    add_library(heyoka::SLEEF UNKNOWN IMPORTED)
    set_target_properties(heyoka::SLEEF PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HEYOKA_SLEEF_INCLUDE_DIR}"
        IMPORTED_LINK_INTERFACE_LANGUAGES "C" IMPORTED_LOCATION "${HEYOKA_SLEEF_LIBRARY}")
endif()
