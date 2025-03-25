# NOTE: don't find a specific version as we already checked
# outside that the Boost version is appropriate.
find_package(Boost QUIET COMPONENTS program_options CONFIG)
if(NOT ${Boost_FOUND})
    message(STATUS "Boost not found in CONFIG mode, retrying in MODULE mode.")
    find_package(Boost QUIET MODULE COMPONENTS program_options)
endif()
if(NOT ${Boost_FOUND})
    message(FATAL_ERROR "Could not locate Boost in either CONFIG or MODULE mode.")
endif()

function(ADD_HEYOKA_UTIL arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE heyoka Boost::boost Boost::program_options fmt::fmt)
  target_compile_options(${arg1} PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
  )
  # Setup the C++ standard.
  target_compile_features(${arg1} PRIVATE cxx_std_20)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
  add_test(${arg1} ${arg1})
endfunction()

ADD_HEYOKA_UTIL(eop_data_updater)
