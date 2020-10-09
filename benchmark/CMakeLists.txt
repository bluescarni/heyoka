# NOTE: don't find a specific version as we already checked
# outside that the Boost version is appropriate.
find_package(Boost REQUIRED COMPONENTS program_options)
if(NOT TARGET Boost::program_options)
    message(STATUS "The 'Boost::program_options' imported target is missing, creating it.")
    add_library(Boost::program_options UNKNOWN IMPORTED)
    set_target_properties(Boost::program_options PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}"
    )
    set_target_properties(Boost::program_options PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${Boost_PROGRAM_OPTIONS_LIBRARY}"
    )
endif()

# Find the xtensor bits.
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

function(ADD_HEYOKA_BENCHMARK arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE heyoka Boost::program_options xtensor xtensor-blas)
  target_compile_definitions(${arg1} PRIVATE XTENSOR_USE_FLENS_BLAS)
  target_compile_options(${arg1} PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
  )
  # Let's setup the target C++ standard, but only if the user did not provide it manually.
  if(NOT CMAKE_CXX_STANDARD)
    set_property(TARGET ${arg1} PROPERTY CXX_STANDARD 17)
  endif()
  set_property(TARGET ${arg1} PROPERTY CXX_STANDARD_REQUIRED YES)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
endfunction()

ADD_HEYOKA_BENCHMARK(evaluate_dbl)
ADD_HEYOKA_BENCHMARK(genetics)
ADD_HEYOKA_BENCHMARK(flops_benchmark_dbl)
ADD_HEYOKA_BENCHMARK(taylor_jet_batch_benchmark)
ADD_HEYOKA_BENCHMARK(two_body_long_term)
ADD_HEYOKA_BENCHMARK(two_body_step)
ADD_HEYOKA_BENCHMARK(two_body_step_batch)
ADD_HEYOKA_BENCHMARK(taylor_ANN)
ADD_HEYOKA_BENCHMARK(apophis)
ADD_HEYOKA_BENCHMARK(stiff_equation)
ADD_HEYOKA_BENCHMARK(mascon_model)
ADD_HEYOKA_BENCHMARK(outer_ss_jet_benchmark)
ADD_HEYOKA_BENCHMARK(outer_ss_long_term)
ADD_HEYOKA_BENCHMARK(outer_ss_long_term_batch)
ADD_HEYOKA_BENCHMARK(n_body_creation)
