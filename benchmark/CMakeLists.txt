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
find_package(xtensor REQUIRED CONFIG)
find_package(xtensor-blas REQUIRED CONFIG)

function(ADD_HEYOKA_BENCHMARK arg1)
  add_executable(${arg1} ${arg1}.cpp)
  # NOTE: fmt was already located in the main CMakeLists.
  target_link_libraries(${arg1} PRIVATE heyoka Boost::boost Boost::program_options xtensor xtensor-blas fmt::fmt spdlog::spdlog)
  target_compile_definitions(${arg1} PRIVATE XTENSOR_USE_FLENS_BLAS PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)
  target_compile_options(${arg1} PRIVATE
    "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
  )
  # Setup the C++ standard.
  target_compile_features(${arg1} PRIVATE cxx_std_17)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
endfunction()

# ADD_HEYOKA_BENCHMARK(evaluate_dbl)
# ADD_HEYOKA_BENCHMARK(genetics)
ADD_HEYOKA_BENCHMARK(taylor_jet_batch_benchmark)
ADD_HEYOKA_BENCHMARK(two_body_long_term)
ADD_HEYOKA_BENCHMARK(two_body_step)
ADD_HEYOKA_BENCHMARK(two_body_step_batch)
ADD_HEYOKA_BENCHMARK(taylor_ANN)
ADD_HEYOKA_BENCHMARK(taylor_ANN_2)
ADD_HEYOKA_BENCHMARK(apophis)
ADD_HEYOKA_BENCHMARK(stiff_equation)
ADD_HEYOKA_BENCHMARK(mascon_models)
ADD_HEYOKA_BENCHMARK(outer_ss_jet_benchmark)
ADD_HEYOKA_BENCHMARK(outer_ss_long_term)
ADD_HEYOKA_BENCHMARK(outer_ss_long_term_batch)
ADD_HEYOKA_BENCHMARK(n_body_creation)
ADD_HEYOKA_BENCHMARK(poly_coll)
ADD_HEYOKA_BENCHMARK(ss_maker)
ADD_HEYOKA_BENCHMARK(taylor_jl_01)
ADD_HEYOKA_BENCHMARK(event_allocations)
ADD_HEYOKA_BENCHMARK(event_overhead)
ADD_HEYOKA_BENCHMARK(ss_event_overhead)
ADD_HEYOKA_BENCHMARK(h_oscillator_lt)
ADD_HEYOKA_BENCHMARK(mb)
ADD_HEYOKA_BENCHMARK(vsop2013_elliptic)
ADD_HEYOKA_BENCHMARK(vsop2013_cartesian)
ADD_HEYOKA_BENCHMARK(pendulum_bench)
ADD_HEYOKA_BENCHMARK(pendulum_odeint)
ADD_HEYOKA_BENCHMARK(pcr3bp)
ADD_HEYOKA_BENCHMARK(pcr3bp_odeint)
ADD_HEYOKA_BENCHMARK(ss_3bp)
ADD_HEYOKA_BENCHMARK(ss_3bp_odeint)
ADD_HEYOKA_BENCHMARK(hh_event)
ADD_HEYOKA_BENCHMARK(burrau_3bp)
ADD_HEYOKA_BENCHMARK(bandf)
ADD_HEYOKA_BENCHMARK(bandf_odeint)
