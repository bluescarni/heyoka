# Find the xtensor bits, if requested.
if(HEYOKA_WITH_XTENSOR)
  find_package(xtensor REQUIRED)
  find_package(xtensor-blas REQUIRED)
endif()

function(ADD_HEYOKA_BENCHMARK arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE heyoka)
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

ADD_HEYOKA_BENCHMARK(taylor_eqq_with_param)

if(HEYOKA_WITH_XTENSOR)
  function(ADD_HEYOKA_XTENSOR_BENCHMARK arg1)
    ADD_HEYOKA_BENCHMARK(${arg1})
    target_link_libraries(${arg1} PRIVATE xtensor xtensor-blas)
    target_compile_definitions(${arg1} PRIVATE HEYOKA_WITH_XTENSOR PRIVATE XTENSOR_USE_FLENS_BLAS)
  endfunction()

  ADD_HEYOKA_XTENSOR_BENCHMARK(outer_ss_long_term)
endif()
