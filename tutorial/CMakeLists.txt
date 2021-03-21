# Find the xtensor bits.
find_package(xtensor REQUIRED CONFIG)

function(ADD_HEYOKA_TUTORIAL arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE heyoka xtensor)
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

ADD_HEYOKA_TUTORIAL(pendulum)
ADD_HEYOKA_TUTORIAL(ex_system)
ADD_HEYOKA_TUTORIAL(adaptive_basic)
ADD_HEYOKA_TUTORIAL(adaptive_opt)
ADD_HEYOKA_TUTORIAL(pendulum_param)
ADD_HEYOKA_TUTORIAL(forced_damped_pendulum)
ADD_HEYOKA_TUTORIAL(d_output)
ADD_HEYOKA_TUTORIAL(batch_mode)
ADD_HEYOKA_TUTORIAL(event_basic)
