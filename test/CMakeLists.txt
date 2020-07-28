add_library(heyoka_test STATIC catch_main.cpp)
target_compile_options(heyoka_test PRIVATE
  "$<$<CONFIG:Debug>:${HEYOKA_CXX_FLAGS_DEBUG}>"
  "$<$<CONFIG:Release>:${HEYOKA_CXX_FLAGS_RELEASE}>"
  "$<$<CONFIG:RelWithDebInfo>:${HEYOKA_CXX_FLAGS_RELEASE}>"
  "$<$<CONFIG:MinSizeRel>:${HEYOKA_CXX_FLAGS_RELEASE}>"
)

# Let's setup the target C++ standard, but only if the user did not provide it manually.
if(NOT CMAKE_CXX_STANDARD)
  set_property(TARGET heyoka_test PROPERTY CXX_STANDARD 17)
endif()
set_property(TARGET heyoka_test PROPERTY CXX_STANDARD_REQUIRED YES)
set_property(TARGET heyoka_test PROPERTY CXX_EXTENSIONS NO)

function(ADD_HEYOKA_TESTCASE arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE heyoka_test heyoka)
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
  add_test(${arg1} ${arg1})
endfunction()

ADD_HEYOKA_TESTCASE(add_expression)
ADD_HEYOKA_TESTCASE(expression)
ADD_HEYOKA_TESTCASE(taylor_div)
ADD_HEYOKA_TESTCASE(taylor_log)
ADD_HEYOKA_TESTCASE(taylor_mul)
ADD_HEYOKA_TESTCASE(taylor_pow)
ADD_HEYOKA_TESTCASE(taylor_sincos)
ADD_HEYOKA_TESTCASE(taylor_const_sys)
ADD_HEYOKA_TESTCASE(taylor_no_decomp_sys)
ADD_HEYOKA_TESTCASE(two_body)
ADD_HEYOKA_TESTCASE(e3bp)
#ADD_HEYOKA_TESTCASE(gp)
