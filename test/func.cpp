// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"

using namespace heyoka;

struct func_00 : func_base {
    func_00() : func_base("f", {}) {}
    explicit func_00(std::vector<expression> args) : func_base("f", std::move(args)) {}
};

struct func_01 {
};

TEST_CASE("func minimal")
{
    using Catch::Matchers::Message;

    func f(func_00{{"x"_var, "y"_var}});
    REQUIRE(f.get_type_index() == typeid(func_00));
    REQUIRE(f.get_display_name() == "f");
    REQUIRE(f.get_args() == std::vector{"x"_var, "y"_var});

    REQUIRE(!detail::func_has_codegen_dbl_v<func_00>);

    llvm_state s;
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {nullptr, nullptr}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {nullptr}), std::invalid_argument,
                           Message("Inconsistent number of arguments supplied to the double codegen for the function "
                                   "'f': 2 arguments were expected, but 1 arguments were provided instead"));

    REQUIRE(!std::is_constructible_v<func, func_01>);

    auto orig_ptr = f.get_ptr();
    REQUIRE(orig_ptr == static_cast<const func &>(f).get_ptr());

    auto f2(f);
    REQUIRE(orig_ptr != f2.get_ptr());
    REQUIRE(f2.get_type_index() == typeid(func_00));
    REQUIRE(f2.get_display_name() == "f");
    REQUIRE(f2.get_args() == std::vector{"x"_var, "y"_var});

    auto f3(std::move(f));
    REQUIRE(orig_ptr == f3.get_ptr());

    f = f3;
    REQUIRE(f.get_ptr() != f3.get_ptr());

    f = std::move(f3);
    REQUIRE(f.get_ptr() == orig_ptr);
}

struct func_02 : func_base {
    func_02() : func_base("f", {}) {}
    explicit func_02(std::vector<expression> args) : func_base("f", std::move(args)) {}

    llvm::Value *codegen_dbl(llvm_state &, const std::vector<llvm::Value *> &) const
    {
        return nullptr;
    }
};

struct func_03 : func_base {
    func_03() : func_base("f", {}) {}
    explicit func_03(std::vector<expression> args) : func_base("f", std::move(args)) {}

    llvm::Value *codegen_ldbl(llvm_state &, const std::vector<llvm::Value *> &) const
    {
        return nullptr;
    }
};

#if defined(HEYOKA_HAVE_REAL128)

struct func_04 : func_base {
    func_04() : func_base("f", {}) {}
    explicit func_04(std::vector<expression> args) : func_base("f", std::move(args)) {}

    llvm::Value *codegen_f128(llvm_state &, const std::vector<llvm::Value *> &) const
    {
        return nullptr;
    }
};

#endif

TEST_CASE("func codegen")
{
    using Catch::Matchers::Message;

    func f(func_02{{}});

    llvm_state s;
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {}), std::invalid_argument,
                           Message("The double codegen for the function 'f' returned a null pointer"));
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), not_implemented_error,
                           Message("long double codegen is not implemented for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), not_implemented_error,
                           Message("real128 codegen is not implemented for the function 'f'"));
#endif

    f = func(func_03{{}});
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), std::invalid_argument,
                           Message("The long double codegen for the function 'f' returned a null pointer"));
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), not_implemented_error,
                           Message("real128 codegen is not implemented for the function 'f'"));
#endif

#if defined(HEYOKA_HAVE_REAL128)
    f = func(func_04{{}});
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), std::invalid_argument,
                           Message("The real128 codegen for the function 'f' returned a null pointer"));
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), not_implemented_error,
                           Message("long double codegen is not implemented for the function 'f'"));
#endif
}
