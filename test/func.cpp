// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
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
    func_00(const std::string &name) : func_base(name, {}) {}
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
    REQUIRE(f.args() == std::vector{"x"_var, "y"_var});

    REQUIRE_THROWS_MATCHES(func{func_00{""}}, std::invalid_argument,
                           Message("Cannot create a function with no display name"));

    llvm_state s;
    auto fake_val = reinterpret_cast<llvm::Value *>(&s);
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {fake_val, fake_val}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.codegen_dbl(s, {nullptr, nullptr}), std::invalid_argument,
        Message("Null pointer detected in the array of values passed to func::codegen_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.codegen_ldbl(s, {nullptr, nullptr}), std::invalid_argument,
        Message("Null pointer detected in the array of values passed to func::codegen_ldbl() for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(
        f.codegen_f128(s, {nullptr, nullptr}), std::invalid_argument,
        Message("Null pointer detected in the array of values passed to func::codegen_f128() for the function 'f'"));
#endif
    REQUIRE_THROWS_MATCHES(f.diff(""), not_implemented_error,
                           Message("The derivative is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.eval_dbl({{}}, {}), not_implemented_error,
                           Message("double eval is not implemented for the function 'f'"));
    std::vector<double> tmp;
    REQUIRE_THROWS_MATCHES(f.eval_batch_dbl(tmp, {{}}, {}), not_implemented_error,
                           Message("double batch eval is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.eval_num_dbl({1., 1.}), not_implemented_error,
                           Message("double numerical eval is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.eval_num_dbl({}), std::invalid_argument,
        Message("Inconsistent number of arguments supplied to the double numerical evaluation of the function 'f': 2 "
                "arguments were expected, but 0 arguments were provided instead"));
    REQUIRE_THROWS_MATCHES(f.deval_num_dbl({1., 1.}, 0), not_implemented_error,
                           Message("double numerical eval of the derivative is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.deval_num_dbl({1.}, 0), std::invalid_argument,
                           Message("Inconsistent number of arguments supplied to the double numerical evaluation of "
                                   "the derivative of function 'f': 2 "
                                   "arguments were expected, but 1 arguments were provided instead"));
    REQUIRE_THROWS_MATCHES(f.deval_num_dbl({1., 1.}, 2), std::invalid_argument,
                           Message("Invalid index supplied to the double numerical evaluation of the derivative of "
                                   "function 'f': index 2 was supplied, but the number of arguments is only 2"));

    REQUIRE(!std::is_constructible_v<func, func_01>);

    auto orig_ptr = f.get_ptr();
    REQUIRE(orig_ptr == static_cast<const func &>(f).get_ptr());

    auto f2(f);
    REQUIRE(orig_ptr != f2.get_ptr());
    REQUIRE(f2.get_type_index() == typeid(func_00));
    REQUIRE(f2.get_display_name() == "f");
    REQUIRE(f2.args() == std::vector{"x"_var, "y"_var});

    auto f3(std::move(f));
    REQUIRE(orig_ptr == f3.get_ptr());

    f = f3;
    REQUIRE(f.get_ptr() != f3.get_ptr());

    f = std::move(f3);
    REQUIRE(f.get_ptr() == orig_ptr);

    f = func{func_00{{"x"_var, "y"_var}}};
    auto [b, e] = f.get_mutable_args_it();
    *b = "z"_var;
    REQUIRE(f.args() == std::vector{"z"_var, "y"_var});

    auto a = 0;
    auto fake_ptr = reinterpret_cast<llvm::Value *>(&a);
    REQUIRE_THROWS_MATCHES(f.taylor_diff_dbl(s, {nullptr, nullptr}, nullptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Null par_ptr detected in func::taylor_diff_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_dbl(s, {nullptr, nullptr}, fake_ptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_diff_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_diff_dbl(s, {nullptr, nullptr}, fake_ptr, 0, 2, 2, 1), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_diff_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_dbl(s, {nullptr, nullptr}, fake_ptr, 2, 1, 2, 1), not_implemented_error,
                           Message("double Taylor diff is not implemented for the function 'f'"));

    REQUIRE_THROWS_MATCHES(f.taylor_diff_ldbl(s, {nullptr, nullptr}, nullptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Null par_ptr detected in func::taylor_diff_ldbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_ldbl(s, {nullptr, nullptr}, fake_ptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_diff_ldbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_diff_ldbl(s, {nullptr, nullptr}, fake_ptr, 0, 2, 2, 1), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_diff_ldbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_ldbl(s, {nullptr, nullptr}, fake_ptr, 2, 1, 2, 1), not_implemented_error,
                           Message("long double Taylor diff is not implemented for the function 'f'"));

#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.taylor_diff_f128(s, {nullptr, nullptr}, nullptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Null par_ptr detected in func::taylor_diff_f128() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_f128(s, {nullptr, nullptr}, fake_ptr, 2, 2, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_diff_f128() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_diff_f128(s, {nullptr, nullptr}, fake_ptr, 0, 2, 2, 1), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_diff_f128() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_f128(s, {nullptr, nullptr}, fake_ptr, 2, 1, 2, 1), not_implemented_error,
                           Message("float128 Taylor diff is not implemented for the function 'f'"));
#endif

    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_dbl(s, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_c_diff_func_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_dbl(s, 0, 2), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_c_diff_func_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_dbl(s, 2, 1), not_implemented_error,
                           Message("double Taylor diff in compact mode is not implemented for the function 'f'"));

    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_ldbl(s, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_c_diff_func_ldbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_ldbl(s, 0, 2), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_c_diff_func_ldbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_ldbl(s, 2, 1), not_implemented_error,
                           Message("long double Taylor diff in compact mode is not implemented for the function 'f'"));

#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_f128(s, 2, 0), std::invalid_argument,
                           Message("Zero batch size detected in func::taylor_c_diff_func_f128() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_f128(s, 0, 2), std::invalid_argument,
        Message("Zero number of u variables detected in func::taylor_c_diff_func_f128() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_c_diff_func_f128(s, 2, 1), not_implemented_error,
                           Message("float128 Taylor diff in compact mode is not implemented for the function 'f'"));
#endif

    std::vector<expression> empty;
    f = func{func_00{{"x"_var, "y"_var}}};
    std::move(f).taylor_decompose(empty);
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
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {nullptr}), std::invalid_argument,
                           Message("Inconsistent number of arguments supplied to the double codegen for the function "
                                   "'f': 0 arguments were expected, but 1 arguments were provided instead"));
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), not_implemented_error,
                           Message("long double codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.codegen_ldbl(s, {nullptr}), std::invalid_argument,
        Message("Inconsistent number of arguments supplied to the long double codegen for the function "
                "'f': 0 arguments were expected, but 1 arguments were provided instead"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), not_implemented_error,
                           Message("float128 codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {nullptr}), std::invalid_argument,
                           Message("Inconsistent number of arguments supplied to the float128 codegen for the function "
                                   "'f': 0 arguments were expected, but 1 arguments were provided instead"));
#endif

    f = func(func_03{{}});
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), std::invalid_argument,
                           Message("The long double codegen for the function 'f' returned a null pointer"));
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), not_implemented_error,
                           Message("float128 codegen is not implemented for the function 'f'"));
#endif

#if defined(HEYOKA_HAVE_REAL128)
    f = func(func_04{{}});
    REQUIRE_THROWS_MATCHES(f.codegen_f128(s, {}), std::invalid_argument,
                           Message("The float128 codegen for the function 'f' returned a null pointer"));
    REQUIRE_THROWS_MATCHES(f.codegen_dbl(s, {}), not_implemented_error,
                           Message("double codegen is not implemented for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.codegen_ldbl(s, {}), not_implemented_error,
                           Message("long double codegen is not implemented for the function 'f'"));
#endif
}

struct func_05 : func_base {
    func_05() : func_base("f", {}) {}
    explicit func_05(std::vector<expression> args) : func_base("f", std::move(args)) {}

    expression diff(const std::string &) const
    {
        return 42_dbl;
    }
};

TEST_CASE("func diff")
{
    auto f = func(func_05{});

    REQUIRE(f.diff("x") == 42_dbl);
}

struct func_06 : func_base {
    func_06() : func_base("f", {}) {}
    explicit func_06(std::vector<expression> args) : func_base("f", std::move(args)) {}

    double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const
    {
        return 42;
    }
};

TEST_CASE("func eval_dbl")
{
    auto f = func(func_06{});

    REQUIRE(f.eval_dbl({{}}, {}) == 42);
}

struct func_07 : func_base {
    func_07() : func_base("f", {}) {}
    explicit func_07(std::vector<expression> args) : func_base("f", std::move(args)) {}

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const
    {
    }
};

TEST_CASE("func eval_batch_dbl")
{
    auto f = func(func_07{});

    std::vector<double> tmp;
    REQUIRE_NOTHROW(f.eval_batch_dbl(tmp, {{}}, {}));
}

struct func_08 : func_base {
    func_08() : func_base("f", {}) {}
    explicit func_08(std::vector<expression> args) : func_base("f", std::move(args)) {}

    double eval_num_dbl(const std::vector<double> &) const
    {
        return 42;
    }
};

TEST_CASE("func eval_num_dbl")
{
    auto f = func(func_08{{"x"_var}});

    REQUIRE(f.eval_num_dbl({1.}) == 42);
}

struct func_09 : func_base {
    func_09() : func_base("f", {}) {}
    explicit func_09(std::vector<expression> args) : func_base("f", std::move(args)) {}

    double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const
    {
        return 43;
    }
};

TEST_CASE("func deval_num_dbl")
{
    auto f = func(func_09{{"x"_var}});

    REQUIRE(f.deval_num_dbl({1.}, 0) == 43);
}

struct func_10 : func_base {
    func_10() : func_base("f", {}) {}
    explicit func_10(std::vector<expression> args) : func_base("f", std::move(args)) {}

    std::vector<expression>::size_type taylor_decompose(std::vector<expression> &u_vars_defs) &&
    {
        u_vars_defs.emplace_back("foo");

        return u_vars_defs.size() - 1u;
    }
};

struct func_10a : func_base {
    func_10a() : func_base("f", {}) {}
    explicit func_10a(std::vector<expression> args) : func_base("f", std::move(args)) {}

    std::vector<expression>::size_type taylor_decompose(std::vector<expression> &u_vars_defs) &&
    {
        u_vars_defs.emplace_back("foo");

        return u_vars_defs.size();
    }
};

TEST_CASE("func taylor_decompose")
{
    using Catch::Matchers::Message;

    auto f = func(func_10{{"x"_var}});

    std::vector<expression> u_vars_defs;
    REQUIRE(std::move(f).taylor_decompose(u_vars_defs) == 0u);
    REQUIRE(u_vars_defs == std::vector{"foo"_var});

    f = func(func_10a{{"x"_var}});

    REQUIRE_THROWS_MATCHES(
        std::move(f).taylor_decompose(u_vars_defs), std::invalid_argument,
        Message("Invalid value returned by the Taylor decomposition function for the function 'f': "
                "the return value is 2, which is not less than the current size of the decomposition "
                "(2)"));
}

struct func_12 : func_base {
    func_12() : func_base("f", {}) {}
    explicit func_12(std::vector<expression> args) : func_base("f", std::move(args)) {}

    llvm::Value *taylor_diff_dbl(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                 std::uint32_t, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
    llvm::Value *taylor_diff_ldbl(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Value *taylor_diff_f128(llvm_state &, const std::vector<llvm::Value *> &, llvm::Value *, std::uint32_t,
                                  std::uint32_t, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
#endif
};

TEST_CASE("func taylor diff")
{
    using Catch::Matchers::Message;

    auto f = func(func_12{});

    llvm_state s;
    auto a = 0;
    auto fake_ptr = reinterpret_cast<llvm::Value *>(&a);
    REQUIRE_THROWS_MATCHES(f.taylor_diff_dbl(s, {}, fake_ptr, 1, 2, 3, 4), std::invalid_argument,
                           Message("Null return value detected in func::taylor_diff_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(f.taylor_diff_ldbl(s, {}, fake_ptr, 1, 2, 3, 4), std::invalid_argument,
                           Message("Null return value detected in func::taylor_diff_ldbl() for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(f.taylor_diff_f128(s, {}, fake_ptr, 1, 2, 3, 4), std::invalid_argument,
                           Message("Null return value detected in func::taylor_diff_f128() for the function 'f'"));
#endif
}

struct func_13 : func_base {
    func_13() : func_base("f", {}) {}
    explicit func_13(std::vector<expression> args) : func_base("f", std::move(args)) {}

    llvm::Function *taylor_c_diff_func_dbl(llvm_state &, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
    llvm::Function *taylor_c_diff_func_ldbl(llvm_state &, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
#if defined(HEYOKA_HAVE_REAL128)
    llvm::Function *taylor_c_diff_func_f128(llvm_state &, std::uint32_t, std::uint32_t) const
    {
        return nullptr;
    }
#endif
};

TEST_CASE("func taylor c_diff")
{
    using Catch::Matchers::Message;

    auto f = func(func_13{});

    llvm_state s;
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_dbl(s, 3, 4), std::invalid_argument,
        Message("Null return value detected in func::taylor_c_diff_func_dbl() for the function 'f'"));
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_ldbl(s, 2, 3), std::invalid_argument,
        Message("Null return value detected in func::taylor_c_diff_func_ldbl() for the function 'f'"));
#if defined(HEYOKA_HAVE_REAL128)
    REQUIRE_THROWS_MATCHES(
        f.taylor_c_diff_func_f128(s, 2, 4), std::invalid_argument,
        Message("Null return value detected in func::taylor_c_diff_func_f128() for the function 'f'"));
#endif
}

TEST_CASE("func swap")
{
    using std::swap;

    auto f1 = func(func_10{{"x"_var}});
    auto f2 = func(func_12{{"y"_var}});

    swap(f1, f2);

    REQUIRE(f1.get_type_index() == typeid(func_12));
    REQUIRE(f2.get_type_index() == typeid(func_10));
    REQUIRE(f1.args() == std::vector{"y"_var});
    REQUIRE(f2.args() == std::vector{"x"_var});

    REQUIRE(std::is_nothrow_swappable_v<func>);
}

TEST_CASE("func ostream")
{
    auto f1 = func(func_10{{"x"_var, "y"_var}});

    std::ostringstream oss;
    oss << f1;

    REQUIRE(oss.str() == "f(x, y)");

    oss.str("");

    f1 = func(func_10{{"y"_var}});

    oss << f1;

    REQUIRE(oss.str() == "f(y)");
}

TEST_CASE("func hash")
{
    auto f1 = func(func_10{{"x"_var, "y"_var}});

    REQUIRE_NOTHROW(hash(f1));

    std::cout << "Hash value for f1: " << hash(f1) << '\n';
}

struct func_14 : func_base {
    func_14(std::string name = "pippo", std::vector<expression> args = {}) : func_base(std::move(name), std::move(args))
    {
    }
    explicit func_14(std::vector<expression> args) : func_base("f", std::move(args)) {}
};

TEST_CASE("func eq ineq")
{
    auto f1 = func(func_10{{"x"_var, "y"_var}});

    REQUIRE(f1 == f1);
    REQUIRE(!(f1 != f1));
    REQUIRE(hash(f1) == hash(f1));

    // Differing arguments.
    auto f2 = func(func_10{{"y"_var, "x"_var}});

    REQUIRE(f1 != f2);
    REQUIRE(!(f1 == f2));

    auto f3 = func(func_14{{"x"_var, "y"_var}});
    auto f4 = func(func_14{"g", {"x"_var, "y"_var}});

    // Differing names.
    REQUIRE(f3 != f4);
    REQUIRE(!(f3 == f4));

    // Differing underlying types.
    f3 = func(func_10{{"x"_var, "y"_var}});
    f4 = func(func_14{{"x"_var, "y"_var}});

    REQUIRE(f3 != f4);
    REQUIRE(!(f3 == f4));
}

TEST_CASE("func get_variables")
{
    auto f1 = func(func_10{{}});
    REQUIRE(get_variables(f1).empty());

    f1 = func(func_10{{0_dbl}});
    REQUIRE(get_variables(f1).empty());

    f1 = func(func_10{{0_dbl, "x"_var}});
    REQUIRE(get_variables(f1) == std::vector<std::string>{"x"});

    f1 = func(func_10{{0_dbl, "y"_var, "x"_var}});
    REQUIRE(get_variables(f1) == std::vector<std::string>{"x", "y"});
    f1 = func(func_10{{0_dbl, "y"_var, "x"_var, 1_dbl, "x"_var, "y"_var, "z"_var}});
    REQUIRE(get_variables(f1) == std::vector<std::string>{"x", "y", "z"});
}

TEST_CASE("func rename_variables")
{
    auto f1 = func(func_10{{}});
    auto f2 = f1;
    rename_variables(f1, {{}});
    REQUIRE(f2 == f1);

    f1 = func(func_10{{0_dbl, "x"_var}});
    f2 = f1;
    rename_variables(f1, {{}});
    REQUIRE(f2 == f1);

    rename_variables(f1, {{"x", "y"}});
    REQUIRE(f2 != f1);
    REQUIRE(get_variables(f1) == std::vector<std::string>{"y"});
    rename_variables(f1, {{"x", "y"}});
    REQUIRE(get_variables(f1) == std::vector<std::string>{"y"});

    f1 = func(func_10{{"x"_var, 0_dbl, "z"_var, "y"_var}});
    rename_variables(f1, {{"x", "y"}});
    REQUIRE(f2 != f1);
    REQUIRE(get_variables(f1) == std::vector<std::string>{"y", "z"});
}

TEST_CASE("func diff free func")
{
    using Catch::Matchers::Message;

    auto f1 = func(func_05{{}});

    REQUIRE(diff(f1, "x") == 42_dbl);

    f1 = func(func_00{});
    REQUIRE_THROWS_MATCHES(diff(f1, ""), not_implemented_error,
                           Message("The derivative is not implemented for the function 'f'"));
}

struct func_15 : func_base {
    func_15(std::string name = "pippo", std::vector<expression> args = {}) : func_base(std::move(name), std::move(args))
    {
    }
    explicit func_15(std::vector<expression> args) : func_base("f", std::move(args)) {}
};

TEST_CASE("func subs")
{
    auto f1 = func(func_15{{"x"_var, "y"_var}});

    auto f2 = subs(expression{f1}, {{}});
    REQUIRE(f2 == expression{f1});

    f2 = subs(expression{f1}, {{"x", "z"_var}});
    REQUIRE(f2 == expression{func(func_15{{"z"_var, "y"_var}})});

    f2 = subs(expression{f1}, {{"x", "z"_var}, {"y", 42_dbl}});
    REQUIRE(f2 == expression{func(func_15{{"z"_var, 42_dbl}})});
}
