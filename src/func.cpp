// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>

namespace heyoka
{

func_base::func_base(std::string display_name, std::vector<expression> args)
    : m_display_name(std::move(display_name)), m_args(std::make_unique<std::vector<expression>>(std::move(args)))
{
    if (m_display_name.empty()) {
        throw std::invalid_argument("Cannot create a function with no display name");
    }
}

func_base::func_base(const func_base &other)
    : m_display_name(other.m_display_name), m_args(std::make_unique<std::vector<expression>>(other.get_args()))
{
}

func_base::func_base(func_base &&) noexcept = default;

func_base &func_base::operator=(const func_base &other)
{
    if (this != &other) {
        *this = func_base(other);
    }

    return *this;
}

func_base &func_base::operator=(func_base &&) noexcept = default;

func_base::~func_base() = default;

const std::string &func_base::get_display_name() const
{
    return m_display_name;
}

const std::vector<expression> &func_base::get_args() const
{
    assert(m_args);

    return *m_args;
}

namespace detail
{

func_inner_base::~func_inner_base() = default;

}

func::func(const func &f) : m_ptr(f.ptr()->clone()) {}

func::func(func &&) noexcept = default;

func &func::operator=(const func &f)
{
    if (this != &f) {
        *this = func(f);
    }

    return *this;
}

func &func::operator=(func &&) noexcept = default;

func::~func() = default;

// Just two small helpers to make sure that whenever we require
// access to the pointer it actually points to something.
const detail::func_inner_base *func::ptr() const
{
    assert(m_ptr.get() != nullptr);
    return m_ptr.get();
}

detail::func_inner_base *func::ptr()
{
    assert(m_ptr.get() != nullptr);
    return m_ptr.get();
}

std::type_index func::get_type_index() const
{
    return ptr()->get_type_index();
}

const void *func::get_ptr() const
{
    return ptr()->get_ptr();
}

void *func::get_ptr()
{
    return ptr()->get_ptr();
}

const std::string &func::get_display_name() const
{
    return ptr()->get_display_name();
}

const std::vector<expression> &func::get_args() const
{
    return ptr()->get_args();
}

llvm::Value *func::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != get_args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_display_name(), get_args().size(), v.size()));
    }

    auto ret = ptr()->codegen_dbl(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The double codegen for the function '{}' returned a null pointer"_format(get_display_name()));
    }

    return ret;
}

llvm::Value *func::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != get_args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the long double codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_display_name(), get_args().size(), v.size()));
    }

    auto ret = ptr()->codegen_ldbl(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The long double codegen for the function '{}' returned a null pointer"_format(get_display_name()));
    }

    return ret;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *func::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != get_args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the real128 codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_display_name(), get_args().size(), v.size()));
    }

    auto ret = ptr()->codegen_f128(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The real128 codegen for the function '{}' returned a null pointer"_format(get_display_name()));
    }

    return ret;
}

#endif

expression func::diff(const std::string &s) const
{
    return ptr()->diff(s);
}

double func::eval_dbl(const std::unordered_map<std::string, double> &m) const
{
    return ptr()->eval_dbl(m);
}

void func::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &m) const
{
    ptr()->eval_batch_dbl(out, m);
}

double func::eval_num_dbl(const std::vector<double> &v) const
{
    if (v.size() != get_args().size()) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double numerical evaluation of the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_display_name(), get_args().size(), v.size()));
    }

    return ptr()->eval_num_dbl(v);
}

} // namespace heyoka
