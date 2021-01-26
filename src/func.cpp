// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container_hash/hash.hpp>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

func_base::func_base(std::string name, std::vector<expression> args) : m_name(std::move(name)), m_args(std::move(args))
{
    if (m_name.empty()) {
        throw std::invalid_argument("Cannot create a function with no name");
    }
}

func_base::func_base(const func_base &) = default;

func_base::func_base(func_base &&) noexcept = default;

func_base &func_base::operator=(const func_base &) = default;

func_base &func_base::operator=(func_base &&) noexcept = default;

func_base::~func_base() = default;

const std::string &func_base::get_name() const
{
    return m_name;
}

const std::vector<expression> &func_base::args() const
{
    return m_args;
}

std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> func_base::get_mutable_args_it()
{
    return {m_args.begin(), m_args.end()};
}

namespace detail
{

namespace
{

// Helper to check if a vector of llvm values contains
// a nullptr.
bool llvm_valvec_has_null(const std::vector<llvm::Value *> &v)
{
    return std::any_of(v.begin(), v.end(), [](llvm::Value *p) { return p == nullptr; });
}

} // namespace

// Default implementation of Taylor decomposition for a function.
// NOTE: this is a generalisation of the implementation
// for the binary operators.
void func_default_td_impl(func_base &fb, std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs)
{
    for (auto r = fb.get_mutable_args_it(); r.first != r.second; ++r.first) {
        if (const auto dres = taylor_decompose_in_place(std::move(*r.first), u_vars_defs)) {
            *r.first = expression{variable{"u_" + li_to_string(dres)}};
        }
    }
}

func_inner_base::~func_inner_base() = default;

} // namespace detail

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

const std::string &func::get_name() const
{
    return ptr()->get_name();
}

const std::vector<expression> &func::args() const
{
    return ptr()->args();
}

std::pair<std::vector<expression>::iterator, std::vector<expression>::iterator> func::get_mutable_args_it()
{
    return ptr()->get_mutable_args_it();
}

llvm::Value *func::codegen_dbl(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    if (detail::llvm_valvec_has_null(v)) {
        throw std::invalid_argument(
            "Null pointer detected in the array of values passed to func::codegen_dbl() for the function '{}'"_format(
                get_name()));
    }

    auto ret = ptr()->codegen_dbl(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The double codegen for the function '{}' returned a null pointer"_format(get_name()));
    }

    return ret;
}

llvm::Value *func::codegen_ldbl(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the long double codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    if (detail::llvm_valvec_has_null(v)) {
        throw std::invalid_argument(
            "Null pointer detected in the array of values passed to func::codegen_ldbl() for the function '{}'"_format(
                get_name()));
    }

    auto ret = ptr()->codegen_ldbl(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The long double codegen for the function '{}' returned a null pointer"_format(get_name()));
    }

    return ret;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *func::codegen_f128(llvm_state &s, const std::vector<llvm::Value *> &v) const
{
    using namespace fmt::literals;

    if (v.size() != args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the float128 codegen for the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    if (detail::llvm_valvec_has_null(v)) {
        throw std::invalid_argument(
            "Null pointer detected in the array of values passed to func::codegen_f128() for the function '{}'"_format(
                get_name()));
    }

    auto ret = ptr()->codegen_f128(s, v);

    if (ret == nullptr) {
        throw std::invalid_argument(
            "The float128 codegen for the function '{}' returned a null pointer"_format(get_name()));
    }

    return ret;
}

#endif

expression func::diff(const std::string &s) const
{
    return ptr()->diff(s);
}

double func::eval_dbl(const std::unordered_map<std::string, double> &m, const std::vector<double> &pars) const
{
    return ptr()->eval_dbl(m, pars);
}

void func::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &m,
                          const std::vector<double> &pars) const
{
    ptr()->eval_batch_dbl(out, m, pars);
}

double func::eval_num_dbl(const std::vector<double> &v) const
{
    if (v.size() != args().size()) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double numerical evaluation of the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    return ptr()->eval_num_dbl(v);
}

double func::deval_num_dbl(const std::vector<double> &v, std::vector<double>::size_type i) const
{
    using namespace fmt::literals;

    if (v.size() != args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double numerical evaluation of the derivative of function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    if (i >= v.size()) {
        throw std::invalid_argument(
            "Invalid index supplied to the double numerical evaluation of the derivative of function '{}': index {} was supplied, but the number of arguments is only {}"_format(
                get_name(), args().size(), v.size()));
    }

    return ptr()->deval_num_dbl(v, i);
}

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
func::taylor_decompose(std::vector<std::pair<expression, std::vector<std::uint32_t>>> &u_vars_defs) &&
{
    auto ret = std::move(*ptr()).taylor_decompose(u_vars_defs);

    if (ret >= u_vars_defs.size()) {
        using namespace fmt::literals;

        throw std::invalid_argument(
            "Invalid value returned by the Taylor decomposition function for the function '{}': "
            "the return value is {}, which is not less than the current size of the decomposition "
            "({})"_format(get_name(), ret, u_vars_defs.size()));
    }

    return ret;
}

llvm::Value *func::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (par_ptr == nullptr) {
        throw std::invalid_argument(
            "Null par_ptr detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    if (time_ptr == nullptr) {
        throw std::invalid_argument(
            "Null time_ptr detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    auto retval = ptr()->taylor_diff_dbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

llvm::Value *func::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (par_ptr == nullptr) {
        throw std::invalid_argument(
            "Null par_ptr detected in func::taylor_diff_ldbl() for the function '{}'"_format(get_name()));
    }

    if (time_ptr == nullptr) {
        throw std::invalid_argument(
            "Null time_ptr detected in func::taylor_diff_ldbl() for the function '{}'"_format(get_name()));
    }

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_diff_ldbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_diff_ldbl() for the function '{}'"_format(get_name()));
    }

    auto retval = ptr()->taylor_diff_ldbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_diff_ldbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *func::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (par_ptr == nullptr) {
        throw std::invalid_argument(
            "Null par_ptr detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    if (time_ptr == nullptr) {
        throw std::invalid_argument(
            "Null time_ptr detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    auto retval = ptr()->taylor_diff_f128(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#endif

llvm::Function *func::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_dbl(s, n_uvars, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

llvm::Function *func::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_ldbl(s, n_uvars, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *func::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size) const
{
    using namespace fmt::literals;

    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_f128() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_f128() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_f128(s, n_uvars, batch_size);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_c_diff_func_f128() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#endif

void swap(func &a, func &b) noexcept
{
    std::swap(a.m_ptr, b.m_ptr);
}

std::ostream &operator<<(std::ostream &os, const func &f)
{
    os << f.get_name() << '(';

    const auto &args = f.args();
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        os << args[i];
        if (i != args.size() - 1u) {
            os << ", ";
        }
    }

    return os << ')';
}

std::size_t hash(const func &f)
{
    // NOTE: the hash value is computed by combining the hash values of:
    // - the function name,
    // - the function inner type index,
    // - the arguments' hashes.
    std::size_t seed = std::hash<std::string>{}(f.get_name());

    boost::hash_combine(seed, f.get_type_index());

    for (const auto &arg : f.args()) {
        boost::hash_combine(seed, hash(arg));
    }

    return seed;
}

bool operator==(const func &a, const func &b)
{
    return a.get_name() == b.get_name() && a.get_type_index() == b.get_type_index() && a.args() == b.args();
}

bool operator!=(const func &a, const func &b)
{
    return !(a == b);
}

std::vector<std::string> get_variables(const func &f)
{
    std::vector<std::string> ret;

    for (const auto &arg : f.args()) {
        auto tmp = get_variables(arg);
        ret.insert(ret.end(), std::make_move_iterator(tmp.begin()), std::make_move_iterator(tmp.end()));
        std::sort(ret.begin(), ret.end());
        ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
    }

    return ret;
}

void rename_variables(func &f, const std::unordered_map<std::string, std::string> &repl_map)
{
    for (auto [b, e] = f.get_mutable_args_it(); b != e; ++b) {
        rename_variables(*b, repl_map);
    }
}

// NOTE: implementing this in-place would perform better.
expression subs(const func &f, const std::unordered_map<std::string, expression> &smap)
{
    auto tmp = f;

    for (auto [b, e] = tmp.get_mutable_args_it(); b != e; ++b) {
        *b = subs(*b, smap);
    }

    return expression{std::move(tmp)};
}

expression diff(const func &f, const std::string &s)
{
    return f.diff(s);
}

double eval_dbl(const func &f, const std::unordered_map<std::string, double> &map, const std::vector<double> &pars)
{
    return f.eval_dbl(map, pars);
}

void eval_batch_dbl(std::vector<double> &out_values, const func &f,
                    const std::unordered_map<std::string, std::vector<double>> &map, const std::vector<double> &pars)
{
    f.eval_batch_dbl(out_values, map, pars);
}

double eval_num_dbl(const func &f, const std::vector<double> &in)
{
    return f.eval_num_dbl(in);
}

double deval_num_dbl(const func &f, const std::vector<double> &in, std::vector<double>::size_type d)
{
    return f.deval_num_dbl(in, d);
}

void update_node_values_dbl(std::vector<double> &node_values, const func &f,
                            const std::unordered_map<std::string, double> &map,
                            const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    // We have to recurse first as to make sure node_values is filled before being accessed later.
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        update_node_values_dbl(node_values, f.args()[i], map, node_connections, node_counter);
    }
    // Then we compute
    std::vector<double> in_values(f.args().size());
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    node_values[node_id] = eval_num_dbl(f, in_values);
}

void update_grad_dbl(std::unordered_map<std::string, double> &grad, const func &f,
                     const std::unordered_map<std::string, double> &map, const std::vector<double> &node_values,
                     const std::vector<std::vector<std::size_t>> &node_connections, std::size_t &node_counter,
                     double acc)
{
    const auto node_id = node_counter;
    node_counter++;
    std::vector<double> in_values(f.args().size());
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        in_values[i] = node_values[node_connections[node_id][i]];
    }
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        auto value = deval_num_dbl(f, in_values, i);
        update_grad_dbl(grad, f.args()[i], map, node_values, node_connections, node_counter, acc * value);
    }
}

void update_connections(std::vector<std::vector<std::size_t>> &node_connections, const func &f,
                        std::size_t &node_counter)
{
    const auto node_id = node_counter;
    node_counter++;
    node_connections.push_back(std::vector<std::size_t>(f.args().size()));
    for (decltype(f.args().size()) i = 0u; i < f.args().size(); ++i) {
        node_connections[node_id][i] = node_counter;
        update_connections(node_connections, f.args()[i], node_counter);
    };
}

std::vector<std::pair<expression, std::vector<std::uint32_t>>>::size_type
taylor_decompose_in_place(func &&f, std::vector<std::pair<expression, std::vector<std::uint32_t>>> &dc)
{
    return std::move(f).taylor_decompose(dc);
}

llvm::Value *taylor_diff_dbl(llvm_state &s, const func &f, const std::vector<std::uint32_t> &deps,
                             const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                             std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return f.taylor_diff_dbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
}

llvm::Value *taylor_diff_ldbl(llvm_state &s, const func &f, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return f.taylor_diff_ldbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_diff_f128(llvm_state &s, const func &f, const std::vector<std::uint32_t> &deps,
                              const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                              std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size)
{
    return f.taylor_diff_f128(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size);
}

#endif

llvm::Function *taylor_c_diff_func_dbl(llvm_state &s, const func &f, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return f.taylor_c_diff_func_dbl(s, n_uvars, batch_size);
}

llvm::Function *taylor_c_diff_func_ldbl(llvm_state &s, const func &f, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return f.taylor_c_diff_func_ldbl(s, n_uvars, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *taylor_c_diff_func_f128(llvm_state &s, const func &f, std::uint32_t n_uvars, std::uint32_t batch_size)
{
    return f.taylor_c_diff_func_f128(s, n_uvars, batch_size);
}

#endif

} // namespace heyoka
