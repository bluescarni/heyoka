// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/version.hpp>

// NOTE: the header for hash_combine changed in version 1.67.
#if (BOOST_VERSION / 100000 > 1) || (BOOST_VERSION / 100000 == 1 && BOOST_VERSION / 100 % 1000 >= 67)

#include <boost/container_hash/hash.hpp>

#else

#include <boost/functional/hash.hpp>

#endif

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

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

// Default implementation of to_stream() for func.
void func_default_to_stream_impl(std::ostream &os, const func_base &f)
{
    os << f.get_name() << '(';

    const auto &args = f.args();
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        os << args[i];
        if (i != args.size() - 1u) {
            os << ", ";
        }
    }

    os << ')';
}

func_inner_base::~func_inner_base() = default;

namespace
{

struct null_func : func_base {
    null_func() : func_base("null_func", {}) {}
};

} // namespace

} // namespace detail

func::func(std::unique_ptr<detail::func_inner_base> p) : m_ptr(p.release()) {}

func::func() : func(detail::null_func{}) {}

func::func(const func &) = default;

func::func(func &&) noexcept = default;

func &func::operator=(const func &) = default;

func &func::operator=(func &&) noexcept = default;

func::~func() = default;

// NOTE: this creates a new func containing
// a copy of the inner object: this means that
// the function arguments are shallow-copied and
// NOT deep-copied.
func func::copy() const
{
    return func{m_ptr->clone()};
}

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

std::vector<expression> func::fetch_gradient(const std::string &target) const
{
    // Check if we have the gradient.
    if (!ptr()->has_gradient()) {
        throw not_implemented_error("Cannot compute the derivative of the function '{}' with respect to a {}, because "
                                    "the function does not provide neither a diff() "
                                    "nor a gradient() member function"_format(get_name(), target));
    }

    // Fetch the gradient.
    auto grad = ptr()->gradient();

    // Check it.
    const auto arity = args().size();
    if (grad.size() != arity) {
        throw std::invalid_argument(
            "Inconsistent gradient returned by the function '{}': a vector of {} elements was expected, but the number of elements is {} instead"_format(
                get_name(), arity, grad.size()));
    }

    return grad;
}

expression func::diff(std::unordered_map<const void *, expression> &func_map, const std::string &s) const
{
    // Run the specialised diff implementation,
    // if available.
    if (ptr()->has_diff_var()) {
        return ptr()->diff(func_map, s);
    }

    const auto arity = args().size();

    // Fetch the gradient.
    auto grad = fetch_gradient("variable");

    // Compute the total derivative.
    std::vector<expression> prod;
    prod.reserve(arity);
    for (decltype(args().size()) i = 0; i < arity; ++i) {
        prod.push_back(std::move(grad[i]) * detail::diff(func_map, args()[i], s));
    }

    return sum(std::move(prod));
}

expression func::diff(std::unordered_map<const void *, expression> &func_map, const param &p) const
{
    // Run the specialised diff implementation,
    // if available.
    if (ptr()->has_diff_par()) {
        return ptr()->diff(func_map, p);
    }

    const auto arity = args().size();

    // Fetch the gradient.
    auto grad = fetch_gradient("parameter");

    // Compute the total derivative.
    std::vector<expression> prod;
    prod.reserve(arity);
    for (decltype(args().size()) i = 0; i < arity; ++i) {
        prod.push_back(std::move(grad[i]) * detail::diff(func_map, args()[i], p));
    }

    return sum(std::move(prod));
}

double func::eval_dbl(const std::unordered_map<std::string, double> &m, const std::vector<double> &pars) const
{
    return ptr()->eval_dbl(m, pars);
}

long double func::eval_ldbl(const std::unordered_map<std::string, long double> &m,
                            const std::vector<long double> &pars) const
{
    return ptr()->eval_ldbl(m, pars);
}

#if defined(HEYOKA_HAVE_REAL128)
mppp::real128 func::eval_f128(const std::unordered_map<std::string, mppp::real128> &m,
                              const std::vector<mppp::real128> &pars) const
{
    return ptr()->eval_f128(m, pars);
}
#endif
void func::eval_batch_dbl(std::vector<double> &out, const std::unordered_map<std::string, std::vector<double>> &m,
                          const std::vector<double> &pars) const
{
    ptr()->eval_batch_dbl(out, m, pars);
}

double func::eval_num_dbl(const std::vector<double> &v) const
{
    if (v.size() != args().size()) {
        throw std::invalid_argument(
            "Inconsistent number of arguments supplied to the double numerical evaluation of the function '{}': {} arguments were expected, but {} arguments were provided instead"_format(
                get_name(), args().size(), v.size()));
    }

    return ptr()->eval_num_dbl(v);
}

double func::deval_num_dbl(const std::vector<double> &v, std::vector<double>::size_type i) const
{
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

namespace detail
{

namespace
{

// Perform the decomposition of the arguments of a function. After this operation,
// each argument will be either:
// - a variable,
// - a number,
// - a param.
void func_td_args(func &fb, std::unordered_map<const void *, taylor_dc_t::size_type> &func_map, taylor_dc_t &dc)
{
    for (auto r = fb.get_mutable_args_it(); r.first != r.second; ++r.first) {
        if (const auto dres = taylor_decompose(func_map, *r.first, dc)) {
            *r.first = expression{variable{"u_{}"_format(dres)}};
        }

        assert(std::holds_alternative<variable>(r.first->value()) || std::holds_alternative<number>(r.first->value())
               || std::holds_alternative<param>(r.first->value()));
    }
}

} // namespace

} // namespace detail

taylor_dc_t::size_type func::taylor_decompose(std::unordered_map<const void *, taylor_dc_t::size_type> &func_map,
                                              taylor_dc_t &dc) const
{
    const auto f_id = get_ptr();

    if (auto it = func_map.find(f_id); it != func_map.end()) {
        // We already decomposed the current function, fetch the result
        // from the cache.
        return it->second;
    }

    // Make a shallow copy: this will be a new function,
    // but its arguments will be shallow-copied from this.
    auto f_copy = copy();

    // Decompose the arguments. This will overwrite
    // the arguments in f_copy with their decomposition.
    detail::func_td_args(f_copy, func_map, dc);

    // Run the decomposition.
    taylor_dc_t::size_type ret = 0;
    if (f_copy.ptr()->has_taylor_decompose()) {
        // Custom implementation.
        ret = std::move(*f_copy.ptr()).taylor_decompose(dc);
    } else {
        // Default implementation: append f_copy and return the index
        // at which it was appended.
        dc.emplace_back(std::move(f_copy), std::vector<std::uint32_t>{});
        ret = dc.size() - 1u;
    }

    if (ret == 0u) {
        throw std::invalid_argument("The return value for the Taylor decomposition of a function can never be zero");
    }

    if (ret >= dc.size()) {
        throw std::invalid_argument(
            "Invalid value returned by the Taylor decomposition function for the function '{}': "
            "the return value is {}, which is not less than the current size of the decomposition "
            "({})"_format(get_name(), ret, dc.size()));
    }

    // Update the cache before exiting.
    [[maybe_unused]] const auto [_, flag] = func_map.insert(std::pair{f_id, ret});
    assert(flag);

    return ret;
}

llvm::Value *func::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                   const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                   std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                   std::uint32_t batch_size, bool high_accuracy) const
{
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

    auto retval
        = ptr()->taylor_diff_dbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_diff_dbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

llvm::Value *func::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &deps,
                                    const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                    std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx,
                                    std::uint32_t batch_size, bool high_accuracy) const
{
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

    auto retval
        = ptr()->taylor_diff_ldbl(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);

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
                                    std::uint32_t batch_size, bool high_accuracy) const
{
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

    auto retval
        = ptr()->taylor_diff_f128(s, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_diff_f128() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#endif

llvm::Function *func::taylor_c_diff_func_dbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                             bool high_accuracy) const
{
    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_dbl(s, n_uvars, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_c_diff_func_dbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

llvm::Function *func::taylor_c_diff_func_ldbl(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                              bool high_accuracy) const
{
    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_ldbl(s, n_uvars, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            "Null return value detected in func::taylor_c_diff_func_ldbl() for the function '{}'"_format(get_name()));
    }

    return retval;
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Function *func::taylor_c_diff_func_f128(llvm_state &s, std::uint32_t n_uvars, std::uint32_t batch_size,
                                              bool high_accuracy) const
{
    if (batch_size == 0u) {
        throw std::invalid_argument(
            "Zero batch size detected in func::taylor_c_diff_func_f128() for the function '{}'"_format(get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(
            "Zero number of u variables detected in func::taylor_c_diff_func_f128() for the function '{}'"_format(
                get_name()));
    }

    auto retval = ptr()->taylor_c_diff_func_f128(s, n_uvars, batch_size, high_accuracy);

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
    f.ptr()->to_stream(os);

    return os;
}

std::size_t hash(const func &f)
{
    // NOTE: the initial hash value is computed by combining the hash values of:
    // - the function name,
    // - the function inner type index,
    // - the arguments' hashes.
    std::size_t seed = std::hash<std::string>{}(f.get_name());

    boost::hash_combine(seed, f.get_type_index());

    for (const auto &arg : f.args()) {
        boost::hash_combine(seed, hash(arg));
    }

    // Combine with the extra hash value too.
    boost::hash_combine(seed, f.ptr()->extra_hash());

    return seed;
}

bool operator==(const func &a, const func &b)
{
    // Check if the underlying object is the same.
    if (a.m_ptr == b.m_ptr) {
        return true;
    }

    // NOTE: the initial comparison considers:
    // - the function name,
    // - the function inner type index,
    // - the arguments.
    // If they are all equal, the extra equality comparison logic
    // is also run.
    if (a.get_name() == b.get_name() && a.get_type_index() == b.get_type_index() && a.args() == b.args()) {
        return a.ptr()->extra_equal_to(b);
    } else {
        return false;
    }
}

bool operator!=(const func &a, const func &b)
{
    return !(a == b);
}

double eval_dbl(const func &f, const std::unordered_map<std::string, double> &map, const std::vector<double> &pars)
{
    return f.eval_dbl(map, pars);
}

long double eval_ldbl(const func &f, const std::unordered_map<std::string, long double> &map,
                      const std::vector<long double> &pars)
{
    return f.eval_ldbl(map, pars);
}

#if defined(HEYOKA_HAVE_REAL128)

mppp::real128 eval_f128(const func &f, const std::unordered_map<std::string, mppp::real128> &map,
                        const std::vector<mppp::real128> &pars)
{
    return f.eval_f128(map, pars);
}

#endif

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

} // namespace heyoka
