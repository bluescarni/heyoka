// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <limits>
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

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <fmt/format.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

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

func_inner_base::func_inner_base() = default;

func_inner_base::~func_inner_base() = default;

namespace
{

struct null_func : func_base {
    null_func() : func_base("null_func", {}) {}
};

} // namespace

} // namespace detail

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT(heyoka::detail::null_func)

HEYOKA_BEGIN_NAMESPACE

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

// NOTE: time dependency here means **intrinsic** time
// dependence of the function. That is, we are not concerned
// with the arguments' time dependence and, e.g., cos(time)
// is **not** time-dependent according to this definition.
bool func::is_time_dependent() const
{
    return ptr()->is_time_dependent();
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

std::vector<expression> func::fetch_gradient(const std::string &target) const
{
    // Check if we have the gradient.
    if (!ptr()->has_gradient()) {
        throw not_implemented_error(
            fmt::format("Cannot compute the derivative of the function '{}' with respect to a {}, because "
                        "the function does not provide neither a diff() "
                        "nor a gradient() member function",
                        get_name(), target));
    }

    // Fetch the gradient.
    auto grad = ptr()->gradient();

    // Check it.
    const auto arity = args().size();
    if (grad.size() != arity) {
        throw std::invalid_argument(fmt::format("Inconsistent gradient returned by the function '{}': a vector of {} "
                                                "elements was expected, but the number of elements is {} instead",
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
            fmt::format("Inconsistent number of arguments supplied to the double numerical evaluation of the function "
                        "'{}': {} arguments were expected, but {} arguments were provided instead",
                        get_name(), args().size(), v.size()));
    }

    return ptr()->eval_num_dbl(v);
}

double func::deval_num_dbl(const std::vector<double> &v, std::vector<double>::size_type i) const
{
    if (v.size() != args().size()) {
        throw std::invalid_argument(fmt::format(
            "Inconsistent number of arguments supplied to the double numerical evaluation of the derivative of "
            "function '{}': {} arguments were expected, but {} arguments were provided instead",
            get_name(), args().size(), v.size()));
    }

    if (i >= v.size()) {
        throw std::invalid_argument(
            fmt::format("Invalid index supplied to the double numerical evaluation of the derivative of function '{}': "
                        "index {} was supplied, but the number of arguments is only {}",
                        get_name(), args().size(), v.size()));
    }

    return ptr()->deval_num_dbl(v, i);
}

llvm::Value *func::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                             llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride, std::uint32_t batch_size,
                             bool high_accuracy) const
{
    return ptr()->llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *func::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                       bool high_accuracy) const
{
    return ptr()->llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);
}

namespace detail
{

namespace
{

// Perform the Taylor decomposition of the arguments of a function. After this operation,
// each argument will be either:
// - a variable,
// - a number,
// - a param.
void func_td_args(func &fb, std::unordered_map<const void *, taylor_dc_t::size_type> &func_map, taylor_dc_t &dc)
{
    for (auto r = fb.get_mutable_args_it(); r.first != r.second; ++r.first) {
        if (const auto dres = taylor_decompose(func_map, *r.first, dc)) {
            *r.first = expression{fmt::format("u_{}", dres)};
        } else {
            assert(std::holds_alternative<variable>(r.first->value())
                   || std::holds_alternative<number>(r.first->value())
                   || std::holds_alternative<param>(r.first->value()));
        }

        assert(std::holds_alternative<variable>(r.first->value()) || std::holds_alternative<number>(r.first->value())
               || std::holds_alternative<param>(r.first->value()));
    }
}

// Perform the decomposition of the arguments of a function. After this operation,
// each argument will be either:
// - a variable,
// - a number,
// - a param.
void func_d_args(func &fb, std::unordered_map<const void *, std::vector<expression>::size_type> &func_map,
                 std::vector<expression> &dc)
{
    for (auto r = fb.get_mutable_args_it(); r.first != r.second; ++r.first) {
        if (const auto dres = decompose(func_map, *r.first, dc)) {
            *r.first = expression{fmt::format("u_{}", *dres)};
        } else {
            assert(std::holds_alternative<variable>(r.first->value())
                   || std::holds_alternative<number>(r.first->value())
                   || std::holds_alternative<param>(r.first->value()));
        }

        assert(std::holds_alternative<variable>(r.first->value()) || std::holds_alternative<number>(r.first->value())
               || std::holds_alternative<param>(r.first->value()));
    }
}

} // namespace

} // namespace detail

std::vector<expression>::size_type
func::decompose(std::unordered_map<const void *, std::vector<expression>::size_type> &func_map,
                std::vector<expression> &dc) const
{
    const auto *const f_id = get_ptr();

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
    detail::func_d_args(f_copy, func_map, dc);

    // Append f_copy and return the index at which it was appended.
    const auto ret = dc.size();
    dc.emplace_back(std::move(f_copy));

    // Update the cache before exiting.
    [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
    assert(flag);

    return ret;
}

taylor_dc_t::size_type func::taylor_decompose(std::unordered_map<const void *, taylor_dc_t::size_type> &func_map,
                                              taylor_dc_t &dc) const
{
    const auto *const f_id = get_ptr();

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
        ret = dc.size();
        dc.emplace_back(std::move(f_copy), std::vector<std::uint32_t>{});
    }

    if (ret == 0u) {
        throw std::invalid_argument("The return value for the Taylor decomposition of a function can never be zero");
    }

    if (ret >= dc.size()) {
        throw std::invalid_argument(
            fmt::format("Invalid value returned by the Taylor decomposition function for the function '{}': "
                        "the return value is {}, which is not less than the current size of the decomposition "
                        "({})",
                        get_name(), ret, dc.size()));
    }

    // Update the cache before exiting.
    [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
    assert(flag);

    return ret;
}

llvm::Value *func::taylor_diff(llvm_state &s, llvm::Type *fp_t, const std::vector<std::uint32_t> &deps,
                               const std::vector<llvm::Value *> &arr, llvm::Value *par_ptr, llvm::Value *time_ptr,
                               std::uint32_t n_uvars, std::uint32_t order, std::uint32_t idx, std::uint32_t batch_size,
                               bool high_accuracy) const
{
    if (fp_t == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null floating-point type detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    if (par_ptr == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null par_ptr detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    if (time_ptr == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null time_ptr detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    if (batch_size == 0u) {
        throw std::invalid_argument(
            fmt::format("Zero batch size detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(fmt::format(
            "Zero number of u variables detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    auto *retval
        = ptr()->taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null return value detected in func::taylor_diff() for the function '{}'", get_name()));
    }

    return retval;
}

llvm::Function *func::taylor_c_diff_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t n_uvars,
                                         std::uint32_t batch_size, bool high_accuracy) const
{
    if (fp_t == nullptr) {
        throw std::invalid_argument(fmt::format(
            "Null floating-point type detected in func::taylor_c_diff_func() for the function '{}'", get_name()));
    }

    if (batch_size == 0u) {
        throw std::invalid_argument(
            fmt::format("Zero batch size detected in func::taylor_c_diff_func() for the function '{}'", get_name()));
    }

    if (n_uvars == 0u) {
        throw std::invalid_argument(fmt::format(
            "Zero number of u variables detected in func::taylor_c_diff_func() for the function '{}'", get_name()));
    }

    auto *retval = ptr()->taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null return value detected in func::taylor_c_diff_func() for the function '{}'", get_name()));
    }

    return retval;
}

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

namespace detail
{

// Helper to perform the codegen for a parameter during the creation
// of a compiled function in non-compact mode.
llvm::Value *cfunc_nc_param_codegen(llvm_state &s, const param &p, std::uint32_t batch_size, llvm::Type *fp_t,
                                    llvm::Value *par_ptr, llvm::Value *stride)
{
    auto &builder = s.builder();

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_t);

    // Determine the index into the parameter array.
    auto *arr_idx = builder.CreateMul(stride, to_size_t(s, builder.getInt32(p.idx())));

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr, arr_idx);

    // Load and return.
    return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
}

// Helper to implement the llvm_eval_*() methods in the func interface
// used to create compiled functions in non-compact mode.
llvm::Value *llvm_eval_helper(const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &g,
                              const func_base &f, llvm_state &s, llvm::Type *fp_t,
                              const std::vector<llvm::Value *> &eval_arr, llvm::Value *par_ptr, llvm::Value *stride,
                              std::uint32_t batch_size, bool high_accuracy)
{
    assert(g);

    auto &builder = s.builder();

    // Codegen the function arguments.
    std::vector<llvm::Value *> llvm_args;
    for (const auto &arg : f.args()) {
        std::visit(
            [&](const auto &v) {
                using type = uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, variable>) {
                    // Fetch the index of the variable argument.
                    const auto u_idx = uname_to_index(v.name());
                    assert(u_idx < eval_arr.size());

                    // Fetch the corresponding value from eval_arr.
                    llvm_args.push_back(eval_arr[u_idx]);
                } else if constexpr (std::is_same_v<type, number>) {
                    // Codegen the number argument.
                    llvm_args.push_back(vector_splat(builder, llvm_codegen(s, fp_t, v), batch_size));
                } else if constexpr (std::is_same_v<type, param>) {
                    // Codegen the parameter argument.
                    llvm_args.push_back(cfunc_nc_param_codegen(s, v, batch_size, fp_t, par_ptr, stride));
                } else {
                    assert(false); // LCOV_EXCL_LINE
                }
            },
            arg.value());
    }

    // Run the generator and return the result.
    return g(llvm_args, high_accuracy);
}

// NOTE: precondition on name: must be conforming to LLVM requirements for
// function names, and must not contain "." (as we use it as a separator in
// the mangling scheme).
std::pair<std::string, std::vector<llvm::Type *>> llvm_c_eval_func_name_args(llvm::LLVMContext &c, llvm::Type *fp_t,
                                                                             const std::string &name,
                                                                             std::uint32_t batch_size,
                                                                             const std::vector<expression> &args)
{
    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_t);

    // Init the name.
    auto fname = fmt::format("heyoka.llvm_c_eval.{}.", name);

    // Init the vector of arguments:
    // - idx of the u variable which is being evaluated,
    // - eval array (pointer to val_t),
    // - par ptr (pointer to scalar),
    // - time ptr (pointer to scalar),
    // - stride value.
    std::vector<llvm::Type *> fargs{llvm::Type::getInt32Ty(c), llvm::PointerType::getUnqual(val_t),
                                    llvm::PointerType::getUnqual(ext_fp_t), llvm::PointerType::getUnqual(ext_fp_t),
                                    to_llvm_type<std::size_t>(c)};

    // Add the mangling and LLVM arg types for the argument types.
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        // Name mangling.
        fname += std::visit([](const auto &v) { return cm_mangle(v); }, args[i].value());

        // Add the arguments separator, if we are not at the
        // last argument.
        if (i != args.size() - 1u) {
            fname += '_';
        }

        // Add the LLVM function argument type.
        fargs.push_back(std::visit(
            [&](const auto &v) -> llvm::Type * {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, number>) {
                    // For numbers, the argument is passed as a scalar
                    // floating-point value.
                    return val_t->getScalarType();
                } else if constexpr (std::is_same_v<type, variable> || std::is_same_v<type, param>) {
                    // For vars and params, the argument is an index
                    // in an array.
                    return llvm::Type::getInt32Ty(c);
                } else {
                    // LCOV_EXCL_START
                    assert(false);
                    throw;
                    // LCOV_EXCL_STOP
                }
            },
            args[i].value()));
    }

    // Close the argument list with a ".".
    // NOTE: this will result in a ".." in the name
    // if the function has zero arguments.
    fname += '.';

    // Finally, add the mangling for the floating-point type.
    fname += llvm_mangle_type(val_t);

    return std::make_pair(std::move(fname), std::move(fargs));
}

llvm::Function *llvm_c_eval_func_helper(const std::string &name,
                                        const std::function<llvm::Value *(const std::vector<llvm::Value *> &, bool)> &g,
                                        const func_base &fb, llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                        bool high_accuracy)
{
    // LCOV_EXCL_START
    assert(g);
    assert(batch_size > 0u);
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the type for external loading.
    auto *ext_fp_t = llvm_ext_type(fp_t);

    // Fetch the vector floating-point type.
    auto *val_t = make_vector_type(fp_t, batch_size);

    const auto na_pair = llvm_c_eval_func_name_args(context, fp_t, name, batch_size, fb.args());
    const auto &fname = na_pair.first;
    const auto &fargs = na_pair.second;

    // Try to see if we already created the function.
    auto *f = md.getFunction(fname);

    if (f == nullptr) {
        // The function was not created before, do it now.

        // Fetch the current insertion block.
        auto *orig_bb = builder.GetInsertBlock();

        // The return type is val_t.
        auto *ft = llvm::FunctionType::get(val_t, fargs, false);
        // Create the function
        f = llvm::Function::Create(ft, llvm::Function::InternalLinkage, fname, &md);
        assert(f != nullptr);

        // Create a new basic block to start insertion into.
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", f));

        // Fetch the necessary arguments.
        auto *eval_arr = f->args().begin() + 1;
        auto *par_ptr = f->args().begin() + 2;
        auto *stride = f->args().begin() + 4;

        // Create the arguments for g.
        std::vector<llvm::Value *> g_args;
        for (decltype(fb.args().size()) i = 0; i < fb.args().size(); ++i) {
            auto *arg = std::visit( // LCOV_EXCL_LINE
                [&](const auto &v) -> llvm::Value * {
                    using type = detail::uncvref_t<decltype(v)>;

                    auto *const cur_f_arg = f->args().begin() + 5 + i;

                    if constexpr (std::is_same_v<type, number>) {
                        // NOTE: number arguments are passed directly as
                        // FP constants when f is invoked. We just need to splat them.
                        return vector_splat(builder, cur_f_arg, batch_size);
                    } else if constexpr (std::is_same_v<type, variable>) {
                        // NOTE: for variables, the u index is passed to f.
                        return cfunc_c_load_eval(s, val_t, eval_arr, cur_f_arg);
                    } else if constexpr (std::is_same_v<type, param>) {
                        // NOTE: for params, we have to load the value from par_ptr.
                        // NOTE: the overflow check is done in add_cfunc_impl().

                        // LCOV_EXCL_START
                        assert(llvm::isa<llvm::PointerType>(par_ptr->getType()));
                        assert(!llvm::cast<llvm::PointerType>(par_ptr->getType())->isVectorTy());
                        // LCOV_EXCL_STOP

                        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr,
                                                              builder.CreateMul(stride, to_size_t(s, cur_f_arg)));

                        return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
                    } else {
                        // LCOV_EXCL_START
                        assert(false);
                        throw;
                        // LCOV_EXCL_STOP
                    }
                },
                fb.args()[i].value());

            g_args.push_back(arg);
        }

        // Compute the return value.
        auto *ret = g(g_args, high_accuracy);
        assert(ret != nullptr);

        // Return it.
        builder.CreateRet(ret);

        // Verify.
        s.verify_function(f);

        // Restore the original insertion block.
        builder.SetInsertPoint(orig_bb);
    } else {
        // LCOV_EXCL_START
        // The function was created before. Check if the signatures match.
        // NOTE: there could be a mismatch if the function was created
        // and then optimised - optimisation might remove arguments which are compile-time
        // constants.
        if (!compare_function_signature(f, val_t, fargs)) {
            throw std::invalid_argument(fmt::format(
                "Inconsistent function signature for the evaluation of {}() in compact mode detected", name));
        }
        // LCOV_EXCL_STOP
    }

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE
