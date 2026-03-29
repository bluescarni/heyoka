// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/func_args.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to check the name of a func_base right after construction.
void func_base_check_name(const std::string &name)
{
    if (name.empty()) [[unlikely]] {
        throw std::invalid_argument("Cannot create a function with no name");
    }
}

} // namespace

} // namespace detail

func_base::func_base(std::string name, std::vector<expression> args, bool shared)
    : m_name(std::move(name)), m_llvm_name(m_name), m_args(std::move(args), shared)
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(std::string name, func_args::shared_args_t sargs)
    : m_name(std::move(name)), m_llvm_name(m_name), m_args(std::move(sargs))
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(std::string name, func_args fargs)
    : m_name(std::move(name)), m_llvm_name(m_name), m_args(std::move(fargs))
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(std::string name, std::string llvm_name, std::vector<expression> args, bool shared)
    : m_name(std::move(name)), m_llvm_name(std::move(llvm_name)), m_args(std::move(args), shared)
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(std::string name, std::string llvm_name, func_args::shared_args_t sargs)
    : m_name(std::move(name)), m_llvm_name(std::move(llvm_name)), m_args(std::move(sargs))
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(std::string name, std::string llvm_name, func_args fargs)
    : m_name(std::move(name)), m_llvm_name(std::move(llvm_name)), m_args(std::move(fargs))
{
    detail::func_base_check_name(m_name);
}

func_base::func_base(const func_base &) = default;

func_base::func_base(func_base &&) noexcept = default;

func_base &func_base::operator=(const func_base &) = default;

func_base &func_base::operator=(func_base &&) noexcept = default;

func_base::~func_base() = default;

void func_base::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_name;
    ar << m_llvm_name;
    ar << m_args;
}

void func_base::load(boost::archive::binary_iarchive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<func_base>::type::value)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Unable to load a func_base object: the archive version ({}) is too old", version));
    }
    // LCOV_EXCL_STOP

    ar >> m_name;
    ar >> m_llvm_name;
    ar >> m_args;
}

const std::string &func_base::get_name() const noexcept
{
    return m_name;
}

const std::string &func_base::get_llvm_name() const noexcept
{
    return m_llvm_name;
}

const func_args &func_base::get_func_args() const noexcept
{
    return m_args;
}

const std::vector<expression> &func_base::args() const noexcept
{
    return m_args.get_args();
}

func_args::shared_args_t func_base::shared_args() const noexcept
{
    return m_args.get_shared_args();
}

// NOTE: here we are enforcing that the two overloads are called consistently with how the function stores its
// arguments. The idea is that the way the function stores its arguments is an invariant property of the function.
//
// NOTE: we use assertions here instead of throws because the checks are already done in make_copy_with_new_args().
void func_base::replace_args(std::vector<expression> new_args)
{
    assert(!shared_args());

    m_args = func_args(std::move(new_args));
}

void func_base::replace_args(func_args::shared_args_t new_args)
{
    assert(shared_args());
    assert(new_args);

    m_args = func_args(std::move(new_args));
}

namespace detail
{

namespace
{

// Default implementation of to_stream() for func.
template <typename Base>
void func_default_to_stream_impl(std::ostringstream &oss, const Base &f)
{
    oss << f.get_name() << '(';

    const auto &args = f.args();
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        stream_expression(oss, args[i]);
        if (i != args.size() - 1u) {
            oss << ", ";
        }
    }

    oss << ')';
}

} // namespace

void func_default_to_stream(std::ostringstream &oss, const func_base &f)
{
    func_default_to_stream_impl(oss, f);
}

null_func::null_func() : func_base("null_func", std::vector<expression>{}) {}

void null_func::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << boost::serialization::base_object<func_base>(*this);
}

void null_func::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> boost::serialization::base_object<func_base>(*this);
}

} // namespace detail

func::func() = default;

func::func(const func &) noexcept = default;

func::func(func &&) noexcept = default;

func &func::operator=(const func &) noexcept = default;

func &func::operator=(func &&) noexcept = default;

func::~func() = default;

void func::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_func;
}

void func::load(boost::archive::binary_iarchive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<func>::type::value)) [[unlikely]] {
        throw std::invalid_argument("Cannot load a function instance from an older archive");
    }
    // LCOV_EXCL_STOP

    ar >> m_func;
}

const void *func::get_ptr() const noexcept
{
    return raw_value_ptr(m_func);
}

const std::vector<expression> &func::args() const noexcept
{
    return m_func->args();
}

func_args::shared_args_t func::shared_args() const noexcept
{
    return m_func->shared_args();
}

func func::make_copy_with_new_args(std::vector<expression> new_args) const
{
    if (shared_args()) [[unlikely]] {
        throw std::invalid_argument(
            "Cannot invoke func::make_copy_with_new_args() with a non-shared arguments set if the "
            "function manages its arguments via a shared reference");
    }

    const auto orig_size = args().size();

    if (new_args.size() != orig_size) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The set of new arguments passed to func::make_copy_with_new_args() "
                                                "has a size of {}, but the number of arguments "
                                                "of the original function is {} (the two sizes must be equal)",
                                                new_args.size(), orig_size));
    }

    // NOTE: this will end up invoking the copy constructor
    // of the internal user-defined function.
    func ret;
    ret.m_func = copy(m_func);

    // Replace the arguments.
    ret.m_func->replace_args(std::move(new_args));

    return ret;
} // LCOV_EXCL_LINE

func func::make_copy_with_new_args(func_args::shared_args_t new_args) const
{
    if (!new_args) [[unlikely]] {
        throw std::invalid_argument("Cannot invoke func::make_copy_with_new_args() with a null pointer argument");
    }
    if (!shared_args()) [[unlikely]] {
        throw std::invalid_argument("Cannot invoke func::make_copy_with_new_args() with a shared arguments set if the "
                                    "function does not manage its arguments via a shared reference");
    }

    const auto orig_size = args().size();

    if (new_args->size() != orig_size) [[unlikely]] {
        throw std::invalid_argument(fmt::format("The set of new arguments passed to func::make_copy_with_new_args() "
                                                "has a size of {}, but the number of arguments "
                                                "of the original function is {} (the two sizes must be equal)",
                                                new_args->size(), orig_size));
    }

    // NOTE: this will end up invoking the copy constructor
    // of the internal user-defined function.
    func ret;
    ret.m_func = copy(m_func);

    // Replace the arguments.
    ret.m_func->replace_args(std::move(new_args));

    return ret;
}

std::type_index func::get_type_index() const
{
    return value_type_index(m_func);
}

std::vector<expression> func::gradient() const
{
    // Check if we have the gradient.
    if (!m_func->has_gradient()) [[unlikely]] {
        throw not_implemented_error(fmt::format("Cannot compute derivatives for the function '{}', because "
                                                "the function does not provide a gradient() member function",
                                                get_name()));
    }

    // Fetch the gradient.
    auto grad = m_func->gradient();

    // Check it.
    const auto arity = args().size();
    if (grad.size() != arity) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Inconsistent gradient returned by the function '{}': a vector of {} "
                                                "elements was expected, but the number of elements is {} instead",
                                                get_name(), arity, grad.size()));
    }

    return grad;
}

// NOTE: time dependency here means **intrinsic** time
// dependence of the function. That is, we are not concerned
// with the arguments' time dependence and, e.g., cos(time)
// is **not** time-dependent according to this definition.
bool func::is_time_dependent() const
{
    return m_func->is_time_dependent();
}

const std::string &func::get_name() const noexcept
{
    return m_func->get_name();
}

const std::string &func::get_llvm_name() const noexcept
{
    return m_func->get_llvm_name();
}

const func_args &func::get_func_args() const noexcept
{
    return m_func->get_func_args();
}

void func::to_stream(std::ostringstream &oss) const
{
    m_func->to_stream(oss);
}

llvm::Value *func::llvm_evaluate(llvm_state &s, const std::vector<llvm::Value *> &args, llvm::Type *val_t,
                                 llvm::Value *time_ptr, const bool high_accuracy) const
{
    return m_func->llvm_evaluate(s, args, val_t, time_ptr, high_accuracy);
}

namespace detail
{

// NOTE: this is a small helper to perform the Taylor decomposition of the input function fn (which will be consumed by
// the operation) into the decomposition dc. The decomposition will be performed either by invoking the custom
// taylor_decompose() member function from the UDF (if available) or by just appending fn to dc (the default behaviour).
// It is assumed that the arguments of fn have already been decomposed.
//
// We implement this as a separate external (friend) function (rather than, e.g., a member function) because we do not
// want to have mutating functions in the public API of func.
taylor_dc_t::size_type func_taylor_decompose_impl(func &&fn, taylor_dc_t &dc)
{
    assert(std::ranges::none_of(fn.args(), [](const auto &ex) { return std::holds_alternative<func>(ex.value()); }));

    taylor_dc_t::size_type ret = 0;
    if (fn.m_func->has_taylor_decompose()) {
        // Custom implementation.
        ret = std::move(*fn.m_func).taylor_decompose(dc);
    } else {
        // Default implementation: append fn and return the index
        // at which it was appended.
        ret = dc.size();
        dc.emplace_back(std::move(fn), std::vector<std::uint32_t>{});
    }

    // Checks on the return value.
    if (ret == 0u) [[unlikely]] {
        throw std::invalid_argument("The return value for the Taylor decomposition of a function can never be zero");
    }

    if (ret >= dc.size()) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("Invalid value returned by the Taylor decomposition of a function: the return value is {}, "
                        "which is not less than the current size of the decomposition ({})",
                        ret, dc.size()));
    }

    return ret;
}

} // namespace detail

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
        = m_func->taylor_diff(s, fp_t, deps, arr, par_ptr, time_ptr, n_uvars, order, idx, batch_size, high_accuracy);

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

    auto *retval = m_func->taylor_c_diff_func(s, fp_t, n_uvars, batch_size, high_accuracy);

    if (retval == nullptr) {
        throw std::invalid_argument(
            fmt::format("Null return value detected in func::taylor_c_diff_func() for the function '{}'", get_name()));
    }

    return retval;
}

void swap(func &a, func &b) noexcept
{
    using std::swap;
    swap(a.m_func, b.m_func);
}

bool operator==(const func &a, const func &b) noexcept
{
    // Check if the underlying object is the same.
    if (same_value(a.m_func, b.m_func)) {
        return true;
    }

    // NOTE: the comparison considers the function name and the arguments.
    if (a.get_name() != b.get_name()) {
        return false;
    }

    // Check if the vectors of arguments are the same object - this
    // could happen when the arguments are stored in a shared pointer.
    if (&a.args() == &b.args()) {
        return true;
    }

    // Compare the arguments.
    return a.args() == b.args();
}

bool operator!=(const func &a, const func &b) noexcept
{
    return !(a == b);
}

// NOTE: this comparison has no mathematical meaning, it is used
// only to impose a strict ordering on functions. Like operator==(),
// this comparison considers, in order:
//
// - the function name,
// - the arguments.
bool operator<(const func &a, const func &b)
{
    if (same_value(a.m_func, b.m_func)) {
        // Same object, a is NOT less than b.
        return false;
    }

    // Check the names.
    if (a.get_name() < b.get_name()) {
        return true;
    }

    if (b.get_name() < a.get_name()) {
        return false;
    }

    assert(a.get_name() == b.get_name());

    // The names are equal, check the arguments next.

    // First we check if the vectors of arguments are the same object - this
    // could happen when the arguments are stored in a shared pointer.
    if (&a.args() == &b.args()) {
        return false;
    }

    // Run a lexicographical compare.
    if (std::lexicographical_compare(a.args().begin(), a.args().end(), b.args().begin(), b.args().end(),
                                     std::less<expression>{})) {
        return true;
    }

    if (std::lexicographical_compare(b.args().begin(), b.args().end(), a.args().begin(), a.args().end(),
                                     std::less<expression>{})) {
        return false;
    }

    assert(a.args() == b.args());

    // a and b are equivalent.
    assert(a == b);

    return false;
}

namespace detail
{

// Helper to perform the codegen for a parameter during the creation
// of a compiled function in non-compact mode.
llvm::Value *cfunc_nc_param_codegen(llvm_state &s, const param &p, std::uint32_t batch_size, llvm::Type *fp_t,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    llvm::Value *par_ptr, llvm::Value *stride)
{
    auto &builder = s.builder();

    // Fetch the type for external loading.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Determine the index into the parameter array.
    auto *arr_idx = builder.CreateMul(stride, to_size_t(s, builder.getInt32(p.idx())));

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, par_ptr, arr_idx);

    // Load and return.
    return ext_load_vector_from_memory(s, fp_t, ptr, batch_size);
}

} // namespace detail

HEYOKA_END_NAMESPACE

// s11n implementation for null_func.
//
// NOLINTNEXTLINE(cert-err58-cpp,bugprone-throwing-static-initialization)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::null_func)
