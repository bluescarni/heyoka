// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container_hash/hash.hpp>

#include <fmt/core.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/cm_utils.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/tanuki.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/exceptions.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

func_iface::~func_iface() = default;

std::vector<expression> func_iface::fetch_gradient(const std::string &target) const
{
    // Check if we have the gradient.
    if (!has_gradient()) {
        throw not_implemented_error(
            fmt::format("Cannot compute the derivative of the function '{}' with respect to a {}, because "
                        "the function does not provide neither a diff() "
                        "nor a gradient() member function",
                        get_name(), target));
    }

    // Fetch the gradient.
    auto grad = gradient();

    // Check it.
    const auto arity = args().size();
    if (grad.size() != arity) {
        throw std::invalid_argument(fmt::format("Inconsistent gradient returned by the function '{}': a vector of {} "
                                                "elements was expected, but the number of elements is {} instead",
                                                get_name(), arity, grad.size()));
    }

    return grad;
}

} // namespace detail

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

void func_base::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_name;
    ar << m_args;
}

void func_base::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_name;
    ar >> m_args;
}

const std::string &func_base::get_name() const noexcept
{
    return m_name;
}

const std::vector<expression> &func_base::args() const noexcept
{
    return m_args;
}

std::pair<expression *, expression *> func_base::get_mutable_args_range()
{
    return {m_args.data(), m_args.data() + m_args.size()};
}

namespace detail
{

// Default implementation of to_stream() for func.
void func_default_to_stream_impl(std::ostringstream &oss, const func_base &f)
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

null_func::null_func() : func_base("null_func", {}) {}

namespace
{

// Helper to invoke the copy() function via ADL.
auto make_adl_copy(const auto &x)
{
    return copy(x);
}

// Perform the Taylor decomposition of the arguments of a function. This function
// will return a vector of arguments in which each element will be either:
// - a variable,
// - a number,
// - a param.
std::vector<expression> func_td_args(const auto &fb, funcptr_map<taylor_dc_t::size_type> &func_map, taylor_dc_t &dc)
{
    std::vector<expression> retval;
    retval.reserve(fb.args().size());

    for (const auto &arg : fb.args()) {
        const auto dres = taylor_decompose(func_map, arg, dc);

        if (dres == 0u) {
            assert(std::holds_alternative<variable>(arg.value()) || std::holds_alternative<number>(arg.value())
                   || std::holds_alternative<param>(arg.value()));

            retval.push_back(arg);
        } else {
            retval.emplace_back(fmt::format("u_{}", dres));
        }
    }

    return retval;
} // LCOV_EXCL_LINE

// Perform the decomposition of the arguments of a function. This function
// will return a vector of arguments in which each element will be either:
// - a variable,
// - a number,
// - a param.
std::vector<expression> func_d_args(const auto &fb, funcptr_map<std::vector<expression>::size_type> &func_map,
                                    std::vector<expression> &dc)
{
    std::vector<expression> retval;
    retval.reserve(fb.args().size());

    for (const auto &arg : fb.args()) {
        const auto dres = decompose(func_map, arg, dc);

        if (dres) {
            retval.emplace_back(fmt::format("u_{}", *dres));
        } else {
            assert(std::holds_alternative<variable>(arg.value()) || std::holds_alternative<number>(arg.value())
                   || std::holds_alternative<param>(arg.value()));

            retval.push_back(arg);
        }
    }

    return retval;
} // LCOV_EXCL_LINE

} // namespace

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
    if (version < static_cast<unsigned>(boost::serialization::version<func>::type::value)) {
        throw std::invalid_argument("Cannot load a function instance from an older archive");
    }
    // LCOV_EXCL_STOP

    ar >> m_func;
}

const void *func::get_ptr() const
{
    return raw_value_ptr(m_func);
}

const std::vector<expression> &func::args() const
{
    return m_func->args();
}

func func::copy(const std::vector<expression> &new_args) const
{
    const auto orig_size = args().size();

    if (new_args.size() != orig_size) {
        throw std::invalid_argument(
            fmt::format("The set of new arguments passed to func::copy() has a size of {}, but the number of arguments "
                        "of the original function is {} (the two sizes must be equal)",
                        new_args.size(), orig_size));
    }

    // NOTE: this will end up invoking the copy constructor
    // of the internal user-defined function.
    // NOTE: need to go through the make_adl_copy() wrapper
    // in order to avoid ambiguities.
    func ret;
    ret.m_func = detail::make_adl_copy(m_func);

    // NOTE: a user-defined function might have a funky
    // implementation of the copy constructor which, e.g.,
    // changes the arity.
    // LCOV_EXCL_START
    if (ret.args().size() != orig_size) {
        throw std::invalid_argument(fmt::format("The copy constructor of a user-defined function changed the arity"
                                                "from {} to {} - this is not allowed",
                                                orig_size, ret.args().size()));
    }
    // LCOV_EXCL_STOP

    // Copy over the new arguments.
    auto *it = ret.m_func->get_mutable_args_range().first;
    for (decltype(new_args.size()) i = 0; i < new_args.size(); ++i, ++it) {
        *it = new_args[i];
    }

    return ret;
}

std::type_index func::get_type_index() const
{
    return value_type_index(m_func);
}

expression func::diff(detail::funcptr_map<expression> &func_map, const std::string &s) const
{
    // Run the specialised diff implementation,
    // if available.
    if (m_func->has_diff_var()) {
        return m_func->diff(func_map, s);
    }

    const auto arity = args().size();

    // Fetch the gradient.
    auto grad = m_func->fetch_gradient("variable");

    // Compute the total derivative.
    std::vector<expression> prod;
    prod.reserve(arity);
    for (decltype(args().size()) i = 0; i < arity; ++i) {
        prod.push_back(grad[i] * detail::diff(func_map, args()[i], s));
    }

    return sum(prod);
}

expression func::diff(detail::funcptr_map<expression> &func_map, const param &p) const
{
    // Run the specialised diff implementation,
    // if available.
    if (m_func->has_diff_par()) {
        return m_func->diff(func_map, p);
    }

    const auto arity = args().size();

    // Fetch the gradient.
    auto grad = m_func->fetch_gradient("parameter");

    // Compute the total derivative.
    std::vector<expression> prod;
    prod.reserve(arity);
    for (decltype(args().size()) i = 0; i < arity; ++i) {
        prod.push_back(grad[i] * detail::diff(func_map, args()[i], p));
    }

    return sum(prod);
}

std::vector<expression>::size_type func::decompose(detail::funcptr_map<std::vector<expression>::size_type> &func_map,
                                                   std::vector<expression> &dc) const
{
    const auto *const f_id = get_ptr();

    if (auto it = func_map.find(f_id); it != func_map.end()) {
        // We already decomposed the current function, fetch the result
        // from the cache.
        return it->second;
    }

    // Compute the decomposed arguments.
    const auto new_args = detail::func_d_args(*this, func_map, dc);

    // Make a copy of this containing the new arguments.
    auto f_copy = copy(new_args);

    // Append f_copy and return the index at which it was appended.
    const auto ret = dc.size();
    dc.emplace_back(std::move(f_copy));

    // Update the cache before exiting.
    [[maybe_unused]] const auto [_, flag] = func_map.emplace(f_id, ret);
    assert(flag);

    return ret;
}

// NOTE: time dependency here means **intrinsic** time
// dependence of the function. That is, we are not concerned
// with the arguments' time dependence and, e.g., cos(time)
// is **not** time-dependent according to this definition.
bool func::is_time_dependent() const
{
    return m_func->is_time_dependent();
}

expression func::normalise() const
{
    if (m_func->has_normalise()) {
        return m_func->normalise();
    } else {
        return expression{*this};
    }
}

const std::string &func::get_name() const
{
    return m_func->get_name();
}

void func::to_stream(std::ostringstream &oss) const
{
    m_func->to_stream(oss);
}

llvm::Value *func::llvm_eval(llvm_state &s, llvm::Type *fp_t, const std::vector<llvm::Value *> &eval_arr,
                             llvm::Value *par_ptr, llvm::Value *time_ptr, llvm::Value *stride, std::uint32_t batch_size,
                             bool high_accuracy) const
{
    return m_func->llvm_eval(s, fp_t, eval_arr, par_ptr, time_ptr, stride, batch_size, high_accuracy);
}

llvm::Function *func::llvm_c_eval_func(llvm_state &s, llvm::Type *fp_t, std::uint32_t batch_size,
                                       bool high_accuracy) const
{
    return m_func->llvm_c_eval_func(s, fp_t, batch_size, high_accuracy);
}

taylor_dc_t::size_type func::taylor_decompose(detail::funcptr_map<taylor_dc_t::size_type> &func_map,
                                              taylor_dc_t &dc) const
{
    const auto *const f_id = get_ptr();

    if (auto it = func_map.find(f_id); it != func_map.end()) {
        // We already decomposed the current function, fetch the result
        // from the cache.
        return it->second;
    }

    // Compute the decomposed arguments.
    const auto new_args = detail::func_td_args(*this, func_map, dc);

    // Make a copy of this containing the new arguments.
    auto f_copy = copy(new_args);

    // Run the decomposition.
    taylor_dc_t::size_type ret = 0;
    if (f_copy.m_func->has_taylor_decompose()) {
        // Custom implementation.
        ret = std::move(*f_copy.m_func).taylor_decompose(dc);
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

// NOTE: it is very important that the implementation here is kept exactly in sync
// with the implementation of expression hashing for non-recursive expressions. If one
// changes, the other must as well.
std::size_t func::hash(detail::funcptr_map<std::size_t> &func_map) const
{
    // NOTE: the hash value is computed by combining the hash values of:
    // - the function name,
    // - the arguments' hashes.
    std::size_t seed = std::hash<std::string>{}(get_name());

    for (const auto &arg : args()) {
        boost::hash_combine(seed, detail::hash(func_map, arg));
    }

    return seed;
}

bool operator==(const func &a, const func &b) noexcept
{
    // Check if the underlying object is the same.
    if (same_value(a.m_func, b.m_func)) {
        return true;
    }

    // NOTE: the comparison considers the function name and the arguments.
    return a.get_name() == b.get_name() && a.args() == b.args();
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
    assert(std::find(name.begin(), name.end(), '.') == name.end());

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

    const auto [fname, fargs] = llvm_c_eval_func_name_args(context, fp_t, name, batch_size, fb.args());

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
    }

    return f;
}

} // namespace detail

HEYOKA_END_NAMESPACE

// s11n implementation for null_func.
// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::null_func)
