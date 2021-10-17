// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka::detail
{

namespace
{

std::string taylor_c_diff_mangle(const variable &)
{
    return "var";
}

std::string taylor_c_diff_mangle(const number &)
{
    return "num";
}

std::string taylor_c_diff_mangle(const param &)
{
    return "par";
}

} // namespace

// NOTE: precondition on name: must be conforming to LLVM requirements for
// function names, and must not contain "." (as we use it as a separator in
// the mangling scheme).
std::pair<std::string, std::vector<llvm::Type *>>
taylor_c_diff_func_name_args_impl(llvm::LLVMContext &context, const std::string &name, llvm::Type *val_t,
                                  std::uint32_t n_uvars, const std::vector<std::variant<variable, number, param>> &args,
                                  std::uint32_t n_hidden_deps)
{
    assert(val_t != nullptr);
    assert(n_uvars > 0u);

    // Init the name.
    auto fname = "heyoka.taylor_c_diff.{}."_format(name);

    // Init the vector of arguments:
    // - diff order,
    // - idx of the u variable whose diff is being computed,
    // - diff array (pointer to val_t),
    // - par ptr (pointer to scalar),
    // - time ptr (pointer to scalar).
    std::vector<llvm::Type *> fargs{
        llvm::Type::getInt32Ty(context), llvm::Type::getInt32Ty(context), llvm::PointerType::getUnqual(val_t),
        llvm::PointerType::getUnqual(val_t->getScalarType()), llvm::PointerType::getUnqual(val_t->getScalarType())};

    // Add the mangling and LLVM arg types for the argument types. Also, detect if
    // we have variables in the arguments.
    bool with_var = false;
    for (decltype(args.size()) i = 0; i < args.size(); ++i) {
        // Detect variable.
        if (std::holds_alternative<variable>(args[i])) {
            with_var = true;
        }

        // Name mangling.
        fname += std::visit([](const auto &v) { return taylor_c_diff_mangle(v); }, args[i]);

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
                } else {
                    // For vars and params, the argument is an index
                    // in an array.
                    return llvm::Type::getInt32Ty(context);
                }
            },
            args[i]));
    }

    // Close the argument list with a ".".
    // NOTE: this will result in a ".." in the name
    // if the function has zero arguments.
    fname += '.';

    // If we have variables in the arguments, add mangling
    // for n_uvars.
    if (with_var) {
        fname += "n_uvars_{}."_format(n_uvars);
    }

    // Finally, add the mangling for the floating-point type.
    fname += llvm_mangle_type(val_t);

    // Fill in the hidden dependency arguments. These are all indices.
    fargs.insert(fargs.end(), boost::numeric_cast<decltype(fargs.size())>(n_hidden_deps),
                 llvm::Type::getInt32Ty(context));

    return std::make_pair(std::move(fname), std::move(fargs));
}

namespace
{

template <typename T>
llvm::Value *taylor_codegen_numparam_num(llvm_state &s, const number &num, std::uint32_t batch_size)
{
    return vector_splat(s.builder(), codegen<T>(s, num), batch_size);
}

llvm::Value *taylor_codegen_numparam_par(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    assert(batch_size > 0u);

    auto &builder = s.builder();

    // Determine the index into the parameter array.
    // LCOV_EXCL_START
    if (p.idx() > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("Overflow detected in the computation of the index into a parameter array");
    }
    // LCOV_EXCL_STOP
    const auto arr_idx = static_cast<std::uint32_t>(p.idx() * batch_size);

    // Compute the pointer to load from.
    auto *ptr = builder.CreateInBoundsGEP(par_ptr, {builder.getInt32(arr_idx)});

    // Load.
    return load_vector_from_memory(builder, ptr, batch_size);
}

} // namespace

llvm::Value *taylor_codegen_numparam_dbl(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<double>(s, num, batch_size);
}

llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<long double>(s, num, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_codegen_numparam_f128(llvm_state &s, const number &num, llvm::Value *, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_num<mppp::real128>(s, num, batch_size);
}

#endif

llvm::Value *taylor_codegen_numparam_dbl(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

llvm::Value *taylor_codegen_numparam_ldbl(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *taylor_codegen_numparam_f128(llvm_state &s, const param &p, llvm::Value *par_ptr, std::uint32_t batch_size)
{
    return taylor_codegen_numparam_par(s, p, par_ptr, batch_size);
}

#endif

// Codegen helpers for number/param for use in the generic c_diff implementations.
llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, const number &, llvm::Value *n, llvm::Value *,
                                            std::uint32_t batch_size)
{
    return vector_splat(s.builder(), n, batch_size);
}

llvm::Value *taylor_c_diff_numparam_codegen(llvm_state &s, const param &, llvm::Value *p, llvm::Value *par_ptr,
                                            std::uint32_t batch_size)
{
    auto &builder = s.builder();

    // Fetch the pointer into par_ptr.
    // NOTE: the overflow check is done in taylor_compute_jet().
    auto *ptr = builder.CreateInBoundsGEP(par_ptr, {builder.CreateMul(p, builder.getInt32(batch_size))});

    return load_vector_from_memory(builder, ptr, batch_size);
}

// Helper to fetch the derivative of order 'order' of the u variable at index u_idx from the
// derivative array 'arr'. The total number of u variables is n_uvars.
llvm::Value *taylor_fetch_diff(const std::vector<llvm::Value *> &arr, std::uint32_t u_idx, std::uint32_t order,
                               std::uint32_t n_uvars)
{
    // Sanity check.
    assert(u_idx < n_uvars);

    // Compute the index.
    const auto idx = static_cast<decltype(arr.size())>(order) * n_uvars + u_idx;
    assert(idx < arr.size());

    return arr[idx];
}

// Load the derivative of order 'order' of the u variable u_idx from the array of Taylor derivatives diff_arr.
// n_uvars is the total number of u variables.
llvm::Value *taylor_c_load_diff(llvm_state &s, llvm::Value *diff_arr, std::uint32_t n_uvars, llvm::Value *order,
                                llvm::Value *u_idx)
{
    auto &builder = s.builder();

    // NOTE: overflow check has already been done to ensure that the
    // total size of diff_arr fits in a 32-bit unsigned integer.
    auto *ptr = builder.CreateInBoundsGEP(
        diff_arr, {builder.CreateAdd(builder.CreateMul(order, builder.getInt32(n_uvars)), u_idx)});

    return builder.CreateLoad(ptr);
}

} // namespace heyoka::detail
