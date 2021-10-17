// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
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

} // namespace heyoka::detail
