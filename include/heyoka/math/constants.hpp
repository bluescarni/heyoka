// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_CONSTANTS_HPP
#define HEYOKA_MATH_CONSTANTS_HPP

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <typeindex>
#include <vector>

#include <heyoka/callable.hpp>
#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// This is the null string function, used in the default
// constructor of the constant class.
class HEYOKA_DLL_PUBLIC null_constant_func
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }

public:
    std::string operator()(unsigned) const;
};

// Implementation of the pi string function.
class HEYOKA_DLL_PUBLIC pi_constant_func
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }

public:
    std::string operator()(unsigned) const;
};

} // namespace detail

// NOTE: like for user-defined functions, here we are relying
// on the user to remember to give different names to different
// constants in order for the comparison/hashing/etc. primitives
// to make sense. If/when we implement a mechanism for automatic
// function name registration to prevent naming collisions, we should
// do the same for constants.
class HEYOKA_DLL_PUBLIC constant : public func_base
{
public:
    using str_func_t = callable<std::string(unsigned)>;

private:
    // NOTE: this mutable is a bit unfortunate but we will
    // have to live with this for the time being. Fixing this
    // needs support for const qualified callables.
    mutable str_func_t m_str_func;
    std::optional<std::string> m_repr;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    [[nodiscard]] HEYOKA_DLL_LOCAL llvm::Constant *make_llvm_const(llvm_state &, llvm::Type *) const;

public:
    constant();
    explicit constant(std::string, str_func_t, std::optional<std::string> = {});

    [[nodiscard]] std::type_index get_str_func_t() const;

    [[nodiscard]] std::string operator()(unsigned) const;

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                           const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                           std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool) const;
};

HEYOKA_DLL_PUBLIC extern const expression pi;

HEYOKA_END_NAMESPACE

HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::null_constant_func, std::string, unsigned)

HEYOKA_S11N_CALLABLE_EXPORT_KEY(heyoka::detail::pi_constant_func, std::string, unsigned)

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::constant)

#endif
