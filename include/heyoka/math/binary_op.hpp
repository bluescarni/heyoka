// Copyright 2020, 2021, 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_BINARY_OP_HPP
#define HEYOKA_MATH_BINARY_OP_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

namespace heyoka
{

namespace detail
{

class HEYOKA_DLL_PUBLIC binary_op : public func_base
{
public:
    enum class type { add, sub, mul, div };

private:
    type m_type;

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
        ar &m_type;
    }

    template <typename T>
    HEYOKA_DLL_LOCAL expression diff_impl(std::unordered_map<const void *, expression> &, const T &) const;

public:
    binary_op();
    explicit binary_op(type, expression, expression);

    void to_stream(std::ostream &) const;

    bool extra_equal_to(const func &) const;

    std::size_t extra_hash() const;

    type op() const;
    const expression &lhs() const;
    const expression &rhs() const;

    expression diff(std::unordered_map<const void *, expression> &, const std::string &) const;
    expression diff(std::unordered_map<const void *, expression> &, const param &) const;

    double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    long double eval_ldbl(const std::unordered_map<std::string, long double> &, const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)
    mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                            const std::vector<mppp::real128> &) const;
#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const;
};

} // namespace detail

HEYOKA_DLL_PUBLIC expression add(expression, expression);

HEYOKA_DLL_PUBLIC expression sub(expression, expression);

HEYOKA_DLL_PUBLIC expression mul(expression, expression);

HEYOKA_DLL_PUBLIC expression div(expression, expression);

} // namespace heyoka

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::binary_op)

#endif
