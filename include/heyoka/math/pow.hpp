// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MATH_POW_HPP
#define HEYOKA_MATH_POW_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

class HEYOKA_DLL_PUBLIC pow_impl : public func_base
{
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<func_base>(*this);
    }

public:
    pow_impl();
    explicit pow_impl(expression, expression);

    void to_stream(std::ostringstream &) const;

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] expression normalise() const;

    [[nodiscard]] double eval_dbl(const std::unordered_map<std::string, double> &, const std::vector<double> &) const;
    [[nodiscard]] long double eval_ldbl(const std::unordered_map<std::string, long double> &,
                                        const std::vector<long double> &) const;
#if defined(HEYOKA_HAVE_REAL128)

    [[nodiscard]] mppp::real128 eval_f128(const std::unordered_map<std::string, mppp::real128> &,
                                          const std::vector<mppp::real128> &) const;

#endif

    void eval_batch_dbl(std::vector<double> &, const std::unordered_map<std::string, std::vector<double>> &,
                        const std::vector<double> &) const;
    [[nodiscard]] double eval_num_dbl(const std::vector<double> &) const;
    [[nodiscard]] double deval_num_dbl(const std::vector<double> &, std::vector<double>::size_type) const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                             const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *, std::uint32_t,
                             std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t, bool) const;
};

// NOTE: this struct stores a std::function for the evaluation (in LLVM) of an exponentiation.
// In the general case, the exponentiation is performed via a direct call to llvm_pow(). If the
// exponent is a small integral or a small integral half, then the exponentiation is implemented
// via multiplications, divisions and calls to llvm_sqrt(). The 'algo' enum signals the selected
// implementation. The 'exp' member is empty in the general case. Otherwise, it contains either
// the integral exponent (for the small integral optimisation), or twice the small integral half
// exponent (for the small integral half optimisation). The 'suffix' member contains a string
// uniquely encoding the 'algo' and 'exp' data members.
struct pow_eval_algo {
    enum class type : int { general, pos_small_int, neg_small_int, pos_small_half, neg_small_half };

    using eval_t = std::function<llvm::Value *(llvm_state &, const std::vector<llvm::Value *> &)>;

    type algo{-1};
    eval_t eval_f;
    std::optional<boost::safe_numerics::safe<std::int64_t>> exp;
    std::string suffix;
};

pow_eval_algo get_pow_eval_algo(const pow_impl &);

} // namespace detail

HEYOKA_DLL_PUBLIC expression pow(expression, expression);
HEYOKA_DLL_PUBLIC expression pow(expression, float);
HEYOKA_DLL_PUBLIC expression pow(expression, double);
HEYOKA_DLL_PUBLIC expression pow(expression, long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC expression pow(expression, mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DLL_PUBLIC expression pow(expression, mppp::real);

#endif

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::detail::pow_impl)

#endif
