// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_DAYFRAC_HPP
#define HEYOKA_MODEL_DAYFRAC_HPP

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

// Implementation of dayfrac().
class HEYOKA_DLL_PUBLIC dayfrac_impl : public func_base
{
    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    dayfrac_impl();
    explicit dayfrac_impl(expression);

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

HEYOKA_DLL_PUBLIC expression dayfrac_func_impl(expression);

template <typename... KwArgs>
auto dayfrac_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    return std::tuple{std::move(time_expr)};
}

} // namespace detail

inline constexpr auto dayfrac_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>>{};

// Function to transform the input expression into the number of days elapsed since January 1st.
//
// The input time expression tt is assumed to represent the number of TT days elapsed since the epoch of J2000. The
// return expression evaluates to the number of TT days elapsed since January 1st 00:00 UTC of the calendar year of tt.
inline constexpr auto dayfrac = []<typename... KwArgs>
    requires igor::validate<dayfrac_kw_cfg, KwArgs...>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression { return std::apply(detail::dayfrac_func_impl, detail::dayfrac_opts(kw_args...)); };

} // namespace model

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::dayfrac_impl)

#endif
