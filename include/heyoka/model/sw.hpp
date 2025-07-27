// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_SW_HPP
#define HEYOKA_MODEL_SW_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/sw_data.hpp>

// NOTE: for the representation of SW data, we adopt a piecewise constant approximation where the switch points are
// given by the dates in the sw dataset. Within each time interval, an SW quantity is approximated as SW(t) = c0 (where
// the value of the c0 constant changes from interval to interval). In the expression system, we implement, for
// each SW quantity, an unary function returning the SW quantity at the given input time.

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

class HEYOKA_DLL_PUBLIC sw_impl : public func_base
{
    std::string m_sw_name;
    // NOTE: we wrap the sw data into an optional because
    // we do not want to pay the cost of storing the full sw data
    // for a default-constructed object, which is anyway only used
    // during serialisation.
    std::optional<sw_data> m_sw_data;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    sw_impl();
    explicit sw_impl(std::string, expression, sw_data);

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

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *
llvm_get_sw_func(llvm_state &, llvm::Type *, std::uint32_t, const sw_data &, const char *,
                 llvm::Value *(*)(llvm_state &, const sw_data &, llvm::Type *));

[[nodiscard]] HEYOKA_DLL_PUBLIC expression Ap_avg_func_impl(expression, sw_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression f107_func_impl(expression, sw_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression f107a_center81_func_impl(expression, sw_data);

template <typename... KwArgs>
auto sw_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Time expression (defaults to heyoka::time).
    auto time_expr = expression(p(kw::time_expr, heyoka::time));

    // SW data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::sw_data)) {
            return p(kw::sw_data);
        } else {
            return sw_data{};
        }
    }();

    return std::tuple{std::move(time_expr), std::move(data)};
}

} // namespace detail

inline constexpr auto sw_kw_cfg = igor::config<kw::descr::constructible_from<expression, kw::time_expr>,
                                               kw::descr::same_as<kw::sw_data, sw_data>>{};

inline constexpr auto Ap_avg = []<typename... KwArgs>
    requires igor::validate<sw_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::Ap_avg_func_impl, detail::sw_common_opts(kw_args...));
};

inline constexpr auto f107 = []<typename... KwArgs>
    requires igor::validate<sw_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression { return std::apply(detail::f107_func_impl, detail::sw_common_opts(kw_args...)); };

inline constexpr auto f107a_center81 = []<typename... KwArgs>
    requires igor::validate<sw_kw_cfg, KwArgs...>
// NOTLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
(KwArgs &&...kw_args) -> expression {
    return std::apply(detail::f107a_center81_func_impl, detail::sw_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::sw_impl)

#endif
