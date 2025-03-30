// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_EOP_HPP
#define HEYOKA_MODEL_EOP_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/eop_data.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/s11n.hpp>

// NOTE: for the representation of EOP data, we adopt a piecewise linear approximation where the switch points are given
// by the dates in the eop dataset. Within each time interval, an EOP quantity is approximated as EOP(t) = c0 + c1*t
// (where the values of the c0 and c1 constants change from interval to interval). In the expression system, we
// implement, for each EOP quantity, two unary functions which return respectively the EOP quantity and its first-order
// derivative at the given input time.

HEYOKA_BEGIN_NAMESPACE

namespace model
{

namespace detail
{

class HEYOKA_DLL_PUBLIC eop_impl : public func_base
{
    std::string m_eop_name;
    // NOTE: we wrap the eop data into an optional because
    // we do not want to pay the cost of storing the full eop data
    // for a default-constructed object, which is anyway only used
    // during serialisation.
    std::optional<eop_data> m_eop_data;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    eop_impl();
    explicit eop_impl(std::string, expression, eop_data);

    [[nodiscard]] std::vector<expression> gradient() const;

    [[nodiscard]] llvm::Value *llvm_eval(llvm_state &, llvm::Type *, const std::vector<llvm::Value *> &, llvm::Value *,
                                         llvm::Value *, llvm::Value *, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *llvm_c_eval_func(llvm_state &, llvm::Type *, std::uint32_t, bool) const;

    [[nodiscard]] taylor_dc_t::size_type taylor_decompose(taylor_dc_t &) &&;

    [[nodiscard]] llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const std::vector<std::uint32_t> &,
                                           const std::vector<llvm::Value *> &, llvm::Value *, llvm::Value *,
                                           std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, bool) const;

    [[nodiscard]] llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, std::uint32_t, std::uint32_t,
                                                     bool) const;
};

class HEYOKA_DLL_PUBLIC eopp_impl : public func_base
{
    std::string m_eop_name;
    std::optional<eop_data> m_eop_data;

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    eopp_impl();
    explicit eopp_impl(std::string, expression, eop_data);

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

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *llvm_get_era_erap_func(llvm_state &, llvm::Type *, std::uint32_t,
                                                                       const eop_data &);

[[nodiscard]] HEYOKA_DLL_PUBLIC llvm::Function *
llvm_get_eop_func(llvm_state &, llvm::Type *, std::uint32_t, const eop_data &, const char *,
                  llvm::Value *(*)(llvm_state &, const eop_data &, llvm::Type *));

[[nodiscard]] HEYOKA_DLL_PUBLIC expression era_func_impl(expression, eop_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression erap_func_impl(expression, eop_data);

[[nodiscard]] HEYOKA_DLL_PUBLIC expression pm_x_func_impl(expression, eop_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression pm_xp_func_impl(expression, eop_data);

[[nodiscard]] HEYOKA_DLL_PUBLIC expression pm_y_func_impl(expression, eop_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression pm_yp_func_impl(expression, eop_data);

[[nodiscard]] HEYOKA_DLL_PUBLIC expression dX_func_impl(expression, eop_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression dXp_func_impl(expression, eop_data);

[[nodiscard]] HEYOKA_DLL_PUBLIC expression dY_func_impl(expression, eop_data);
[[nodiscard]] HEYOKA_DLL_PUBLIC expression dYp_func_impl(expression, eop_data);

template <typename... KwArgs>
auto eop_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "This function accepts only named arguments");

    // Time expression (defaults to heyoka::time).
    auto time_expr = [&p]() {
        if constexpr (p.has(kw::time_expr)) {
            return expression{p(kw::time_expr)};
        } else {
            return heyoka::time;
        }
    }();

    // EOP data (defaults to def-cted).
    auto data = [&p]() {
        if constexpr (p.has(kw::eop_data)) {
            return p(kw::eop_data);
        } else {
            return eop_data{};
        }
    }();

    return std::tuple{std::move(time_expr), std::move(data)};
}

} // namespace detail

inline constexpr auto era = [](const auto &...kw_args) -> expression {
    return std::apply(detail::era_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto erap = [](const auto &...kw_args) -> expression {
    return std::apply(detail::erap_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto pm_x = [](const auto &...kw_args) -> expression {
    return std::apply(detail::pm_x_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto pm_xp = [](const auto &...kw_args) -> expression {
    return std::apply(detail::pm_xp_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto pm_y = [](const auto &...kw_args) -> expression {
    return std::apply(detail::pm_y_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto pm_yp = [](const auto &...kw_args) -> expression {
    return std::apply(detail::pm_yp_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto dX = [](const auto &...kw_args) -> expression {
    return std::apply(detail::dX_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto dXp = [](const auto &...kw_args) -> expression {
    return std::apply(detail::dXp_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto dY = [](const auto &...kw_args) -> expression {
    return std::apply(detail::dY_func_impl, detail::eop_common_opts(kw_args...));
};

inline constexpr auto dYp = [](const auto &...kw_args) -> expression {
    return std::apply(detail::dYp_func_impl, detail::eop_common_opts(kw_args...));
};

} // namespace model

HEYOKA_END_NAMESPACE

HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::eop_impl)
HEYOKA_S11N_FUNC_EXPORT_KEY(heyoka::model::detail::eopp_impl)

#endif
