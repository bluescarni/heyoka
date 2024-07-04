// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_MODEL_SGP4_HPP
#define HEYOKA_MODEL_SGP4_HPP

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/mdspan.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

HEYOKA_DLL_PUBLIC std::vector<expression> sgp4();

namespace detail
{

// Small helper struct only used to store the results of sgp4_build_funcs().
// It contains the init/tprop vector functions and their input variables,
// plus the derivatives of the Cartesian state wrt the original elements
// as a dtens object (if derivatives are requested).
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS sgp4_prop_funcs {
    std::pair<std::vector<expression>, std::vector<expression>> init;
    std::pair<std::vector<expression>, std::vector<expression>> tprop;
    std::optional<dtens> dt;
};

HEYOKA_DLL_PUBLIC sgp4_prop_funcs sgp4_build_funcs(std::uint32_t);

HEYOKA_DLL_PUBLIC void sgp4_compile_funcs(const std::function<void()> &, const std::function<void()> &);

} // namespace detail

template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS sgp4_propagator
{
    struct impl;

    std::unique_ptr<impl> m_impl;

    template <typename Input, typename... KwArgs>
    static auto parse_ctor_args(const Input &in, const KwArgs &...kw_args)
    {
        if (in.extent(1) == 0u) [[unlikely]] {
            throw std::invalid_argument("Cannot initialise an sgp4_propagator with an empty list of satellites");
        }

        // Make own copy of the TLE elements + bstar + epoch for each satellite.
        std::vector<T> sat_buffer;
        sat_buffer.reserve(boost::safe_numerics::safe<decltype(sat_buffer.size())>(in.extent(1)) * 9);
        for (auto i = 0u; i < 9u; ++i) {
            for (std::size_t j = 0u; j < in.extent(1); ++j) {
                sat_buffer.push_back(in(i, j));
            }
        }

        igor::parser p{kw_args...};

        // Differentiation order (defaults to zero, no derivatives).
        std::uint32_t order = 0;
        if constexpr (p.has(kw::diff_order)) {
            if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::diff_order))>>) {
                order = boost::numeric_cast<std::uint32_t>(p(kw::diff_order));
            } else {
                static_assert(heyoka::detail::always_false_v<KwArgs...>,
                              "The diff_order keyword argument must be of integral type.");
            }
        }

        // Build the functions to be compiled.
        auto funcs = detail::sgp4_build_funcs(order);

        // Compile them.
        // NOTE: it is important here to use as_const_kwarg() so that we don't end up with moved-from
        // keyword arguments the second time we use kw_args.
        // NOTE: in order to perform parallel compilation, we need to use the sgp4_compile_funcs() helper
        // because we want to avoid having TBB in the public interface.
        cfunc<T> cf_init, cf_prop;
        detail::sgp4_compile_funcs(
            [&]() {
                cf_init = cfunc<T>(std::move(funcs.init.first), std::move(funcs.init.second),
                                   igor::as_const_kwarg(kw_args)...);
            },
            [&]() {
                cf_prop = cfunc<T>(std::move(funcs.tprop.first), std::move(funcs.tprop.second),
                                   igor::as_const_kwarg(kw_args)...);
            });

        return std::make_tuple(std::move(sat_buffer), std::move(cf_init), std::move(cf_prop), std::move(funcs.dt));
    }
    struct ptag {
    };
    explicit sgp4_propagator(ptag, std::tuple<std::vector<T>, cfunc<T>, cfunc<T>, std::optional<dtens>>);

public:
    // Julian date with fractional correction.
    struct date {
        T jd = 0;
        T frac = 0;
    };

    sgp4_propagator() noexcept;
    template <typename LayoutPolicy, typename AccessorPolicy, typename... KwArgs>
        requires(!igor::has_unnamed_arguments<KwArgs...>())
    explicit sgp4_propagator(
        mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>, LayoutPolicy, AccessorPolicy> sat_list,
        const KwArgs &...kw_args)
        : sgp4_propagator(ptag{}, parse_ctor_args(sat_list, kw_args...))
    {
    }
    sgp4_propagator(const sgp4_propagator &);
    sgp4_propagator(sgp4_propagator &&) noexcept;
    sgp4_propagator &operator=(const sgp4_propagator &);
    sgp4_propagator &operator=(sgp4_propagator &&) noexcept;
    ~sgp4_propagator();

    [[nodiscard]] std::uint32_t get_n_sats() const;

    template <typename U>
    using in_1d = mdspan<const U, dextents<std::size_t, 1>>;
    template <typename U>
    using in_2d = mdspan<const U, dextents<std::size_t, 2>>;
    using out_2d = mdspan<T, dextents<std::size_t, 2>>;
    using out_3d = mdspan<T, dextents<std::size_t, 3>>;
    void operator()(out_2d, in_1d<T>);
    void operator()(out_2d, in_1d<date>);
    void operator()(out_3d, in_2d<T>);
    void operator()(out_3d, in_2d<date>);
};

// Prevent implicit instantiations.
extern template class sgp4_propagator<float>;
extern template class sgp4_propagator<double>;

} // namespace model

HEYOKA_END_NAMESPACE

#endif
