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
#include <span>
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
#include <heyoka/s11n.hpp>

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

// NOTE: a couple of ideas for performance improvements:
// - simultaneous computation of sin/cos for SLEEF,
// - partitioning of the satellite list into simplified
//   (perigee < 220km) and non-simplified dynamics. This would
//   allow to get rid of the select() calls in the time propagation
//   function, as we would then have 2 different functions for the simplified
//   and non-simplified tprop. Getting rid of the select()s would allow
//   to avoid unnecessary computations. The issue with this approach
//   is that we would need to alter the original ordering of the satellites.
//   Perhaps a similar approach could work for the deep space part too?
template <typename T>
    requires std::same_as<T, double> || std::same_as<T, float>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS sgp4_propagator
{
    struct impl;

    std::unique_ptr<impl> m_impl;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    template <typename Input, typename... KwArgs>
    static auto parse_ctor_args(const Input &in, const KwArgs &...kw_args)
    {
        if (in.data_handle() == nullptr) [[unlikely]] {
            throw std::invalid_argument("Cannot initialise an sgp4_propagator with a null list of satellites");
        }
        if (in.extent(1) == 0u) [[unlikely]] {
            throw std::invalid_argument("Cannot initialise an sgp4_propagator with an empty list of satellites");
        }

        // Make own copy of the TLE elements + bstar + epoch for each satellite.
        std::vector<T> sat_buffer;
        sat_buffer.reserve(boost::safe_numerics::safe<decltype(sat_buffer.size())>(in.extent(1)) * 9);
        for (auto i = 0u; i < 9u; ++i) {
            for (std::size_t j = 0; j < in.extent(1); ++j) {
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
        // NOTE: in order to perform parallel compilation, we need to use the sgp4_compile_funcs() helper
        // because we want to avoid having TBB in the public interface.
        cfunc<T> cf_init, cf_prop;
        detail::sgp4_compile_funcs(
            [&]() {
                // NOTE: it is important here to use as_const_kwarg() so that we don't end up with moved-from
                // keyword arguments the second time we use kw_args.
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

    HEYOKA_DLL_LOCAL void check_with_diff(const char *) const;

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

    [[nodiscard]] std::uint32_t get_nsats() const;
    [[nodiscard]] std::uint32_t get_nouts() const noexcept;
    [[nodiscard]] mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>> get_sat_data() const;

    void replace_sat_data(mdspan<const T, extents<std::size_t, 9, std::dynamic_extent>>);

    [[nodiscard]] std::uint32_t get_diff_order() const noexcept;
    [[nodiscard]] const std::vector<expression> &get_diff_args() const;
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_dslice(std::uint32_t) const;
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> get_dslice(std::uint32_t, std::uint32_t) const;
    [[nodiscard]] const dtens::sv_idx_t &get_mindex(std::uint32_t) const;

    template <typename U>
    using in_1d = mdspan<const U, dextents<std::size_t, 1>>;
    template <typename U>
    using in_2d = mdspan<const U, dextents<std::size_t, 2>>;
    using out_2d = mdspan<T, dextents<std::size_t, 2>>;
    using out_3d = mdspan<T, dextents<std::size_t, 3>>;
    // NOTE: it is important to document properly the non-overlapping
    // memory requirement for the input arguments.
    // NOTE: because of the use of an internal buffer to convert
    // dates to tsinces, the date overloads of these operators are
    // never thread-safe. This needs to be documented properly.
    // Perhaps this can be fixed one day if we implement the conversion
    // to dates on-the-fly via double-length primitives.
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
