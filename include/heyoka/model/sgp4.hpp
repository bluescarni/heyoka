// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/config.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model
{

HEYOKA_DLL_PUBLIC std::vector<expression> sgp4(const std::vector<expression> & = {});

HEYOKA_DLL_PUBLIC bool gpe_is_deep_space(double, double, double);

namespace detail
{

// Small helper struct only used to store the results of sgp4_build_funcs(). It contains the init/tprop vector functions
// and their input variables, plus the derivatives of the Cartesian state wrt the original elements as a dtens object
// (if derivatives are requested).
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS sgp4_prop_funcs {
    std::pair<std::vector<expression>, std::vector<expression>> init;
    std::pair<std::vector<expression>, std::vector<expression>> tprop;
    std::optional<dtens> dt;
};

HEYOKA_DLL_PUBLIC sgp4_prop_funcs sgp4_build_funcs(std::uint32_t);

HEYOKA_DLL_PUBLIC void sgp4_compile_funcs(const std::function<void()> &, const std::function<void()> &);

} // namespace detail

// NOTE: a couple of ideas for performance improvements:
//
// - simultaneous computation of sin/cos for SLEEF,
// - partitioning of the satellite list into simplified (perigee < 220km) and non-simplified dynamics. This would allow
//   to get rid of the select() calls in the time propagation function, as we would then have 2 different functions for
//   the simplified and non-simplified tprop. Getting rid of the select()s would allow to avoid unnecessary
//   computations. The issue with this approach is that we would need to alter the original ordering of the satellites.
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

        const igor::parser p{kw_args...};

        // Differentiation order (defaults to zero, no derivatives).
        const auto order = boost::numeric_cast<std::uint32_t>(p(kw::diff_order, 0));

        // Build the functions to be compiled.
        auto funcs = detail::sgp4_build_funcs(order);

        // Compile them.
        //
        // NOTE: in order to perform parallel compilation, we need to use the sgp4_compile_funcs() helper because we
        // want to avoid having TBB in the public interface.
        //
        // NOTE: thanks to the checks in llvm_state and cfunc, we know that the keyword arguments are safe for multiple
        // and concurrent usages.
        cfunc<T> cf_init, cf_prop;
        detail::sgp4_compile_funcs(
            [&cf_init, &funcs, &kw_args...]() {
                cf_init = igor::filter_invoke<cfunc<T>::ctor_kw_cfg>(
                    [&funcs](const auto &...args) {
                        return cfunc<T>(std::move(funcs.init.first), std::move(funcs.init.second), args...);
                    },
                    kw_args...);
            },
            [&cf_prop, &funcs, &kw_args...]() {
                cf_prop = igor::filter_invoke<cfunc<T>::ctor_kw_cfg>(
                    [&funcs](const auto &...args) {
                        return cfunc<T>(std::move(funcs.tprop.first), std::move(funcs.tprop.second), args...);
                    },
                    kw_args...);
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

    // kwargs configuration for the constructor.
    static constexpr auto ctor_kw_cfg = cfunc<T>::ctor_kw_cfg | igor::config<kw::descr::integral<kw::diff_order>>{};

    // NOTE: the GPE data is expected as a 9 x n span, where n is the number of satellites and the rows represent:
    //
    // - the mean motion (in [rad / min]),
    // - the eccentricity,
    // - the inclination (in [rad]),
    // - the right ascension of the ascending node (in [rad]),
    // - the argument of perigee (in [rad]),
    // - the mean anomaly (in [rad]),
    // - the BSTAR drag term (in whatever unit is used by the SGP4 GPEs),
    // - the reference epoch (as a Julian date),
    // - a fractional correction to the epoch (in Julian days).
    //
    // Julian dates are to be provided in the UTC scale of time. Internal conversion to TAI will ensure correct
    // propagation across leap seconds.
    template <typename LayoutPolicy, typename AccessorPolicy, typename... KwArgs>
        requires igor::validate<ctor_kw_cfg, KwArgs...>
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

    // NOTE: it is important to document the weak exception safety guarantee here: in the (very unlikely) case that
    // invoking the init cfunc throws (due to TBB primitives failing), we may end up with inconsistent data in the
    // propagator that will lead to wrong results for successive invocations of the call operator. I think this is an ok
    // compromise for the sake of performance: providing strong exception safety is trivial if we make temp copy of the
    // internal data, but that means allocating. If this becomes a problem, we may think of a more complicated scheme
    // involving internal temp buffers in order to provide better exception safety.
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
    // NOTE: it is important to document properly the non-overlapping memory requirement for the input arguments.
    //
    // NOTE: Julian dates are to be provided in the UTC scale of time. Internal conversion to TAI will ensure correct
    // propagation across leap seconds.
    void operator()(out_2d, in_1d<T>) const;
    void operator()(out_2d, in_1d<date>) const;
    void operator()(out_3d, in_2d<T>) const;
    void operator()(out_3d, in_2d<date>) const;
};

// Prevent implicit instantiations.
extern template class sgp4_propagator<float>;
extern template class sgp4_propagator<double>;

// Helpers to convert between UTC and TAI Julian dates.
HEYOKA_DLL_PUBLIC std::pair<double, double> jd_utc_to_tai(double, double);
HEYOKA_DLL_PUBLIC std::pair<double, double> jd_tai_to_utc(double, double);

namespace detail
{

// Boost s11n class version history for the sgp4 class:
//
// - 1: removed the temporary internal buffers in favour of thread_local buffers.
inline constexpr int sgp4_propagator_s11n_version = 1;

} // namespace detail

} // namespace model

HEYOKA_END_NAMESPACE

// Set the Boost s11n class version for the sgp4_propagator class.
BOOST_CLASS_VERSION(heyoka::model::sgp4_propagator<float>, heyoka::model::detail::sgp4_propagator_s11n_version);
BOOST_CLASS_VERSION(heyoka::model::sgp4_propagator<double>, heyoka::model::detail::sgp4_propagator_s11n_version);

#endif
