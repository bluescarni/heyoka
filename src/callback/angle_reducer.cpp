// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/serialization/unordered_set.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callback/angle_reducer.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

// NOTE: possible improvements:
// - JIT compiled implementation (but how to deal
//   with batch mode? recompile in pre-hook as needed?);
// - double-length implementation to improve the accuracy
//   of the reduction;
// - clamp result to ensure with 100% certainty that
//   the reduced angle falls within the range;
// - allow for other ranges, such as [-pi / pi).

HEYOKA_BEGIN_NAMESPACE

namespace callback
{

class angle_reducer::impl
{
public:
    std::unordered_set<expression> m_var_set;
    std::vector<std::size_t> m_ind;

private:
    // Serialisation.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_var_set;
        ar & m_ind;
    }

    impl() = default;

public:
    explicit impl(std::unordered_set<expression> var_set) : m_var_set(std::move(var_set)) {}
};

angle_reducer::angle_reducer() = default;

angle_reducer::angle_reducer(const angle_reducer &other)
    : m_impl(other.m_impl ? std::make_unique<impl>(*other.m_impl) : nullptr)
{
}

angle_reducer::angle_reducer(angle_reducer &&) noexcept = default;

angle_reducer &angle_reducer::operator=(const angle_reducer &other)
{
    if (this != &other) {
        *this = angle_reducer(other);
    }

    return *this;
}

angle_reducer &angle_reducer::operator=(angle_reducer &&) noexcept = default;

angle_reducer::~angle_reducer() = default;

void angle_reducer::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

void angle_reducer::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        ar >> m_impl;
        // LCOV_EXCL_START
    } catch (...) {
        *this = angle_reducer{};
        throw;
    }
    // LCOV_EXCL_STOP
}

void angle_reducer::validate_and_construct(std::unordered_set<expression> var_set)
{
    if (var_set.empty()) {
        throw std::invalid_argument(
            "The list of expressions passed to the constructor of angle_reducer cannot be empty");
    }

    if (std::ranges::any_of(var_set, [](const auto &ex) { return !std::holds_alternative<variable>(ex.value()); })) {
        throw std::invalid_argument(
            "The list of expressions passed to the constructor of angle_reducer can contain only variables");
    }

    m_impl = std::make_unique<impl>(std::move(var_set));
}

namespace detail
{

namespace
{

constexpr auto invalid_ar_msg = "Cannot use an angle_reducer which was default-constructed or moved-from";

} // namespace

} // namespace detail

template <typename TA>
void angle_reducer::pre_hook_impl(TA &ta)
{
    if (!m_impl) {
        throw std::invalid_argument(detail::invalid_ar_msg);
    }

    // Reset and rebuild the vector of indices.
    auto &ind_v = m_impl->m_ind;
    const auto &var_set = m_impl->m_var_set;

    ind_v.clear();
    for (decltype(ta.get_state_vars().size()) i = 0; i < ta.get_state_vars().size(); ++i) {
        const auto &ex = ta.get_state_vars()[i];

        assert(std::holds_alternative<variable>(ex.value()));

        if (var_set.contains(ex)) {
            ind_v.push_back(boost::numeric_cast<std::size_t>(i));
        }
    }
}

template <typename T>
void angle_reducer::pre_hook(taylor_adaptive<T> &ta)
{
    pre_hook_impl(ta);
}

template <typename T>
void angle_reducer::pre_hook(taylor_adaptive_batch<T> &ta)
{
    pre_hook_impl(ta);
}

namespace detail
{

namespace
{

// Helpers to fetch the constant 2*pi.
template <typename T>
T get_twopi_const(const taylor_adaptive<T> &)
{
    return 2 * boost::math::constants::pi<T>();
}

template <typename T>
T get_twopi_const(const taylor_adaptive_batch<T> &)
{
    return 2 * boost::math::constants::pi<T>();
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 get_twopi_const(const taylor_adaptive<mppp::real128> &)
{
    return 2 * mppp::pi_128;
}

template <>
mppp::real128 get_twopi_const(const taylor_adaptive_batch<mppp::real128> &)
{
    return 2 * mppp::pi_128;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
mppp::real get_twopi_const(const taylor_adaptive<mppp::real> &ta)
{
    auto ret = mppp::real_pi(ta.get_prec());
    mul_2ui(ret, ret, 1u);

    return ret;
} // LCOV_EXCL_LINE

#endif

} // namespace

} // namespace detail

template <typename T>
bool angle_reducer::operator()(taylor_adaptive<T> &ta)
{
    if (!m_impl) {
        throw std::invalid_argument(detail::invalid_ar_msg);
    }

    // Validate m_ind against the integrator object.
    const auto &ind_v = m_impl->m_ind;
    assert(std::ranges::is_sorted(ind_v));
    if (!ind_v.empty() && ind_v.back() >= ta.get_state_vars().size()) {
        throw std::invalid_argument(
            fmt::format("Inconsistent state detected in angle_reducer: the last index in the indices vector has a "
                        "value of {}, but the number of state variables is only {}",
                        ind_v.back(), ta.get_state_vars().size()));
    }

    // Fetch the 2pi constant.
    const auto twopi_const = detail::get_twopi_const(ta);

    // Run the reduction.
    auto *sptr = ta.get_state_data();
    for (const auto ind : ind_v) {
        using std::floor;
        sptr[ind] -= twopi_const * floor(sptr[ind] / twopi_const);
    }

    return true;
}

template <typename T>
bool angle_reducer::operator()(taylor_adaptive_batch<T> &ta)
{
    if (!m_impl) {
        throw std::invalid_argument(detail::invalid_ar_msg);
    }

    // Validate m_ind against the integrator object.
    const auto &ind_v = m_impl->m_ind;
    assert(std::ranges::is_sorted(ind_v));
    if (!ind_v.empty() && ind_v.back() >= ta.get_state_vars().size()) {
        throw std::invalid_argument(
            fmt::format("Inconsistent state detected in angle_reducer: the last index in the indices vector has a "
                        "value of {}, but the number of state variables is only {}",
                        ind_v.back(), ta.get_state_vars().size()));
    }

    // Fetch the 2pi constant.
    const auto twopi_const = detail::get_twopi_const(ta);

    // Fetch the batch size.
    const auto batch_size = ta.get_batch_size();

    // Run the reduction.
    auto *sptr = ta.get_state_data();
    for (const auto ind : ind_v) {
        const auto base_idx = ind * batch_size;

        for (std::uint32_t i = 0; i < batch_size; ++i) {
            using std::floor;
            sptr[base_idx + i] -= twopi_const * floor(sptr[base_idx + i] / twopi_const);
        }
    }

    return true;
}

std::ostream &operator<<(std::ostream &os, const angle_reducer &ar)
{
    if (ar.m_impl) {
        os << fmt::format("Angle reducer: {}", ar.m_impl->m_var_set);
    } else {
        os << "Angle reducer (default constructed)";
    }

    return os;
}

// Explicit insantiations.
#define HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(F)                                                          \
    template HEYOKA_DLL_PUBLIC void angle_reducer::pre_hook(taylor_adaptive<F> &);                                     \
    template HEYOKA_DLL_PUBLIC bool angle_reducer::operator()(taylor_adaptive<F> &);

#define HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH(F)                                                           \
    template HEYOKA_DLL_PUBLIC void angle_reducer::pre_hook(taylor_adaptive_batch<F> &);                               \
    template HEYOKA_DLL_PUBLIC bool angle_reducer::operator()(taylor_adaptive_batch<F> &);

HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(float)
HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(double)
HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(long double)

HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH(float)
HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH(double)
HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(mppp::real128)
HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR(mppp::real)

#endif

#undef HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_SCALAR
#undef HEYOKA_CALLBACK_ANGLE_REDUCER_TEMPLATE_INST_BATCH

} // namespace callback

HEYOKA_END_NAMESPACE

// Serialisation support.
#define HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT(F)                                                         \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::callback::angle_reducer, F)                                     \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(heyoka::callback::angle_reducer, F)

HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT(float)
HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT(double)
HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(heyoka::callback::angle_reducer, mppp::real)

#endif

#undef HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_IMPLEMENT
