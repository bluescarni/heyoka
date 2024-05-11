// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cstdint>
#include <ios>
#include <locale>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include <boost/core/demangle.hpp>

#include <fmt/core.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/string_conv.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Implementation of the streaming operator for the scalar integrators.
template <typename T>
std::ostream &taylor_adaptive_stream_impl(std::ostream &os, const taylor_adaptive<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::boolalpha;

    oss << "C++ datatype            : " << boost::core::demangle(typeid(T).name()) << '\n';

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        oss << "Precision               : " << ta.get_prec() << " bits\n";
    }

#endif

    oss << "Tolerance               : " << fp_to_string(ta.get_tol()) << '\n';
    oss << "High accuracy           : " << ta.get_high_accuracy() << '\n';
    oss << "Compact mode            : " << ta.get_compact_mode() << '\n';
    oss << "Taylor order            : " << ta.get_order() << '\n';
    oss << "Dimension               : " << ta.get_dim() << '\n';
    oss << "Time                    : " << fp_to_string(ta.get_time()) << '\n';
    oss << "State                   : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << fp_to_string(ta.get_state()[i]);
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    if (!ta.get_pars().empty()) {
        oss << "Parameters              : [";
        for (decltype(ta.get_pars().size()) i = 0; i < ta.get_pars().size(); ++i) {
            oss << fp_to_string(ta.get_pars()[i]);
            if (i != ta.get_pars().size() - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";
    }

    if (ta.with_events()) {
        if (!ta.get_t_events().empty()) {
            oss << "N of terminal events    : " << ta.get_t_events().size() << '\n';
        }

        if (!ta.get_nt_events().empty()) {
            oss << "N of non-terminal events: " << ta.get_nt_events().size() << '\n';
        }
    }

    return os << oss.str();
}

// Implementation of the streaming operator for the batch integrators.
template <typename T>
std::ostream &taylor_adaptive_batch_stream_impl(std::ostream &os, const taylor_adaptive_batch<T> &ta)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::boolalpha;

    oss << "C++ datatype            : " << boost::core::demangle(typeid(T).name()) << '\n';
    oss << "Tolerance               : " << fp_to_string(ta.get_tol()) << '\n';
    oss << "High accuracy           : " << ta.get_high_accuracy() << '\n';
    oss << "Compact mode            : " << ta.get_compact_mode() << '\n';
    oss << "Taylor order            : " << ta.get_order() << '\n';
    oss << "Dimension               : " << ta.get_dim() << '\n';
    oss << "Batch size              : " << ta.get_batch_size() << '\n';
    oss << "Time                    : [";
    for (decltype(ta.get_time().size()) i = 0; i < ta.get_time().size(); ++i) {
        oss << fp_to_string(ta.get_time()[i]);
        if (i != ta.get_time().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";
    oss << "State                   : [";
    for (decltype(ta.get_state().size()) i = 0; i < ta.get_state().size(); ++i) {
        oss << fp_to_string(ta.get_state()[i]);
        if (i != ta.get_state().size() - 1u) {
            oss << ", ";
        }
    }
    oss << "]\n";

    if (!ta.get_pars().empty()) {
        oss << "Parameters              : [";
        for (decltype(ta.get_pars().size()) i = 0; i < ta.get_pars().size(); ++i) {
            oss << fp_to_string(ta.get_pars()[i]);
            if (i != ta.get_pars().size() - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";
    }

    if (ta.with_events()) {
        if (!ta.get_t_events().empty()) {
            oss << "N of terminal events    : " << ta.get_t_events().size() << '\n';
        }

        if (!ta.get_nt_events().empty()) {
            oss << "N of non-terminal events: " << ta.get_nt_events().size() << '\n';
        }
    }

    return os << oss.str();
}

} // namespace

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<float> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<double> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<long double> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<mppp::real128> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive<mppp::real> &ta)
{
    return detail::taylor_adaptive_stream_impl(os, ta);
}

#endif

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<float> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<double> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<long double> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const taylor_adaptive_batch<mppp::real128> &ta)
{
    return detail::taylor_adaptive_batch_stream_impl(os, ta);
}

#endif

#define HEYOKA_TAYLOR_ENUM_STREAM_CASE(val)                                                                            \
    case val:                                                                                                          \
        os << #val;                                                                                                    \
        break

std::ostream &operator<<(std::ostream &os, taylor_outcome oc)
{
    switch (oc) {
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::success);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::step_limit);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::time_limit);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::err_nf_state);
        HEYOKA_TAYLOR_ENUM_STREAM_CASE(taylor_outcome::cb_stop);
        default:
            if (oc >= taylor_outcome{0}) {
                // Continuing terminal event.
                os << fmt::format("taylor_outcome::terminal_event_{} (continuing)", static_cast<std::int64_t>(oc));
            } else if (oc > taylor_outcome::success) {
                // Stopping terminal event.
                os << fmt::format("taylor_outcome::terminal_event_{} (stopping)", -static_cast<std::int64_t>(oc) - 1);
            } else {
                // Unknown value.
                os << "taylor_outcome::??";
            }
    }

    return os;
}

#undef HEYOKA_TAYLOR_ENUM_STREAM_CASE

std::ostream &operator<<(std::ostream &os, taylor_ad_mode m)
{
    switch (m) {
        case taylor_ad_mode::classic:
            os << "classic";
            break;
        case taylor_ad_mode::tseries:
            os << "tseries";
            break;
        default:
            os << "invalid";
    }

    return os;
}

HEYOKA_END_NAMESPACE
