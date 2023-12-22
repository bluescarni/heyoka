// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_CALLBACK_ANGLE_REDUCER_HPP
#define HEYOKA_CALLBACK_ANGLE_REDUCER_HPP

#include <heyoka/config.hpp>

#include <concepts>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <unordered_set>
#include <utility>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/step_callback.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace callback
{

class HEYOKA_DLL_PUBLIC angle_reducer
{
    class impl;

    std::unique_ptr<impl> m_impl;

    void validate_and_construct(std::unordered_set<expression>);

    template <typename TA>
    HEYOKA_DLL_LOCAL void pre_hook_impl(TA &);

    // Serialisation.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_impl;
    }

public:
    angle_reducer();
    angle_reducer(const angle_reducer &);
    angle_reducer(angle_reducer &&) noexcept;
    angle_reducer &operator=(const angle_reducer &);
    angle_reducer &operator=(angle_reducer &&) noexcept;
    ~angle_reducer();

    template <typename R>
        requires(!std::same_as<angle_reducer, std::remove_cvref_t<R>>) && std::ranges::input_range<R>
                && std::constructible_from<expression, std::iter_reference_t<std::ranges::iterator_t<R>>>
    explicit angle_reducer(R &&r)
    {
        // Turn into lvalue.
        auto &&rl = std::forward<R>(r);

        std::unordered_set<expression> var_set;
        for (auto &&x : rl) {
            var_set.emplace(std::forward<decltype(x)>(x));
        }

        validate_and_construct(std::move(var_set));
    }

    template <typename T>
    void pre_hook(taylor_adaptive<T> &);
    template <typename T>
    bool operator()(taylor_adaptive<T> &);

    template <typename T>
    void pre_hook(taylor_adaptive_batch<T> &);
    template <typename T>
    bool operator()(taylor_adaptive_batch<T> &);
};

// Prevent implicit instantiations of the member functions.
#define HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST(F)                                                                   \
    extern template bool angle_reducer::operator()<F>(taylor_adaptive<F> &);                                           \
    extern template void angle_reducer::pre_hook<F>(taylor_adaptive<F> &);                                             \
    extern template bool angle_reducer::operator()<F>(taylor_adaptive_batch<F> &);                                     \
    extern template void angle_reducer::pre_hook<F>(taylor_adaptive_batch<F> &);

HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST(float)
HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST(double)
HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

extern template bool angle_reducer::operator()<mppp::real>(taylor_adaptive<mppp::real> &);
extern template void angle_reducer::pre_hook<mppp::real>(taylor_adaptive<mppp::real> &);

#endif

#undef HEYOKA_CALLBACK_ANGLE_REDUCER_EXTERN_INST

} // namespace callback

HEYOKA_END_NAMESPACE

// Serialisation macros for angle_reducer.
#define HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY(F)                                                               \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::callback::angle_reducer, F)                                           \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::callback::angle_reducer, F)

HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY(float)
HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY(double)
HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::callback::angle_reducer, mppp::real)

#endif

#undef HEYOKA_CALLBACK_ANGLE_REDUCER_S11N_EXPORT_KEY

#endif
