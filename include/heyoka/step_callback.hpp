// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_STEP_CALLBACK_HPP
#define HEYOKA_STEP_CALLBACK_HPP

#include <heyoka/config.hpp>

#include <concepts>
#include <initializer_list>
#include <type_traits>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/tanuki.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

// NOTE: the step callback needs its own class because we want to
// give the ability to (optionally) define additional member
// functions in the callback object (beside the call operator).

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Forward declaration of the step_cb interface.
template <typename>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_iface;

// Default implementation of the step_cb interface.
// If T has an empty (i.e., invalid) callable_iface_impl, this class
// will be empty too. Otherwise, it will inherit the call
// operator from callable_iface_impl and the default (no-op)
// pre_hook() implementation from step_cb_iface.
template <typename Base, typename Holder, typename T, typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_iface_impl : callable_iface_impl<Base, Holder, T, bool, TA &> {
};

// Concept checking for the presence of the
// pre_hook() member function.
template <typename T, typename TA>
concept with_pre_hook = requires(T &x, TA &ta) { static_cast<void>(x.pre_hook(ta)); };

// Implementation of the step_cb interface for
// objects providing the pre_hook() member function.
template <typename Base, typename Holder, typename T, typename TA>
    requires
    // NOTE: this first concept requirement is needed to prevent objects
    // implementing pre_hook() *without* a valid callable_iface_impl
    // triggering a hard error due to pre_hook() being marked final.
    std::derived_from<callable_iface_impl<Base, Holder, T, bool, TA &>, step_cb_iface<TA>>
        && with_pre_hook<std::remove_reference_t<std::unwrap_reference_t<T>>, TA>
        struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_iface_impl<Base, Holder, T, TA>
    : callable_iface_impl<Base, Holder, T, bool, TA &>,
    tanuki::iface_impl_helper<callable_iface_impl<Base, Holder, T, bool, TA &>, Holder> {
    void pre_hook(TA &ta) final
    {
        static_cast<void>(this->value().pre_hook(ta));
    }
};

// Definition of the step_cb interface.
template <typename TA>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_iface : callable_iface<bool, TA &> {
    // Default implementation of pre_hook() is a no-op.
    virtual void pre_hook(TA &) {}

    template <typename Base, typename Holder, typename T>
    using impl = step_cb_iface_impl<Base, Holder, T, TA>;
};

// Implementation of the reference interface.
template <typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_ref_iface {
    template <typename Wrap>
    struct impl : callable_ref_iface<bool, TA &>::template impl<Wrap> {
        using ta_t = TA;
        TANUKI_REF_IFACE_MEMFUN(pre_hook)
    };
};

// Configuration.
template <typename TA>
inline constexpr auto step_cb_wrap_config = tanuki::config<empty_callable, step_cb_ref_iface<TA>>{
    // Similarly to std::function, ensure that step_callback can store
    // in static storage pointers and reference wrappers.
    // NOTE: reference wrappers are not guaranteed to have the size
    // of a pointer, but in practice that should always be the case.
    .static_size = tanuki::holder_size<bool (*)(TA &), step_cb_iface<TA>>,
    .pointer_interface = false,
    .explicit_ctor = tanuki::wrap_ctor::always_implicit};

// Definition of the step_cb wrap.
template <typename TA>
using step_cb_wrap_t = tanuki::wrap<step_cb_iface<TA>, step_cb_wrap_config<TA>>;

} // namespace detail

// NOTE: ideally here we would want to prevent the implicit instantiations
// of step callbacks via extern template and explicitly instantiate in the
// .cpp as usual. However, I cannot get this to work on GCC, and it seems like
// this is a known issue:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104634
// Hopefully this will be fixed - in the meantime if necessary we could do
// the explicit instantiation only on clang (where this seems to work without
// issues) and MSVC.
// NOTE: this issue seems now to be fixed for GCC >= 13.3.
template <typename T>
using step_callback = detail::step_cb_wrap_t<taylor_adaptive<T>>;

template <typename T>
using step_callback_batch = detail::step_cb_wrap_t<taylor_adaptive_batch<T>>;

namespace detail
{

template <typename, bool>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS step_callback_set_impl;

template <typename T, bool Batch>
void swap(step_callback_set_impl<T, Batch> &, step_callback_set_impl<T, Batch> &) noexcept;

template <typename T, bool Batch>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS step_callback_set_impl
{
    template <typename T2, bool Batch2>
    friend void swap(step_callback_set_impl<T2, Batch2> &, step_callback_set_impl<T2, Batch2> &) noexcept;

public:
    using step_cb_t = std::conditional_t<Batch, step_callback_batch<T>, step_callback<T>>;
    using ta_t = typename step_cb_t::ta_t;
    using size_type = typename std::vector<step_cb_t>::size_type;

private:
    std::vector<step_cb_t> m_cbs;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_cbs;
    }

public:
    step_callback_set_impl() noexcept;
    explicit step_callback_set_impl(std::vector<step_cb_t>);
    step_callback_set_impl(std::initializer_list<step_cb_t>);
    step_callback_set_impl(const step_callback_set_impl &);
    step_callback_set_impl(step_callback_set_impl &&) noexcept;
    step_callback_set_impl &operator=(const step_callback_set_impl &);
    step_callback_set_impl &operator=(step_callback_set_impl &&) noexcept;
    ~step_callback_set_impl();

    [[nodiscard]] size_type size() const noexcept;
    const step_cb_t &operator[](size_type) const;
    step_cb_t &operator[](size_type);

    bool operator()(ta_t &);
    void pre_hook(ta_t &);
};

// Prevent implicit instantiations.
#define HEYOKA_SCS_EXTERN_TEMPLATE(T)                                                                                  \
    extern template class step_callback_set_impl<T, true>;                                                             \
    extern template class step_callback_set_impl<T, false>;                                                            \
    extern template void swap(step_callback_set_impl<T, true> &, step_callback_set_impl<T, true> &) noexcept;          \
    extern template void swap(step_callback_set_impl<T, false> &, step_callback_set_impl<T, false> &) noexcept;

HEYOKA_SCS_EXTERN_TEMPLATE(float)
HEYOKA_SCS_EXTERN_TEMPLATE(double)
HEYOKA_SCS_EXTERN_TEMPLATE(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_SCS_EXTERN_TEMPLATE(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_SCS_EXTERN_TEMPLATE(mppp::real)

#endif

#undef HEYOKA_SCS_EXTERN_TEMPLATE

} // namespace detail

template <typename T>
using step_callback_set = detail::step_callback_set_impl<T, false>;

template <typename T>
using step_callback_batch_set = detail::step_callback_set_impl<T, true>;

HEYOKA_END_NAMESPACE

// Serialisation macros.
// NOTE: by default, we build a custom name and pass it to TANUKI_S11N_WRAP_EXPORT_KEY2.
// This allows us to reduce the size of the final guid wrt to what TANUKI_S11N_WRAP_EXPORT_KEY
// would synthesise, and thus to ameliorate the "class name too long" issue.
// NOLINTBEGIN
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(udc, F)                                                                   \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, "heyoka::step_callback<" #F ">@" #udc,                                           \
                                 heyoka::detail::step_cb_iface<heyoka::taylor_adaptive<F>>)
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(udc, gid, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::step_cb_iface<heyoka::taylor_adaptive<F>>)
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(udc, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::step_cb_iface<heyoka::taylor_adaptive<F>>)

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(udc, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, "heyoka::step_callback_batch<" #F ">@" #udc,                                     \
                                 heyoka::detail::step_cb_iface<heyoka::taylor_adaptive_batch<F>>)
#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(udc, gid, F)                                                       \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::step_cb_iface<heyoka::taylor_adaptive_batch<F>>)
#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(udc, F)                                                       \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::step_cb_iface<heyoka::taylor_adaptive_batch<F>>)
// NOLINTEND

#define HEYOKA_S11N_STEP_CALLBACK_EXPORT(T, F)                                                                         \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(T, F)                                                                         \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(T, F)

#define HEYOKA_S11N_STEP_CALLBACK_EXPORT2(T, gid, F)                                                                   \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(T, gid, F)                                                                   \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(T, F)

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(T, F)                                                                   \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(T, F)                                                                   \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(T, F)

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT2(T, gid, F)                                                             \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(T, gid, F)                                                             \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(T, F)

// Enable serialisation support for step_callback_set.
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::step_callback_set<float>, float)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::step_callback_set<double>, double)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::step_callback_set<long double>, long double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::step_callback_batch_set<float>, float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::step_callback_batch_set<double>, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::step_callback_batch_set<long double>, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::step_callback_set<mppp::real128>, mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::step_callback_batch_set<mppp::real128>, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::step_callback_set<mppp::real>, mppp::real)

#endif

// Export the s11n keys for default-constructed step callbacks.
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::detail::empty_callable, float)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::detail::empty_callable, double)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::detail::empty_callable, long double)

HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::detail::empty_callable, float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::detail::empty_callable, double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::detail::empty_callable, long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::detail::empty_callable, mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(heyoka::detail::empty_callable, mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(heyoka::detail::empty_callable, mppp::real)

#endif

#endif
