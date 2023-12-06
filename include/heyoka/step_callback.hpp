// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_STEP_CALLBACK_HPP
#define HEYOKA_STEP_CALLBACK_HPP

#include <heyoka/config.hpp>

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

// Declaration of the pre_hook interface template.
template <typename, typename, typename>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS pre_hook_iface {
};

// Declaration of the pre_hook interface.
template <typename TA>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS pre_hook_iface<void, void, TA> {
    virtual ~pre_hook_iface() = default;
    // Default implementation of pre_hook() is a no-op.
    virtual void pre_hook(TA &) {}
};

// Concept checking for the presence of the
// pre_hook() member function.
template <typename T, typename TA>
concept with_pre_hook = requires(T &x, TA &ta) { static_cast<void>(x.pre_hook(ta)); };

// Implementation of the pre_hook interface for
// objects providing the pre_hook() member function.
template <typename Holder, typename T, typename TA>
    requires with_pre_hook<std::remove_reference_t<std::unwrap_reference_t<T>>, TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS pre_hook_iface<Holder, T, TA>
    : virtual pre_hook_iface<void, void, TA>, tanuki::iface_impl_helper<Holder, T, pre_hook_iface, TA> {
    void pre_hook(TA &ta) final
    {
        static_cast<void>(this->value().pre_hook(ta));
    }
};

// Definition of the pre_hook wrap. This is defined only for convenience
// and never used directly.
template <typename TA>
using pre_hook_wrap_t = tanuki::wrap<pre_hook_iface, tanuki::default_config, TA>;

// Implementation of the reference interface.
template <typename Wrap, typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_ref_iface_impl : callable_ref_iface_impl<Wrap, bool, TA &> {
    using ta_t = TA;
    TANUKI_REF_IFACE_MEMFUN(pre_hook)
};

template <typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_ref_iface {
    template <typename Wrap>
    using type = step_cb_ref_iface_impl<Wrap, TA>;
};

// Helper to shorten the definition of the step_cb interface template.
template <typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_cb_ifaceT {
    template <typename Holder, typename T>
    using type = tanuki::composite_wrap_interfaceT<callable<bool(TA &)>, pre_hook_wrap_t<TA>>::template type<Holder, T>;
};

// Configuration.
template <typename TA>
constexpr auto step_cb_wrap_config = tanuki::config<bool (*)(TA &), step_cb_ref_iface<TA>::template type>{
    // Similarly to std::function, ensure that callable can store
    // in static storage pointers and reference wrappers.
    // NOTE: reference wrappers are not guaranteed to have the size
    // of a pointer, but in practice that should always be the case.
    .static_size = tanuki::holder_size<bool (*)(TA &), step_cb_ifaceT<TA>::template type>,
    .pointer_interface = false,
    .explicit_generic_ctor = false};

// Definition of the step_cb wrap.
template <typename TA>
using step_cb_wrap_t = tanuki::wrap<step_cb_ifaceT<TA>::template type, step_cb_wrap_config<TA>>;

} // namespace detail

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
// NOLINTBEGIN
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(udc, F)                                                                   \
    TANUKI_S11N_WRAP_EXPORT_KEY(udc, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive<F>>::type)
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(udc, gid, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive<F>>::type)
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(udc, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive<F>>::type)

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(udc, F)                                                             \
    TANUKI_S11N_WRAP_EXPORT_KEY(udc, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive_batch<F>>::type)
#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(udc, gid, F)                                                       \
    TANUKI_S11N_WRAP_EXPORT_KEY2(udc, gid, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive_batch<F>>::type)
#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(udc, F)                                                       \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(udc, heyoka::detail::step_cb_ifaceT<heyoka::taylor_adaptive_batch<F>>::type)
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
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(heyoka::step_callback_set<float>, "heyoka::step_callback_set<float>", float)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(heyoka::step_callback_set<double>, "heyoka::step_callback_set<double>", double)
HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(heyoka::step_callback_set<long double>, "heyoka::step_callback_set<long double>",
                                      long double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(heyoka::step_callback_batch_set<float>,
                                            "heyoka::step_callback_batch_set<float>", float)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(heyoka::step_callback_batch_set<double>,
                                            "heyoka::step_callback_batch_set<double>", double)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(heyoka::step_callback_batch_set<long double>,
                                            "heyoka::step_callback_batch_set<long double>", long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(heyoka::step_callback_set<mppp::real128>,
                                      "heyoka::step_callback_set<mppp::real128>", mppp::real128)
HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY2(heyoka::step_callback_batch_set<mppp::real128>,
                                            "heyoka::step_callback_batch_set<mppp::real128>", mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY2(heyoka::step_callback_set<mppp::real>, "heyoka::step_callback_set<mppp::real>",
                                      mppp::real)

#endif

#endif
