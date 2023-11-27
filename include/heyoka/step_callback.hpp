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
#include <memory>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/callable.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

// NOTE: the step callback needs its own class (rather
// than using std::function directly) because we want to
// give the ability to (optionally) define additional member
// functions in the callback object (beside the call operator).

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_callback_inner_base {
    step_callback_inner_base() = default;
    step_callback_inner_base(const step_callback_inner_base &) = delete;
    step_callback_inner_base(step_callback_inner_base &&) = delete;
    step_callback_inner_base &operator=(const step_callback_inner_base &) = delete;
    step_callback_inner_base &operator=(step_callback_inner_base &&) = delete;
    virtual ~step_callback_inner_base() = default;

    [[nodiscard]] virtual std::unique_ptr<step_callback_inner_base> clone() const = 0;

    virtual bool operator()(TA &) = 0;
    virtual void pre_hook(TA &) = 0;

    [[nodiscard]] virtual std::type_index get_type_index() const = 0;

private:
    // Serialization (empty, no data members).
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }
};

template <typename T, typename TA>
using step_callback_pre_hook_t = decltype(std::declval<std::add_lvalue_reference_t<T>>().pre_hook(
    std::declval<std::add_lvalue_reference_t<TA>>()));

template <typename T, typename TA>
inline constexpr bool step_callback_has_pre_hook_v = std::is_same_v<detected_t<step_callback_pre_hook_t, T, TA>, void>;

template <typename T, typename TA>
struct HEYOKA_DLL_PUBLIC_INLINE_CLASS step_callback_inner final : step_callback_inner_base<TA> {
    T m_value;

    // We just need the def ctor, delete everything else.
    step_callback_inner() = default;
    step_callback_inner(const step_callback_inner &) = delete;
    step_callback_inner(step_callback_inner &&) = delete;
    step_callback_inner &operator=(const step_callback_inner &) = delete;
    step_callback_inner &operator=(step_callback_inner &&) = delete;
    ~step_callback_inner() final = default;

    // Constructors from T (copy and move variants).
    explicit step_callback_inner(const T &x) : m_value(x) {}
    explicit step_callback_inner(T &&x) : m_value(std::move(x)) {}

    // The clone method, used in the copy constructor.
    std::unique_ptr<step_callback_inner_base<TA>> clone() const final
    {
        return std::make_unique<step_callback_inner>(m_value);
    }

    bool operator()(TA &ta) final
    {
        return m_value(ta);
    }
    void pre_hook(TA &ta) final
    {
        if constexpr (step_callback_has_pre_hook_v<T, TA>) {
            m_value.pre_hook(ta);
        }
    }

    [[nodiscard]] std::type_index get_type_index() const final
    {
        return typeid(T);
    }

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<step_callback_inner_base<TA>>(*this);
        ar & m_value;
    }
};

template <typename TA>
class HEYOKA_DLL_PUBLIC step_callback_impl
{
    std::unique_ptr<step_callback_inner_base<TA>> m_ptr;

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & m_ptr;
    }

    // Meta-programming for the generic ctor.

    // Detection of the call operator.
    template <typename T>
    using step_callback_call_t
        = decltype(std::declval<std::add_lvalue_reference_t<T>>()(std::declval<std::add_lvalue_reference_t<TA>>()));

    // Detection of the candidate internal type from a generic T.
    template <typename T>
    using internal_type = std::conditional_t<std::is_function_v<uncvref_t<T>>, std::decay_t<T>, uncvref_t<T>>;

    template <typename T>
    using generic_ctor_enabler = std::enable_if_t<
        std::conjunction_v<
            // Must not compete with copy/move ctors.
            std::negation<std::is_same<uncvref_t<T>, step_callback_impl>>,
            // The internal type must have the correct call operator.
            std::is_same<bool, detected_t<step_callback_call_t, internal_type<T>>>,
            // The internal type must be copy/move constructible and destructible.
            std::conjunction<std::is_copy_constructible<internal_type<T>>, std::is_move_constructible<internal_type<T>>,
                             std::is_destructible<internal_type<T>>>>,
        int>;

public:
    using ta_t = TA;

    step_callback_impl();

    template <typename T, generic_ctor_enabler<T &&> = 0>
    // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
    step_callback_impl(T &&f)
    {
        using uT = detail::uncvref_t<T>;

        // NOTE: if f is a nullptr or an empty callable or
        // std::function, leave this empty.
        // NOTE: I do not think that we need to guard against uT being
        // an empty step_callback here, since:
        // - if we try to construct from a step_callback with the same signature
        //   as this, then we end up in the copy/move ctor (and not here), and
        // - if we try to construct from a step_callback with a different signature
        //   (meaning a different TA), then the check on step_callback_call_t would
        //   fail due to the lack of implicit conversions between TAs.
        if constexpr (detail::is_any_std_func_v<uT> || is_any_callable<uT>::value) {
            if (!f) {
                return;
            }
        }

        if constexpr (std::is_pointer_v<uT> || std::is_member_object_pointer_v<uT>) {
            if (f == nullptr) {
                return;
            }
        }

        m_ptr = std::make_unique<step_callback_inner<internal_type<T &&>, TA>>(std::forward<T>(f));
    }

    step_callback_impl(const step_callback_impl &);
    step_callback_impl(step_callback_impl &&) noexcept;
    step_callback_impl &operator=(const step_callback_impl &);
    step_callback_impl &operator=(step_callback_impl &&) noexcept;
    ~step_callback_impl();

    explicit operator bool() const noexcept;

    bool operator()(TA &);
    void pre_hook(TA &);

    void swap(step_callback_impl &) noexcept;

    [[nodiscard]] std::type_index get_type_index() const;

    // Extraction.
    template <typename T>
    const T *extract() const noexcept
    {
        if (!m_ptr) {
            return nullptr;
        }

        auto p = dynamic_cast<const step_callback_inner<T, TA> *>(m_ptr.get());
        return p == nullptr ? nullptr : &(p->m_value);
    }
    template <typename T>
    T *extract() noexcept
    {
        if (!m_ptr) {
            return nullptr;
        }

        auto p = dynamic_cast<step_callback_inner<T, TA> *>(m_ptr.get());
        return p == nullptr ? nullptr : &(p->m_value);
    }
};

template <typename TA>
HEYOKA_DLL_PUBLIC void swap(step_callback_impl<TA> &, step_callback_impl<TA> &) noexcept;

} // namespace detail

template <typename T>
using step_callback = detail::step_callback_impl<taylor_adaptive<T>>;

template <typename T>
using step_callback_batch = detail::step_callback_impl<taylor_adaptive_batch<T>>;

namespace detail
{

template <typename, bool>
class HEYOKA_DLL_PUBLIC step_callback_set_impl;

template <typename T, bool Batch>
HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<T, Batch> &, step_callback_set_impl<T, Batch> &) noexcept;

template <typename T, bool Batch>
class HEYOKA_DLL_PUBLIC step_callback_set_impl
{
    template <typename T2, bool Batch2>
    friend HEYOKA_DLL_PUBLIC void swap(step_callback_set_impl<T2, Batch2> &,
                                       step_callback_set_impl<T2, Batch2> &) noexcept;

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

    size_type size() const noexcept;
    const step_cb_t &operator[](size_type) const;
    step_cb_t &operator[](size_type);

    bool operator()(ta_t &);
    void pre_hook(ta_t &);
};

} // namespace detail

template <typename T>
using step_callback_set = detail::step_callback_set_impl<T, false>;

template <typename T>
using step_callback_batch_set = detail::step_callback_set_impl<T, true>;

HEYOKA_END_NAMESPACE

// Disable Boost.Serialization tracking for the implementation details of step_callback.
BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive<float>>,
                     boost::serialization::track_never)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive<double>>,
                     boost::serialization::track_never)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive<long double>>,
                     boost::serialization::track_never)

#if defined(HEYOKA_HAVE_REAL128)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive<mppp::real128>>,
                     boost::serialization::track_never)

#endif

#if defined(HEYOKA_HAVE_REAL)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive<mppp::real>>,
                     boost::serialization::track_never)

#endif

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive_batch<float>>,
                     boost::serialization::track_never)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive_batch<double>>,
                     boost::serialization::track_never)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive_batch<long double>>,
                     boost::serialization::track_never)

#if defined(HEYOKA_HAVE_REAL128)

BOOST_CLASS_TRACKING(heyoka::detail::step_callback_inner_base<heyoka::taylor_adaptive_batch<mppp::real128>>,
                     boost::serialization::track_never)

#endif

// NOTE: these are verbatim re-implementations of the BOOST_CLASS_EXPORT_KEY
// and BOOST_CLASS_EXPORT_IMPLEMENT macros, which do not work well with class templates.
// NOLINTBEGIN
#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(T, F)                                                                     \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct guid_defined<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>> : boost::mpl::true_ {      \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>>()                      \
    {                                                                                                                  \
        /* NOTE: the stringize here will produce a name enclosed by brackets. */                                       \
        return BOOST_PP_STRINGIZE((heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>));               \
    }                                                                                                                  \
    }

#define HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(T, F)                                                               \
    namespace boost::archive::detail::extra_detail                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>> {                             \
        static guid_initializer<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>> const &g;          \
    };                                                                                                                 \
    guid_initializer<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>> const                         \
        &init_guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>>::g                              \
        = ::boost::serialization::singleton<guid_initializer<                                                          \
            heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive<F>>>>::get_mutable_instance()               \
              .export_guid();                                                                                          \
    }
// NOLINTEND

#define HEYOKA_S11N_STEP_CALLBACK_EXPORT(T, F)                                                                         \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_KEY(T, F)                                                                         \
    HEYOKA_S11N_STEP_CALLBACK_EXPORT_IMPLEMENT(T, F)

// NOLINTBEGIN
#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(T, F)                                                               \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct guid_defined<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>>                      \
        : boost::mpl::true_ {                                                                                          \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>>()                \
    {                                                                                                                  \
        /* NOTE: the stringize here will produce a name enclosed by brackets. */                                       \
        return BOOST_PP_STRINGIZE((heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>));         \
    }                                                                                                                  \
    }

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_IMPLEMENT(T, F)                                                         \
    namespace boost::archive::detail::extra_detail                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>> {                       \
        static guid_initializer<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>> const &g;    \
    };                                                                                                                 \
    guid_initializer<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>> const                   \
        &init_guid<heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>>::g                        \
        = ::boost::serialization::singleton<guid_initializer<                                                          \
            heyoka::detail::step_callback_inner<T, heyoka::taylor_adaptive_batch<F>>>>::get_mutable_instance()         \
              .export_guid();                                                                                          \
    }
// NOLINTEND

#define HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT(T, F)                                                                   \
    HEYOKA_S11N_STEP_CALLBACK_BATCH_EXPORT_KEY(T, F)                                                                   \
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

#endif
