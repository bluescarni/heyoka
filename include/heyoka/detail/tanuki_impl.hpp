// Copyright 2023 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the tanuki library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef TANUKI_TANUKI_HPP
#define TANUKI_TANUKI_HPP

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#if defined(TANUKI_WITH_BOOST_S11N)

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>

#endif

// Versioning.
#define TANUKI_VERSION_STRING "1.0.0"
#define TANUKI_VERSION_MAJOR 1
#define TANUKI_VERSION_MINOR 0
#define TANUKI_VERSION_PATCH 0

// No unique address setup.
#if defined(_MSC_VER)

#define TANUKI_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]

#else

#define TANUKI_NO_UNIQUE_ADDRESS [[no_unique_address]]

#endif

// ABI tag setup.
#if defined(__GNUC__) || defined(__clang__)

#define TANUKI_ABI_TAG_ATTR __attribute__((abi_tag))

#else

#define TANUKI_ABI_TAG_ATTR

#endif

#define TANUKI_BEGIN_NAMESPACE                                                                                         \
    namespace tanuki                                                                                                   \
    {                                                                                                                  \
    inline namespace v1 TANUKI_ABI_TAG_ATTR                                                                            \
    {

#define TANUKI_END_NAMESPACE                                                                                           \
    }                                                                                                                  \
    }

// Clang concept bugs.
// NOTE: perhaps we can put more specific version checking
// here once we figure out exactly which versions work ok.
#if defined(__clang__)

#define TANUKI_CLANG_BUGGY_CONCEPTS

#endif

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

TANUKI_BEGIN_NAMESPACE

namespace detail
{

#if defined(TANUKI_CLANG_BUGGY_CONCEPTS)

// NOTE: we employ the detection idiom in order to work
// around certain bugs in clang's concepts implementation.

// http://en.cppreference.com/w/cpp/experimental/is_detected
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector {
    using value_t = std::false_type;
    using type = Default;
};

struct nonesuch {
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const &) = delete;
    nonesuch(nonesuch &&) noexcept = delete;
    nonesuch &operator=(nonesuch const &) = delete;
    nonesuch &operator=(nonesuch &&) noexcept = delete;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template <template <class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

#endif

// NOTE: an argument of this tag type is appended
// to the signature of all member functions in value_iface. The purpose
// is to prevent the user from accidentally implementing
// functions from value_iface in the interface implementations.
struct vtag {
};

// Type-trait to detect instances of std::reference_wrapper.
template <typename T>
struct is_reference_wrapper : std::false_type {
};

template <typename T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {
};

// Interface containing methods to interact
// with the value in the holder class.
// NOTE: templating this on IFace is not strictly necessary,
// as we could just use void * instead of IFace * and then
// cast back to Iface * as needed. However, templating
// gives a higher degree of type safety as there's no risk
// of casting to the wrong type in the wrap class (which already
// does enough memory shenanigans). Perhaps in the future
// we can reconsider if we want to reduce binary bloat.
template <typename IFace>
struct value_iface {
    value_iface() = default;
    value_iface(const value_iface &) = delete;
    value_iface(value_iface &&) noexcept = delete;
    value_iface &operator=(const value_iface &) = delete;
    value_iface &operator=(value_iface &&) noexcept = delete;
    virtual ~value_iface() = default;

    // Access to the value and its type.
    [[nodiscard]] virtual void *value_ptr(vtag) noexcept = 0;
    [[nodiscard]] virtual std::type_index value_type_index(vtag) const noexcept = 0;
    [[nodiscard]] virtual bool is_reference(vtag) const noexcept = 0;

    // Methods to implement virtual copy/move primitives for the holder class.
    [[nodiscard]] virtual std::pair<IFace *, value_iface *> clone(vtag) const = 0;
    [[nodiscard]] virtual std::pair<IFace *, value_iface *> copy_init_holder(void *, vtag) const = 0;
    [[nodiscard]] virtual std::pair<IFace *, value_iface *> move_init_holder(void *, vtag) && noexcept = 0;
    virtual void copy_assign_value_to(value_iface *, vtag) const = 0;
    virtual void move_assign_value_to(value_iface *, vtag) && noexcept = 0;
    virtual void copy_assign_value_from(const void *, vtag) = 0;
    virtual void move_assign_value_from(void *, vtag) noexcept = 0;
    virtual void swap_value(value_iface *, vtag) noexcept = 0;

#if defined(TANUKI_WITH_BOOST_S11N)

private:
    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &, unsigned)
    {
    }

#endif
};

// Concept to detect if a type is default initialisable without throwing.
template <typename T>
concept nothrow_default_initializable
    = std::default_initializable<T> && noexcept(::new (static_cast<void *>(nullptr)) T)
      && std::is_nothrow_constructible_v<T> && noexcept(T{});

// Concept to detect if T is an rvalue reference without cv qualifications.
template <typename T>
concept noncv_rvalue_reference
    = std::is_rvalue_reference_v<T> && std::same_as<std::remove_cvref_t<T>, std::remove_reference_t<T>>;

// NOTE: constrain value types to be non-cv qualified objects.
template <typename T>
concept valid_value_type = std::is_object_v<T> && (!std::is_const_v<T>)&&(!std::is_volatile_v<T>)&&std::destructible<T>;

#if defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wexceptions"

#elif defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"

#elif defined(_MSC_VER)

#pragma warning(push)
#pragma warning(disable : 4297)

#endif

template <typename T, template <typename, typename, typename...> typename IFaceT, typename... Args>
// NOTE: ideally, we would like to put here the checks about IFaceT, e.g.,
// the interface implementation must derive from the interface, it must
// be destructible, etc. However, because we are using the CRTP
// and passing the holder as a template parameter to the interface
// impl, the checks cannot go here because holder
// is still an incomplete type. Thus, the interface checks are placed in
// the ctible_holder concept instead (defined elsewhere). As an unfortunate
// consequence, a holder with an invalid IFaceT might end up being instantiated,
// and we must thus take care of coping with an invalid IFaceT throughout
// the implementation of this class (see for instance the static checks
// in clone() and friends). This is not 100% foolproof as we cannot put
// static checks on the destructor (since it is virtual), thus a non-dtible
// interface impl will still trigger a hard-error - however this is a corner case
// I think we can live with for the time being.
// NOTE: this situation might be resolved in C++23 with the "deducing this"
// feature, which should allow us to avoid passing the holder as a template
// parameter to the interface implementation when implementing the CRTP. See here:
// https://devblogs.microsoft.com/cppblog/cpp23-deducing-this/
// NOTE: it seems like "deducing this" may also help with the interface template
// and with iface_impl_helper (no more Holder parameter?).
    requires valid_value_type<T>
struct holder final : public value_iface<IFaceT<void, void, Args...>>,
                      public IFaceT<holder<T, IFaceT, Args...>, T, Args...> {
    TANUKI_NO_UNIQUE_ADDRESS T m_value;

    // Make sure we don't end up accidentally copying/moving
    // this class.
    holder(const holder &) = delete;
    holder(holder &&) noexcept = delete;
    holder &operator=(const holder &) = delete;
    holder &operator=(holder &&) noexcept = delete;

    // NOTE: this may not end up be noexcept because the value
    // or the interface implementation might throw on destruction.
    // In any case, the wrap dtor is marked noexcept, so, similarly
    // to move operations, if the value or the interface implementation
    // throw on destruction, the program will terminate.
    ~holder() final = default;

    // NOTE: special-casing to avoid the single-argument ctor
    // potentially competing with the copy/move ctors.
    template <typename U>
        requires(!std::same_as<holder, std::remove_cvref_t<U>>)
                && std::constructible_from<T, U &&>
                // NOTE: we need the interface implementation to be:
                // - default initable,
                // - destructible.
                && std::default_initializable<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>
                && std::destructible<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>
    explicit holder(U &&x) noexcept(std::is_nothrow_constructible_v<T, U &&>
                                    && nothrow_default_initializable<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>)
        : m_value(std::forward<U>(x))
    {
    }
    template <typename... U>
        requires(sizeof...(U) != 1u) && std::constructible_from<T, U &&...>
                && std::default_initializable<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>
                && std::destructible<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>
    explicit holder(U &&...x) noexcept(std::is_nothrow_constructible_v<T, U &&...>
                                       && nothrow_default_initializable<IFaceT<holder<T, IFaceT, Args...>, T, Args...>>)
        : m_value(std::forward<U>(x)...)
    {
    }

    // NOTE: mark everything else as private so that it is going to be
    // unreachable from the interface implementation.
private:
    [[nodiscard]] std::type_index value_type_index(vtag) const noexcept final
    {
        return typeid(T);
    }
    [[nodiscard]] void *value_ptr(vtag) noexcept final
    {
        return std::addressof(m_value);
    }

    [[nodiscard]] bool is_reference(vtag) const noexcept final
    {
        return is_reference_wrapper<T>::value;
    }

    // Clone this, and cast the result to the two bases.
    [[nodiscard]] std::pair<IFaceT<void, void, Args...> *, value_iface<IFaceT<void, void, Args...>> *>
    clone(vtag) const final
    {
        // NOTE: the std::convertible_to check is to avoid a hard error when instantiating a holder
        // with an invalid interface implementation.
        if constexpr (std::copy_constructible<T> && std::convertible_to<holder *, IFaceT<void, void, Args...> *>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            auto *ret = new holder(m_value);
            return {ret, ret};
        } else {
            throw std::invalid_argument("Attempting to clone a non-copyable value type");
        }
    }
    // Copy-init a new holder into the storage beginning at ptr.
    // Then cast the result to the two bases and return.
    [[nodiscard]] std::pair<IFaceT<void, void, Args...> *, value_iface<IFaceT<void, void, Args...>> *>
    copy_init_holder(void *ptr, vtag) const final
    {
        if constexpr (std::copy_constructible<T> && std::convertible_to<holder *, IFaceT<void, void, Args...> *>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            auto *ret = ::new (ptr) holder(m_value);
            return {ret, ret};
        } else {
            throw std::invalid_argument("Attempting to copy-construct a non-copyable value type");
        }
    }
    // Move-init a new holder into the storage beginning at ptr.
    // Then cast the result to the two bases and return.
    [[nodiscard]] std::pair<IFaceT<void, void, Args...> *, value_iface<IFaceT<void, void, Args...>> *>
    // NOLINTNEXTLINE(bugprone-exception-escape)
    move_init_holder(void *ptr, vtag) && noexcept final
    {
        if constexpr (std::move_constructible<T> && std::convertible_to<holder *, IFaceT<void, void, Args...> *>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            auto *ret = ::new (ptr) holder(std::move(m_value));
            return {ret, ret};
        } else {
            throw std::invalid_argument("Attempting to move-construct a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Copy-assign m_value into the m_value of v_iface.
    void copy_assign_value_to(value_iface<IFaceT<void, void, Args...>> *v_iface, vtag) const final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            // NOTE: I don't think it is necessary to invoke launder here,
            // as value_ptr() just does a static cast to void *. Since we are assuming that
            // copy_assign_value_to() is called only when assigning holders containing
            // the same T, the conversion chain should boil down to T * -> void * -> T *, which
            // does not require laundering.
            assert(typeid(T) == v_iface->value_type_index(vtag{}));
            *static_cast<T *>(v_iface->value_ptr(vtag{})) = m_value;
        } else {
            throw std::invalid_argument("Attempting to copy-assign a non-copyable value type");
        }
    }
    // Move-assign m_value into the m_value of v_iface.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void move_assign_value_to(value_iface<IFaceT<void, void, Args...>> *v_iface, vtag) && noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            assert(typeid(T) == v_iface->value_type_index(vtag{}));
            *static_cast<T *>(v_iface->value_ptr(vtag{})) = std::move(m_value);
        } else {
            throw std::invalid_argument("Attempting to move-assign a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Copy-assign the object of type T assumed to be stored in ptr into m_value.
    void copy_assign_value_from(const void *ptr, vtag) final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            m_value = *static_cast<const T *>(ptr);
        } else {
            throw std::invalid_argument("Attempting to copy-assign a non-copyable value type");
        }
    }
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void move_assign_value_from(void *ptr, vtag) noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            m_value = std::move(*static_cast<T *>(ptr));
        } else {
            throw std::invalid_argument("Attempting to move-assign a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Swap m_value with the m_value of v_iface.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void swap_value(value_iface<IFaceT<void, void, Args...>> *v_iface, vtag) noexcept final
    {
        if constexpr (std::swappable<T>) {
            assert(typeid(T) == v_iface->value_type_index(vtag{}));

            using std::swap;
            swap(m_value, *static_cast<T *>(v_iface->value_ptr(vtag{})));
        } else {
            throw std::invalid_argument("Attempting to swap a non-swappable value type"); // LCOV_EXCL_LINE
        }
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    // Serialization.
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &boost::serialization::base_object<value_iface<IFaceT<void, void, Args...>>>(*this);
        ar & m_value;
    }

#endif
};

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning(pop)

#endif

// Implementation of basic storage for the wrap class.
template <typename IFace, std::size_t StaticStorageSize, std::size_t StaticStorageAlignment>
struct wrap_storage {
    // NOTE: static storage optimisation enabled. The m_p_iface member is used as a flag:
    // if it is null, then the current storage type is dynamic and the interface pointer
    // (which may be null for the invalid state) is stored in static_storage. If m_p_iface
    // is *not* null, then the current storage type is static and both m_p_iface and m_pv_iface
    // point somewhere in static_storage.
    static_assert(StaticStorageSize > 0u);

    // NOTE: replacing static_storage with an union (IFace *, std::byte[]) may allow us to get rid
    // of the reinterpret_casts and perhaps even to make wrap constexpr-friendly. However, there are
    // some patterns we use (e.g., void * cast and back) which, as of C++20, still cannot be used
    // in constant expressions. Something to investigate for later versions of the standard?

    // NOTE: the static storage is used to store an IFace * in dynamic
    // storage mode, thus it has minimum size and alignment requirements.
    alignas(std::max(StaticStorageAlignment,
                     alignof(IFace *))) std::byte static_storage[std::max(StaticStorageSize, sizeof(IFace *))];
    IFace *m_p_iface;
    value_iface<IFace> *m_pv_iface;
};

template <typename IFace, std::size_t StaticStorageAlignment>
struct wrap_storage<IFace, 0, StaticStorageAlignment> {
    IFace *m_p_iface;
    value_iface<IFace> *m_pv_iface;
};

// NOTE: this is used to check that a config instance
// is a specialisation from the primary config template.
struct config_base {
};

} // namespace detail

// Helpers to determine the size and alignment of a holder instance, given the value
// type T, template interface IFaceT and arguments Args for IFaceT.
template <typename T, template <typename, typename, typename...> typename IFaceT, typename... Args>
inline constexpr auto holder_size = sizeof(detail::holder<T, IFaceT, Args...>);

template <typename T, template <typename, typename, typename...> typename IFaceT, typename... Args>
inline constexpr auto holder_align = alignof(detail::holder<T, IFaceT, Args...>);

// Configuration settings for the wrap class.
// NOTE: the DefaultValueType is subject to the constraints
// for valid value types.
template <typename DefaultValueType = void>
    requires std::same_as<DefaultValueType, void> || detail::valid_value_type<DefaultValueType>
struct config final : detail::config_base {
    using default_value_type = DefaultValueType;

    // Size of the static storage.
    std::size_t static_size = 48;
    // Alignment of the static storage.
    std::size_t static_alignment = alignof(std::max_align_t);
    // Default constructor initialises to the invalid state.
    bool invalid_default_ctor = false;
    // Provide pointer interface.
    bool pointer_interface = true;
    // Explicitness of the generic ctor.
    bool explicit_generic_ctor = true;
    // Enable copy construction/assignment.
    bool copyable = true;
    // Enable move construction/assignment.
    bool movable = true;
    // Enable swap.
    bool swappable = true;
};

// Default configuration for the wrap class.
inline constexpr auto default_config = config{};

namespace detail
{

template <std::size_t N>
concept power_of_two = (N > 0u) && ((N & (N - 1u)) == 0u);

// Concept for checking that Cfg is a valid config instance.
template <auto Cfg>
concept valid_config =
    // This checks that decltype(Cfg) is a specialisation from the primary config template.
    std::derived_from<std::remove_const_t<decltype(Cfg)>, config_base> &&
    // The static alignment value must be a power of 2.
    power_of_two<Cfg.static_alignment>;

} // namespace detail

// Default reference interface implementation.
template <typename, template <typename, typename, typename...> typename, typename...>
struct ref_iface {
};

#define TANUKI_REF_IFACE_MEMFUN(name)                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) & noexcept(                                                                        \
        noexcept(iface_ptr(*static_cast<JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...)))                  \
        -> decltype(iface_ptr(*static_cast<JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...))                \
    {                                                                                                                  \
        return iface_ptr(*static_cast<Wrap *>(this))->name(std::forward<MemFunArgs>(args)...);                         \
    }                                                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) const & noexcept(                                                                  \
        noexcept(iface_ptr(*static_cast<const JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...)))            \
        -> decltype(iface_ptr(*static_cast<const JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...))          \
    {                                                                                                                  \
        return iface_ptr(*static_cast<const Wrap *>(this))->name(std::forward<MemFunArgs>(args)...);                   \
    }                                                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) && noexcept(                                                                       \
        noexcept(std::move(*iface_ptr(*static_cast<JustWrap *>(this))).name(std::forward<MemFunArgs>(args)...)))       \
        -> decltype(std::move(*iface_ptr(*static_cast<JustWrap *>(this))).name(std::forward<MemFunArgs>(args)...))     \
    {                                                                                                                  \
        return std::move(*iface_ptr(*static_cast<Wrap *>(this))).name(std::forward<MemFunArgs>(args)...);              \
    }                                                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) const && noexcept(                                                                 \
        noexcept(std::move(*iface_ptr(*static_cast<const JustWrap *>(this))).name(std::forward<MemFunArgs>(args)...))) \
        -> decltype(std::move(*iface_ptr(*static_cast<const JustWrap *>(this)))                                        \
                        .name(std::forward<MemFunArgs>(args)...))                                                      \
    {                                                                                                                  \
        return std::move(*iface_ptr(*static_cast<const Wrap *>(this))).name(std::forward<MemFunArgs>(args)...);        \
    }

namespace detail
{

// Meta-programming to establish a holder value type
// from an argument of type T.
// This is used in the generic ctor/assignment of wrap.
// By default, the value type is T itself without reference
// or cv qualifications. For function types, let it decay so that
// the stored value is a function pointer.
template <typename T>
using value_t_from_arg = std::conditional_t<std::is_function_v<std::remove_cvref_t<T>>,
                                            std::decay_t<std::remove_cvref_t<T>>, std::remove_cvref_t<T>>;

// These two concepts are used in the implementation of the wrap constructors.

// Check if we can construct from U a valid Holder for the interface IFace.
// Holder is expected to be an instance of the "holder" class defined earlier.
// Cfg must be a valid config instance.
template <typename Holder, typename IFace, auto Cfg, typename... U>
concept ctible_holder =
    // These checks are for verifying that:
    // - IFace is a base of the interface implementation, and
    // - all interface requirements have been implemented, and
    // - we can construct the value type from the variadic args, and
    // - the value type T satisfies the conditions to be stored in a holder.
    std::constructible_from<Holder, U...> && std::derived_from<Holder, IFace> &&
    // Alignment checks: if we are going to use dynamic storage, then no checks are needed
    // as new() takes care of proper alignment; otherwise, we need to ensure that the static
    // storage is sufficiently aligned.
    (sizeof(Holder) > Cfg.static_size || alignof(Holder) <= Cfg.static_alignment);

} // namespace detail

// Type used to indicate emplace construction in the wrap class.
template <typename>
struct in_place_type {
};

template <typename T>
inline constexpr auto in_place = in_place_type<T>{};

namespace detail
{

// Helper to detect if T is an in_place_type. This is used
// to avoid ambiguities in the wrap class between the nullary emplace ctor
// and the generic ctor.
template <typename>
struct is_in_place_type : std::false_type {
};

template <typename T>
struct is_in_place_type<in_place_type<T>> : std::true_type {
};

// Type trait to check if T is a reference wrapper
// whose type, after the removal of cv-qualifiers, is U.
template <typename T, typename U>
struct is_reference_wrapper_for : std::false_type {
};

template <typename T, typename U>
struct is_reference_wrapper_for<std::reference_wrapper<T>, U>
    : std::bool_constant<std::same_as<std::remove_cv_t<T>, U>> {
};

} // namespace detail

// Concept to detect if either:
// - T is the same as U, or
// - T is a reference wrapper whose type, after the
//   removal of cv-qualifiers, is U.
template <typename T, typename U>
concept same_or_ref_for = std::same_as<T, U> || detail::is_reference_wrapper_for<T, U>::value;

// The wrap class.
template <template <typename, typename, typename...> typename IFaceT, auto Cfg = default_config, typename... Args>
    requires std::is_polymorphic_v<IFaceT<void, void, Args...>>
                 && std::has_virtual_destructor_v<IFaceT<void, void, Args...>> && detail::valid_config<Cfg>
class wrap : private detail::wrap_storage<IFaceT<void, void, Args...>, Cfg.static_size, Cfg.static_alignment>,
             // NOTE: the reference interface is not supposed to hold any data: it will always
             // be def-inited (even when copying/moving a wrap object), its assignment operators
             // will never be invoked, it will never be swapped, etc. This needs to be documented.
             public ref_iface<wrap<IFaceT, Cfg, Args...>, IFaceT, Args...>
{
    // Aliases for the two interfaces.
    using iface_t = IFaceT<void, void, Args...>;
    using value_iface_t = detail::value_iface<iface_t>;

    // Alias for the reference interface.
    using ref_iface_t = ref_iface<wrap<IFaceT, Cfg, Args...>, IFaceT, Args...>;

    // The default value type.
    using default_value_t = typename decltype(Cfg)::default_value_type;

    // Shortcut for the holder type corresponding to the value type T.
    template <typename T>
    using holder_t = detail::holder<T, IFaceT, Args...>;

    // Helpers to fetch the interface pointers and the storage type when
    // static storage is enabled.
    std::tuple<const iface_t *, const value_iface_t *, bool> stype() const noexcept
        requires(Cfg.static_size > 0u)
    {
        if (this->m_p_iface == nullptr) {
            // Dynamic storage.
            const auto *ret = *std::launder(reinterpret_cast<iface_t *const *>(this->static_storage));
            // NOTE: if one interface pointer is null, the other must be as well, and vice-versa.
            // Null interface pointers with dynamic storage indicate that this object is in the
            // invalid state.
            assert((ret == nullptr) == (this->m_pv_iface == nullptr));
            return {ret, this->m_pv_iface, false};
        } else {
            // Static storage.
            // NOTE: with static storage, the interface pointers cannot be null.
            assert(this->m_p_iface != nullptr && this->m_pv_iface != nullptr);
            return {this->m_p_iface, this->m_pv_iface, true};
        }
    }
    std::tuple<iface_t *, value_iface_t *, bool> stype() noexcept
        requires(Cfg.static_size > 0u)
    {
        if (this->m_p_iface == nullptr) {
            auto *ret = *std::launder(reinterpret_cast<iface_t **>(this->static_storage));
            assert((ret == nullptr) == (this->m_pv_iface == nullptr));
            return {ret, this->m_pv_iface, false};
        } else {
            assert(this->m_p_iface != nullptr && this->m_pv_iface != nullptr);
            return {this->m_p_iface, this->m_pv_iface, true};
        }
    }

    // Implementation of generic construction. This will constrcut
    // a holder with value type T using the construction argument(s) x.
    template <typename T, typename... U>
    void ctor_impl(U &&...x) noexcept(sizeof(holder_t<T>) <= Cfg.static_size
                                      && std::is_nothrow_constructible_v<holder_t<T>, U &&...>)
    {
        if constexpr (Cfg.static_size == 0u) {
            // Static storage disabled.
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            auto d_ptr = new holder_t<T>(std::forward<U>(x)...);
            this->m_p_iface = d_ptr;
            this->m_pv_iface = d_ptr;
        } else {
            if constexpr (sizeof(holder_t<T>) <= Cfg.static_size) {
                // Static storage.
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                auto *d_ptr = ::new (this->static_storage) holder_t<T>(std::forward<U>(x)...);
                this->m_p_iface = d_ptr;
                this->m_pv_iface = d_ptr;
            } else {
                // Dynamic storage.
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                auto d_ptr = new holder_t<T>(std::forward<U>(x)...);
                ::new (this->static_storage) iface_t *(d_ptr);
                this->m_p_iface = nullptr;
                this->m_pv_iface = d_ptr;
            }
        }
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        if constexpr (Cfg.static_size == 0u) {
            // Store the pointer to the value interface.
            ar << this->m_pv_iface;
        } else {
            const auto [_, pv_iface, st] = stype();

            // Store the storage type.
            ar << st;
            // Store the pointer to the value interface.
            ar << pv_iface;
        }
    }
    // NOTE: as I have understood, when deserialising a pointer Boost
    // allocates memory only the *first* time the pointer is encountered
    // in an archive:
    // https://stackoverflow.com/questions/62105624/how-does-boostserialization-allocate-memory-when-deserializing-through-a-point
    // In our case, we disable object tracking for value_iface and I also
    // think that there are no situations in which a pointer to value_iface
    // could be shared amongst multiple wraps. Thus, we should be ok assuming
    // that the pointer coming out of des11n has always been created with a new
    // call.
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        if constexpr (Cfg.static_size == 0u) {
            // Load the serialised pointer.
            value_iface_t *pv_iface = nullptr;
            ar >> pv_iface;
            assert(pv_iface != nullptr);

            // NOTE: from now on, all is noexcept.

            // Destroy the current object.
            destroy();

            // Assign the new pointers.
            this->m_pv_iface = pv_iface;
            this->m_p_iface = dynamic_cast<iface_t *>(pv_iface);
            assert(this->m_p_iface != nullptr);
        } else {
            // Recover the storage type.
            bool st{};
            ar >> st;

            // Load the serialised pointer.
            value_iface_t *pv_iface = nullptr;
            ar >> pv_iface;
            assert(pv_iface != nullptr);

            // NOTE: from now on, all is noexcept.

            // Destroy the current object.
            destroy();

            if (st) {
                // Move-init the value from pv_iface.
                auto [new_p_iface, new_pv_iface]
                    = std::move(*pv_iface).move_init_holder(this->static_storage, detail::vtag{});
                this->m_p_iface = new_p_iface;
                this->m_pv_iface = new_pv_iface;

                // Clean up pv_iface.
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                delete pv_iface;
            } else {
                assert(dynamic_cast<iface_t *>(pv_iface) != nullptr);
                ::new (this->static_storage) iface_t *(dynamic_cast<iface_t *>(pv_iface));
                this->m_p_iface = nullptr;
                this->m_pv_iface = pv_iface;
            }
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

#endif

public:
    // Store the configuration.
    static constexpr auto cfg = Cfg;

    wrap() noexcept(detail::nothrow_default_initializable<ref_iface_t>)
        requires(Cfg.invalid_default_ctor) && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.static_size != 0u) {
            // Init the interface pointer to null.
            ::new (this->static_storage) iface_t *(nullptr);
        }

        // NOTE: if static storage is enabled, this will indicate
        // that dynamic storage is being employed. Otherwise, this will
        // set the interface pointer to null.
        this->m_p_iface = nullptr;
        this->m_pv_iface = nullptr;
    }
    wrap() noexcept(noexcept(this->ctor_impl<default_value_t>()) && detail::nothrow_default_initializable<ref_iface_t>)
        requires(!Cfg.invalid_default_ctor) && std::default_initializable<ref_iface_t> &&
                // A default value type must have been specified
                // in the configuration.
                (!std::same_as<void, default_value_t>) &&
                // We must be able to value-init the holder.
                detail::ctible_holder<
#if defined(TANUKI_CLANG_BUGGY_CONCEPTS)
                    // NOTE: this seems due to this bug:
                    // https://github.com/llvm/llvm-project/issues/55945
                    detail::detected_t<holder_t, default_value_t>,
#else
                    holder_t<default_value_t>,
#endif
                    iface_t, Cfg>
    {
        ctor_impl<default_value_t>();
    }

    // Generic ctor from a wrappable value.
    template <typename T>
        requires std::default_initializable<ref_iface_t> &&
                 // Must not compete with the emplace ctor.
                 (!detail::is_in_place_type<std::remove_cvref_t<T>>::value) &&
                 // Must not compete with copy/move.
                 (!std::same_as<std::remove_cvref_t<T>, wrap>) &&
                 // We must be able to construct a holder from x.
                 detail::ctible_holder<holder_t<detail::value_t_from_arg<T &&>>, iface_t, Cfg, T &&>
    explicit(Cfg.explicit_generic_ctor)
        // NOLINTNEXTLINE(bugprone-forwarding-reference-overload,cppcoreguidelines-pro-type-member-init,hicpp-member-init,google-explicit-constructor,hicpp-explicit-conversions)
        wrap(T &&x) noexcept(noexcept(this->ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x)))
                             && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
    }

    // NOTE: this will *value-init* if no args
    // are provided. This must be documented well.
    template <typename T, typename... U>
        requires std::default_initializable<ref_iface_t> &&
                 // Forbid emplacing a wrap inside a wrap.
                 (!std::same_as<T, wrap>) &&
                 // We must be able to construct a holder from args.
                 detail::ctible_holder<holder_t<T>, iface_t, Cfg, U &&...>
    explicit wrap(in_place_type<T>, U &&...args) noexcept(noexcept(this->ctor_impl<T>(std::forward<U>(args)...))
                                                          && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<T>(std::forward<U>(args)...);
    }

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    wrap(const wrap &other)
        requires(Cfg.copyable) && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.static_size == 0u) {
            // Static storage disabled.
            std::tie(this->m_p_iface, this->m_pv_iface) = other.m_pv_iface->clone(detail::vtag{});
        } else {
            const auto [_, pv_iface, st] = other.stype();

            if (st) {
                // Other has static storage.
                std::tie(this->m_p_iface, this->m_pv_iface)
                    = pv_iface->copy_init_holder(this->static_storage, detail::vtag{});
            } else {
                // Other has dynamic storage.
                auto [new_p_iface, new_pv_iface] = pv_iface->clone(detail::vtag{});
                ::new (this->static_storage) iface_t *(new_p_iface);
                this->m_p_iface = nullptr;
                this->m_pv_iface = new_pv_iface;
            }
        }
    }

private:
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    void move_init_from(wrap &&other) noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            // Static storage disabled.
            // Shallow copy the pointers.
            this->m_p_iface = other.m_p_iface;
            this->m_pv_iface = other.m_pv_iface;

            // Invalidate other.
            other.m_p_iface = nullptr;
            other.m_pv_iface = nullptr;
        } else {
            const auto [p_iface, pv_iface, st] = other.stype();

            if (st) {
                // Other has static storage.
                std::tie(this->m_p_iface, this->m_pv_iface)
                    = std::move(*pv_iface).move_init_holder(this->static_storage, detail::vtag{});
            } else {
                // Other has dynamic storage.
                ::new (this->static_storage) iface_t *(p_iface);
                this->m_p_iface = nullptr;
                this->m_pv_iface = pv_iface;

                // Invalidate other.
                // NOTE: re-initing with new() is ok here: we know that
                // other.static_storage contains a pointer and we can overwrite
                // it with another pointer without calling the destructor first.
                ::new (other.static_storage) iface_t *(nullptr);
                assert(other.m_p_iface == nullptr);
                other.m_pv_iface = nullptr;
            }
        }
    }

public:
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    wrap(wrap &&other) noexcept
        requires(Cfg.movable) && std::default_initializable<ref_iface_t>
    {
        move_init_from(std::move(other));
    }

private:
    void destroy() noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            // NOTE: if one pointer is null, the other one must be as well.
            assert((this->m_p_iface == nullptr) == (this->m_pv_iface == nullptr));

            delete this->m_p_iface;
        } else {
            const auto [p_iface, _, st] = stype();

            if (st) {
                p_iface->~iface_t();
            } else {
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                delete p_iface;
            }
        }
    }

public:
    ~wrap()
        requires std::destructible<ref_iface_t>
    {
        destroy();
    }

    // Move assignment.
    wrap &operator=(wrap &&other) noexcept
        requires(Cfg.movable)
    {
        // Handle self-assign.
        if (this == std::addressof(other)) {
            return *this;
        }

        // Handle invalid object.
        if (is_invalid(*this)) {
            // No need to destroy, just move init
            // from other is sufficient.
            move_init_from(std::move(other));
            return *this;
        }

        // Handle different internal types.
        if (value_type_index(*this) != value_type_index(other)) {
            destroy();
            move_init_from(std::move(other));
            return *this;
        }

        // The internal types are the same.
        if constexpr (Cfg.static_size == 0u) {
            // For dynamic storage, swap the pointers.
            std::swap(this->m_p_iface, other.m_p_iface);
            std::swap(this->m_pv_iface, other.m_pv_iface);
        } else {
            const auto [p_iface0, pv_iface0, st0] = stype();
            const auto [p_iface1, pv_iface1, st1] = other.stype();

            // The storage flags must match, as they depend only
            // on the internal types.
            assert(st0 == st1);

            if (st0) {
                // For static storage, directly move assign the internal value.
                std::move(*pv_iface1).move_assign_value_to(pv_iface0, detail::vtag{});
            } else {
                // For dynamic storage, swap the pointers.
                assert(this->m_p_iface == nullptr);
                assert(other.m_p_iface == nullptr);

                std::swap(*std::launder(reinterpret_cast<iface_t **>(this->static_storage)),
                          *std::launder(reinterpret_cast<iface_t **>(other.static_storage)));
                std::swap(this->m_pv_iface, other.m_pv_iface);
            }
        }

        return *this;
    }

    // Copy assignment.
    wrap &operator=(const wrap &other)
        requires(Cfg.copyable)
    {
        // Handle self-assign.
        if (this == std::addressof(other)) {
            return *this;
        }

        // Handle invalid object or different internal types.
        if (is_invalid(*this) || value_type_index(*this) != value_type_index(other)) {
            *this = wrap(other);
            return *this;
        }

        // The internal types are the same.
        if constexpr (Cfg.static_size == 0u) {
            // Assign the internal value.
            other.m_pv_iface->copy_assign_value_to(this->m_pv_iface, detail::vtag{});
        } else {
            const auto [p_iface0, pv_iface0, st0] = stype();
            const auto [p_iface1, pv_iface1, st1] = other.stype();

            // The storage flags must match, as they depend only
            // on the internal types.
            assert(st0 == st1);

            // Assign the internal value.
            pv_iface1->copy_assign_value_to(pv_iface0, detail::vtag{});
        }

        return *this;
    }

    // Generic assignment.
    template <typename T>
        requires
        // NOTE: not 100% sure about this, but it seems consistent
        // for generic assignment to be enabled only if copy/move
        // assignment are as well.
        (Cfg.copyable) && (Cfg.movable) &&
        // Must not compete with copy/move assignment.
        (!std::same_as<std::remove_cvref_t<T>, wrap>) &&
        // We must be able to construct a holder from x.
        detail::ctible_holder<holder_t<detail::value_t_from_arg<T &&>>, iface_t, Cfg, T &&>
        wrap &operator=(T &&x)
    {
        // Handle invalid object.
        if (is_invalid(*this)) {
            ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
            return *this;
        }

        // Handle different internal types.
        if (value_type_index(*this) != typeid(detail::value_t_from_arg<T &&>)) {
            destroy();

            try {
                ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
            } catch (...) {
                // NOTE: if ctor_impl fails there's no cleanup required.
                // Invalidate this before rethrowing.
                if constexpr (Cfg.static_size == 0u) {
                    this->m_p_iface = nullptr;
                    this->m_pv_iface = nullptr;
                } else {
                    ::new (this->static_storage) iface_t *(nullptr);
                    this->m_p_iface = nullptr;
                    this->m_pv_iface = nullptr;
                }

                throw;
            }

            return *this;
        }

        if constexpr (std::is_function_v<std::remove_cvref_t<T &&>>) {
            // NOTE: we need a special case if x is a function. The reason for this
            // is that we cannot take the address of a function and then cast it directly to void *,
            // as required by copy/move_assign_value_from(). See here:
            // https://stackoverflow.com/questions/36645660/why-cant-i-cast-a-function-pointer-to-void
            // Thus, we need to create a temporary pointer to the function and use its address
            // in copy/move_assign_value_from() instead.
            auto *fptr = std::addressof(x);
            this->m_pv_iface->copy_assign_value_from(&fptr, detail::vtag{});
        } else {
            // The internal types are the same, do directly copy/move assignment.
            if constexpr (detail::noncv_rvalue_reference<T &&>) {
                this->m_pv_iface->move_assign_value_from(std::addressof(x), detail::vtag{});
            } else {
                this->m_pv_iface->copy_assign_value_from(std::addressof(x), detail::vtag{});
            }
        }

        return *this;
    }

    const iface_t *operator->() const noexcept
        requires(Cfg.pointer_interface)
    {
        return iface_ptr(*this);
    }
    iface_t *operator->() noexcept
        requires(Cfg.pointer_interface)
    {
        return iface_ptr(*this);
    }

    const iface_t &operator*() const noexcept
        requires(Cfg.pointer_interface)
    {
        return *iface_ptr(*this);
    }
    iface_t &operator*() noexcept
        requires(Cfg.pointer_interface)
    {
        return *iface_ptr(*this);
    }

    // Free functions interface.

    // NOTE: w is invalid if either its storage type is dynamic and
    // it has been moved from (note that this also includes the case
    // in which w has been swapped with an invalid object),
    // or if generic assignment failed.
    // In an invalid wrap, the interface pointers are set to null,
    // and the static storage (if enabled) also stores a null pointer.
    // The only valid operations on an invalid object are:
    //
    // - invocation of is_invalid(),
    // - destruction,
    // - copy/move assignment from, and swapping with, a valid wrap,
    // - generic assignment.
    [[nodiscard]] friend bool is_invalid(const wrap &w) noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            assert((w.m_p_iface == nullptr) == (w.m_pv_iface == nullptr));
            return w.m_p_iface == nullptr;
        } else {
            return std::get<0>(w.stype()) == nullptr;
        }
    }

    [[nodiscard]] friend std::type_index value_type_index(const wrap &w) noexcept
    {
        // NOTE: the value interface pointer can be accessed regardless of whether
        // or not static storage is enabled.
        return w.m_pv_iface->value_type_index(detail::vtag{});
    }

    [[nodiscard]] friend const iface_t *iface_ptr(const wrap &w) noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            return w.m_p_iface;
        } else {
            return std::get<0>(w.stype());
        }
    }
    [[nodiscard]] friend iface_t *iface_ptr(wrap &w) noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            return w.m_p_iface;
        } else {
            return std::get<0>(w.stype());
        }
    }

    friend void swap(wrap &w1, wrap &w2) noexcept
        requires(Cfg.swappable)
    {
        // Handle self swap.
        if (std::addressof(w1) == std::addressof(w2)) {
            return;
        }

        // Handle invalid arguments.
        const auto inv1 = is_invalid(w1);
        const auto inv2 = is_invalid(w2);

        if (inv1 && inv2) {
            // Both w1 and w2 are invalid, do nothing.
            return;
        }

        if (inv1) {
            // w1 is invalid, w2 is not: move-assign w2 to w1.
            // This may or may not
            // leave w2 in the invalid state.
            w1 = std::move(w2);
            return;
        }

        if (inv2) {
            // Opposite of the above.
            w2 = std::move(w1);
            return;
        }

        // Handle different types with the canonical swap() implementation.
        if (value_type_index(w1) != value_type_index(w2)) {
            auto temp(std::move(w1));
            w1 = std::move(w2);
            w2 = std::move(temp);
            return;
        }

        // The types are the same.
        if constexpr (Cfg.static_size == 0u) {
            // For dynamic storage, swap the pointers.
            std::swap(w1.m_p_iface, w2.m_p_iface);
            std::swap(w1.m_pv_iface, w2.m_pv_iface);
        } else {
            const auto [p_iface1, pv_iface1, st1] = w1.stype();
            const auto [p_iface2, pv_iface2, st2] = w2.stype();

            // The storage flags must match, as they depend only
            // on the internal types.
            assert(st1 == st2);

            if (st1) {
                // For static storage, directly swap the internal values.
                pv_iface2->swap_value(pv_iface1, detail::vtag{});
            } else {
                // For dynamic storage, swap the pointers.
                assert(w1.m_p_iface == nullptr);
                assert(w2.m_p_iface == nullptr);

                std::swap(*std::launder(reinterpret_cast<iface_t **>(w1.static_storage)),
                          *std::launder(reinterpret_cast<iface_t **>(w2.static_storage)));
                std::swap(w1.m_pv_iface, w2.m_pv_iface);
            }
        }
    }

    [[nodiscard]] friend bool has_static_storage(const wrap &w) noexcept
    {
        if constexpr (Cfg.static_size == 0u) {
            return false;
        } else {
            return std::get<2>(w.stype());
        }
    }

    [[nodiscard]] friend const void *raw_ptr(const wrap &w) noexcept
    {
        return w.m_pv_iface->value_ptr(detail::vtag{});
    }
    [[nodiscard]] friend void *raw_ptr(wrap &w) noexcept
    {
        return w.m_pv_iface->value_ptr(detail::vtag{});
    }

    [[nodiscard]] friend bool contains_reference(const wrap &w) noexcept
    {
        return w.m_pv_iface->is_reference(detail::vtag{});
    }
};

namespace detail
{

template <typename>
struct is_any_wrap_impl : std::false_type {
};

template <template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
struct is_any_wrap_impl<wrap<IFaceT, Cfg, Args...>> : std::true_type {
};

} // namespace detail

// Concept to detect any wrap instance.
template <typename T>
concept any_wrap = detail::is_any_wrap_impl<T>::value;

namespace detail
{

// Machinery to detect the interface of a wrap.
template <typename>
struct iface_from_wrap_impl {
};

template <template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
struct iface_from_wrap_impl<wrap<IFaceT, Cfg, Args...>> {
    using type = IFaceT<void, void, Args...>;
};

template <typename Wrap>
using wrap_interface_t = typename detail::iface_from_wrap_impl<Wrap>::type;

// Machinery to detect the interface implementation of a wrap.
template <typename>
struct iface_impl_from_wrap_impl {
};

template <template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
struct iface_impl_from_wrap_impl<wrap<IFaceT, Cfg, Args...>> {
    template <typename Holder, typename T>
    using type = IFaceT<Holder, T, Args...>;
};

template <typename Wrap, typename Holder, typename T>
using wrap_interface_impl_t = typename detail::iface_impl_from_wrap_impl<Wrap>::template type<Holder, T>;

// Machinery for the definition of the composite wrap.
template <typename, typename, typename, typename, typename...>
struct composite_wrap_iface;

template <typename Wrap0, typename Wrap1, typename... WrapN>
struct composite_wrap_iface<void, void, Wrap0, Wrap1, WrapN...> : virtual public wrap_interface_t<Wrap0>,
                                                                  virtual public wrap_interface_t<Wrap1>,
                                                                  virtual public wrap_interface_t<WrapN>... {
};

template <typename Holder, typename T, typename Wrap0, typename Wrap1, typename... WrapN>
struct composite_wrap_iface : composite_wrap_iface<void, void, Wrap0, Wrap1, WrapN...>,
                              public wrap_interface_impl_t<Wrap0, Holder, T>,
                              public wrap_interface_impl_t<Wrap1, Holder, T>,
                              public wrap_interface_impl_t<WrapN, Holder, T>... {
};

template <typename Wrap0, typename Wrap1, typename... WrapN>
struct composite_wrap_iface_selector {
    template <typename Holder, typename T>
    using type = composite_wrap_iface<Holder, T, Wrap0, Wrap1, WrapN...>;
};

} // namespace detail

// Composite wrap.
template <any_wrap Wrap0, any_wrap Wrap1, any_wrap... WrapN>
using composite_wrap = wrap<detail::composite_wrap_iface_selector<Wrap0, Wrap1, WrapN...>::template type>;

// Composite wrap with custom config.
template <auto Cfg, any_wrap Wrap0, any_wrap Wrap1, any_wrap... WrapN>
    requires detail::valid_config<Cfg>
using composite_cwrap = wrap<detail::composite_wrap_iface_selector<Wrap0, Wrap1, WrapN...>::template type, Cfg>;

// Helper that can be used to reduce typing in an
// interface implementation. This implements value()
// helpers for fetching the value held in Holder,
// automatically unwrapping it in case it is
// a std::reference_wrapper.
// NOTE: the IFaceT and Args... arguments are the interface
// template and its arguments. They are unused in the implementation
// of this class, but it is useful to have them because they
// will allow to disambiguate iface_impl_helper bases when
// implementing composite wrappers.
template <typename Holder, typename T, template <typename, typename, typename...> typename IFaceT, typename... Args>
struct iface_impl_helper {
    auto &value() noexcept
    {
        // NOTE: check to make sure that iface_impl_helper is used as a base of the
        // interface implementation.
        static_assert(std::is_base_of_v<iface_impl_helper, IFaceT<Holder, T, Args...>>,
                      "iface_impl_helper must be used as a base for the interface implementation.");

        auto &val = static_cast<Holder *>(this)->m_value;

        if constexpr (detail::is_reference_wrapper<T>::value) {
            return val.get();
        } else {
            return val;
        }
    }
    const auto &value() const noexcept
    {
        static_assert(std::is_base_of_v<iface_impl_helper, IFaceT<Holder, T, Args...>>,
                      "iface_impl_helper must be used as a base for the interface implementation.");

        const auto &val = static_cast<const Holder *>(this)->m_value;

        if constexpr (detail::is_reference_wrapper<T>::value) {
            return val.get();
        } else {
            return val;
        }
    }
};

template <template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
bool has_dynamic_storage(const wrap<IFaceT, Cfg, Args...> &w) noexcept
{
    return !has_static_storage(w);
}

template <typename T, template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
const T *value_ptr(const wrap<IFaceT, Cfg, Args...> &w) noexcept
{
    return value_type_index(w) == typeid(T) ? static_cast<const T *>(raw_ptr(w)) : nullptr;
}

template <typename T, template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
T *value_ptr(wrap<IFaceT, Cfg, Args...> &w) noexcept
{
    return value_type_index(w) == typeid(T) ? static_cast<T *>(raw_ptr(w)) : nullptr;
}

template <typename T, template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
const T &value_ref(const wrap<IFaceT, Cfg, Args...> &w)
{
    const auto *ptr = value_ptr<T>(w);
    return ptr ? *ptr : throw std::bad_cast{};
}

template <typename T, template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
T &value_ref(wrap<IFaceT, Cfg, Args...> &w)
{
    auto *ptr = value_ptr<T>(w);
    return ptr ? *ptr : throw std::bad_cast{};
}

template <typename T, template <typename, typename, typename...> typename IFaceT, auto Cfg, typename... Args>
bool value_isa(const wrap<IFaceT, Cfg, Args...> &w) noexcept
{
    return value_ptr<T>(w) != nullptr;
}

TANUKI_END_NAMESPACE

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#if defined(TANUKI_WITH_BOOST_S11N)

namespace boost::serialization
{

// NOTE: disable address tracking for value_iface. We do not need it as value_iface
// pointers are never shared, and it might only create issues when
// deserialising into a function-local pointer which is then copied
// into the wrap storage.
template <typename IFace>
struct tracking_level<tanuki::detail::value_iface<IFace>> {
    using tag = mpl::integral_c_tag;
    using type = mpl::int_<track_never>;
    BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
    BOOST_STATIC_ASSERT(
        (mpl::greater<implementation_level<tanuki::detail::value_iface<IFace>>, mpl::int_<primitive_type>>::value));
};

} // namespace boost::serialization

// NOTE: these are verbatim re-implementations of the BOOST_CLASS_EXPORT_KEY
// and BOOST_CLASS_EXPORT_IMPLEMENT macros, which do not work well with class templates.
#define TANUKI_S11N_WRAP_EXPORT_KEY(...)                                                                               \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct guid_defined<tanuki::detail::holder<__VA_ARGS__>> : boost::mpl::true_ {                                     \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::holder<__VA_ARGS__>>()                                                     \
    {                                                                                                                  \
        /* NOTE: the stringize here will produce a name enclosed by brackets. */                                       \
        return BOOST_PP_STRINGIZE((tanuki::detail::holder<__VA_ARGS__>));                                              \
    }                                                                                                                  \
    }

#define TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(...)                                                                         \
    namespace boost::archive::detail::extra_detail                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<tanuki::detail::holder<__VA_ARGS__>> {                                                            \
        static guid_initializer<tanuki::detail::holder<__VA_ARGS__>> const &g;                                         \
    };                                                                                                                 \
    guid_initializer<tanuki::detail::holder<__VA_ARGS__>> const &init_guid<tanuki::detail::holder<__VA_ARGS__>>::g     \
        = ::boost::serialization::singleton<                                                                           \
              guid_initializer<tanuki::detail::holder<__VA_ARGS__>>>::get_mutable_instance()                           \
              .export_guid();                                                                                          \
    }

#define TANUKI_S11N_WRAP_EXPORT(...)                                                                                   \
    TANUKI_S11N_WRAP_EXPORT_KEY(__VA_ARGS__)                                                                           \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(__VA_ARGS__)

#endif

#undef TANUKI_ABI_TAG_ATTR
#undef TANUKI_NO_UNIQUE_ADDRESS

#if defined(TANUKI_CLANG_BUGGY_CONCEPTS)

#undef TANUKI_CLANG_BUGGY_CONCEPTS

#endif

#endif
