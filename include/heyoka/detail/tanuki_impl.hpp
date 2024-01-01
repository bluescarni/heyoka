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
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))

// Headers for GCC-style demangle implementation. This is available
// also for clang, both with libstdc++ and libc++.
#include <cstdlib>
#include <cxxabi.h>

#endif

#if defined(TANUKI_WITH_BOOST_S11N)

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>

#endif

// Versioning.
#define TANUKI_VERSION_MAJOR 1
#define TANUKI_VERSION_MINOR 0
#define TANUKI_VERSION_PATCH 0
#define TANUKI_ABI_VERSION 1

// NOTE: indirection to allow token pasting/stringification:
// https://stackoverflow.com/questions/24991208/expand-a-macro-in-a-macro
#define TANUKI_VERSION_STRING__(maj, min, pat) #maj "." #min "." #pat
#define TANUKI_VERSION_STRING_(maj, min, pat) TANUKI_VERSION_STRING__(maj, min, pat)
#define TANUKI_VERSION_STRING TANUKI_VERSION_STRING_(TANUKI_VERSION_MAJOR, TANUKI_VERSION_MINOR, TANUKI_VERSION_PATCH)

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

#define TANUKI_BEGIN_NAMESPACE__(abiver)                                                                               \
    namespace tanuki                                                                                                   \
    {                                                                                                                  \
    inline namespace v##abiver TANUKI_ABI_TAG_ATTR                                                                     \
    {

#define TANUKI_BEGIN_NAMESPACE_(abiver) TANUKI_BEGIN_NAMESPACE__(abiver)

#define TANUKI_BEGIN_NAMESPACE TANUKI_BEGIN_NAMESPACE_(TANUKI_ABI_VERSION)

#define TANUKI_END_NAMESPACE                                                                                           \
    }                                                                                                                  \
    }

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#endif

// Visibility setup.
// NOTE: the idea here is as follows:
// - on Windows there is apparently no need to set up
//   dllimport/dllexport on class templates;
// - on non-Windows platforms with known compilers,
//   we mark several class templates as visible. This is apparently
//   necessary at least in some situations involving, for
//   instance, the export/registration macros in Boost
//   serialisation. libc++ also, for instance, usually marks
//   public class templates as visible;
// - otherwise, we do not implement any visibility attribute.
#if defined(_WIN32) || defined(__CYGWIN__)

#define TANUKI_VISIBLE

#elif defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)

#define TANUKI_VISIBLE __attribute__((visibility("default")))

#else

#define TANUKI_VISIBLE

#endif

TANUKI_BEGIN_NAMESPACE

// Semantics for the wrap class.
// NOTE: this needs to be marked as visibile because
// the value_iface class depends on it. If we do not, we have
// the usual s11n-related visibility issues on OSX.
enum class TANUKI_VISIBLE wrap_semantics { value, reference };

namespace detail
{

// Helper to demangle a type name.
inline std::string demangle(const char *s)
{
#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))
    // NOTE: wrap std::free() in a local lambda, so we avoid
    // potential ambiguities when taking the address of std::free().
    // See:
    // https://stackoverflow.com/questions/27440953/stdunique-ptr-for-c-functions-that-need-free
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory, hicpp-no-malloc)
    auto deleter = [](void *ptr) { std::free(ptr); };

    // NOTE: abi::__cxa_demangle will return a pointer allocated by std::malloc, which we will delete via std::free().
    std::unique_ptr<char, decltype(deleter)> res{::abi::__cxa_demangle(s, nullptr, nullptr, nullptr), deleter};

    // NOTE: return the original string if demangling fails.
    return res ? std::string(res.get()) : std::string(s);
#else
    // If no demangling is available, just return the mangled name.
    // NOTE: MSVC already returns the demangled name from typeid.
    return std::string(s);
#endif
}

// Type-trait to detect instances of std::reference_wrapper.
template <typename T>
struct is_reference_wrapper : std::false_type {
};

template <typename T>
struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {
};

// This is a base class for value_iface, used
// to check that an interface implementation
// is correctly inheriting from its Base.
struct value_iface_base {
};

// Interface containing methods to interact
// with the value in the holder class.
// NOTE: we need to template value_iface on the semantics
// so that we can selectively disable address tracking
// in Boost.serialisation when employing value semantics
// (we do not want to disable it for reference semantics
// since we need it for correct shared_ptr s11n).
template <typename IFace, wrap_semantics Sem>
struct TANUKI_VISIBLE value_iface : public IFace, value_iface_base {
    value_iface() = default;
    value_iface(const value_iface &) = delete;
    value_iface(value_iface &&) noexcept = delete;
    value_iface &operator=(const value_iface &) = delete;
    value_iface &operator=(value_iface &&) noexcept = delete;
    virtual ~value_iface() = default;

    // NOTE: don't make these pure virtual as this prevents us from asserting
    // default-constructability of the implementation interface when doing concept
    // checking.

    // LCOV_EXCL_START
    // Access to the value and its type.
    [[nodiscard]] virtual void *_tanuki_value_ptr() noexcept
    {
        assert(false);
        return {};
    }
    [[nodiscard]] virtual std::type_index _tanuki_value_type_index() const noexcept
    {
        assert(false);
        return typeid(void);
    }
    [[nodiscard]] virtual bool _tanuki_is_reference() const noexcept
    {
        assert(false);
        return {};
    }

    // Methods to implement virtual copy/move primitives for the holder class.
    [[nodiscard]] virtual value_iface *_tanuki_clone() const
    {
        assert(false);
        return {};
    }
    [[nodiscard]] virtual value_iface *_tanuki_copy_init_holder(void *) const
    {
        assert(false);
        return {};
    }
    [[nodiscard]] virtual value_iface *_tanuki_move_init_holder(void *) && noexcept
    {
        assert(false);
        return {};
    }
    virtual void _tanuki_copy_assign_value_to(value_iface *) const
    {
        assert(false);
    }
    virtual void _tanuki_move_assign_value_to(value_iface *) && noexcept
    {
        assert(false);
    }
    virtual void _tanuki_copy_assign_value_from(const void *)
    {
        assert(false);
    }
    virtual void _tanuki_move_assign_value_from(void *) noexcept
    {
        assert(false);
    }
    virtual void _tanuki_swap_value(value_iface *) noexcept
    {
        assert(false);
    }
    // LCOV_EXCL_STOP

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

// NOTE: constrain value types to be non-cv qualified destructible objects.
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

} // namespace detail

// Composite interface.
template <typename IFace0, typename IFace1, typename... IFaceN>
struct TANUKI_VISIBLE composite_iface : public IFace0, public IFace1, public IFaceN... {
};

namespace detail
{

// Detection of a composite interface.
template <typename>
struct is_composite_interface : std::false_type {
};

template <typename IFace0, typename IFace1, typename... IFaceN>
struct is_composite_interface<composite_iface<IFace0, IFace1, IFaceN...>> : std::true_type {
};

// Private base for the unspecialised iface_impl.
struct iface_impl_base {
};

} // namespace detail

// Definition of the external interface implementation
// customisation point. Derives from detail::iface_impl_base
// in order to detect specialisations.
// NOTE: prohibit the definition of an external implementation
// for the composite interface.
template <typename IFace, typename Base, typename Holder, typename T>
    requires(!detail::is_composite_interface<IFace>::value)
struct iface_impl final : detail::iface_impl_base {
};

namespace detail
{

// Detect the presence of an external interface implementation.
template <typename IFace, typename Base, typename Holder, typename T>
concept iface_has_external_impl = !std::derived_from<iface_impl<IFace, Base, Holder, T>, iface_impl_base>;

// Detect the presence of an intrusive interface implementation.
template <typename IFace, typename Base, typename Holder, typename T>
concept iface_has_intrusive_impl = requires() { typename IFace::template impl<Base, Holder, T>; };

// Helper to fetch the interface implementation.
template <typename, typename, typename, typename>
struct get_iface_impl {
};

// External interface implementation.
// NOTE: this will take the precedence in case an intrusive
// implementation is also available.
template <typename IFace, typename Base, typename Holder, typename T>
    requires iface_has_external_impl<IFace, Base, Holder, T>
struct get_iface_impl<IFace, Base, Holder, T> {
    using type = iface_impl<IFace, Base, Holder, T>;
};

// Intrusive interface implementation.
template <typename IFace, typename Base, typename Holder, typename T>
    requires iface_has_intrusive_impl<IFace, Base, Holder, T> && (!iface_has_external_impl<IFace, Base, Holder, T>)
struct get_iface_impl<IFace, Base, Holder, T> {
    using type = typename IFace::template impl<Base, Holder, T>;
};

// Meta-programming to select the implementation of an interface.

// By default, the Base for the interface implementation is
// value_iface of IFace (which transitively makes IFace also a base for
// the implementation).
template <typename IFace, typename Holder, typename T, wrap_semantics Sem>
struct impl_from_iface_impl : get_iface_impl<IFace, value_iface<IFace, Sem>, Holder, T> {
};

// For composite interfaces, we synthesize a class hierarchy in which every
// implementation derives from the previous one, and the first implementation
// derives from value_iface of the composite interface.
template <typename Holder, typename T, typename CurIFace, typename CurBase, typename NextIFace, typename... IFaceN>
struct c_iface_assembler {
    using cur_impl = typename get_iface_impl<CurIFace, CurBase, Holder, T>::type;
    using type = typename c_iface_assembler<Holder, T, NextIFace, cur_impl, IFaceN...>::type;
};

template <typename Holder, typename T, typename CurIFace, typename CurBase, typename LastIFace>
struct c_iface_assembler<Holder, T, CurIFace, CurBase, LastIFace> {
    using cur_impl = typename get_iface_impl<CurIFace, CurBase, Holder, T>::type;
    using type = typename get_iface_impl<LastIFace, cur_impl, Holder, T>::type;
};

template <typename Holder, typename T, wrap_semantics Sem, typename IFace0, typename IFace1, typename... IFaceN>
struct impl_from_iface_impl<composite_iface<IFace0, IFace1, IFaceN...>, Holder, T, Sem> {
    using type =
        typename c_iface_assembler<Holder, T, IFace0, value_iface<composite_iface<IFace0, IFace1, IFaceN...>, Sem>,
                                   IFace1, IFaceN...>::type;
};

// Helper alias.
template <typename IFace, typename Holder, typename T, wrap_semantics Sem>
using impl_from_iface = typename impl_from_iface_impl<IFace, Holder, T, Sem>::type;

// Concept to check that the interface IFace has an implementation
// for the value type T.
template <typename IFace, typename Holder, typename T, wrap_semantics Sem>
concept iface_has_impl = requires() {
    typename impl_from_iface<IFace, Holder, T, Sem>;
    // NOTE: this will check that the implementation derives
    // from its Base (e.g., the check will fail in case of
    // an empty implementation).
    requires std::derived_from<impl_from_iface<IFace, Holder, T, Sem>, value_iface_base>;
};

// Class for holding an instance of the value type T.
// The inheritance diagram is:
// holder -> iface impl -> value_iface -> iface.
// NOTE: this class has several conceptual requirements which
// are checked in wrap::ctor_impl().
template <typename T, typename IFace, wrap_semantics Sem>
struct TANUKI_VISIBLE holder final : public impl_from_iface<IFace, holder<T, IFace, Sem>, T, Sem> {
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
    explicit holder(U &&x) noexcept(
        std::is_nothrow_constructible_v<T, U &&>
        && nothrow_default_initializable<impl_from_iface<IFace, holder<T, IFace, Sem>, T, Sem>>)
        : m_value(std::forward<U>(x))
    {
    }
    template <typename... U>
        requires(sizeof...(U) != 1u)
    explicit holder(U &&...x) noexcept(
        std::is_nothrow_constructible_v<T, U &&...>
        && nothrow_default_initializable<impl_from_iface<IFace, holder<T, IFace, Sem>, T, Sem>>)
        : m_value(std::forward<U>(x)...)
    {
    }

    // NOTE: mark everything else as private so that it is going to be
    // unreachable from the interface implementation.
private:
    [[nodiscard]] std::type_index _tanuki_value_type_index() const noexcept final
    {
        return typeid(T);
    }
    [[nodiscard]] void *_tanuki_value_ptr() noexcept final
    {
        return std::addressof(m_value);
    }

    [[nodiscard]] bool _tanuki_is_reference() const noexcept final
    {
        return is_reference_wrapper<T>::value;
    }

    // Clone this, and cast the result to the value interface.
    [[nodiscard]] value_iface<IFace, Sem> *_tanuki_clone() const final
    {
        if constexpr (std::copy_constructible<T>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            return new holder(m_value);
        } else {
            throw std::invalid_argument("Attempting to clone a non-copyable value type");
        }
    }
    // Copy-init a new holder into the storage beginning at ptr.
    // Then cast the result to the value interface and return.
    [[nodiscard]] value_iface<IFace, Sem> *_tanuki_copy_init_holder(void *ptr) const final
    {
        if constexpr (std::copy_constructible<T>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            return ::new (ptr) holder(m_value);
        } else {
            throw std::invalid_argument("Attempting to copy-construct a non-copyable value type");
        }
    }
    // Move-init a new holder into the storage beginning at ptr.
    // Then cast the result to the value interface and return.
    [[nodiscard]] value_iface<IFace, Sem> *
    // NOLINTNEXTLINE(bugprone-exception-escape)
    _tanuki_move_init_holder(void *ptr) && noexcept final
    {
        if constexpr (std::move_constructible<T>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            return ::new (ptr) holder(std::move(m_value));
        } else {
            throw std::invalid_argument("Attempting to move-construct a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Copy-assign m_value into the m_value of v_iface.
    void _tanuki_copy_assign_value_to(value_iface<IFace, Sem> *v_iface) const final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            // NOTE: I don't think it is necessary to invoke launder here,
            // as value_ptr() just does a static cast to void *. Since we are assuming that
            // copy_assign_value_to() is called only when assigning holders containing
            // the same T, the conversion chain should boil down to T * -> void * -> T *, which
            // does not require laundering.
            assert(typeid(T) == v_iface->_tanuki_value_type_index());
            *static_cast<T *>(v_iface->_tanuki_value_ptr()) = m_value;
        } else {
            throw std::invalid_argument("Attempting to copy-assign a non-copyable value type");
        }
    }
    // Move-assign m_value into the m_value of v_iface.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void _tanuki_move_assign_value_to(value_iface<IFace, Sem> *v_iface) && noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            assert(typeid(T) == v_iface->_tanuki_value_type_index());
            *static_cast<T *>(v_iface->_tanuki_value_ptr()) = std::move(m_value);
        } else {
            throw std::invalid_argument("Attempting to move-assign a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Copy-assign the object of type T assumed to be stored in ptr into m_value.
    void _tanuki_copy_assign_value_from(const void *ptr) final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            m_value = *static_cast<const T *>(ptr);
        } else {
            throw std::invalid_argument("Attempting to copy-assign a non-copyable value type");
        }
    }
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void _tanuki_move_assign_value_from(void *ptr) noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            m_value = std::move(*static_cast<T *>(ptr));
        } else {
            throw std::invalid_argument("Attempting to move-assign a non-movable value type"); // LCOV_EXCL_LINE
        }
    }
    // Swap m_value with the m_value of v_iface.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    void _tanuki_swap_value(value_iface<IFace, Sem> *v_iface) noexcept final
    {
        if constexpr (std::swappable<T>) {
            assert(typeid(T) == v_iface->_tanuki_value_type_index());

            using std::swap;
            swap(m_value, *static_cast<T *>(v_iface->_tanuki_value_ptr()));
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
        ar &boost::serialization::base_object<value_iface<IFace, Sem>>(*this);
        ar & m_value;
    }

#endif
};

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning(pop)

#endif

// Detection of the holder value type.
template <typename>
struct holder_value {
};

template <typename T, typename IFace, wrap_semantics Sem>
struct holder_value<holder<T, IFace, Sem>> {
    using type = T;
};

} // namespace detail

namespace detail
{

// Implementation of storage for the wrap class. This will be used
// to store an instance of the holder type.
template <typename IFace, std::size_t StaticStorageSize, std::size_t StaticStorageAlignment, wrap_semantics Sem>
struct TANUKI_VISIBLE wrap_storage {
    static_assert(StaticStorageSize > 0u);
    static_assert(Sem == wrap_semantics::value);

    // Static storage optimisation enabled.
    // The active storage is dynamic if either m_pv_iface is null (which indicates the
    // invalid state) or if it points somewhere outside static_storage. Otherwise,
    // the active storage is static and m_pv_iface points somewhere within
    // static_storage.
    value_iface<IFace, Sem> *m_pv_iface;
    alignas(StaticStorageAlignment) std::byte static_storage[StaticStorageSize];
};

template <typename IFace, std::size_t StaticStorageAlignment>
struct TANUKI_VISIBLE wrap_storage<IFace, 0, StaticStorageAlignment, wrap_semantics::value> {
    value_iface<IFace, wrap_semantics::value> *m_pv_iface;
};

template <typename IFace, std::size_t StaticStorageSize, std::size_t StaticStorageAlignment>
struct TANUKI_VISIBLE wrap_storage<IFace, StaticStorageSize, StaticStorageAlignment, wrap_semantics::reference> {
    std::shared_ptr<value_iface<IFace, wrap_semantics::reference>> m_pv_iface;
};

// NOTE: this is used to check that a config instance
// is a specialisation from the primary config template.
struct config_base {
};

} // namespace detail

// Helpers to determine the size and alignment of a holder instance, given the value
// type T and the interface IFace.
// NOTE: here we have the complication that holder technically depends on the
// wrap semantics. We do not want to complicate the interface of these helpers
// requiring the user to pass in the semantics as well, so instead we assert
// that size/alignment are the same regardless of semantics (which should always
// be the case).
template <typename T, typename IFace>
    requires(sizeof(detail::holder<T, IFace, wrap_semantics::value>)
             == sizeof(detail::holder<T, IFace, wrap_semantics::reference>))
inline constexpr auto holder_size = sizeof(detail::holder<T, IFace, wrap_semantics::value>);

template <typename T, typename IFace>
    requires(alignof(detail::holder<T, IFace, wrap_semantics::value>)
             == alignof(detail::holder<T, IFace, wrap_semantics::reference>))
inline constexpr auto holder_align = alignof(detail::holder<T, IFace, wrap_semantics::value>);

// Default implementation of the reference interface.
struct TANUKI_VISIBLE no_ref_iface {
    template <typename>
    struct impl {
    };
};

// Composite reference interface.
template <typename IFace0, typename IFace1, typename... IFaceN>
struct TANUKI_VISIBLE composite_ref_iface {
    template <typename Wrap>
    struct impl : public IFace0::template impl<Wrap>,
                  public IFace1::template impl<Wrap>,
                  public IFaceN::template impl<Wrap>... {
    };
};

// Enum to select the explicitness of the generic wrap ctors.
enum class wrap_ctor { always_explicit, ref_implicit, always_implicit };

// Configuration settings for the wrap class.
// NOTE: the DefaultValueType is subject to the constraints
// for valid value types.
template <typename DefaultValueType = void, typename RefIFace = no_ref_iface>
    requires std::same_as<DefaultValueType, void> || detail::valid_value_type<DefaultValueType>
struct TANUKI_VISIBLE config final : detail::config_base {
    using default_value_type = DefaultValueType;

    // Size of the static storage.
    std::size_t static_size = 48;
    // Alignment of the static storage.
    std::size_t static_align = alignof(std::max_align_t);
    // Default constructor initialises to the invalid state.
    bool invalid_default_ctor = false;
    // Provide pointer interface.
    bool pointer_interface = true;
    // Explicitness of the generic ctor.
    wrap_ctor explicit_ctor = wrap_ctor::always_explicit;
    // Semantics.
    wrap_semantics semantics = wrap_semantics::value;
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
    power_of_two<Cfg.static_align> &&
    // Cfg.explicit_ctor must be set to one of the valid enumerators.
    (Cfg.explicit_ctor >= wrap_ctor::always_explicit && Cfg.explicit_ctor <= wrap_ctor::always_implicit) &&
    // Cfg.semantics must be one of the two valid enumerators.
    (Cfg.semantics == wrap_semantics::value || Cfg.semantics == wrap_semantics::reference);

// Machinery to infer the reference interface from a config instance.
template <typename>
struct cfg_ref_type {
};

template <typename DefaultValueType, typename RefIFace>
struct cfg_ref_type<config<DefaultValueType, RefIFace>> {
    using type = RefIFace;
};

} // namespace detail

// Helpers to ease the definition of a reference interface.
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

// Implementation of the pointer interface for the wrap
// class, conditionally-enabled depending on the configuration.
template <bool Enable, typename Wrap, typename IFace>
struct wrap_pointer_iface {
    const IFace *operator->() const noexcept
    {
        return iface_ptr(*static_cast<const Wrap *>(this));
    }
    IFace *operator->() noexcept
    {
        return iface_ptr(*static_cast<Wrap *>(this));
    }

    const IFace &operator*() const noexcept
    {
        return *iface_ptr(*static_cast<const Wrap *>(this));
    }
    IFace &operator*() noexcept
    {
        return *iface_ptr(*static_cast<Wrap *>(this));
    }
};

template <typename Wrap, typename IFace>
struct wrap_pointer_iface<false, Wrap, IFace> {
};

} // namespace detail

// Helper to unwrap a std::reference_wrapper and remove reference
// and cv qualifiers from the result.
template <typename T>
using unwrap_cvref_t = std::remove_cvref_t<std::unwrap_reference_t<T>>;

// The wrap class.
template <typename IFace, auto Cfg = default_config>
    requires std::is_polymorphic_v<IFace> && std::has_virtual_destructor_v<IFace> && detail::valid_config<Cfg>
class TANUKI_VISIBLE wrap
    : private detail::wrap_storage<IFace, Cfg.static_size, Cfg.static_align, Cfg.semantics>,
      // NOTE: the reference interface is not supposed to hold any data: it will always
      // be def-inited (even when copying/moving a wrap object), its assignment operators
      // will never be invoked, it will never be swapped, etc. This needs to be documented.
      public detail::cfg_ref_type<std::remove_const_t<decltype(Cfg)>>::type::template impl<wrap<IFace, Cfg>>,
      public detail::wrap_pointer_iface<Cfg.pointer_interface, wrap<IFace, Cfg>, IFace>
{
    // Aliases for the two interfaces.
    using iface_t = IFace;
    using value_iface_t = detail::value_iface<iface_t, Cfg.semantics>;

    // Alias for the reference interface.
    using ref_iface_t =
        // NOTE: clang 14 needs the typename here, hopefully this is not harmful to other compilers.
        typename detail::cfg_ref_type<std::remove_const_t<decltype(Cfg)>>::type::template impl<wrap<IFace, Cfg>>;

    // The default value type.
    using default_value_t = typename decltype(Cfg)::default_value_type;

    // Shortcut for the holder type corresponding to the value type T.
    template <typename T>
    using holder_t = detail::holder<T, IFace, Cfg.semantics>;

    // Helper to detect the type of storage in use. Returns true for static
    // storage, false for dynamic storage (including the invalid state).
    [[nodiscard]] bool stype() const noexcept
        requires(Cfg.semantics == wrap_semantics::value && Cfg.static_size > 0u)
    {
        const auto *ptr = reinterpret_cast<const std::byte *>(this->m_pv_iface);

        // NOTE: this is not 100% portable, for the following reasons:
        // - if ptr is not within static_storage, the results of the comparisons are unspecified;
        // - even if we used std::less & co. (instead of builtin comparisons), in principle static_storage
        //   could be interleaved with another object while at the same time respecting the total
        //   pointer ordering guarantees given by the standard.
        // In pratice, this should be ok an all commonly-used platforms.
        // NOTE: it seems like the only truly portable way of implementing this is to compare ptr
        // to the addresses of all elements in static_storage. Unfortunately, it seems like compilers
        // are not able to optimise this to a simple pointer comparison.
        return ptr >= this->static_storage && ptr < this->static_storage + sizeof(this->static_storage);
    }

    // Implementation of generic construction. This will constrcut
    // a holder with value type T using the construction argument(s) x.
    // NOTE: it is important that the checks on the
    // interface implementation come *before* the checks on T.
    // This helps breaking infinite recursions that can arise when
    // the generic constructor of wrap is implicit
    // (see also the test_inf_loop_bug test).
    template <typename T, typename... U>
        requires
        // The interface must have an implementation for T.
        detail::iface_has_impl<iface_t, holder_t<T>, T, Cfg.semantics> &&
        // The implementation must be default-initable and
        // destructible.
        std::default_initializable<detail::impl_from_iface<iface_t, holder_t<T>, T, Cfg.semantics>>
        && std::destructible<detail::impl_from_iface<iface_t, holder_t<T>, T, Cfg.semantics>> &&
        // T must be a valid value type.
        detail::valid_value_type<T> &&
        // T must be constructible from the construction arguments.
        std::constructible_from<T, U &&...>
        void ctor_impl(U &&...x) noexcept(Cfg.semantics == wrap_semantics::value
                                          && sizeof(holder_t<T>) <= Cfg.static_size
                                          && alignof(holder_t<T>) <= Cfg.static_align
                                          && std::is_nothrow_constructible_v<holder_t<T>, U &&...>)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            if constexpr (sizeof(holder_t<T>) > Cfg.static_size || alignof(holder_t<T>) > Cfg.static_align) {
                // Static storage is disabled, or the type is overaligned, or
                // there is not enough room in static storage.
                // Use dynamic memory allocation.
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                this->m_pv_iface = new holder_t<T>(std::forward<U>(x)...);
            } else {
                // Static storage is enabled and there is enough room. Construct in-place.
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                this->m_pv_iface = ::new (this->static_storage) holder_t<T>(std::forward<U>(x)...);
            }
        } else {
            this->m_pv_iface = std::make_shared<holder_t<T>>(std::forward<U>(x)...);
        }
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    // Serialisation.
    // NOTE: serialisation support has certain prerequisites:
    // - the value type must be default-initialisable,
    // - the value type must be move-ctible (when using value semantics),
    // - the value type must not be over-aligned (when using value semantics,
    //   not sure if it applies to reference semantics as well).
    // The first two come from the way pointer serialisation works
    // in Boost (i.e., serialisation via pointer to base requires a
    // default constructor and dynamic allocation of an object instance,
    // from which we do a move-init of the holder when using value semantics).
    // The last one I think comes from the way memory is allocated during des11n,
    // i.e., see here:
    // https://github.com/boostorg/serialization/blob/a20c4d97c37e5f437c8ba78f296830edb79cff9e/include/boost/archive/detail/iserializer.hpp#L241
    // Perhaps by providing a custom new operator to the value interface
    // class we can implement proper over-alignment of dynamically-allocated memory.
    // NOTE: support for portable archives would requires some additional logic
    // in case of value semantics with static storage enabled: the sizeof
    // types will vary across platforms and thus we cannot assume that an archive
    // that contains a wrap storing a value in static storage on a platform can
    // be deserialised in static storage on another platform. Perhaps we can
    // add a function in the value interface to report the sizeof?
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        if constexpr (Cfg.semantics == wrap_semantics::value && Cfg.static_size > 0u) {
            // Store the storage type.
            ar << stype();
        }

        // Store the pointer to the value interface.
        ar << this->m_pv_iface;
    }
    // NOTE: as I have understood, when deserialising a pointer Boost
    // allocates memory only the *first* time the pointer is encountered
    // in an archive:
    // https://stackoverflow.com/questions/62105624/how-does-boostserialization-allocate-memory-when-deserializing-through-a-point
    // In our case, when employing value semantics we disable
    // object tracking for value_iface and I also
    // think that there are no situations in which a pointer to value_iface
    // could be shared amongst multiple wraps. Thus, we should be ok assuming
    // that the pointer coming out of des11n has always been created with a new
    // call.
    void load(boost::archive::binary_iarchive &ar, unsigned)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            if constexpr (Cfg.static_size == 0u) {
                // Load the serialised pointer.
                value_iface_t *pv_iface = nullptr;
                ar >> pv_iface;

                // NOTE: from now on, all is noexcept.

                // Destroy the current object.
                destroy();

                // Assign the deserialised pointer.
                this->m_pv_iface = pv_iface;
            } else {
                // Recover the storage type.
                bool st{};
                ar >> st;

                // Load the serialised pointer.
                value_iface_t *pv_iface = nullptr;
                ar >> pv_iface;

                // NOTE: the only way pv_iface can be null
                // is if the storage type is dynamic.
                assert(pv_iface != nullptr || !st);

                // NOTE: from now on, all is noexcept.

                // Destroy the current object.
                destroy();

                if (st) {
                    // Move-init the value from pv_iface.
                    this->m_pv_iface = std::move(*pv_iface)._tanuki_move_init_holder(this->static_storage);

                    // Clean up pv_iface.
                    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                    delete pv_iface;
                } else {
                    // Assign the deserialised pointer.
                    this->m_pv_iface = pv_iface;
                }
            }
        } else {
            // NOTE: not sure what the guarantees from Boost in case
            // of exceptions are here. Just in case, ensure we reset the wrap
            // to the invalid state in case of exceptions before rethrowing.
            try {
                ar >> this->m_pv_iface;
                // LCOV_EXCL_START
            } catch (...) {
                this->m_pv_iface = nullptr;
                throw;
            }
            // LCOV_EXCL_STOP
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

#endif

public:
    // Store the configuration.
    static constexpr auto cfg = Cfg;

    // Default initialisation into the invalid state.
    wrap() noexcept(detail::nothrow_default_initializable<ref_iface_t>)
        requires(Cfg.invalid_default_ctor) && std::default_initializable<ref_iface_t>
    {
        // NOTE: for reference semantics, the default ctor
        // of shared_ptr already does the right thing.
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            this->m_pv_iface = nullptr;
        }
    }

    // Default initialisation into the default value type.
    // NOTE: need to document that this value-inits.
    // NOTE: the extra W template argument appearing
    // here and in the other ctors/assignment operators
    // is to work around compiler issues: earlier versions
    // of clang error out complaining about an incomplete type
    // if we just use wrap instead of W. Older clang versions
    // also suffer from this bug
    //
    // https://github.com/llvm/llvm-project/issues/55945
    //
    // I.e., trailing-style concept checks may not short
    // circuit, which is particularly problematic for a default
    // constructor.
    template <typename W = wrap>
        requires(requires(W &w) {
            requires !Cfg.invalid_default_ctor;
            requires std::default_initializable<ref_iface_t>;
            // A default value type must have been specified
            // in the configuration.
            requires !std::same_as<void, default_value_t>;
            // We must be able to invoke the construction function.
            w.template ctor_impl<default_value_t>();
        })
    wrap() noexcept(noexcept(this->ctor_impl<default_value_t>()) && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<default_value_t>();
    }

    // Generic ctor from a wrappable value.
    template <typename T, typename W = wrap>
        requires(requires(W &w, T &&x) {
            requires std::default_initializable<ref_iface_t>;
            // Must not compete with the emplace ctor.
            requires !detail::is_in_place_type<std::remove_cvref_t<T>>::value;
            // Must not compete with copy/move.
            requires !std::same_as<std::remove_cvref_t<T>, W>;
            // We must be able to invoke the construction function.
            w.template ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
        })
    explicit(Cfg.explicit_ctor < wrap_ctor::always_implicit)
        // NOLINTNEXTLINE(bugprone-forwarding-reference-overload,google-explicit-constructor,hicpp-explicit-conversions)
        wrap(T &&x) noexcept(noexcept(this->ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x)))
                             && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
    }

    // Generic ctor from std::reference_wrapper.
    template <typename T, typename W = wrap>
        requires(requires(W &w, std::reference_wrapper<T> ref) {
            requires std::default_initializable<ref_iface_t>;
            // We must be able to invoke the construction function.
            w.template ctor_impl<std::reference_wrapper<T>>(std::move(ref));
        })
    explicit(Cfg.explicit_ctor == wrap_ctor::always_explicit)
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        wrap(std::reference_wrapper<T> ref) noexcept(
            noexcept(this->ctor_impl<std::reference_wrapper<T>>(std::move(ref)))
            && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<std::reference_wrapper<T>>(std::move(ref));
    }

    // Generic in-place initialisation.
    // NOTE: this will *value-init* if no args
    // are provided. This must be documented well.
    template <typename T, typename W = wrap, typename... U>
        requires(requires(W &w, U &&...args) {
            requires std::default_initializable<ref_iface_t>;
            // Forbid emplacing a wrap inside a wrap.
            requires !std::same_as<T, W>;
            w.template ctor_impl<T>(std::forward<U>(args)...);
        })
    explicit wrap(in_place_type<T>, U &&...args) noexcept(noexcept(this->ctor_impl<T>(std::forward<U>(args)...))
                                                          && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<T>(std::forward<U>(args)...);
    }

    wrap(const wrap &other) noexcept(Cfg.semantics == wrap_semantics::reference)
        requires(Cfg.copyable) && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            if constexpr (Cfg.static_size == 0u) {
                // Static storage disabled.
                this->m_pv_iface = other.m_pv_iface->_tanuki_clone();
            } else {
                if (other.stype()) {
                    // Other has static storage.
                    this->m_pv_iface = other.m_pv_iface->_tanuki_copy_init_holder(this->static_storage);
                } else {
                    // Other has dynamic storage.
                    this->m_pv_iface = other.m_pv_iface->_tanuki_clone();
                }
            }
        } else {
            this->m_pv_iface = other.m_pv_iface;
        }
    }

private:
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    void move_init_from(wrap &&other) noexcept
        requires(Cfg.semantics == wrap_semantics::value)
    {
        if constexpr (Cfg.static_size == 0u) {
            // Static storage disabled.
            // Shallow copy the pointer.
            this->m_pv_iface = other.m_pv_iface;

            // Invalidate other.
            other.m_pv_iface = nullptr;
        } else {
            auto *pv_iface = other.m_pv_iface;

            if (other.stype()) {
                // Other has static storage.
                this->m_pv_iface = std::move(*pv_iface)._tanuki_move_init_holder(this->static_storage);
            } else {
                // Other has dynamic storage.
                this->m_pv_iface = pv_iface;

                // Invalidate other.
                other.m_pv_iface = nullptr;
            }
        }
    }

public:
    wrap(wrap &&other) noexcept
        requires(Cfg.movable) && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            move_init_from(std::move(other));
        } else {
            this->m_pv_iface = std::move(other.m_pv_iface);
        }
    }

private:
    void destroy() noexcept
        requires(Cfg.semantics == wrap_semantics::value)
    {
        if constexpr (Cfg.static_size == 0u) {
            delete this->m_pv_iface;
        } else {
            if (stype()) {
                this->m_pv_iface->~value_iface_t();
            } else {
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                delete this->m_pv_iface;
            }
        }
    }

public:
    ~wrap()
        requires std::destructible<ref_iface_t>
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            destroy();
        }
    }

    // Move assignment.
    wrap &operator=(wrap &&other) noexcept
        requires(Cfg.movable)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
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

            // Handle different internal types (which means
            // in general also different storage types).
            if (value_type_index(*this) != value_type_index(other)) {
                destroy();
                move_init_from(std::move(other));
                return *this;
            }

            // The internal types are the same.
            if constexpr (Cfg.static_size == 0u) {
                // For dynamic storage, swap the pointer.
                std::swap(this->m_pv_iface, other.m_pv_iface);
            } else {
                // The storage flags must match, as they depend only
                // on the internal types.
                assert(stype() == other.stype());

                if (stype()) {
                    // For static storage, directly move assign the internal value.
                    std::move(*other.m_pv_iface)._tanuki_move_assign_value_to(this->m_pv_iface);
                } else {
                    // For dynamic storage, swap the pointer.
                    std::swap(this->m_pv_iface, other.m_pv_iface);
                }
            }
        } else {
            this->m_pv_iface = std::move(other.m_pv_iface);
        }

        return *this;
    }

    // Copy assignment.
    wrap &operator=(const wrap &other) noexcept(Cfg.semantics == wrap_semantics::reference)
        requires(Cfg.copyable)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Handle self-assign.
            if (this == std::addressof(other)) {
                return *this;
            }

            // Handle invalid object or different internal types.
            if (is_invalid(*this) || value_type_index(*this) != value_type_index(other)) {
                *this = wrap(other);
                return *this;
            }

            // The internal types are the same and this is valid.

            // NOTE: no need to branch on the storage type or static_size
            // here, as everything happens through the value interface.
            if constexpr (Cfg.static_size > 0u) {
                // The storage flags must match, as they depend only
                // on the internal types.
                assert(stype() == other.stype());
            }

            // Assign the internal value.
            other.m_pv_iface->_tanuki_copy_assign_value_to(this->m_pv_iface);
        } else {
            this->m_pv_iface = other.m_pv_iface;
        }

        return *this;
    }

    // Generic assignment.
    template <typename T, typename W = wrap>
        requires(requires(W &w, T &&x) {
            // NOTE: not 100% sure about this, but it seems consistent
            // for generic assignment to be enabled only if copy/move
            // assignment are as well.
            requires Cfg.copyable;
            requires Cfg.movable;
            // Must not compete with copy/move assignment.
            requires !std::same_as<std::remove_cvref_t<T>, W>;
            w.template ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
        })
    wrap &operator=(T &&x)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Handle invalid object.
            if (is_invalid(*this)) {
                ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
                return *this;
            }

            // Handle different types.
            if (value_type_index(*this) != typeid(detail::value_t_from_arg<T &&>)) {
                destroy();

                try {
                    ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
                } catch (...) {
                    // NOTE: if ctor_impl fails there's no cleanup required.
                    // Invalidate this before rethrowing.
                    this->m_pv_iface = nullptr;

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
                this->m_pv_iface->_tanuki_copy_assign_value_from(&fptr);
            } else {
                // The internal types are the same, do directly copy/move assignment.
                if constexpr (detail::noncv_rvalue_reference<T &&>) {
                    this->m_pv_iface->_tanuki_move_assign_value_from(std::addressof(x));
                } else {
                    this->m_pv_iface->_tanuki_copy_assign_value_from(std::addressof(x));
                }
            }
        } else {
            ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
        }

        return *this;
    }

    // Free functions interface.

    // NOTE: w is invalid when the value interface pointer is set to null.
    // This can happen if w has been moved from (note that this also includes the case
    // in which w has been swapped with an invalid object),
    // if generic assignment failed, or, in case of reference semantics,
    // if deserialisation threw an exception.
    // The only valid operations on an invalid object are:
    // - invocation of is_invalid(),
    // - destruction,
    // - copy/move assignment from, and swapping with, a valid wrap,
    // - generic assignment.
    [[nodiscard]] friend bool is_invalid(const wrap &w) noexcept
    {
        return w.m_pv_iface == nullptr;
    }

    [[nodiscard]] friend std::type_index value_type_index(const wrap &w) noexcept
    {
        return w.m_pv_iface->_tanuki_value_type_index();
    }

    [[nodiscard]] friend const iface_t *iface_ptr(const wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }
    [[nodiscard]] friend iface_t *iface_ptr(wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }

    friend void swap(wrap &w1, wrap &w2) noexcept
        requires(Cfg.swappable)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
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

            // Handle different internal types (which means in general
            // also different storage types) with the canonical swap()
            // implementation.
            if (value_type_index(w1) != value_type_index(w2)) {
                auto temp(std::move(w1));
                w1 = std::move(w2);
                w2 = std::move(temp);
                return;
            }

            // The types are the same.
            if constexpr (Cfg.static_size == 0u) {
                // For dynamic storage, swap the pointers.
                std::swap(w1.m_pv_iface, w2.m_pv_iface);
            } else {
                // The storage flags must match, as they depend only
                // on the internal types.
                assert(w1.stype() == w2.stype());

                if (w1.stype()) {
                    // For static storage, directly swap the internal values.
                    w2.m_pv_iface->_tanuki_swap_value(w1.m_pv_iface);
                } else {
                    // For dynamic storage, swap the pointers.
                    std::swap(w1.m_pv_iface, w2.m_pv_iface);
                }
            }
        } else {
            std::swap(w1.m_pv_iface, w2.m_pv_iface);
        }
    }

    [[nodiscard]] friend bool has_static_storage(const wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::reference || Cfg.static_size == 0u) {
            return false;
        } else {
            return w.stype();
        }
    }

    [[nodiscard]] friend const void *raw_value_ptr(const wrap &w) noexcept
    {
        return w.m_pv_iface->_tanuki_value_ptr();
    }
    [[nodiscard]] friend void *raw_value_ptr(wrap &w) noexcept
    {
        return w.m_pv_iface->_tanuki_value_ptr();
    }

    [[nodiscard]] friend bool contains_reference(const wrap &w) noexcept
    {
        return w.m_pv_iface->_tanuki_is_reference();
    }
};

namespace detail
{

template <typename>
struct is_any_wrap_impl : std::false_type {
};

template <typename IFace, auto Cfg>
struct is_any_wrap_impl<wrap<IFace, Cfg>> : std::true_type {
};

} // namespace detail

// Concept to detect any wrap instance.
template <typename T>
concept any_wrap = detail::is_any_wrap_impl<T>::value;

namespace detail
{

// Base class for iface_impl_helper.
struct iface_impl_helper_base {
};

// Helper to determine if the non-const value() getter
// overload in iface_impl_helper can be marked as noexcept.
template <typename Holder>
consteval bool iface_impl_value_getter_is_noexcept()
{
    using T = typename detail::holder_value<Holder>::type;

    if constexpr (detail::is_reference_wrapper<T>::value) {
        if constexpr (std::is_const_v<std::remove_reference_t<std::unwrap_reference_t<T>>>) {
            return false;
        }
    }

    return true;
}

} // namespace detail

// Helper that can be used to reduce typing in an
// interface implementation. This implements value()
// helpers for fetching the value held in Holder,
// automatically unwrapping it in case it is
// a std::reference_wrapper.
// NOTE: only Holder is strictly necessary here. However,
// we require Base to be passed in as well, so that,
// in composite interfaces:
// - we don't end up inheriting the same class from
//   mulitple places,
// - we can detect if iface_impl_helper has already been
//   used in an interface implementation.
template <typename Base, typename Holder>
struct iface_impl_helper : public detail::iface_impl_helper_base {
    template <typename H = Holder>
    auto &value() noexcept(detail::iface_impl_value_getter_is_noexcept<H>())
    {
        using T = typename detail::holder_value<Holder>::type;

        auto &val = static_cast<Holder *>(this)->m_value;

        if constexpr (detail::is_reference_wrapper<T>::value) {
            if constexpr (std::is_const_v<std::remove_reference_t<std::unwrap_reference_t<T>>>) {
                // NOLINTNEXTLINE(google-readability-casting)
                throw std::runtime_error("Invalid access to a const reference of type '"
                                         + detail::demangle(typeid(std::unwrap_reference_t<T>).name())
                                         + "' via a non-const member function");

                // LCOV_EXCL_START
                return *static_cast<unwrap_cvref_t<T> *>(nullptr);
                // LCOV_EXCL_STOP
            } else {
                return val.get();
            }
        } else {
            return val;
        }
    }
    const auto &value() const noexcept
    {
        using T = typename detail::holder_value<Holder>::type;

        const auto &val = static_cast<const Holder *>(this)->m_value;

        if constexpr (detail::is_reference_wrapper<T>::value) {
            return val.get();
        } else {
            return val;
        }
    }
};

// Specialisation of iface_impl_helper if Base derives
// from iface_impl_helper_base. This indicates that
// we are in a composite interface situation, in which we want
// to avoid defining another set of value() helpers (this would
// lead to ambiguities).
template <typename Base, typename Holder>
    requires std::derived_from<Base, detail::iface_impl_helper_base>
struct iface_impl_helper<Base, Holder> {
};

template <typename IFace, auto Cfg>
bool has_dynamic_storage(const wrap<IFace, Cfg> &w) noexcept
{
    return !has_static_storage(w);
}

template <typename T, typename IFace, auto Cfg>
const T *value_ptr(const wrap<IFace, Cfg> &w) noexcept
{
    return value_type_index(w) == typeid(T) ? static_cast<const T *>(raw_value_ptr(w)) : nullptr;
}

template <typename T, typename IFace, auto Cfg>
T *value_ptr(wrap<IFace, Cfg> &w) noexcept
{
    return value_type_index(w) == typeid(T) ? static_cast<T *>(raw_value_ptr(w)) : nullptr;
}

template <typename T, typename IFace, auto Cfg>
const T &value_ref(const wrap<IFace, Cfg> &w)
{
    const auto *ptr = value_ptr<T>(w);
    return ptr ? *ptr : throw std::bad_cast{};
}

template <typename T, typename IFace, auto Cfg>
T &value_ref(wrap<IFace, Cfg> &w)
{
    auto *ptr = value_ptr<T>(w);
    return ptr ? *ptr : throw std::bad_cast{};
}

template <typename T, typename IFace, auto Cfg>
bool value_isa(const wrap<IFace, Cfg> &w) noexcept
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

// NOTE: disable address tracking for value_iface when employing value semantics.
// We do not need it as value_iface pointers are never shared, and it might only create issues when
// deserialising into a function-local pointer which is then copied
// into the wrap storage.
template <typename IFace>
struct tracking_level<tanuki::detail::value_iface<IFace, tanuki::wrap_semantics::value>> {
    using tag = mpl::integral_c_tag;
    using type = mpl::int_<track_never>;
    BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
    BOOST_STATIC_ASSERT(
        (mpl::greater<implementation_level<tanuki::detail::value_iface<IFace, tanuki::wrap_semantics::value>>,
                      mpl::int_<primitive_type>>::value));
};

} // namespace boost::serialization

// NOTE: these are verbatim re-implementations of the BOOST_CLASS_EXPORT_KEY(2)
// and BOOST_CLASS_EXPORT_IMPLEMENT macros, which do not work well with class templates.
// NOTE: the use of __VA_ARGS__ here is kind of a hacky convenience, as it allows us
// to pass in arguments containing commas without additional mucking around.
// NOTE: in these macros we are always exporting/implementing both semantics
// variants.
#define TANUKI_S11N_WRAP_EXPORT_KEY(ud_type, ...)                                                                      \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <tanuki::wrap_semantics Sem>                                                                              \
    struct guid_defined<tanuki::detail::holder<ud_type, __VA_ARGS__, Sem>> : boost::mpl::true_ {                       \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>()             \
    {                                                                                                                  \
        return "tanuki::wrap<" #__VA_ARGS__ ">@" #ud_type "#val";                                                      \
    }                                                                                                                  \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>()         \
    {                                                                                                                  \
        return "tanuki::wrap<" #__VA_ARGS__ ">@" #ud_type "#ref";                                                      \
    }                                                                                                                  \
    }

#define TANUKI_S11N_WRAP_EXPORT_KEY2(ud_type, gid, ...)                                                                \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <tanuki::wrap_semantics Sem>                                                                              \
    struct guid_defined<tanuki::detail::holder<ud_type, __VA_ARGS__, Sem>> : boost::mpl::true_ {                       \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>()             \
    {                                                                                                                  \
        return gid "#val";                                                                                             \
    }                                                                                                                  \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>()         \
    {                                                                                                                  \
        return gid "#ref";                                                                                             \
    }                                                                                                                  \
    }

#define TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(ud_type, ...)                                                                \
    namespace boost::archive::detail::extra_detail                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> {                    \
        static guid_initializer<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> const &g; \
    };                                                                                                                 \
    template <>                                                                                                        \
    struct init_guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> {                \
        static guid_initializer<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> const \
            &g;                                                                                                        \
    };                                                                                                                 \
    guid_initializer<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> const                \
        &init_guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>::g                     \
        = ::boost::serialization::singleton<guid_initializer<                                                          \
            tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>>::get_mutable_instance()      \
              .export_guid();                                                                                          \
    guid_initializer<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> const            \
        &init_guid<tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>::g                 \
        = ::boost::serialization::singleton<guid_initializer<                                                          \
            tanuki::detail::holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>>::get_mutable_instance()  \
              .export_guid();                                                                                          \
    }

#define TANUKI_S11N_WRAP_EXPORT(ud_type, ...)                                                                          \
    TANUKI_S11N_WRAP_EXPORT_KEY(ud_type, __VA_ARGS__)                                                                  \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(ud_type, __VA_ARGS__)

#define TANUKI_S11N_WRAP_EXPORT2(ud_type, gid, ...)                                                                    \
    TANUKI_S11N_WRAP_EXPORT_KEY2(ud_type, gid, __VA_ARGS__)                                                            \
    TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(ud_type, __VA_ARGS__)

#endif

#undef TANUKI_ABI_TAG_ATTR
#undef TANUKI_NO_UNIQUE_ADDRESS
#undef TANUKI_VISIBLE

#endif
