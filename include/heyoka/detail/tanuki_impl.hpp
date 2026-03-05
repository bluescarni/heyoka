// Copyright 2023, 2024, 2025 Francesco Biscani (bluescarni@gmail.com)
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

// Headers for GCC-style demangle implementation. This is available also for clang, both with libstdc++ and libc++.
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

#if !defined(NDEBUG)

// NOTE: this is used in pointer alignment checks at runtime in debug mode.
#include <boost/align/is_aligned.hpp>

#endif

#endif

// Versioning.
#define TANUKI_VERSION_MAJOR 2
#define TANUKI_VERSION_MINOR 0
#define TANUKI_VERSION_PATCH 0
#define TANUKI_ABI_VERSION 2

// NOTE: indirection to allow token pasting/stringification:
//
// https://stackoverflow.com/questions/24991208/expand-a-macro-in-a-macro
#define TANUKI_VERSION_STRING_U(maj, min, pat) #maj "." #min "." #pat
#define TANUKI_VERSION_STRING_(maj, min, pat) TANUKI_VERSION_STRING_U(maj, min, pat)
#define TANUKI_VERSION_STRING TANUKI_VERSION_STRING_(TANUKI_VERSION_MAJOR, TANUKI_VERSION_MINOR, TANUKI_VERSION_PATCH)

// No unique address setup.
#if defined(_MSC_VER)

#define TANUKI_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]

#else

#define TANUKI_NO_UNIQUE_ADDRESS [[no_unique_address]]

#endif

// Detect the presence of C++23's "explicit this" feature.
//
// Normally can do this via the standard __cpp_explicit_this_parameter ifdef, except that for some reason MSVC>=19.32
// and clang>=18 support the feature but do not set the ifdef.
#if __cpp_explicit_this_parameter >= 202110L                                                                           \
    || (__clang_major__ >= 18 && __cplusplus >= 202302L && !defined(__apple_build_version__))                          \
    || (_MSC_VER >= 1932 && _MSVC_LANG > 202002L)

#define TANUKI_HAVE_EXPLICIT_THIS

#endif

// ABI tag setup.
#if defined(__GNUC__) || defined(__clang__)

#define TANUKI_ABI_TAG_ATTR __attribute__((abi_tag))

#else

#define TANUKI_ABI_TAG_ATTR

#endif

#define TANUKI_BEGIN_NAMESPACE_U(abiver)                                                                               \
    namespace tanuki                                                                                                   \
    {                                                                                                                  \
    inline namespace v##abiver TANUKI_ABI_TAG_ATTR                                                                     \
    {

#define TANUKI_BEGIN_NAMESPACE_(abiver) TANUKI_BEGIN_NAMESPACE_U(abiver)

#define TANUKI_BEGIN_NAMESPACE TANUKI_BEGIN_NAMESPACE_(TANUKI_ABI_VERSION)

#define TANUKI_END_NAMESPACE                                                                                           \
    }                                                                                                                  \
    }

#if defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"

#if !defined(__clang__)

#pragma GCC diagnostic ignored "-Wsuggest-final-methods"

#endif

#endif

// Visibility setup.
//
// NOTE: the idea here is as follows:
//
// - on Windows there is apparently no need to set up dllimport/dllexport on class templates;
// - on non-Windows platforms with known compilers, we mark several class templates as visible. This is apparently
//   necessary at least in some situations involving, for instance, the export/registration macros in Boost
//   serialisation. libc++ also, for instance, usually marks public class templates as visible;
// - otherwise, we do not implement any visibility attribute.
#if defined(_WIN32) || defined(__CYGWIN__)

#define TANUKI_VISIBLE

#elif defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)

#define TANUKI_VISIBLE __attribute__((visibility("default")))

#else

#define TANUKI_VISIBLE

#endif

TANUKI_BEGIN_NAMESPACE

// Helper to demangle a type name.
inline std::string demangle(const char *s)
{
#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))
    // NOTE: wrap std::free() in a local lambda, so we avoid potential ambiguities when taking the address of
    // std::free(). See:
    //
    // https://stackoverflow.com/questions/27440953/stdunique-ptr-for-c-functions-that-need-free
    //
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc, cppcoreguidelines-owning-memory, hicpp-no-malloc)
    auto deleter = [](void *ptr) { std::free(ptr); };

    // NOTE: abi::__cxa_demangle will return a pointer allocated by std::malloc, which we will delete via std::free().
    const std::unique_ptr<char, decltype(deleter)> res{::abi::__cxa_demangle(s, nullptr, nullptr, nullptr), deleter};

    // NOTE: return the original string if demangling fails.
    return res ? std::string(res.get()) : std::string(s);
#else
    // If no demangling is available, just return the mangled name.
    //
    // NOTE: MSVC already returns the demangled name from typeid.
    return std::string(s);
#endif
}

// Semantics for the wrap class.
//
// NOTE: this needs to be marked as visibile because the _tanuki_value_iface class depends on it. If we do not, we have
// the usual s11n-related visibility issues on OSX.
//
// NOLINTNEXTLINE(performance-enum-size)
enum class TANUKI_VISIBLE wrap_semantics { value, reference };

// Helper to unwrap a std::reference_wrapper and remove reference and cv qualifiers from the result.
template <typename T>
using unwrap_cvref_t = std::remove_cvref_t<std::unwrap_reference_t<T>>;

namespace detail
{

#if defined(TANUKI_HAVE_EXPLICIT_THIS)

// Implementation of std::forward_like(), at this time still missing in some compilers. See:
//
// https://en.cppreference.com/w/cpp/utility/forward_like
template <typename T, typename U>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
[[nodiscard]] constexpr auto &&forward_like(U &&x) noexcept
{
    constexpr bool is_adding_const = std::is_const_v<std::remove_reference_t<T>>;
    if constexpr (std::is_lvalue_reference_v<T &&>) {
        if constexpr (is_adding_const) {
            return std::as_const(x);
        } else {
            // NOLINTNEXTLINE(readability-redundant-casting)
            return static_cast<U &>(x);
        }
    } else {
        if constexpr (is_adding_const) {
            return std::move(std::as_const(x));
        } else {
            // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
            return std::move(x);
        }
    }
}

#endif

// LCOV_EXCL_START

// std::unreachable() implementation:
//
// https://en.cppreference.com/w/cpp/utility/unreachable
[[noreturn]] inline void unreachable()
{
#if defined(__GNUC__) || defined(__clang__)
    __builtin_unreachable();
#elif defined(_MSC_VER)
    __assume(false);
#endif
}

// LCOV_EXCL_STOP

// Type-trait to detect instances of std::reference_wrapper.
template <typename>
inline constexpr bool is_reference_wrapper_v = false;

template <typename T>
inline constexpr bool is_reference_wrapper_v<std::reference_wrapper<T>> = true;

// Implementation of the concept to detect any wrap instance. This will be specialised after the definition of the wrap
// class.
template <typename>
inline constexpr bool is_any_wrap_v = false;

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-final-methods"
#pragma GCC diagnostic ignored "-Wsuggest-final-types"

#endif

// NOTE: in traditional OOP, the class hierarchy would be:
//
// iface_impl -> iface
//
// In order to implement type erasure, we are interposing additional classes to the traditional OOP hierarchy:
//
// _tanuki_holder -> iface_impl -> _tanuki_value_iface -> iface
//
// _tanuki_value_iface contains the abstract functions used to interact with the type-erased value. _tanuki_holder
// stores the type-erased value and implements the _tanuki_value_iface interface. The wrap class stores an instance of
// _tanuki_holder and contains a pointer to _tanuki_value_iface, so that it can use both the iface and
// _tanuki_value_iface API.

// Interface for interacting with type-erased values.
//
// NOTE: we need to template _tanuki_value_iface on the semantics so that we can selectively disable address tracking in
// Boost.serialisation when employing value semantics (we do not want to disable it for reference semantics since we
// need it for correct shared_ptr s11n).
template <typename IFace, wrap_semantics Sem>
struct TANUKI_VISIBLE _tanuki_value_iface : public IFace {
    _tanuki_value_iface() = default;
    _tanuki_value_iface(const _tanuki_value_iface &) = delete;
    _tanuki_value_iface(_tanuki_value_iface &&) noexcept = delete;
    _tanuki_value_iface &operator=(const _tanuki_value_iface &) = delete;
    _tanuki_value_iface &operator=(_tanuki_value_iface &&) noexcept = delete;
    // NOTE: it is important that this is virtual because we will be deleting through pointers to _tanuki_value_iface.
    // Mark it also as noexcept as we do not want to bother supporting values/interfaces which may throw on destruction.
    //
    // NOLINTNEXTLINE(hicpp-use-override,modernize-use-override)
    virtual ~_tanuki_value_iface() noexcept = default;

    // NOTE: we want to provide an implementation for the virtual functions (instead of keeping them pure virtual). This
    // allows us to check for correct interface implementations through their default-constructibility, and to determine
    // the noexcept-ness of the _tanuki_holder constructors.

    // LCOV_EXCL_START

    // Access to the value and its type.
    [[nodiscard]] virtual void *_tanuki_value_ptr() noexcept
    {
        unreachable();
    }
    [[nodiscard]] virtual std::type_index _tanuki_value_type_index() const noexcept
    {
        unreachable();
    }
    [[nodiscard]] virtual bool _tanuki_value_is_reference() const noexcept
    {
        unreachable();
    }

    // Methods to implement virtual copy/move primitives for the _tanuki_holder class.
    [[nodiscard]] virtual _tanuki_value_iface *_tanuki_clone_holder() const
    {
        unreachable();
    }
    [[nodiscard]] virtual std::shared_ptr<_tanuki_value_iface> _tanuki_shared_clone_holder() const
    {
        unreachable();
    }
    [[nodiscard]] virtual _tanuki_value_iface *_tanuki_copy_init_holder(void *) const
    {
        unreachable();
    }
    [[nodiscard]] virtual _tanuki_value_iface *_tanuki_move_init_holder(void *) && noexcept
    {
        unreachable();
    }
    virtual bool _tanuki_copy_assign_value_to(_tanuki_value_iface *) const
    {
        unreachable();
    }
    virtual bool _tanuki_move_assign_value_to(_tanuki_value_iface *) && noexcept
    {
        unreachable();
    }
    virtual bool _tanuki_copy_assign_value_from(const void *)
    {
        unreachable();
    }
    virtual bool _tanuki_move_assign_value_from(void *) noexcept
    {
        unreachable();
    }
    virtual bool _tanuki_swap_value(_tanuki_value_iface *) noexcept
    {
        unreachable();
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    // NOTE: these are used to check that a value type is serialisable.
    [[nodiscard]] virtual bool _tanuki_value_is_default_initializable() const noexcept
    {
        unreachable();
    }
    [[nodiscard]] virtual bool _tanuki_value_is_move_constructible() const noexcept
    {
        unreachable();
    }

#endif

    // LCOV_EXCL_STOP
};

#if defined(TANUKI_WITH_BOOST_S11N)

// NOTE: keep serialisation outside the class in order not to pollute the internal namespace.
template <typename Archive, typename IFace, wrap_semantics Sem>
void serialize(Archive &, _tanuki_value_iface<IFace, Sem> &, const unsigned)
{
}

#endif

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic pop

#endif

// Concept to detect if a type satisfies std::default_initializable without throwing.
template <typename T>
concept nothrow_default_initializable
    = std::default_initializable<T> && noexcept(::new (static_cast<void *>(nullptr)) T)
      && std::is_nothrow_constructible_v<T> && noexcept(T{});

// Concept to detect if T is an rvalue reference without cv qualifications.
template <typename T>
concept noncv_rvalue_reference
    = std::is_rvalue_reference_v<T> && std::same_as<std::remove_cvref_t<T>, std::remove_reference_t<T>>;

} // namespace detail

// Composite interface.
template <typename IFace0, typename IFace1, typename... IFaceN>
// NOLINTNEXTLINE(misc-multiple-inheritance,cppcoreguidelines-virtual-class-destructor)
struct TANUKI_VISIBLE composite_iface : public IFace0, public IFace1, public IFaceN... {
};

// Concept to detect any wrap instance.
template <typename T>
concept any_wrap = detail::is_any_wrap_v<T>;

// Concept checking for value types. Must be non-cv qualified destructible objects.
template <typename T>
concept valid_value_type
    = std::is_object_v<T> && (!std::is_const_v<T>) && (!std::is_volatile_v<T>) && std::destructible<T>;

namespace detail
{

// Detection of a composite interface.
template <typename>
inline constexpr bool is_composite_interface_v = false;

template <typename IFace0, typename IFace1, typename... IFaceN>
inline constexpr bool is_composite_interface_v<composite_iface<IFace0, IFace1, IFaceN...>> = true;

// Private base for the unspecialised iface_impl.
struct iface_impl_base {
};

} // namespace detail

// Definition of the external interface implementation customisation point. Tihs derives from detail::iface_impl_base
// in order to detect specialisations.
//
// NOTE: prohibit the definition of an external implementation for the composite interface.
template <typename IFace, typename Base, typename T>
    requires(!detail::is_composite_interface_v<IFace>)
struct TANUKI_VISIBLE iface_impl final : detail::iface_impl_base {
};

namespace detail
{

// NOTE: this section contains the metaprogramming necessary to determine whether or not an interface has an
// implementation, and to automatically synthesise composite interface implementations.

// Detect the presence of an external or intrusive interface implementation.
//
// NOTE: at this stage, we are only checking for the existence of a specialisation of iface_impl (external) or an 'impl'
// template (intrusive). Further checks are implemented later.
template <typename IFace, typename Base, typename T>
concept iface_has_external_impl = !std::derived_from<iface_impl<IFace, Base, T>, iface_impl_base>;

template <typename IFace, typename Base, typename T>
concept iface_has_intrusive_impl = requires() { typename IFace::template impl<Base, T>; };

// Helper to fetch the implementation of a non-composite interface
template <typename, typename, typename>
struct get_nc_iface_impl {
};

// External interface implementation.
//
// NOTE: this will take the precedence in case an intrusive implementation is also available.
template <typename IFace, typename Base, typename T>
    requires iface_has_external_impl<IFace, Base, T>
struct get_nc_iface_impl<IFace, Base, T> {
    using type = iface_impl<IFace, Base, T>;
};

// Intrusive interface implementation.
template <typename IFace, typename Base, typename T>
    requires iface_has_intrusive_impl<IFace, Base, T> && (!iface_has_external_impl<IFace, Base, T>)
struct get_nc_iface_impl<IFace, Base, T> {
    using type = IFace::template impl<Base, T>;
};

template <typename IFace, typename Base, typename T>
concept with_external_or_intrusive_iface_impl = requires() { typename get_nc_iface_impl<IFace, Base, T>::type; };

// Meta-programming to select the implementation of an interface.
template <typename, typename, wrap_semantics>
struct impl_from_iface_impl {
};

// NOTE: this is a typed wrapper for _tanuki_value_iface.
//
// In the code below, we will ensure that any valid interface implementation derives from _tanuki_typed_value_iface.
// This ensures that a pointer to an interface implementation is implicitly convertible to _tanuki_typed_value_iface,
// which in turn allows us - in the implementation of getval() - to static_cast a pointer to an implementation to the
// holder for that implementation. We would not be able to perform this static_cast without the information about T.
template <typename T, typename IFace, wrap_semantics Sem>
struct TANUKI_VISIBLE _tanuki_typed_value_iface : _tanuki_value_iface<IFace, Sem> {
};

// For non-composite interfaces, the Base for the interface implementation is _tanuki_typed_value_iface<T, IFace, Sem>
// (which transitively makes IFace also a base for the implementation).
template <typename IFace, typename T, wrap_semantics Sem>
    requires
    // NOTE: we add an initial concept check on IFace here in order to avoid instantiating the
    // with_external_or_intrusive_iface_impl concept with a composite interface. This seems to confuse some compilers
    // (e.g., MSVC) since the composite interface may have several 'impl' typedefs inherited from the individual
    // interfaces.
    (!is_composite_interface_v<IFace>)
    && with_external_or_intrusive_iface_impl<IFace, _tanuki_typed_value_iface<T, IFace, Sem>, T>
    struct impl_from_iface_impl<IFace, T, Sem> : get_nc_iface_impl<IFace, _tanuki_typed_value_iface<T, IFace, Sem>, T> {
};

// For composite interfaces, we synthesize a class hierarchy in which every implementation derives from the previous
// one, and the first implementation derives from _tanuki_typed_value_iface of the composite interface.
template <typename T, typename CurIFace, typename CurBase, typename NextIFace, typename... IFaceN>
struct c_iface_assembler {
};

template <typename T, typename CurIFace, typename CurBase, typename NextIFace, typename... IFaceN>
    requires requires() {
        requires with_external_or_intrusive_iface_impl<CurIFace, CurBase, T>;
        typename c_iface_assembler<T, NextIFace, typename get_nc_iface_impl<CurIFace, CurBase, T>::type,
                                   IFaceN...>::type;
    }
struct c_iface_assembler<T, CurIFace, CurBase, NextIFace, IFaceN...> {
    using cur_impl = get_nc_iface_impl<CurIFace, CurBase, T>::type;
    using type = c_iface_assembler<T, NextIFace, cur_impl, IFaceN...>::type;
};

template <typename T, typename CurIFace, typename CurBase, typename LastIFace>
    requires requires() {
        requires with_external_or_intrusive_iface_impl<CurIFace, CurBase, T>;
        typename get_nc_iface_impl<LastIFace, typename get_nc_iface_impl<CurIFace, CurBase, T>::type, T>::type;
    }
struct c_iface_assembler<T, CurIFace, CurBase, LastIFace> {
    using cur_impl = get_nc_iface_impl<CurIFace, CurBase, T>::type;
    using type = get_nc_iface_impl<LastIFace, cur_impl, T>::type;
};

template <typename T, wrap_semantics Sem, typename IFace0, typename IFace1, typename... IFaceN>
    requires requires() {
        typename c_iface_assembler<T, IFace0,
                                   _tanuki_typed_value_iface<T, composite_iface<IFace0, IFace1, IFaceN...>, Sem>,
                                   IFace1, IFaceN...>::type;
    }
struct impl_from_iface_impl<composite_iface<IFace0, IFace1, IFaceN...>, T, Sem> {
    using type
        = c_iface_assembler<T, IFace0, _tanuki_typed_value_iface<T, composite_iface<IFace0, IFace1, IFaceN...>, Sem>,
                            IFace1, IFaceN...>::type;
};

// Helper alias.
template <typename IFace, typename T, wrap_semantics Sem>
    requires requires() { typename impl_from_iface_impl<IFace, T, Sem>::type; }
using impl_from_iface = impl_from_iface_impl<IFace, T, Sem>::type;

// Concept to check that the interface IFace has an implementation for the value type T.
template <typename IFace, typename T, wrap_semantics Sem>
concept iface_has_impl = requires() {
    // NOTE: include the check on the validity of the value type.
    requires valid_value_type<T>;
    typename impl_from_iface<IFace, T, Sem>;
    // The implementation must derive from the interface.
    requires std::derived_from<impl_from_iface<IFace, T, Sem>, IFace>;
    // This will check that:
    //
    // - we can rely on default-initialisation of the interface implementation in the constructors of the holder class,
    // - the interface implementation implements the interface.
    requires std::default_initializable<impl_from_iface<IFace, T, Sem>>;
};

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

// Class for holding type-erased values.
//
// This implements the _tanuki_value_iface interface.
template <typename T, typename IFace, wrap_semantics Sem>
    requires iface_has_impl<IFace, T, Sem>
struct TANUKI_VISIBLE _tanuki_holder final : public impl_from_iface<IFace, T, Sem> {
    TANUKI_NO_UNIQUE_ADDRESS T _tanuki_value;

    // Make sure we don't end up accidentally copying/moving this class.
    _tanuki_holder(const _tanuki_holder &) = delete;
    _tanuki_holder(_tanuki_holder &&) noexcept = delete;
    _tanuki_holder &operator=(const _tanuki_holder &) = delete;
    _tanuki_holder &operator=(_tanuki_holder &&) noexcept = delete;
    ~_tanuki_holder() noexcept final = default;

// NOTE: silence false positives on gcc.
#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"

#endif

    // NOTE: in these constructors we don't have to check for default-constructibility of the interface implementation
    // because this is already checked in the iface_has_impl concept.
    //
    // NOTE: special-casing to avoid the single-argument ctor potentially competing with the copy/move ctors.
    template <typename U>
        requires(!std::same_as<_tanuki_holder, std::remove_cvref_t<U>>) && std::constructible_from<T, U &&>
    explicit _tanuki_holder(U &&x) noexcept(std::is_nothrow_constructible_v<T, U &&>
                                            && nothrow_default_initializable<impl_from_iface<IFace, T, Sem>>)
        : _tanuki_value(std::forward<U>(x))
    {
    }
    template <typename... U>
        requires(sizeof...(U) != 1u) && std::constructible_from<T, U &&...>
    explicit _tanuki_holder(U &&...x) noexcept(std::is_nothrow_constructible_v<T, U &&...>
                                               && nothrow_default_initializable<impl_from_iface<IFace, T, Sem>>)
        : _tanuki_value(std::forward<U>(x)...)
    {
    }

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic pop

#endif

    // Access to the value and its type.
    [[nodiscard]] std::type_index _tanuki_value_type_index() const noexcept final
    {
        return typeid(T);
    }
    [[nodiscard]] void *_tanuki_value_ptr() noexcept final
    {
        return static_cast<void *>(std::addressof(_tanuki_value));
    }

    [[nodiscard]] bool _tanuki_value_is_reference() const noexcept final
    {
        return is_reference_wrapper_v<T>;
    }

    // Copy/move construction primitives.

    // Clone this, and cast the result to the value interface.
    [[nodiscard]] _tanuki_value_iface<IFace, Sem> *_tanuki_clone_holder() const final
    {
        // NOTE: we don't need to check the constructibility of the holder object: since the holder object already
        // exists, we already know that there is a default-constructible interface implementation.
        if constexpr (std::is_copy_constructible_v<T>) {
            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
            return new _tanuki_holder(_tanuki_value);
        }

        // NOTE: we should never reach this point as we are using this function only with value semantics and we are
        // forbidding the creation of a copyable value semantics wrap from a non-copyable value.
        unreachable(); // LCOV_EXCL_LINE
    }
    // Same as above, but return a shared ptr.
    [[nodiscard]] std::shared_ptr<_tanuki_value_iface<IFace, Sem>> _tanuki_shared_clone_holder() const final
    {
        if constexpr (std::is_copy_constructible_v<T>) {
            return std::make_shared<_tanuki_holder>(_tanuki_value);
        } else {
            // NOTE: this is the one case in which we might end up here at runtime. This function is used only in the
            // implementation of the copy() function to force deep copy behaviour when employing reference semantics.
            // But, when reference semantics is active, we are always allowing the construction of a wrap regardless of
            // the copyability of the value type. Hence, we might end up attempting to deep-copy a wrap containing a
            // non-copyable value.
            throw std::invalid_argument("Attempting to clone a non-copyable value type");
        }
    }
    // Copy-init a new holder from this into the storage beginning at ptr. Then cast the result to the value interface
    // and return.
    [[nodiscard]] _tanuki_value_iface<IFace, Sem> *_tanuki_copy_init_holder(void *ptr) const final
    {
        if constexpr (std::is_copy_constructible_v<T>) {
#if defined(TANUKI_WITH_BOOST_S11N)
            assert(boost::alignment::is_aligned(ptr, alignof(T)));
#endif

            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,clang-analyzer-cplusplus.PlacementNew)
            return ::new (ptr) _tanuki_holder(_tanuki_value);
        }

        // NOTE: we should never reach this point as we are using this function only with value semantics and we are
        // forbidding the creation of a copyable value semantics wrap from a non-copyable value.
        unreachable(); // LCOV_EXCL_LINE
    }
    // Move-init a new holder from this into the storage beginning at ptr. Then cast the result to the value interface
    // and return.
    //
    // NOTE: currently we mark this as noexcept, which will lead to clang-tidy warnings if the type-erased value throws
    // on move construction. We do not want to handle the complexity of types with throwing move constructors, hence we
    // silence the warning.
    //
    // NOLINTNEXTLINE(bugprone-exception-escape)
    [[nodiscard]] _tanuki_value_iface<IFace, Sem> *_tanuki_move_init_holder(void *ptr) && noexcept final
    {
        if constexpr (std::is_move_constructible_v<T>) {
#if defined(TANUKI_WITH_BOOST_S11N)
            assert(boost::alignment::is_aligned(ptr, alignof(T)));
#endif

            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,clang-analyzer-cplusplus.PlacementNew)
            return ::new (ptr) _tanuki_holder(std::move(_tanuki_value));
        }

        // NOTE: we should never reach this point as we are using this function only with value semantics and we are
        // forbidding the creation of a movable value semantics wrap from a non-movable value.
        unreachable(); // LCOV_EXCL_LINE
    }

    // Copy/move assignment and swap primitives.

    // If T is copy-assignable, copy-assign _tanuki_value into the _tanuki_value of v_iface and return true. Otherwise,
    // do nothing and return false.
    bool _tanuki_copy_assign_value_to(_tanuki_value_iface<IFace, Sem> *v_iface) const final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            // NOTE: I don't think it is necessary to invoke launder here, as value_ptr() just does a static cast to
            // void *. Since we are assuming that copy_assign_value_to() is called only when assigning holders
            // containing the same T, the conversion chain should boil down to T * -> void * -> T *, which does not
            // require laundering.
            assert(typeid(T) == v_iface->_tanuki_value_type_index());
            *static_cast<T *>(v_iface->_tanuki_value_ptr()) = _tanuki_value;
            return true;
        } else {
            return false;
        }
    }
    // If T is move-assignable, move-assign _tanuki_value into the _tanuki_value of v_iface and return true. Otherwise,
    // do nothing and return false.
    bool _tanuki_move_assign_value_to(_tanuki_value_iface<IFace, Sem> *v_iface) && noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            assert(typeid(T) == v_iface->_tanuki_value_type_index());
            *static_cast<T *>(v_iface->_tanuki_value_ptr()) = std::move(_tanuki_value);
            return true;
        } else {
            return false;
        }
    }
    // If T is copy-assignable, copy-assign the object of type T assumed to be stored in ptr into _tanuki_value and
    // return true. Otherwise, return false.
    bool _tanuki_copy_assign_value_from(const void *ptr) final
    {
        if constexpr (std::is_copy_assignable_v<T>) {
            _tanuki_value = *static_cast<const T *>(ptr);
            return true;
        } else {
            return false;
        }
    }
    // If T is move-assignable, move-assign the object of type T assumed to be stored in ptr into _tanuki_value and
    // return true. Otherwise, return false.
    bool _tanuki_move_assign_value_from(void *ptr) noexcept final
    {
        if constexpr (std::is_move_assignable_v<T>) {
            _tanuki_value = std::move(*static_cast<T *>(ptr));
            return true;
        } else {
            return false;
        }
    }
    // If T is swappable, swap _tanuki_value with the _tanuki_value of v_iface and return true. Otherwise, return false.
    bool _tanuki_swap_value(_tanuki_value_iface<IFace, Sem> *v_iface) noexcept final
    {
        if constexpr (std::swappable<T>) {
            assert(typeid(T) == v_iface->_tanuki_value_type_index());

            using std::swap;
            swap(_tanuki_value, *static_cast<T *>(v_iface->_tanuki_value_ptr()));

            return true;
        } else {
            // NOTE: at the moment I cannot find a way to trigger this, because we end up here only if a swappable wrap
            // contains a non-swappable value. But: a wrap is marked as swappable only if it is move ctible/assignable,
            // which requires a move ctible/assignable value, which (almost?) always means that the value is swappable
            // as well. There may be some way to construct a pathological type that triggers this branch, but so far I
            // have not found it.
            return false; // LCOV_EXCL_LINE
        }
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    [[nodiscard]] bool _tanuki_value_is_default_initializable() const noexcept final
    {
        return std::default_initializable<T>;
    }
    [[nodiscard]] bool _tanuki_value_is_move_constructible() const noexcept final
    {
        return std::is_move_constructible_v<T>;
    }

#endif
};

#if defined(TANUKI_WITH_BOOST_S11N)

// Serialization.
//
// NOTE: keep it outside the class in order not to pollute the namespace.
template <typename Archive, typename T, typename IFace, wrap_semantics Sem>
void serialize(Archive &ar, _tanuki_holder<T, IFace, Sem> &self, const unsigned)
{
    // NOTE: here we are "skipping" the serialisation of intermediate classes in the hierarchy and going directly to the
    // serialisation of _tanuki_value_iface. This is ok, as we do not support state in the interface or its
    // implementation.
    ar &boost::serialization::base_object<_tanuki_value_iface<IFace, Sem>>(self);
    ar & self._tanuki_value;
}

#endif

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#elif defined(_MSC_VER)

#pragma warning(pop)

#endif

// Helper to determine if the non-const overload of getval() is noexcept.
template <typename T>
consteval bool getval_is_noexcept()
{
    if constexpr (is_reference_wrapper_v<T>) {
        return !std::is_const_v<std::remove_reference_t<std::unwrap_reference_t<T>>>;
    } else {
        return true;
    }
}

// Getters for the type-erased value, to be used in the implementation of interfaces.
template <typename T, typename IFace, wrap_semantics Sem>
    requires std::derived_from<_tanuki_holder<T, IFace, Sem>, _tanuki_typed_value_iface<T, IFace, Sem>>
[[nodiscard]] const auto &getval(const _tanuki_typed_value_iface<T, IFace, Sem> *h) noexcept
{
    assert(h != nullptr);
    assert((dynamic_cast<const _tanuki_holder<T, IFace, Sem> *>(h) != nullptr));

    const auto &val = static_cast<const _tanuki_holder<T, IFace, Sem> *>(h)->_tanuki_value;

    if constexpr (is_reference_wrapper_v<T>) {
        return val.get();
    } else {
        return val;
    }
}

template <typename T, typename IFace, wrap_semantics Sem>
    requires std::derived_from<_tanuki_holder<T, IFace, Sem>, _tanuki_typed_value_iface<T, IFace, Sem>>
[[nodiscard]] auto &getval(_tanuki_typed_value_iface<T, IFace, Sem> *h) noexcept(getval_is_noexcept<T>())
{
    assert(h != nullptr);
    assert((dynamic_cast<_tanuki_holder<T, IFace, Sem> *>(h) != nullptr));

    auto &val = static_cast<_tanuki_holder<T, IFace, Sem> *>(h)->_tanuki_value;

    if constexpr (is_reference_wrapper_v<T>) {
        if constexpr (std::is_const_v<std::remove_reference_t<std::unwrap_reference_t<T>>>) {
            // NOLINTNEXTLINE(google-readability-casting)
            throw std::runtime_error("Invalid access to a const reference of type '"
                                     + demangle(typeid(std::unwrap_reference_t<T>).name())
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

} // namespace detail

// Concept to check that the interface IFace has an implementation for the value type T.
//
// NOTE: like in the holder_size helper, we check that the implementation exists for both semantics types. At this time
// it is not possible for an implementation to exist only for one semantics type and hence the double check is
// superfluous, but let us just keep it for consistency.
template <typename IFace, typename T>
concept iface_with_impl = detail::iface_has_impl<IFace, T, wrap_semantics::value>
                          && detail::iface_has_impl<IFace, T, wrap_semantics::reference>;

namespace detail
{

// Implementation of storage for the wrap class. This will be used to store an instance of the holder type.
template <typename IFace, std::size_t StaticStorageSize, std::size_t StaticStorageAlignment, wrap_semantics Sem>
struct TANUKI_VISIBLE wrap_storage {
    static_assert(StaticStorageSize > 0u);
    static_assert(Sem == wrap_semantics::value);

    // Static storage optimisation enabled.
    //
    // The active storage is dynamic if either m_pv_iface is null (which indicates the invalid state) or if it points
    // somewhere outside static_storage. Otherwise, the active storage is static and m_pv_iface points somewhere within
    // static_storage.
    _tanuki_value_iface<IFace, Sem> *m_pv_iface;
    alignas(StaticStorageAlignment) std::byte static_storage[StaticStorageSize];
};

template <typename IFace, std::size_t StaticStorageAlignment>
struct TANUKI_VISIBLE wrap_storage<IFace, 0, StaticStorageAlignment, wrap_semantics::value> {
    _tanuki_value_iface<IFace, wrap_semantics::value> *m_pv_iface;
};

template <typename IFace, std::size_t StaticStorageSize, std::size_t StaticStorageAlignment>
struct TANUKI_VISIBLE wrap_storage<IFace, StaticStorageSize, StaticStorageAlignment, wrap_semantics::reference> {
    std::shared_ptr<_tanuki_value_iface<IFace, wrap_semantics::reference>> m_pv_iface;
};

// NOTE: this is used to check that a config instance is a specialisation from the primary config template.
struct config_base {
};

} // namespace detail

// Helpers to determine the size and alignment of a holder instance, given the value type T and the interface IFace.
//
// NOTE: here we have the complication that holder technically depends on the wrap semantics. We do not want to
// complicate the interface of these helpers requiring the user to pass in the semantics as well, so instead we assert
// that size/alignment are the same regardless of semantics (which should always be the case).
template <typename T, typename IFace>
    requires iface_with_impl<IFace, T>
                 && (sizeof(detail::_tanuki_holder<T, IFace, wrap_semantics::value>)
                     == sizeof(detail::_tanuki_holder<T, IFace, wrap_semantics::reference>))
inline constexpr auto holder_size = sizeof(detail::_tanuki_holder<T, IFace, wrap_semantics::value>);

template <typename T, typename IFace>
    requires iface_with_impl<IFace, T>
                 && (alignof(detail::_tanuki_holder<T, IFace, wrap_semantics::value>)
                     == alignof(detail::_tanuki_holder<T, IFace, wrap_semantics::reference>))
inline constexpr auto holder_align = alignof(detail::_tanuki_holder<T, IFace, wrap_semantics::value>);

// Default implementation of the reference interface.
struct TANUKI_VISIBLE no_ref_iface {
    template <typename>
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
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
//
// NOLINTNEXTLINE(performance-enum-size)
enum class TANUKI_VISIBLE wrap_ctor { always_explicit, ref_implicit, always_implicit };

namespace detail
{

// NOTE: the machinery in this section is used to detect if a type defines in its scope a template type/typedef called
// "impl" which depends on a single parameter. In order to do this, we exploit the fact that if during concept checking
// a substitution error occurs, then the concept is considered not satisfied.

template <template <typename> typename>
struct single_tt {
};

// NOTE: the purpose of this concept is to yield alwas true for any input template-template TT depending on a single
// parameter. We cannot simply use "concept single_tt_id = true" because of reasons that have to do with constraint
// normalisation and illustrated partly here:
//
// https://stackoverflow.com/questions/69823200/gcc-disagrees-with-clang-and-msvc-when-concept-thats-always-true-is-used-to-imp
// https://stackoverflow.com/questions/75442605/c20-concepts-constraint-normalization
//
// Basically, with "concept single_tt_id = true" during the normalisation phase we would end up with the atomic
// constraint "true" and an empty parameter mapping. Thus, it does not matter whether or not the "impl" type/typedef
// exists or not, the concept will always be satisfied because no substitution takes place (due to the parameter mapping
// being empty).
template <template <typename> typename TT>
concept single_tt_id = requires() { typename single_tt<TT>; };

// NOTE: this is where we are checking that T defines an "impl" template in its scope.
template <typename T>
concept with_impl_tt = single_tt_id<T::template impl>;

} // namespace detail

// Concept for checking that RefIFace is a valid reference interface. In C++>=23, anything goes, in C++20 we need to
// make sure the impl typedef exists.
template <typename RefIFace>
concept valid_ref_iface = std::is_class_v<RefIFace> && std::same_as<RefIFace, std::remove_cv_t<RefIFace>>
#if !defined(TANUKI_HAVE_EXPLICIT_THIS)
                          && detail::with_impl_tt<RefIFace>
#endif
    ;

// Configuration settings for the wrap class.
//
// NOTE: the DefaultValueType is subject to the constraints for valid value types.
template <typename DefaultValueType = void, typename RefIFace = no_ref_iface>
    requires(std::same_as<DefaultValueType, void> || valid_value_type<DefaultValueType>) && valid_ref_iface<RefIFace>
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
    // Enable copy construction.
    bool copy_constructible = true;
    // Enable copy assignment.
    bool copy_assignable = true;
    // Enable move construction.
    bool move_constructible = true;
    // Enable move assignment.
    bool move_assignable = true;
};

// Default configuration for the wrap class.
inline constexpr auto default_config = config{};

namespace detail
{

template <std::size_t N>
concept power_of_two = (N > 0u) && ((N & (N - 1u)) == 0u);

} // namespace detail

// Concept for checking that Cfg is a valid config instance.
template <auto Cfg>
concept valid_config =
    // This checks that decltype(Cfg) is a specialisation from the primary config template.
    std::derived_from<std::remove_const_t<decltype(Cfg)>, detail::config_base> &&
    // The static alignment value must be a power of 2.
    detail::power_of_two<Cfg.static_align> &&
    // Cfg.explicit_ctor must be set to one of the valid enumerators.
    (Cfg.explicit_ctor >= wrap_ctor::always_explicit && Cfg.explicit_ctor <= wrap_ctor::always_implicit) &&
    // Cfg.semantics must be one of the two valid enumerators.
    (Cfg.semantics == wrap_semantics::value || Cfg.semantics == wrap_semantics::reference) &&
    // Copy-assignability requires copy-constructibility.
    (!Cfg.copy_assignable || Cfg.copy_constructible) &&
    // Move-assignability requires move-constructibility.
    (!Cfg.move_assignable || Cfg.move_constructible);

// Helpers to ease the definition of a reference interface.
#define TANUKI_REF_IFACE_MEMFUN(name)                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) & noexcept(                                                                        \
        noexcept(iface_ptr(*static_cast<JustWrap *>(this)) -> name(std::forward<MemFunArgs>(args)...)))                \
        ->decltype(iface_ptr(*static_cast<JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...))                 \
    {                                                                                                                  \
        return iface_ptr(*static_cast<Wrap *>(this))->name(std::forward<MemFunArgs>(args)...);                         \
    }                                                                                                                  \
    template <typename JustWrap = Wrap, typename... MemFunArgs>                                                        \
    auto name(MemFunArgs &&...args) const & noexcept(                                                                  \
        noexcept(iface_ptr(*static_cast<const JustWrap *>(this)) -> name(std::forward<MemFunArgs>(args)...)))          \
        ->decltype(iface_ptr(*static_cast<const JustWrap *>(this))->name(std::forward<MemFunArgs>(args)...))           \
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

#if defined(TANUKI_HAVE_EXPLICIT_THIS)

// NOTE: this is the C++23 version of the macro,
// leveraging the "explicit this" feature.
#define TANUKI_REF_IFACE_MEMFUN2(name)                                                                                 \
    template <typename Wrap, typename... MemFunArgs>                                                                   \
    auto name(this Wrap &&self, MemFunArgs &&...args) noexcept(                                                        \
        noexcept(tanuki::detail::forward_like<Wrap>(*iface_ptr(std::forward<Wrap>(self)))                              \
                     .name(std::forward<MemFunArgs>(args)...)))                                                        \
        -> decltype(tanuki::detail::forward_like<Wrap>(*iface_ptr(std::forward<Wrap>(self)))                           \
                        .name(std::forward<MemFunArgs>(args)...))                                                      \
    {                                                                                                                  \
        return tanuki::detail::forward_like<Wrap>(*iface_ptr(std::forward<Wrap>(self)))                                \
            .name(std::forward<MemFunArgs>(args)...);                                                                  \
    }

#endif

namespace detail
{

// Meta-programming to establish a holder value type from an argument of type T. This is used in the generic
// ctor/assignment of wrap. By default, the value type is T itself without reference or cv qualifications. For function
// types, let it decay so that the stored value is a function pointer.
template <typename T>
using value_t_from_arg = std::conditional_t<std::is_function_v<std::remove_cvref_t<T>>,
                                            std::decay_t<std::remove_cvref_t<T>>, std::remove_cvref_t<T>>;

// Helper to detect if T is a std::in_place_type_t. This is used to avoid ambiguities in the wrap class between the
// nullary emplace ctor and the generic ctor.
template <typename>
inline constexpr bool is_in_place_type_v = false;

template <typename T>
inline constexpr bool is_in_place_type_v<std::in_place_type_t<T>> = true;

// Implementation of the pointer interface for the wrap class, conditionally-enabled depending on the configuration.
template <bool Enable, typename Wrap, typename IFace>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
struct TANUKI_VISIBLE wrap_pointer_iface {
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
struct TANUKI_VISIBLE wrap_pointer_iface<false, Wrap, IFace> {
};

} // namespace detail

// Tag structure to construct/assign a wrap into the invalid state.
struct invalid_wrap_t {
};

inline constexpr invalid_wrap_t invalid_wrap{};

namespace detail
{

// NOTE: this section contains code for fetching the reference interface type for a wrap instance.

template <typename>
struct cfg_ref_type {
};

template <typename DefaultValueType, typename RefIFace>
struct cfg_ref_type<config<DefaultValueType, RefIFace>> {
    using type = RefIFace;
};

template <auto Cfg>
using cfg_ref_t = cfg_ref_type<std::remove_const_t<decltype(Cfg)>>::type;

template <typename T, typename Wrap>
struct get_ref_iface {
};

template <typename T, typename Wrap>
    requires with_impl_tt<T>
struct get_ref_iface<T, Wrap> {
    using type = T::template impl<Wrap>;
};

#if defined(TANUKI_HAVE_EXPLICIT_THIS)

template <typename T, typename Wrap>
    requires(!with_impl_tt<T>)
struct get_ref_iface<T, Wrap> {
    using type = T;
};

#endif

template <auto Cfg, typename Wrap>
using get_ref_iface_t = get_ref_iface<cfg_ref_t<Cfg>, Wrap>::type;

// Concept to check that a value type T is consistent with the wrap settings in Cfg: if the wrap is using value
// semantics and it is copy/move-constructible, so must be the value type.
template <typename T, auto Cfg>
concept copy_move_consistent = (Cfg.semantics == wrap_semantics::reference)
                               || ((!Cfg.copy_constructible || std::is_copy_constructible_v<T>)
                                   && (!Cfg.move_constructible || std::is_move_constructible_v<T>));

} // namespace detail

// The wrap class.
template <typename IFace, auto Cfg = default_config>
    requires std::is_class_v<IFace> && std::same_as<IFace, std::remove_cv_t<IFace>> && valid_config<Cfg>
// NOLINTNEXTLINE(misc-multiple-inheritance)
class TANUKI_VISIBLE wrap : private detail::wrap_storage<IFace, Cfg.static_size, Cfg.static_align, Cfg.semantics>,
                            // NOTE: the reference interface is not supposed to hold any data: it will always be
                            // def-inited (even when copying/moving a wrap object), its assignment operators will never
                            // be invoked, it will never be swapped, etc. This needs to be documented.
                            public detail::get_ref_iface_t<Cfg, wrap<IFace, Cfg>>,
                            public detail::wrap_pointer_iface<Cfg.pointer_interface, wrap<IFace, Cfg>, IFace>
{
    // Aliases for the value interface.
    using value_iface_t = detail::_tanuki_value_iface<IFace, Cfg.semantics>;

    // Alias for the reference interface.
    using ref_iface_t = detail::get_ref_iface_t<Cfg, wrap<IFace, Cfg>>;

    // The default value type.
    using default_value_t = decltype(Cfg)::default_value_type;

    // Shortcut for the holder type corresponding to the value type T.
    template <typename T>
    using holder_t = detail::_tanuki_holder<T, IFace, Cfg.semantics>;

    // Helper to detect the type of storage in use. Returns true for static storage, false for dynamic storage
    // (including the invalid state).
    [[nodiscard]] bool stype() const noexcept
        requires(Cfg.semantics == wrap_semantics::value && Cfg.static_size > 0u)
    {
        const auto *ptr = reinterpret_cast<const std::byte *>(this->m_pv_iface);

        // NOTE: although we are using std::less and friends here (and thus avoiding the use of builtin comparison
        // operators, which could in principle be optimised out by the compiler), this is not 100% portable, because in
        // principle static_storage could be interleaved with another object while at the same time respecting the total
        // pointer ordering guarantees given by the standard. This could happen for instance on segmented memory
        // architectures.
        //
        // In pratice, this should be ok an all commonly-used platforms.
        //
        // NOTE: ptr will be null if the storage type is dynamic, hence another assumption here is that nullptr is not
        // included in the storage range of static_storage.
        //
        // NOTE: it seems like the only truly portable way of implementing this is to compare ptr to the addresses of
        // all elements in static_storage. Unfortunately, it seems like compilers are not able to optimise this to a
        // simple pointer comparison.
        return std::greater_equal<void>{}(ptr, this->static_storage)
               && std::less<void>{}(ptr, this->static_storage + sizeof(this->static_storage));
    }

    // Implementation of generic construction. This will construct a holder with value type T using the construction
    // argument(s) x.
    //
    // NOTE: concept checking is performed in the actual constructors.
    template <typename T, typename... U>
    void ctor_impl(U &&...x) noexcept(Cfg.semantics == wrap_semantics::value && sizeof(holder_t<T>) <= Cfg.static_size
                                      && alignof(holder_t<T>) <= Cfg.static_align
                                      && std::is_nothrow_constructible_v<holder_t<T>, U &&...>)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            if constexpr (sizeof(holder_t<T>) > Cfg.static_size || alignof(holder_t<T>) > Cfg.static_align) {
                // Static storage is disabled, or the type is overaligned, or there is not enough room in static
                // storage. Use dynamic memory allocation.
                //
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                this->m_pv_iface = new holder_t<T>(std::forward<U>(x)...);
            } else {
                // Static storage is enabled and there is enough room. Construct in-place.
                //
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
                this->m_pv_iface = ::new (this->static_storage) holder_t<T>(std::forward<U>(x)...);
            }
        } else {
            this->m_pv_iface = std::make_shared<holder_t<T>>(std::forward<U>(x)...);
        }
    }

#if defined(TANUKI_WITH_BOOST_S11N)

    // Serialisation.
    //
    // NOTE: serialisation support has certain prerequisites:
    //
    // - the value type must be default-initialisable,
    // - the value type must be move-ctible (when using value semantics),
    // - the value type must not be over-aligned (when using value semantics,
    //   not sure if it applies to reference semantics as well).
    //
    // The first two come from the way pointer serialisation works in Boost (i.e., serialisation via pointer to base
    // requires a default constructor and dynamic allocation of an object instance, from which we do a move-init of the
    // holder when using value semantics). The last one I think comes from the way memory is allocated during des11n,
    // i.e., see here:
    //
    // https://github.com/boostorg/serialization/blob/a20c4d97c37e5f437c8ba78f296830edb79cff9e/include/boost/archive/detail/iserializer.hpp#L241
    //
    // Perhaps by providing a custom new operator to the value interface class we can implement proper over-alignment of
    // dynamically-allocated memory.
    //
    // NOTE: support for portable archives would requires some additional logic in case of value semantics with static
    // storage enabled: the sizeof types will vary across platforms and thus we cannot assume that an archive that
    // contains a wrap storing a value in static storage on a platform can be deserialised in static storage on another
    // platform. Perhaps we can add a function in the value interface to report the sizeof?
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &ar, unsigned) const
    {
        if (this->m_pv_iface != nullptr && !this->m_pv_iface->_tanuki_value_is_default_initializable()) [[unlikely]] {
            throw std::invalid_argument("Cannot serialise a wrap containing a value of type '"
                                        + demangle(this->m_pv_iface->_tanuki_value_type_index().name())
                                        + "': the type is not default-initializable");
        }

        if constexpr (Cfg.semantics == wrap_semantics::value) {
            if (this->m_pv_iface != nullptr && !this->m_pv_iface->_tanuki_value_is_move_constructible()) [[unlikely]] {
                throw std::invalid_argument("Cannot serialise a wrap containing a value of type '"
                                            + demangle(this->m_pv_iface->_tanuki_value_type_index().name())
                                            + "': the type is not move-constructible");
            }
        }

        if constexpr (Cfg.semantics == wrap_semantics::value && Cfg.static_size > 0u) {
            // Store the storage type.
            ar << stype();
        }

        // Store the pointer to the value interface.
        ar << this->m_pv_iface;
    }
    // NOTE: as I have understood, when deserialising a pointer Boost allocates memory only the *first* time the pointer
    // is encountered in an archive:
    //
    // https://stackoverflow.com/questions/62105624/how-does-boostserialization-allocate-memory-when-deserializing-through-a-point
    //
    // In our case, when employing value semantics we disable object tracking for value_iface and I also think that
    // there are no situations in which a pointer to value_iface could be shared amongst multiple wraps. Thus, we should
    // be ok assuming that the pointer coming out of des11n has always been created with a new call.
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

                // Destroy the current object (this is noexcept).
                destroy();

                if (st) {
                    // NOTE: the storage type is static, thus whatever happens we will have to delete pv_iface. We
                    // accomplish this with a small RAII wrapper.
                    //
                    // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
                    const struct pv_iface_deleter {
                        value_iface_t *m_pv_iface;
                        ~pv_iface_deleter()
                        {
                            delete m_pv_iface;
                        }
                    } pv_iface_cleanup{.m_pv_iface = pv_iface};

                    // Move-init the value from pv_iface.
                    //
                    // NOTE: since we are assuming that we are deserialising a valid wrap, this is guaranteed to work
                    // since we checked on serialisation that the type-erased value supports move-construction.
                    this->m_pv_iface = std::move(*pv_iface)._tanuki_move_init_holder(this->static_storage);

                    // NOTE: when we loaded the serialised pointer, the value contained in the holder was deserialised
                    // into the address pv_iface->_tanuki_value_ptr() (i.e., somewhere in dynamically-allocated memory).
                    // However we now have moved the value into this->m_pv_iface->_tanuki_value_ptr() via
                    // _tanuki_move_init_holder(). Inform the archive of the new address of the value, so that the
                    // address tracking machinery keeps on working. See:
                    //
                    // https://www.boost.org/doc/libs/1_82_0/libs/serialization/doc/special.html#objecttracking
                    try {
                        ar.reset_object_address(this->m_pv_iface->_tanuki_value_ptr(), pv_iface->_tanuki_value_ptr());
                        // LCOV_EXCL_START
                    } catch (...) {
                        // If anything goes wrong, we must first destroy and then set to the invalid state before
                        // re-throwing. This is all noexcept.
                        destroy();
                        this->m_pv_iface = nullptr;
                        throw;
                    }
                    // LCOV_EXCL_STOP
                } else {
                    // Assign the deserialised pointer.
                    this->m_pv_iface = pv_iface;
                }
            }
        } else {
            // NOTE: not sure what the guarantees from Boost in case of exceptions are here. Just in case, ensure we
            // reset the wrap to the invalid state in case of exceptions before rethrowing.
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

    // NOTE: store the ctor explicitness into a separate variable. This helps as a workaround for compiler issues in
    // conditionally explicit constructors.
    static constexpr auto explicit_ctor = Cfg.explicit_ctor;

public:
    // Explicit initialisation into the invalid state.
    explicit wrap(invalid_wrap_t) noexcept(detail::nothrow_default_initializable<ref_iface_t>)
        requires std::default_initializable<ref_iface_t>
    {
        // NOTE: for reference semantics, the default ctor
        // of shared_ptr already does the right thing.
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            this->m_pv_iface = nullptr;
        }
    }

    // Default initialisation into the invalid state.
    //
    // NOTE: there's some repetition here with the invalid state ctor in the noexcept() and requires() clauses. This
    // helps avoiding compiler issues on earlier clang versions.
    wrap() noexcept(detail::nothrow_default_initializable<ref_iface_t>)
        requires(Cfg.invalid_default_ctor) && std::default_initializable<ref_iface_t>
        : wrap(invalid_wrap_t{})
    {
    }

    // Default initialisation into the default value type.
    //
    // NOTE: need to document that this value-inits.
    //
    // NOTE: the extra default template parameter is a workaround for older clang versions:
    //
    // https://github.com/llvm/llvm-project/issues/55945
    //
    // I.e., trailing-style concept checks may not short circuit.
    template <typename = void>
        requires(!Cfg.invalid_default_ctor) && std::default_initializable<ref_iface_t> &&
                // A default value type must have been specified in the configuration.
                (!std::same_as<void, default_value_t>) &&
                // We must be able to construct the holder.
                std::constructible_from<holder_t<default_value_t>> &&
                // Check copy/move consistency.
                detail::copy_move_consistent<default_value_t, Cfg>
    wrap() noexcept(noexcept(this->ctor_impl<default_value_t>()) && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<default_value_t>();
    }

    // Generic ctor from a wrappable value.
    template <typename T>
        requires std::default_initializable<ref_iface_t> &&
                 // Make extra sure this does not compete with the invalid ctor.
                 (!std::same_as<invalid_wrap_t, std::remove_cvref_t<T>>) &&
                 // Must not compete with the emplace ctor.
                 (!detail::is_in_place_type_v<std::remove_cvref_t<T>>) &&
                 // Must not compete with copy/move.
                 (!std::same_as<std::remove_cvref_t<T>, wrap>) &&
                 // We must be able to construct the holder.
                 std::constructible_from<holder_t<detail::value_t_from_arg<T &&>>, T &&> &&
                 // Check copy/move consistency.
                 detail::copy_move_consistent<detail::value_t_from_arg<T &&>, Cfg>
    explicit(explicit_ctor < wrap_ctor::always_implicit)
        // NOLINTNEXTLINE(bugprone-forwarding-reference-overload,google-explicit-constructor,hicpp-explicit-conversions)
        wrap(T &&x) noexcept(noexcept(this->ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x)))
                             && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
    }

    // Generic ctor from std::reference_wrapper.
    //
    // NOTE: this is implemented separately from the generic ctor only in order to work around compiler bugs when the
    // explicit() clause contains complex expressions.
    //
    // NOTE: no need to check for copy_move_consistent here as reference wrappers are always copyable/movable.
    template <typename T>
        requires std::default_initializable<ref_iface_t> &&
                 // We must be able to construct the holder.
                 std::constructible_from<holder_t<std::reference_wrapper<T>>, std::reference_wrapper<T>>
    explicit(explicit_ctor == wrap_ctor::always_explicit)
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        wrap(std::reference_wrapper<T> ref) noexcept(
            noexcept(this->ctor_impl<std::reference_wrapper<T>>(std::move(ref)))
            && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<std::reference_wrapper<T>>(std::move(ref));
    }

    // Generic in-place initialisation.
    //
    // NOTE: this will *value-init* if no args are provided. This must be documented well.
    template <typename T, typename... U>
        requires
        // T must be an a non-cv-qualified object.
        std::is_object_v<T> && std::same_as<T, std::remove_cv_t<T>> && std::default_initializable<ref_iface_t> &&
        // We must be able to construct the holder.
        std::constructible_from<holder_t<T>, U &&...> &&
        // Check copy/move consistency.
        detail::copy_move_consistent<T, Cfg>
        explicit wrap(std::in_place_type_t<T>,
                      U &&...args) noexcept(noexcept(this->ctor_impl<T>(std::forward<U>(args)...))
                                            && detail::nothrow_default_initializable<ref_iface_t>)
    {
        ctor_impl<T>(std::forward<U>(args)...);
    }

private:
    // Implementation of copy-initialisation for value semantics.
    void copy_init_from(const wrap &other)
    {
        static_assert(Cfg.semantics == wrap_semantics::value);

        if (is_valid(other)) {
            if constexpr (Cfg.static_size == 0u) {
                // Static storage disabled.
                this->m_pv_iface = other.m_pv_iface->_tanuki_clone_holder();
            } else {
                if (other.stype()) {
                    // Other has static storage.
                    this->m_pv_iface = other.m_pv_iface->_tanuki_copy_init_holder(this->static_storage);
                } else {
                    // Other has dynamic storage.
                    this->m_pv_iface = other.m_pv_iface->_tanuki_clone_holder();
                }
            }
        } else {
            // Handle initialisation from an invalid wrap.
            this->m_pv_iface = nullptr;
        }
    }

public:
    // Copy constructor.
    wrap(const wrap &other) noexcept(Cfg.semantics == wrap_semantics::reference
                                     && detail::nothrow_default_initializable<ref_iface_t>)
        requires(Cfg.copy_constructible || Cfg.semantics == wrap_semantics::reference)
                && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            copy_init_from(other);
        } else {
            this->m_pv_iface = other.m_pv_iface;
        }
    }

private:
    // Implementation of move-initialisation for value semantics.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    void move_init_from(wrap &&other) noexcept
    {
        static_assert(Cfg.semantics == wrap_semantics::value);

        if (is_valid(other)) {
            if constexpr (Cfg.static_size == 0u) {
                // Static storage disabled. Shallow copy the pointer.
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
        } else {
            // Handle initialisation from an invalid wrap.
            this->m_pv_iface = nullptr;
        }
    }

public:
    // Move constructor.
    wrap(wrap &&other) noexcept
        requires(Cfg.move_constructible || Cfg.semantics == wrap_semantics::reference)
                && std::default_initializable<ref_iface_t>
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            move_init_from(std::move(other));
        } else {
            this->m_pv_iface = std::move(other.m_pv_iface);
        }
    }

private:
    void destroy() noexcept
    {
        static_assert(Cfg.semantics == wrap_semantics::value);

        if constexpr (Cfg.static_size == 0u) {
            delete this->m_pv_iface;
        } else {
            // NOTE: the clang-analyzer-cplusplus.NewDelete suppressions are for a clang-tidy false positive emerging
            // from some tests. clang-tidy complains about double delete/destruction in the random_access_range.cpp
            // test, but we checked that this is not actually happening.
            if (stype()) {
                // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
                this->m_pv_iface->~value_iface_t();
            } else {
                // NOLINTNEXTLINE(cppcoreguidelines-owning-memory,clang-analyzer-cplusplus.NewDelete)
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
        requires(Cfg.move_assignable || Cfg.semantics == wrap_semantics::reference)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Handle self-assign.
            if (this == std::addressof(other)) {
                return *this;
            }

            // Handle invalid this.
            if (is_invalid(*this)) {
                // No need to destroy, just move-init from other is sufficient.
                //
                // NOTE: move_init_from() will work fine if other is invalid.
                move_init_from(std::move(other));
                return *this;
            }

            // this is valid, handle invalid other.
            if (is_invalid(other)) {
                destroy();
                this->m_pv_iface = nullptr;
                return *this;
            }

            // Helper to implement move-assignment via destruction + move-initialisation.
            const auto destroy_and_move_init = [this, &other]() noexcept {
                destroy();
                move_init_from(std::move(other));
            };

            // Handle different internal types (which means in general also different storage types).
            //
            // NOTE: in principle we could check here if both wraps are using dynamic storage. In such a case, we could
            // just swap the pointers. Not sure if this optimisation is worth it though.
            if (value_type_index(*this) != value_type_index(other)) {
                destroy_and_move_init();
                return *this;
            }

            // The internal types are the same.
            if constexpr (Cfg.static_size == 0u) {
                // For dynamic storage, swap the pointer.
                std::swap(this->m_pv_iface, other.m_pv_iface);
            } else {
                // The storage flags must match, as they depend only on the internal types.
                assert(stype() == other.stype());

                if (stype()) {
                    // For static storage, directly move assign the internal value, if possible. Otherwise, destroy and
                    // move-initialise.
                    if (!std::move(*other.m_pv_iface)._tanuki_move_assign_value_to(this->m_pv_iface)) {
                        destroy_and_move_init();
                    }
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
    //
    // NOTE: if the internal types differ or the internal type does not support copy-assignment, this will be left in
    // the invalid state if an exception is thrown during the copy operation.
    //
    // NOTE: clang-tidy does not see that we are handling self-assignment properly.
    //
    // NOLINTNEXTLINE(cert-oop54-cpp)
    wrap &operator=(const wrap &other) noexcept(Cfg.semantics == wrap_semantics::reference)
        requires(Cfg.copy_assignable || Cfg.semantics == wrap_semantics::reference)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Handle self-assign.
            if (this == std::addressof(other)) {
                return *this;
            }

            // Handle invalid this.
            if (is_invalid(*this)) {
                // No need to destroy, just copy-init from other is sufficient.
                //
                // NOTE: copy_init_from() either succeeds or fails, no intermediate state is possible.
                //
                // NOTE: copy_init_from() will work fine if other is invalid.
                copy_init_from(other);
                return *this;
            }

            // this is valid, handle invalid other.
            if (is_invalid(other)) {
                destroy();
                this->m_pv_iface = nullptr;
                return *this;
            }

            // NOTE: the idea here is as follows:
            //
            // - if the internal types are the same and copy-assignment is possible, employ it; otherwise,
            // - destroy the current value and copy-initialise from other.
            if (value_type_index(*this) != value_type_index(other)
                || !other.m_pv_iface->_tanuki_copy_assign_value_to(this->m_pv_iface)) {
                destroy();

                try {
                    // NOTE: copy_init_from() either succeeds or fails, no intermediate state is possible.
                    copy_init_from(other);
                } catch (...) {
                    // NOTE: this is important - we want to mark this as invalid before re-throwing. Like this, this
                    // will be left in the invalid state.
                    //
                    // NOTE: we do not need to do this in the move-assignment operator because we do not allow for
                    // exceptions there.
                    this->m_pv_iface = nullptr;

                    throw;
                }
            }
        } else {
            this->m_pv_iface = other.m_pv_iface;
        }

        return *this;
    }

    // Assignment from the invalid state.
    wrap &operator=(invalid_wrap_t) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Do something only if this is in a valid state.
            if (is_valid(*this)) {
                // Destroy the contained value.
                destroy();

                // Set the invalid state.
                this->m_pv_iface = nullptr;
            }
        } else {
            this->m_pv_iface.reset();
        }

        return *this;
    }

    // Generic assignment.
    template <typename T>
        requires
        // Make extra sure this does not compete with the invalid assignment operator.
        (!std::same_as<invalid_wrap_t, std::remove_cvref_t<T>>) &&
        // Must not compete with copy/move assignment.
        (!std::same_as<std::remove_cvref_t<T>, wrap>) &&
        // We must be able to construct the holder.
        std::constructible_from<holder_t<detail::value_t_from_arg<T &&>>, T &&> &&
        // Check copy/move consistency.
        detail::copy_move_consistent<detail::value_t_from_arg<T &&>, Cfg>
        wrap &operator=(T &&x)
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Handle invalid object.
            if (is_invalid(*this)) {
                ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
                return *this;
            }

            // Helper to perform assignment via destruction + initialisation.
            const auto destroy_and_init = [this, &x]() {
                destroy();

                try {
                    ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
                } catch (...) {
                    // NOTE: this is important - we want to mark this as invalid before re-throwing. Like this, this
                    // will be left in the invalid state.
                    this->m_pv_iface = nullptr;

                    throw;
                }
            };

            // Handle different types.
            if (value_type_index(*this) != typeid(detail::value_t_from_arg<T &&>)) {
                destroy_and_init();
                return *this;
            }

            if constexpr (std::is_function_v<std::remove_cvref_t<T &&>>) {
                // NOTE: we need a special case if x is a function. The reason for this is that we cannot take the
                // address of a function and then cast it directly to void *, as required by
                // copy/move_assign_value_from(). See here:
                //
                // https://stackoverflow.com/questions/36645660/why-cant-i-cast-a-function-pointer-to-void
                //
                // Thus, we need to create a temporary pointer to the function and use its address in
                // copy/move_assign_value_from() instead.
                //
                // NOTE: since we know we are dealing with a function pointer here, we can 1) use a copy operation (no
                // need to bother with moving) and 2) avoid checking the return value of
                // _tanuki_copy_assign_value_from() - we know the assignment must succeed.
                auto *fptr = std::addressof(x);
                [[maybe_unused]] const auto ret
                    = this->m_pv_iface->_tanuki_copy_assign_value_from(static_cast<const void *>(&fptr));
                assert(ret);
            } else {
                // The internal types are the same, attempt to directly copy/move assign.
                bool ret = false;
                if constexpr (detail::noncv_rvalue_reference<T &&>) {
                    ret = this->m_pv_iface->_tanuki_move_assign_value_from(static_cast<void *>(std::addressof(x)));
                } else {
                    ret = this->m_pv_iface->_tanuki_copy_assign_value_from(std::addressof(x));
                }

                if (!ret) {
                    // The internal value does not support copy/move assignment. Resort to destruction + copy/move init.
                    destroy_and_init();
                }
            }
        } else {
            ctor_impl<detail::value_t_from_arg<T &&>>(std::forward<T>(x));
        }

        return *this;
    }

    // Free functions interface.

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"

#endif

    // Emplacement.
    template <typename T, typename... Args>
        requires
        // T must be an a non-cv-qualified object.
        std::is_object_v<T> && std::same_as<T, std::remove_cv_t<T>> &&
        // We must be able to construct the holder.
        std::constructible_from<holder_t<T>, Args &&...> &&
        // Check copy/move consistency.
        detail::copy_move_consistent<T, Cfg>
        friend void emplace(wrap &w, Args &&...args) noexcept(noexcept(w.ctor_impl<T>(std::forward<Args>(args)...)))
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            // Destroy the value in w if necessary.
            if (is_valid(w)) {
                w.destroy();
            }

            try {
                w.template ctor_impl<T>(std::forward<Args>(args)...);
            } catch (...) {
                // NOTE: if ctor_impl fails there's no cleanup required, except to invalidate this before rethrowing.
                w.m_pv_iface = nullptr;

                throw;
            }
        } else {
            w.template ctor_impl<T>(std::forward<Args>(args)...);
        }
    }

#if defined(__GNUC__) && !defined(__clang__)

#pragma GCC diagnostic pop

#endif

    // NOTE: w is invalid when the value interface pointer is set to null. This can happen if w has been moved from
    // (note that this also includes the case in which w has been swapped with an invalid object), if generic assignment
    // or emplacement failed, or, in case of reference semantics, if deserialisation threw an exception. The invalid
    // state can also be explicitly set by constructing/assigning from invalid_wrap_t. The only valid operations on an
    // invalid object are:
    //
    // - invocation of is_invalid()/is_valid(),
    // - destruction,
    // - copy/move assignment from, and swapping with, a valid wrap,
    // - generic assignment,
    // - emplacement.
    [[nodiscard]] friend bool is_invalid(const wrap &w) noexcept
    {
        return w.m_pv_iface == nullptr;
    }

    [[nodiscard]] friend std::type_index value_type_index(const wrap &w) noexcept
    {
        return w.m_pv_iface->_tanuki_value_type_index();
    }

    [[nodiscard]] friend const IFace *iface_ptr(const wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }
    [[nodiscard]] friend const IFace *iface_ptr(const wrap &&w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }
    [[nodiscard]] friend IFace *iface_ptr(wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    [[nodiscard]] friend IFace *iface_ptr(wrap &&w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::value) {
            return w.m_pv_iface;
        } else {
            return w.m_pv_iface.get();
        }
    }

    friend void swap(wrap &w1, wrap &w2) noexcept
        requires(Cfg.move_assignable)
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
                // w1 is invalid, w2 is not: move-assign w2 to w1. This may or may not leave w2 in the invalid state.
                w1 = std::move(w2);
                return;
            }

            if (inv2) {
                // Opposite of the above.
                w2 = std::move(w1);
                return;
            }

            // Canonical swap implementation.
            const auto canonical_swap = [&w1, &w2]() {
                auto temp(std::move(w1));
                w1 = std::move(w2);
                w2 = std::move(temp);
            };

            // Handle different internal types (which means in general also different storage types) with the canonical
            // swap() implementation.
            if (value_type_index(w1) != value_type_index(w2)) {
                canonical_swap();
                return;
            }

            // The types are the same.
            if constexpr (Cfg.static_size == 0u) {
                // For dynamic storage, swap the pointers.
                std::swap(w1.m_pv_iface, w2.m_pv_iface);
            } else {
                // The storage flags must match, as they depend only on the internal types.
                assert(w1.stype() == w2.stype());

                if (w1.stype()) {
                    // For static storage, attempt to directly swap the internal values.
                    if (!w2.m_pv_iface->_tanuki_swap_value(w1.m_pv_iface)) {
                        // The internal value does not support swapping. Resort to the canonical implementation.
                        //
                        // NOTE: at the moment I cannot find a way to trigger this, because we end up here only if a
                        // swappable wrap contains a non-swappable value. But: a wrap is marked as swappable only if it
                        // is move ctible/assignable, which requires a move ctible/assignable value, which (almost?)
                        // always means that the value is swappable as well. There may be some way to construct a
                        // pathological type that triggers this branch, but so far I have not found it.
                        canonical_swap(); // LCOV_EXCL_LINE
                    }
                } else {
                    // For dynamic storage, swap the pointers.
                    std::swap(w1.m_pv_iface, w2.m_pv_iface);
                }
            }
        } else {
            std::swap(w1.m_pv_iface, w2.m_pv_iface);
        }
    }

    // NOTE: a wrap in the invalid state is always considered as being in *dynamic* storage.
    [[nodiscard]] friend bool has_static_storage(const wrap &w) noexcept
    {
        if constexpr (Cfg.semantics == wrap_semantics::reference || Cfg.static_size == 0u) {
            return false;
        } else {
            // NOTE: we are explicitly allowing to call this function on moved-from wraps.
            //
            // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
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
        return w.m_pv_iface->_tanuki_value_is_reference();
    }

    // Specific functions for reference semantics.

    // Deep copy.
    [[nodiscard]] friend wrap copy(const wrap &w)
        requires(Cfg.semantics == wrap_semantics::reference)
    {
        wrap retval(invalid_wrap);
        // NOTE: perform the deep copy only if w is valid. Otherwise, return an invalid wrap.
        if (is_valid(w)) {
            retval.m_pv_iface = w.m_pv_iface->_tanuki_shared_clone_holder();
        } else {
            ;
        }
        return retval;
    }

    // Check if two wraps point to the same underlying value.
    [[nodiscard]] friend bool same_value(const wrap &w1, const wrap &w2) noexcept
        requires(Cfg.semantics == wrap_semantics::reference)
    {
        return w1.m_pv_iface == w2.m_pv_iface;
    }
};

// NOTE: let us add a getval() overload in the tanuki namespace, so that is it invocable also from non-ADL contexts.
template <typename T>
[[nodiscard]] auto getval(T *h) noexcept(noexcept(detail::getval(h))) -> decltype(detail::getval(h))
{
    return detail::getval(h);
}

namespace detail
{

// Specialise is_any_wrap_impl.
template <typename IFace, auto Cfg>
inline constexpr bool is_any_wrap_v<wrap<IFace, Cfg>> = true;

} // namespace detail

template <typename IFace, auto Cfg>
[[nodiscard]] bool is_valid(const wrap<IFace, Cfg> &w) noexcept
{
    return !is_invalid(w);
}

// NOTE: a wrap in the invalid state is always considered as being in *dynamic* storage.
template <typename IFace, auto Cfg>
bool has_dynamic_storage(const wrap<IFace, Cfg> &w) noexcept
{
    return !has_static_storage(w);
}

template <typename T, typename IFace, auto Cfg>
const T *value_ptr(const wrap<IFace, Cfg> &w) noexcept
{
    // NOTE: if T is cv-qualified, always return null. No need to remove reference as we cannot form pointers to
    // references in the return value.
    if constexpr (std::same_as<T, std::remove_cv_t<T>>) {
        return value_type_index(w) == typeid(T) ? static_cast<const T *>(raw_value_ptr(w)) : nullptr;
    } else {
        return nullptr;
    }
}

template <typename T, typename IFace, auto Cfg>
T *value_ptr(wrap<IFace, Cfg> &w) noexcept
{
    if constexpr (std::same_as<T, std::remove_cv_t<T>>) {
        return value_type_index(w) == typeid(T) ? static_cast<T *>(raw_value_ptr(w)) : nullptr;
    } else {
        return nullptr;
    }
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

namespace detail
{

template <typename>
struct cfg_from_wrap {
};

template <typename IFace, auto Cfg>
struct cfg_from_wrap<wrap<IFace, Cfg>> {
    static constexpr auto cfg = Cfg;
};

} // namespace detail

// Fetch the configuration settings for a wrap type.
template <any_wrap W>
inline constexpr auto wrap_cfg = detail::cfg_from_wrap<W>::cfg;

TANUKI_END_NAMESPACE

#if defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#if defined(TANUKI_WITH_BOOST_S11N)

namespace boost::serialization
{

// NOTE: disable address tracking for value_iface when employing value semantics. We do not need it as value_iface
// pointers are never shared, and it might only create issues when deserialising into a function-local pointer which is
// then copied into the wrap storage.
template <typename IFace>
struct tracking_level<tanuki::detail::_tanuki_value_iface<IFace, tanuki::wrap_semantics::value>> {
    using tag = mpl::integral_c_tag;
    using type = mpl::int_<track_never>;
    BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
    BOOST_STATIC_ASSERT(
        (mpl::greater<implementation_level<tanuki::detail::_tanuki_value_iface<IFace, tanuki::wrap_semantics::value>>,
                      mpl::int_<primitive_type>>::value));
};

} // namespace boost::serialization

// NOTE: these are verbatim re-implementations of the BOOST_CLASS_EXPORT_KEY(2) and BOOST_CLASS_EXPORT_IMPLEMENT macros,
// which do not work well with class templates.
//
// NOTE: the use of __VA_ARGS__ here is kind of a hacky convenience, as it allows us to pass in arguments containing
// commas without additional mucking around.
//
// NOTE: in these macros we are always exporting/implementing both semantics variants.
#define TANUKI_S11N_WRAP_EXPORT_KEY(ud_type, ...)                                                                      \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <tanuki::wrap_semantics Sem>                                                                              \
    struct guid_defined<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, Sem>> : boost::mpl::true_ {               \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>()     \
    {                                                                                                                  \
        return "tanuki::wrap<" #__VA_ARGS__ ">@" #ud_type "#val";                                                      \
    }                                                                                                                  \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>() \
    {                                                                                                                  \
        return "tanuki::wrap<" #__VA_ARGS__ ">@" #ud_type "#ref";                                                      \
    }                                                                                                                  \
    }

#define TANUKI_S11N_WRAP_EXPORT_KEY2(ud_type, gid, ...)                                                                \
    namespace boost::serialization                                                                                     \
    {                                                                                                                  \
    template <tanuki::wrap_semantics Sem>                                                                              \
    struct guid_defined<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, Sem>> : boost::mpl::true_ {               \
    };                                                                                                                 \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>()     \
    {                                                                                                                  \
        return gid "#val";                                                                                             \
    }                                                                                                                  \
    template <>                                                                                                        \
    inline const char *guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>() \
    {                                                                                                                  \
        return gid "#ref";                                                                                             \
    }                                                                                                                  \
    }

#define TANUKI_S11N_WRAP_EXPORT_IMPLEMENT(ud_type, ...)                                                                \
    namespace boost::archive::detail::extra_detail                                                                     \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct init_guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> {            \
        static guid_initializer<                                                                                       \
            tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> const &g;             \
    };                                                                                                                 \
    template <>                                                                                                        \
    struct init_guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> {        \
        static guid_initializer<                                                                                       \
            tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> const &g;         \
    };                                                                                                                 \
    guid_initializer<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>> const        \
        &init_guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>::g             \
        = ::boost::serialization::singleton<guid_initializer<tanuki::detail::_tanuki_holder<                           \
            ud_type, __VA_ARGS__, tanuki::wrap_semantics::value>>>::get_mutable_instance()                             \
              .export_guid();                                                                                          \
    guid_initializer<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>> const    \
        &init_guid<tanuki::detail::_tanuki_holder<ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>::g         \
        = ::boost::serialization::singleton<guid_initializer<tanuki::detail::_tanuki_holder<                           \
            ud_type, __VA_ARGS__, tanuki::wrap_semantics::reference>>>::get_mutable_instance()                         \
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
#undef TANUKI_VISIBLE

#endif
