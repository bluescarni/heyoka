// Copyright 2018-2025 Francesco Biscani
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef IGOR_IGOR_HPP
#define IGOR_IGOR_HPP

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>

#define IGOR_VERSION_STRING "1.0.0"
#define IGOR_VERSION_MAJOR 1
#define IGOR_VERSION_MINOR 0
#define IGOR_VERSION_PATCH 0

#if defined(__GNUC__) || defined(__clang__)

#define IGOR_ABI_TAG_ATTR __attribute__((abi_tag))

#else

#define IGOR_ABI_TAG_ATTR

#endif

// NOTE: currently we are employing several quadratic compile-time algorithms, e.g., when it comes to check for
// uniqueness in a set of named arguments. Something like compile-time sorting or (perfect?) hashing would of course
// perform better asymptotically, however at this time they seem to be really difficult to implement for types.
//
// Pointer-based approaches won't work at compile time because comparing pointers with '<' is well-defined only within
// an array (and std::less does not help here because it has a special exception for compile-time behaviour). Hashing
// has similar problems. Name-based approaches are fragile and compiler-specific (e.g., MSVC gives the same "name" to
// all lambdas used as NTTP). Perhaps reflection (or constexpr-friendly typeid(), if we ever get it) could eventually
// help here.

namespace igor
{

inline namespace v1 IGOR_ABI_TAG_ATTR
{

namespace detail
{

// This is the class used to store a reference to a function argument. An object of this type is returned by
// named_argument's assignment operator.
//
// NOTE: the Tag type is is used to establish a correspondence between the tagged reference and the named argument that
// created it.
template <typename Tag, typename T>
    requires(std::is_reference_v<T>)
struct tagged_ref {
    T value;

    using tag_type = Tag;
    using value_type = T;
};

} // namespace detail

// Helper to turn a tagged reference into another tagged reference containing a const reference to the original
// reference. This is useful in order to enforce const reference access semantics to an argument (in the same spirit as
// std::as_const()).
template <typename Tag, typename T>
auto as_const(const detail::tagged_ref<Tag, T> &tc)
{
    return detail::tagged_ref<Tag, decltype(std::as_const(tc.value))>{std::as_const(tc.value)};
}

// Class to represent a named argument.
template <typename Tag, typename ExplicitType = void>
struct named_argument {
    using tag_type = Tag;

    // NOTE: make sure this does not interfere with the copy/move assignment operators.
    template <typename T>
        requires(!std::same_as<named_argument, std::remove_cvref_t<T>>)
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(T &&x) const
    {
        return detail::tagged_ref<Tag, T &&>{std::forward<T>(x)};
    }

    // Add overloads for std::initializer_list as well.
    template <typename T>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(const std::initializer_list<T> &l) const
    {
        return detail::tagged_ref<Tag, const std::initializer_list<T> &>{l};
    }
    template <typename T>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(std::initializer_list<T> &l) const
    {
        return detail::tagged_ref<Tag, std::initializer_list<T> &>{l};
    }
    template <typename T>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(std::initializer_list<T> &&l) const
    {
        return detail::tagged_ref<Tag, std::initializer_list<T> &&>{std::move(l)};
    }
    template <typename T>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(const std::initializer_list<T> &&l) const
    {
        return detail::tagged_ref<Tag, const std::initializer_list<T> &&>{std::move(l)};
    }
};

template <typename Tag, typename ExplicitType>
    requires(std::is_reference_v<ExplicitType>)
struct named_argument<Tag, ExplicitType> {
    using tag_type = Tag;
    using value_type = ExplicitType;

    // NOTE: disable implicit conversion, deduced type needs to be the same as explicit type.
    template <typename T>
        requires std::same_as<T &&, ExplicitType>
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(T &&x) const
    {
        return detail::tagged_ref<Tag, ExplicitType>{std::forward<T>(x)};
    }

    // NOTE: enable implicit conversion with curly braces and copy-list/aggregate initialization with double curly
    // braces.
    //
    // NOLINTNEXTLINE(misc-unconventional-assign-operator, cppcoreguidelines-c-copy-assignment-signature)
    constexpr auto operator=(detail::tagged_ref<Tag, ExplicitType> &&tc) const
    {
        return std::move(tc);
    }

    template <typename T>
        requires(!std::same_as<T &&, ExplicitType>)
    auto operator=(T &&) const = delete; // please use {...} to typed argument implicit conversion
};

// Equality comparison: return true when comparing named arguments with identical tags, false otherwise.
template <typename Tag1, typename ExplicitType1, typename Tag2, typename ExplicitType2>
consteval bool operator==(named_argument<Tag1, ExplicitType1>, named_argument<Tag2, ExplicitType2>)
{
    return std::same_as<Tag1, Tag2>;
}

// Type representing a named argument which
// was not provided in a function call.
struct not_provided_t {
};

// Non-provided named arguments will return a const reference
// to this global object.
inline constexpr not_provided_t not_provided;

namespace detail
{

// Type trait to detect if T is a tagged reference with tag Tag (and any type as second parameter).
template <typename Tag, typename T>
struct is_tagged_ref : std::false_type {
};

template <typename Tag, typename T>
struct is_tagged_ref<Tag, tagged_ref<Tag, T>> : std::true_type {
};

// Type trait/concept to detect if T is a tagged reference (regardless of the tag type or the type of the second
// parameter).
template <typename T>
struct is_tagged_ref_any : std::false_type {
};

template <typename Tag, typename T>
struct is_tagged_ref_any<tagged_ref<Tag, T>> : std::true_type {
};

template <typename T>
concept any_tagged_ref = is_tagged_ref_any<T>::value;

// Type trait to detect named arguments.
template <typename>
struct is_any_named_argument : std::false_type {
};

template <typename Tag, typename ExplicitType>
struct is_any_named_argument<named_argument<Tag, ExplicitType>> : std::true_type {
};

} // namespace detail

// Concept to detect (const) named arguments.
template <auto NA>
concept any_named_argument = detail::is_any_named_argument<std::remove_const_t<decltype(NA)>>::value;

namespace detail
{

// An always-true concept for use below.
template <bool>
concept always_true = true;

} // namespace detail

// Concept to check if V is usable as a validator for the type T.
template <auto V, typename T>
concept valid_descr_validator = requires {
    { V.template operator()<T>() } -> std::same_as<bool>;
    // NOTE: this part checks that the call operator of V is usable at compile-time.
    requires detail::always_true<V.template operator()<T>()>;
};

template <auto NA, auto Validator = []<typename>() { return true; }>
    requires any_named_argument<NA>
struct descr {
    // Configuration options.
    bool required = false;

    // Store a copy of the named argument for later use.
    static constexpr auto na = NA;

    template <typename T>
        requires valid_descr_validator<Validator, T>
    static consteval bool validate()
    {
        return Validator.template operator()<T>();
    }
};

namespace detail
{

// Type trait to detect descriptors.
template <typename>
struct is_any_descr : std::false_type {
};

template <auto NA, auto Validator>
struct is_any_descr<descr<NA, Validator>> : std::true_type {
};

} // namespace detail

// Concept to detect (const) descriptors.
template <auto Descr>
concept any_descr = detail::is_any_descr<std::remove_const_t<decltype(Descr)>>::value;

namespace detail
{

// Function to check that there are no duplicates in the pack of descriptors Descrs.
//
// NOTE: descriptors are compared via their named arguments.
consteval bool no_duplicate_descrs_impl(auto... Descrs)
{
    // Helper to compare one descriptor to all Descrs.
    auto check_one = [](auto cur_descr, auto... all_descrs) {
        return (static_cast<std::size_t>(0) + ... + static_cast<std::size_t>(cur_descr.na == all_descrs.na)) == 1u;
    };

    return (... && check_one(Descrs, Descrs...));
}

} // namespace detail

// Concept to check that there are no duplicates in the pack of descriptors Descrs.
template <auto... Descrs>
concept no_duplicate_descrs = detail::no_duplicate_descrs_impl(Descrs...);

// Configuration structure for named arguments validation.
template <auto... Descrs>
    requires(sizeof...(Descrs) > 0u) && (... && any_descr<Descrs>) && no_duplicate_descrs<Descrs...>
struct config {
    // Config options.
    bool allow_unnamed = false;
    bool allow_extra = false;
};

namespace detail
{

// Type trait to detect an instance of the config class.
template <typename>
struct is_any_config : std::false_type {
};

template <auto... Descrs>
struct is_any_config<config<Descrs...>> : std::true_type {
};

// Concept to detect (const) instances of the config class.
template <auto Cfg>
concept any_config = is_any_config<std::remove_const_t<decltype(Cfg)>>::value;

template <auto Cfg, typename... Args>
concept validate_unnamed_arguments = (Cfg.allow_unnamed) || (any_tagged_ref<std::remove_cvref_t<Args>> && ...);

template <typename Arg, typename... Args>
consteval bool check_one_unique_named_argument()
{
    using arg_u = std::remove_cvref_t<Arg>;

    if constexpr (any_tagged_ref<arg_u>) {
        return (static_cast<std::size_t>(0) + ...
                + static_cast<std::size_t>(std::same_as<arg_u, std::remove_cvref_t<Args>>))
               == 1u;
    } else {
        return true;
    }
}

template <typename Arg, typename... Args>
concept unique_named_argument = check_one_unique_named_argument<Arg, Args...>();

template <typename... Args>
concept validate_no_repeated_named_arguments = (unique_named_argument<Args, Args...> && ...);

template <typename Arg>
consteval bool arg_has_descriptor(auto... descrs)
{
    using arg_u = std::remove_cvref_t<Arg>;

    if constexpr (any_tagged_ref<arg_u>) {
        return (... || std::same_as<typename arg_u::tag_type, typename decltype(descrs.na)::tag_type>);
    } else {
        return true;
    }
}

template <typename>
struct all_args_have_descriptors;

template <auto... Descrs>
struct all_args_have_descriptors<config<Descrs...>> {
    template <typename... Args>
    static constexpr bool value = (... && arg_has_descriptor<Args>(Descrs...));
};

template <typename... Args>
consteval bool check_one_descr_present(auto descr)
{
    if (descr.required) {
        using tag_type = typename decltype(descr.na)::tag_type;

        [[maybe_unused]] auto tags_match = []<typename Arg>() {
            using arg_u = std::remove_cvref_t<Arg>;

            if constexpr (any_tagged_ref<arg_u>) {
                return std::same_as<typename arg_u::tag_type, tag_type>;
            } else {
                return false;
            }
        };

        return (... || tags_match.template operator()<Args>());
    } else {
        return true;
    }
}

template <typename>
struct all_required_arguments_are_present;

template <auto... Descrs>
struct all_required_arguments_are_present<config<Descrs...>> {
    template <typename... Args>
    static constexpr bool value = (... && check_one_descr_present<Args...>(Descrs));
};

template <typename... Args>
consteval bool validate_one_validator([[maybe_unused]] auto descr)
{
    [[maybe_unused]] auto check_single_arg = []<typename Arg>(auto d) {
        using arg_u = std::remove_cvref_t<Arg>;

        if constexpr (any_tagged_ref<arg_u>) {
            if constexpr (std::same_as<typename arg_u::tag_type, typename decltype(d.na)::tag_type>) {
                // NOTE: here we are checking if the validator is properly implemented.
                return requires { d.template validate<typename arg_u::value_type>(); };
            } else {
                return true;
            }
        } else {
            return true;
        }
    };

    return (... && check_single_arg.template operator()<Args>(descr));
}

template <typename>
struct validate_validators;

template <auto... Descrs>
struct validate_validators<config<Descrs...>> {
    template <typename... Args>
    static constexpr bool value = (... && validate_one_validator<Args...>(Descrs));
};

template <typename Arg>
consteval bool validate_one_named_argument(auto... descrs)
{
    using arg_u = std::remove_cvref_t<Arg>;

    if constexpr (any_tagged_ref<arg_u>) {
        auto check_single_descr = [](auto d) {
            if constexpr (std::same_as<typename arg_u::tag_type, typename decltype(d.na)::tag_type>) {
                return d.template validate<typename arg_u::value_type>();
            } else {
                return true;
            }
        };

        return (... && check_single_descr(descrs));
    } else {
        return true;
    }
}

template <typename>
struct validate_named_arguments;

template <auto... Descrs>
struct validate_named_arguments<config<Descrs...>> {
    template <typename... Args>
    static constexpr bool value = (... && validate_one_named_argument<Args>(Descrs...));
};

} // namespace detail

template <auto Cfg, typename... Args>
concept validate = requires {
    // Step 0: check that Cfg is a config instance.
    requires detail::any_config<Cfg>;
    // Step 1: validate the unnamed arguments.
    requires detail::validate_unnamed_arguments<Cfg, Args...>;
    // Step 2: check that there are no duplicate named arguments in Args.
    requires detail::validate_no_repeated_named_arguments<Args...>;
    // Step 3: validate extra named arguments (i.e., those not present in Cfg).
    requires(Cfg.allow_extra)
                || (detail::all_args_have_descriptors<std::remove_const_t<decltype(Cfg)>>::template value<Args...>);
    // Step 4: check the presence of the required named arguments.
    requires(detail::all_required_arguments_are_present<std::remove_const_t<decltype(Cfg)>>::template value<Args...>);
    // Step 5: check the validators.
    requires(detail::validate_validators<std::remove_const_t<decltype(Cfg)>>::template value<Args...>);
    // Step 6: run the validators.
    requires(detail::validate_named_arguments<std::remove_const_t<decltype(Cfg)>>::template value<Args...>);
};

namespace detail
{

// Helper to check that two config instances have no common named arguments.
template <auto... Descrs1, auto... Descrs2>
consteval bool no_common_named_arguments(config<Descrs1...>, config<Descrs2...>)
{
    auto check_one = [](auto cur_descr1, auto... all_descrs2) {
        return (static_cast<std::size_t>(0) + ... + static_cast<std::size_t>(cur_descr1.na == all_descrs2.na)) == 0u;
    };

    return (... && check_one(Descrs1, Descrs2...));
}

} // namespace detail

// Implementation of binary configuration merging.
template <auto... Descrs1, auto... Descrs2>
    requires(
        // NOTE: we need to construct config instances on-the-fly from the descriptors in order to run
        // no_common_named_arguments(). A bit yucky, but correct.
        detail::no_common_named_arguments(config<Descrs1...>{}, config<Descrs2...>{}))
consteval auto operator|(config<Descrs1...> c1, config<Descrs2...> c2)
{
    // NOTE: here we are allowing merging between configurations with different settings. The merged config will adopt
    // the most permissive settings.
    return config<Descrs1..., Descrs2...>{.allow_unnamed = c1.allow_unnamed || c2.allow_unnamed,
                                          .allow_extra = c1.allow_extra || c2.allow_extra};
}

// NOTE: implement some of the parser functionality as free functions, which will then be wrapped by static constexpr
// member functions in the parser class. These free functions can be used where a parser object is not available (e.g.,
// in a requires clause).
template <typename... Args, typename Tag, typename ExplicitType>
consteval bool has([[maybe_unused]] const named_argument<Tag, ExplicitType> &narg)
{
    return (... || detail::is_tagged_ref<Tag, std::remove_cvref_t<Args>>::value);
}

template <typename... Args, typename... Tags, typename... ExplicitTypes>
consteval bool has_all(const named_argument<Tags, ExplicitTypes> &...nargs)
{
    return (... && igor::has<Args...>(nargs));
}

template <typename... Args, typename... Tags, typename... ExplicitTypes>
consteval bool has_any(const named_argument<Tags, ExplicitTypes> &...nargs)
{
    return (... || igor::has<Args...>(nargs));
}

template <typename... Args>
consteval bool has_unnamed_arguments()
{
    return (... || !detail::any_tagged_ref<std::remove_cvref_t<Args>>);
}

template <typename... Args, typename... Tags, typename... ExplicitTypes>
consteval bool has_other_than(const named_argument<Tags, ExplicitTypes> &...nargs)
{
    // NOTE: the first fold expression will return how many of the nargs
    // are in the pack. The second fold expression will return the total number
    // of named arguments in the pack.
    return (static_cast<std::size_t>(0) + ... + static_cast<std::size_t>(igor::has<Args...>(nargs)))
           < (static_cast<std::size_t>(0) + ...
              + static_cast<std::size_t>(detail::any_tagged_ref<std::remove_cvref_t<Args>>));
}

namespace detail
{

// Check if T is a named argument which appears more than once in Args.
template <typename T, typename... Args>
consteval bool is_repeated_named_argument()
{
    using Tu = std::remove_cvref_t<T>;

    if constexpr (any_tagged_ref<Tu>) {
        return (static_cast<std::size_t>(0) + ...
                + static_cast<std::size_t>(std::same_as<Tu, std::remove_cvref_t<Args>>))
               > 1u;
    } else {
        return false;
    }
}

} // namespace detail

template <typename... Args>
consteval bool has_duplicates()
{
    return (... || detail::is_repeated_named_argument<Args, Args...>());
}

// Remove from the set of variadic arguments args the named arguments NArgs.
//
// The result is returned as a tuple of perfectly-forwarded references.
template <auto... NArgs, typename... Args>
    requires(any_named_argument<NArgs> && ...)
constexpr auto reject(Args &&...args)
{
    [[maybe_unused]] auto filter = []<typename T>(T &&x) {
        using Tu = std::remove_cvref_t<T>;

        if constexpr (detail::any_tagged_ref<Tu>) {
            if constexpr ((... || std::same_as<typename decltype(NArgs)::tag_type, typename Tu::tag_type>)) {
                return std::tuple{};
            } else {
                return std::forward_as_tuple(std::forward<T>(x));
            }
        } else {
            return std::forward_as_tuple(std::forward<T>(x));
        }
    };

    return std::tuple_cat(filter(std::forward<Args>(args))...);
}

namespace detail
{

template <typename>
struct reject_na_from_cfg;

template <auto... Descrs>
struct reject_na_from_cfg<config<Descrs...>> {
    template <typename... Args>
    static constexpr auto run_reject(Args &&...args)
    {
        return reject<Descrs.na...>(std::forward<Args>(args)...);
    }
};

} // namespace detail

// Same as the previous overload, except that the named arguments to reject are deduced from the input config.
template <auto Cfg, typename... Args>
    requires(detail::any_config<Cfg>)
constexpr auto reject(Args &&...args)
{
    // Need to go through an auxiliary struct in order to recover the pack of descriptors.
    return detail::reject_na_from_cfg<std::remove_const_t<decltype(Cfg)>>::run_reject(std::forward<Args>(args)...);
}

// Remove from the set of variadic arguments args the named arguments *other than* NArgs.
//
// The result is returned as a tuple of perfectly-forwarded references.
template <auto... NArgs, typename... Args>
    requires(any_named_argument<NArgs> && ...)
constexpr auto filter(Args &&...args)
{
    [[maybe_unused]] auto filter = []<typename T>(T &&x) {
        using Tu = std::remove_cvref_t<T>;

        if constexpr (detail::any_tagged_ref<Tu>) {
            if constexpr ((... || std::same_as<typename decltype(NArgs)::tag_type, typename Tu::tag_type>)) {
                return std::forward_as_tuple(std::forward<T>(x));
            } else {
                return std::tuple{};
            }
        } else {
            return std::forward_as_tuple(std::forward<T>(x));
        }
    };

    return std::tuple_cat(filter(std::forward<Args>(args))...);
}

namespace detail
{

template <typename>
struct filter_na_from_cfg;

template <auto... Descrs>
struct filter_na_from_cfg<config<Descrs...>> {
    template <typename... Args>
    static constexpr auto run_filter(Args &&...args)
    {
        return filter<Descrs.na...>(std::forward<Args>(args)...);
    }
};

} // namespace detail

// Same as the previous overload, except that the named arguments to filter are deduced from the input config.
template <auto Cfg, typename... Args>
    requires(detail::any_config<Cfg>)
constexpr auto filter(Args &&...args)
{
    // Need to go through an auxiliary struct in order to recover the pack of descriptors.
    return detail::filter_na_from_cfg<std::remove_const_t<decltype(Cfg)>>::run_filter(std::forward<Args>(args)...);
}

namespace detail
{

// Type trait to detect if a function object can be invoked with the elements of a tuple as arguments.
template <typename, typename>
struct tuple_invocable;

template <typename F, typename... Args>
struct tuple_invocable<F, std::tuple<Args...>> : std::is_invocable<F, Args...> {
};

} // namespace detail

// Reject named arguments from the set of variadic arguments args and invoke F on the result.
template <auto... CfgOrNArgs, typename F, typename... Args>
    requires requires {
        reject<CfgOrNArgs...>(std::declval<Args>()...);
        requires detail::tuple_invocable<F, decltype(reject<CfgOrNArgs...>(std::declval<Args>()...))>::value;
    }
constexpr decltype(auto) reject_invoke(F &&f, Args &&...args)
{
    return std::apply(std::forward<F>(f), reject<CfgOrNArgs...>(std::forward<Args>(args)...));
}

// Filter named arguments from the set of variadic arguments args and invoke F on the result.
template <auto... CfgOrNArgs, typename F, typename... Args>
    requires requires {
        filter<CfgOrNArgs...>(std::declval<Args>()...);
        requires detail::tuple_invocable<F, decltype(filter<CfgOrNArgs...>(std::declval<Args>()...))>::value;
    }
constexpr decltype(auto) filter_invoke(F &&f, Args &&...args)
{
    return std::apply(std::forward<F>(f), filter<CfgOrNArgs...>(std::forward<Args>(args)...));
}

namespace detail
{

// Implementation of parsers' constructor.
//
// This function will examine all input arguments and return a tuple of references to the tagged reference arguments.
// All other arguments will be discarded.
constexpr auto parser_ctor_impl(const auto &...args)
{
    [[maybe_unused]] auto filter_na = []<typename T>(const T &x) {
        if constexpr (any_tagged_ref<T>) {
            return std::forward_as_tuple(x);
        } else {
            return std::tuple{};
        }
    };

    return std::tuple_cat(filter_na(args)...);
}

} // namespace detail

// Parser for named arguments in a function call.
//
// NOTE: the template arguments are intended to always be deduced via the constructor and never explicitly passed. Thus,
// they should never end up being cvref-qualified.
template <typename... ParseArgs>
    requires(std::same_as<ParseArgs, std::remove_cvref_t<ParseArgs>> && ...)
class parser
{
    using tuple_t = decltype(detail::parser_ctor_impl(std::declval<const ParseArgs &>()...));

    tuple_t m_nargs;

public:
    constexpr explicit parser(const ParseArgs &...parse_args) : m_nargs(detail::parser_ctor_impl(parse_args...)) {}

private:
    // Fetch the value associated to the input named
    // argument narg. If narg is not present, this will
    // return a const ref to a global not_provided_t object.
    template <std::size_t I, typename Tag, typename ExplicitType>
    constexpr decltype(auto) fetch_one_impl([[maybe_unused]] const named_argument<Tag, ExplicitType> &narg) const
    {
        if constexpr (I == std::tuple_size_v<tuple_t>) {
            // NOTE: clang-tidy is wrong here, we do need the cast to const ref otherwise we return a copy of the
            // not_provided object.
            // NOLINTNEXTLINE(readability-redundant-casting)
            return static_cast<const not_provided_t &>(not_provided);
        } else if constexpr (std::same_as<typename std::remove_cvref_t<std::tuple_element_t<I, tuple_t>>::tag_type,
                                          Tag>) {
            if constexpr (std::is_rvalue_reference_v<decltype(std::get<I>(m_nargs).value)>) {
                return std::move(std::get<I>(m_nargs).value);
            } else {
                return std::get<I>(m_nargs).value;
            }
        } else {
            return fetch_one_impl<I + 1u>(narg);
        }
    }

public:
    // Get references to the values associated to the input named arguments.
    template <typename... Tags, typename... ExplicitTypes>
    constexpr decltype(auto) operator()([[maybe_unused]] const named_argument<Tags, ExplicitTypes> &...nargs) const
    {
        if constexpr (sizeof...(Tags) == 0u) {
            return;
        } else if constexpr (sizeof...(Tags) == 1u) {
            return this->fetch_one_impl<0>(nargs...);
        } else {
            return std::forward_as_tuple(this->fetch_one_impl<0>(nargs)...);
        }
    }
    // Get a reference to the value associated to the input named argument, if present, otherwise return a default
    // value. The default value is returned as a new object constructed from perfectly forwarding 'def'.
    //
    // NOTE: T cannot be a named argument, otherwise this will be an ambiguous overload with the other call operator.
    template <typename Tag, typename ExplicitType, typename T>
        requires(!detail::is_any_named_argument<std::remove_cvref_t<T>>::value)
                && std::constructible_from<std::remove_cvref_t<T>, T &&>
    constexpr decltype(auto) operator()(const named_argument<Tag, ExplicitType> &narg, T &&def) const
    {
        // NOTE: this condition is equivalent to invoking has().
        if constexpr ((... || detail::is_tagged_ref<Tag, ParseArgs>::value)) {
            return this->fetch_one_impl<0>(narg);
        } else {
            return std::remove_cvref_t<T>(std::forward<T>(def));
        }
    }
    // Check if the input named argument na is present in the parser.
    template <typename Tag, typename ExplicitType>
    static consteval bool has(const named_argument<Tag, ExplicitType> &narg)
    {
        return igor::has<ParseArgs...>(narg);
    }
    // Check if all the input named arguments nargs are present in the parser.
    template <typename... Tags, typename... ExplicitTypes>
    static consteval bool has_all(const named_argument<Tags, ExplicitTypes> &...nargs)
    {
        return igor::has_all<ParseArgs...>(nargs...);
    }
    // Check if at least one of the input named arguments nargs is present in the parser.
    template <typename... Tags, typename... ExplicitTypes>
    static consteval bool has_any(const named_argument<Tags, ExplicitTypes> &...nargs)
    {
        return igor::has_any<ParseArgs...>(nargs...);
    }
    // Detect the presence of unnamed arguments.
    static consteval bool has_unnamed_arguments()
    {
        return igor::has_unnamed_arguments<ParseArgs...>();
    }
    // Check if the parser contains named arguments other than nargs.
    template <typename... Tags, typename... ExplicitTypes>
    static consteval bool has_other_than(const named_argument<Tags, ExplicitTypes> &...nargs)
    {
        return igor::has_other_than<ParseArgs...>(nargs...);
    }
    // Check if the parser contains duplicate named arguments (that is, check
    // if at least one named argument appears more than once).
    static consteval bool has_duplicates()
    {
        return igor::has_duplicates<ParseArgs...>();
    }
};

template <typename ExplicitType = void, typename T = decltype([] {})>
    requires std::same_as<ExplicitType, void> || (std::is_reference_v<ExplicitType>)
consteval auto make_named_argument()
{
    return named_argument<T, ExplicitType>{};
}

} // namespace v1 IGOR_ABI_TAG_ATTR

} // namespace igor

#define IGOR_MAKE_NAMED_ARGUMENT(arg) inline constexpr auto arg = igor::make_named_argument()

#undef IGOR_ABI_TAG_ATTR

#endif
