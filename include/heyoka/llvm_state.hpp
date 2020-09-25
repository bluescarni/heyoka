// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_LLVM_STATE_HPP
#define HEYOKA_LLVM_STATE_HPP

#include <heyoka/config.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(mname);
IGOR_MAKE_NAMED_ARGUMENT(opt_level);
IGOR_MAKE_NAMED_ARGUMENT(fast_math);
IGOR_MAKE_NAMED_ARGUMENT(save_object_code);

namespace detail
{

// Default value for the opt_level argument.
inline constexpr unsigned default_opt_level = 3;

} // namespace detail

} // namespace kw

class llvm_state;

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

class HEYOKA_DLL_PUBLIC llvm_state
{
    friend std::ostream &operator<<(std::ostream &, const llvm_state &);

    struct jit;

    std::unique_ptr<jit> m_jitter;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<llvm::IRBuilder<>> m_builder;
    std::unordered_map<std::string, llvm::Value *> m_named_values;
    std::unordered_map<std::string, std::pair<std::type_index, std::vector<std::type_index>>> m_sig_map;
    unsigned m_opt_level;
    std::string m_ir_snapshot;
    bool m_use_fast_math;
    std::string m_module_name;
    bool m_save_object_code;
    std::string m_object_code;

    // Check functions and verification.
    HEYOKA_DLL_LOCAL void check_uncompiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_compiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_add_name(const std::string &) const;

    // Implementation details for expressions.
    template <typename T>
    HEYOKA_DLL_LOCAL void add_varargs_expression(const std::string &, const expression &,
                                                 const std::vector<std::string> &);
    template <typename T>
    HEYOKA_DLL_LOCAL void add_vecargs_expression(const std::string &, const expression &);
    template <typename T>
    HEYOKA_DLL_LOCAL void add_vecargs_expressions(const std::string &, const std::vector<expression> &);
    template <typename T>
    HEYOKA_DLL_LOCAL void add_batch_expression_impl(const std::string &, const expression &, std::uint32_t);

    // Implementation details for the variadic constructor.
    template <typename... KwArgs>
    static auto kw_args_ctor_impl(KwArgs &&... kw_args)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments in the construction of an llvm_state contain "
                          "unnamed arguments.");
        } else {
            // Module name (defaults to empty string).
            auto mod_name = [&p]() -> std::string {
                if constexpr (p.has(kw::mname)) {
                    return std::forward<decltype(p(kw::mname))>(p(kw::mname));
                } else {
                    return "";
                }
            }();

            // Optimisation level.
            auto opt_level = [&p]() -> unsigned {
                if constexpr (p.has(kw::opt_level)) {
                    return std::forward<decltype(p(kw::opt_level))>(p(kw::opt_level));
                } else {
                    return kw::detail::default_opt_level;
                }
            }();

            // Fast math flag (defaults to true).
            auto fmath = [&p]() -> bool {
                if constexpr (p.has(kw::fast_math)) {
                    return std::forward<decltype(p(kw::fast_math))>(p(kw::fast_math));
                } else {
                    return true;
                }
            }();

            // Save object code (defaults to false).
            auto socode = [&p]() -> bool {
                if constexpr (p.has(kw::save_object_code)) {
                    return std::forward<decltype(p(kw::save_object_code))>(p(kw::save_object_code));
                } else {
                    return false;
                }
            }();

            return std::tuple{std::move(mod_name), opt_level, fmath, socode};
        }
    }
    explicit llvm_state(std::tuple<std::string, unsigned, bool, bool> &&);

public:
    llvm_state();
    // NOTE: enable the kwargs ctor only if:
    // - there is at least 1 argument (i.e., cannot act as a def ctor),
    // - if there is only 1 argument, it cannot be of type llvm_state
    //   (so that it does not interfere with copy/move ctors).
    template <typename... KwArgs,
              std::enable_if_t<(sizeof...(KwArgs) > 0u)
                                   && (sizeof...(KwArgs) > 1u
                                       || (... && !std::is_same_v<detail::uncvref_t<KwArgs>, llvm_state>)),
                               int> = 0>
    explicit llvm_state(KwArgs &&... kw_args) : llvm_state(kw_args_ctor_impl(std::forward<KwArgs>(kw_args)...))
    {
    }
    llvm_state(const llvm_state &);
    llvm_state(llvm_state &&) noexcept;
    llvm_state &operator=(const llvm_state &);
    llvm_state &operator=(llvm_state &&) noexcept;
    ~llvm_state();

    std::uint32_t vector_size_dbl() const;
    std::uint32_t vector_size_ldbl() const;
#if defined(HEYOKA_HAVE_REAL128)
    std::uint32_t vector_size_f128() const;
#endif
    template <typename T>
    std::uint32_t vector_size() const
    {
        if constexpr (std::is_same_v<T, double>) {
            return vector_size_dbl();
        } else if constexpr (std::is_same_v<T, long double>) {
            return vector_size_ldbl();
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return vector_size_f128();
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    llvm::Module &module();
    llvm::IRBuilder<> &builder();
    llvm::LLVMContext &context();
    unsigned &opt_level();
    std::unordered_map<std::string, llvm::Value *> &named_values();

    const llvm::Module &module() const;
    const llvm::IRBuilder<> &builder() const;
    const llvm::LLVMContext &context() const;
    const unsigned &opt_level() const;
    const std::unordered_map<std::string, llvm::Value *> &named_values() const;

    std::string get_ir() const;
    void dump_object_code(const std::string &) const;

    void verify_function(const std::string &);
    void verify_function(llvm::Function *);

    void optimise();

    void add_nary_function_dbl(const std::string &, const expression &);
    void add_nary_function_ldbl(const std::string &, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
    void add_nary_function_f128(const std::string &, const expression &);
#endif
    template <typename T>
    void add_nary_function(const std::string &name, const expression &ex)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_nary_function_dbl(name, ex);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_nary_function_ldbl(name, ex);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_nary_function_f128(name, ex);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    void add_function_dbl(const std::string &, const expression &);
    void add_function_ldbl(const std::string &, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
    void add_function_f128(const std::string &, const expression &);
#endif
    template <typename T>
    void add_function(const std::string &name, const expression &ex)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_function_dbl(name, ex);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_function_ldbl(name, ex);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_function_f128(name, ex);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    void add_vector_function_dbl(const std::string &, const std::vector<expression> &);
    void add_vector_function_ldbl(const std::string &, const std::vector<expression> &);
#if defined(HEYOKA_HAVE_REAL128)
    void add_vector_function_f128(const std::string &, const std::vector<expression> &);
#endif
    template <typename T>
    void add_vector_function(const std::string &name, const std::vector<expression> &es)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_vector_function_dbl(name, es);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_vector_function_ldbl(name, es);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_vector_function_f128(name, es);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    void add_function_batch_dbl(const std::string &, const expression &, std::uint32_t);
    void add_function_batch_ldbl(const std::string &, const expression &, std::uint32_t);
#if defined(HEYOKA_HAVE_REAL128)
    void add_function_batch_f128(const std::string &, const expression &, std::uint32_t);
#endif
    template <typename T>
    void add_function_batch(const std::string &name, const expression &ex, std::uint32_t batch_size)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_function_batch_dbl(name, ex, batch_size);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_function_batch_ldbl(name, ex, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_function_batch_f128(name, ex, batch_size);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    bool is_compiled() const;

    void compile();

    std::uintptr_t jit_lookup(const std::string &);

private:
    template <typename Tup, std::size_t... S>
    static bool sig_check_args(const std::vector<std::type_index> &v, std::index_sequence<S...>)
    {
        assert(sizeof...(S) == v.size());
        static_assert(sizeof...(S) == std::tuple_size_v<Tup>);

        return ((v[S] == std::type_index(typeid(std::tuple_element_t<S, Tup>))) && ...);
    }
    // This function will check if ptr is compatible with the signature of
    // the function called "name" which was added via one of the add_*()
    // overloads.
    // NOTE: this function is supposed to be called only
    // with a pointer obtained via jit_lookup, thus we don't need
    // compiled/uncompiled checks.
    template <typename Ret, typename... Args>
    auto sig_check(const std::string &name, Ret (*ptr)(Args...)) const
    {
        auto it = m_sig_map.find(name);

        if (it == m_sig_map.end()) {
            // NOTE: this could happen if jit_lookup() in fetch_*() returns a pointer
            // to some object which was not added via the add_*() overloads.
            throw std::invalid_argument("Cannot determine the signature of the function '" + name + "'");
        }

        if (it->second.first != std::type_index(typeid(Ret))) {
            throw std::invalid_argument("Function return type mismatch when trying to fetch the function '" + name
                                        + "' from the compiled module");
        }

        if (sizeof...(Args) != it->second.second.size()) {
            throw std::invalid_argument(
                "Mismatch in the number of function arguments when trying to fetch the function '" + name
                + "' from the compiled module");
        }

        // Check the types of all arguments.
        if (!sig_check_args<std::tuple<Args...>>(it->second.second, std::make_index_sequence<sizeof...(Args)>{})) {
            throw std::invalid_argument("Mismatch in the type of function arguments when trying to fetch the function '"
                                        + name + "' from the compiled module");
        }

        return ptr;
    }

    // Machinery to construct a function pointer
    // type with signature T(T, T, ..., T).
    // This type will be used in the implementation
    // of the fetch_nary_* overloads.
    template <typename T, std::size_t>
    using always_same_t = T;

    template <typename T, std::size_t... S>
    static auto get_vararg_type_impl(std::index_sequence<S...>)
    {
        return static_cast<T (*)(always_same_t<T, S>...)>(nullptr);
    }

    template <typename T, std::size_t N>
    using vararg_f_ptr = decltype(get_vararg_type_impl<T>(std::make_index_sequence<N>{}));

public:
    template <std::size_t N>
    auto fetch_nary_function_dbl(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<double, N>>(jit_lookup(name)));
    }
    template <std::size_t N>
    auto fetch_nary_function_ldbl(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<long double, N>>(jit_lookup(name)));
    }
#if defined(HEYOKA_HAVE_REAL128)
    template <std::size_t N>
    auto fetch_nary_function_f128(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<mppp::real128, N>>(jit_lookup(name)));
    }
#endif
    template <typename T, std::size_t N>
    auto fetch_nary_function(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<T, N>>(jit_lookup(name)));
    }

    template <typename T>
    using sf_t = T (*)(const T *);
    sf_t<double> fetch_function_dbl(const std::string &);
    sf_t<long double> fetch_function_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    sf_t<mppp::real128> fetch_function_f128(const std::string &);
#endif
    template <typename T>
    sf_t<T> fetch_function(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<sf_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    // NOTE: remember documenting that
    // these pointers are restricted.
    template <typename T>
    using vf_t = void (*)(T *, const T *);
    vf_t<double> fetch_vector_function_dbl(const std::string &);
    vf_t<long double> fetch_vector_function_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    vf_t<mppp::real128> fetch_vector_function_f128(const std::string &);
#endif
    template <typename T>
    vf_t<T> fetch_vector_function(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<vf_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    // NOTE: remember documenting that
    // these pointers are restricted.
    template <typename T>
    using sfb_t = void (*)(T *, const T *);
    sfb_t<double> fetch_function_batch_dbl(const std::string &);
    sfb_t<long double> fetch_function_batch_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    sfb_t<mppp::real128> fetch_function_batch_f128(const std::string &);
#endif
    template <typename T>
    sfb_t<T> fetch_function_batch(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<sfb_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }
};

} // namespace heyoka

#endif
