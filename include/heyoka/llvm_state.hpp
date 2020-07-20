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
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC llvm_state
{
    struct jit;

    std::unique_ptr<jit> m_jitter;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<llvm::IRBuilder<>> m_builder;
    std::unordered_map<std::string, llvm::Value *> m_named_values;
    std::unordered_map<std::string, std::pair<std::type_index, std::vector<std::type_index>>> m_sig_map;
    bool m_verify = true;
    unsigned m_opt_level;

    // Check functions and verification.
    HEYOKA_DLL_LOCAL void check_uncompiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_compiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_add_name(const std::string &) const;
    HEYOKA_DLL_LOCAL void verify_function_impl(llvm::Function *);

    // Implementation details for expressions.
    template <typename T>
    HEYOKA_DLL_LOCAL void add_varargs_expression(const std::string &, const expression &,
                                                 const std::vector<std::string> &);
    template <typename T>
    HEYOKA_DLL_LOCAL void add_vecargs_expression(const std::string &, const expression &);
    template <typename T>
    HEYOKA_DLL_LOCAL void add_batch_expression_impl(const std::string &, const expression &, std::uint32_t);

    // Implementation details for Taylor integration.
    template <typename T>
    HEYOKA_DLL_LOCAL auto taylor_add_uvars_diff(const std::string &, const std::vector<expression> &, std::uint32_t,
                                                std::uint32_t);
    template <typename T>
    HEYOKA_DLL_LOCAL auto taylor_add_sv_diff(const std::string &, std::uint32_t, const variable &);
    template <typename T>
    HEYOKA_DLL_LOCAL auto taylor_add_sv_diff(const std::string &, std::uint32_t, const number &);
    template <typename T, typename U>
    HEYOKA_DLL_LOCAL auto add_taylor_jet_impl(const std::string &, U, std::uint32_t);
    template <typename T>
    HEYOKA_DLL_LOCAL void taylor_add_jet_func(const std::string &, const std::vector<expression> &,
                                              const std::vector<llvm::Function *> &, std::uint32_t, std::uint32_t,
                                              std::uint32_t);

public:
    explicit llvm_state(const std::string &, unsigned = 3);
    llvm_state(const llvm_state &) = delete;
    llvm_state(llvm_state &&) noexcept;
    llvm_state &operator=(const llvm_state &) = delete;
    llvm_state &operator=(llvm_state &&) noexcept;
    ~llvm_state();

    llvm::Module &module();
    llvm::IRBuilder<> &builder();
    llvm::LLVMContext &context();
    bool &verify();
    unsigned &opt_level();
    std::unordered_map<std::string, llvm::Value *> &named_values();

    const llvm::Module &module() const;
    const llvm::IRBuilder<> &builder() const;
    const llvm::LLVMContext &context() const;
    const bool &verify() const;
    const unsigned &opt_level() const;
    const std::unordered_map<std::string, llvm::Value *> &named_values() const;

    std::string dump_ir() const;
    std::string dump_function_ir(const std::string &) const;
    void dump_object_code(const std::string &) const;

    void verify_function(const std::string &);

    void optimise();

    void add_expression_dbl(const std::string &, const expression &);
    void add_expression_ldbl(const std::string &, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
    void add_expression_f128(const std::string &, const expression &);
#endif
    template <typename T>
    void add_expression(const std::string &name, const expression &ex)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_expression_dbl(name, ex);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_expression_ldbl(name, ex);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_expression_f128(name, ex);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    void add_vec_expression_dbl(const std::string &, const expression &);
    void add_vec_expression_ldbl(const std::string &, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
    void add_vec_expression_f128(const std::string &, const expression &);
#endif
    template <typename T>
    void add_vec_expression(const std::string &name, const expression &ex)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_vec_expression_dbl(name, ex);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_vec_expression_ldbl(name, ex);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_vec_expression_f128(name, ex);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    void add_batch_expression_dbl(const std::string &, const expression &, std::uint32_t);
    void add_batch_expression_ldbl(const std::string &, const expression &, std::uint32_t);
#if defined(HEYOKA_HAVE_REAL128)
    void add_batch_expression_f128(const std::string &, const expression &, std::uint32_t);
#endif
    template <typename T>
    void add_batch_expression(const std::string &name, const expression &ex, std::uint32_t batch_size)
    {
        if constexpr (std::is_same_v<T, double>) {
            add_batch_expression_dbl(name, ex, batch_size);
        } else if constexpr (std::is_same_v<T, long double>) {
            add_batch_expression_ldbl(name, ex, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            add_batch_expression_f128(name, ex, batch_size);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    std::vector<expression> add_taylor_jet_dbl(const std::string &, std::vector<expression>, std::uint32_t);
    std::vector<expression> add_taylor_jet_ldbl(const std::string &, std::vector<expression>, std::uint32_t);
#if defined(HEYOKA_HAVE_REAL128)
    std::vector<expression> add_taylor_jet_f128(const std::string &, std::vector<expression>, std::uint32_t);
#endif
    template <typename T>
    std::vector<expression> add_taylor_jet(const std::string &name, std::vector<expression> sys,
                                           std::uint32_t max_order)
    {
        if constexpr (std::is_same_v<T, double>) {
            return add_taylor_jet_dbl(name, std::move(sys), max_order);
        } else if constexpr (std::is_same_v<T, long double>) {
            return add_taylor_jet_ldbl(name, std::move(sys), max_order);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return add_taylor_jet_f128(name, std::move(sys), max_order);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    std::vector<expression> add_taylor_jet_dbl(const std::string &, std::vector<std::pair<expression, expression>>,
                                               std::uint32_t);
    std::vector<expression> add_taylor_jet_ldbl(const std::string &, std::vector<std::pair<expression, expression>>,
                                                std::uint32_t);
#if defined(HEYOKA_HAVE_REAL128)
    std::vector<expression> add_taylor_jet_f128(const std::string &, std::vector<std::pair<expression, expression>>,
                                                std::uint32_t);
#endif
    template <typename T>
    std::vector<expression> add_taylor_jet(const std::string &name, std::vector<std::pair<expression, expression>> sys,
                                           std::uint32_t max_order)
    {
        if constexpr (std::is_same_v<T, double>) {
            return add_taylor_jet_dbl(name, std::move(sys), max_order);
        } else if constexpr (std::is_same_v<T, long double>) {
            return add_taylor_jet_ldbl(name, std::move(sys), max_order);
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return add_taylor_jet_f128(name, std::move(sys), max_order);
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

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
    // of the N-ary fetch_expression_* overloads.
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
    auto fetch_expression_dbl(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<double, N>>(jit_lookup(name)));
    }
    template <std::size_t N>
    auto fetch_expression_ldbl(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<long double, N>>(jit_lookup(name)));
    }
#if defined(HEYOKA_HAVE_REAL128)
    template <std::size_t N>
    auto fetch_expression_f128(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<mppp::real128, N>>(jit_lookup(name)));
    }
#endif
    template <typename T, std::size_t N>
    auto fetch_expression(const std::string &name)
    {
        return sig_check(name, reinterpret_cast<vararg_f_ptr<T, N>>(jit_lookup(name)));
    }

    template <typename T>
    using ev_t = T (*)(const T *);
    ev_t<double> fetch_vec_expression_dbl(const std::string &);
    ev_t<long double> fetch_vec_expression_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    ev_t<mppp::real128> fetch_vec_expression_f128(const std::string &);
#endif
    template <typename T>
    ev_t<T> fetch_vec_expression(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<ev_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    // NOTE: remember documenting that
    // these pointers are restricted.
    template <typename T>
    using eb_t = void (*)(T *, const T *);
    eb_t<double> fetch_batch_expression_dbl(const std::string &);
    eb_t<long double> fetch_batch_expression_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    eb_t<mppp::real128> fetch_batch_expression_f128(const std::string &);
#endif
    template <typename T>
    eb_t<T> fetch_batch_expression(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<eb_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }

    template <typename T>
    using tj_t = void (*)(T *, std::uint32_t);
    tj_t<double> fetch_taylor_jet_dbl(const std::string &);
    tj_t<long double> fetch_taylor_jet_ldbl(const std::string &);
#if defined(HEYOKA_HAVE_REAL128)
    tj_t<mppp::real128> fetch_taylor_jet_f128(const std::string &);
#endif
    template <typename T>
    tj_t<T> fetch_taylor_jet(const std::string &name)
    {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, long double>
#if defined(HEYOKA_HAVE_REAL128)
                      || std::is_same_v<T, mppp::real128>
#endif
        ) {
            return sig_check(name, reinterpret_cast<tj_t<T>>(jit_lookup(name)));
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }
};

} // namespace heyoka

#endif
