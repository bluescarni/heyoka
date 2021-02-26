// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_LLVM_STATE_HPP
#define HEYOKA_LLVM_STATE_HPP

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

namespace detail
{

// Helper struct to signal the availability
// of certain features on the host machine.
struct target_features {
    bool sse2 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
};

// NOTE: no need to make this DLL-public as long
// as this is used only in library code.
const target_features &get_target_features();

} // namespace detail

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(mname);
IGOR_MAKE_NAMED_ARGUMENT(opt_level);
IGOR_MAKE_NAMED_ARGUMENT(fast_math);
IGOR_MAKE_NAMED_ARGUMENT(inline_functions);

} // namespace kw

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

class HEYOKA_DLL_PUBLIC llvm_state
{
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

    struct jit;

    std::unique_ptr<jit> m_jitter;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<ir_builder> m_builder;
    unsigned m_opt_level;
    std::string m_ir_snapshot;
    bool m_fast_math;
    std::string m_module_name;
    bool m_inline_functions;

    // Check functions.
    HEYOKA_DLL_LOCAL void check_uncompiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_compiled(const char *) const;

    // Implementation details for the variadic constructor.
    template <typename... KwArgs>
    static auto kw_args_ctor_impl(KwArgs &&...kw_args)
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

            // Optimisation level (defaults to 3).
            auto opt_level = [&p]() -> unsigned {
                if constexpr (p.has(kw::opt_level)) {
                    return std::forward<decltype(p(kw::opt_level))>(p(kw::opt_level));
                } else {
                    return 3;
                }
            }();

            // Fast math flag (defaults to false).
            auto fmath = [&p]() -> bool {
                if constexpr (p.has(kw::fast_math)) {
                    return std::forward<decltype(p(kw::fast_math))>(p(kw::fast_math));
                } else {
                    return false;
                }
            }();

            // Inline functions (defaults to true).
            auto i_func = [&p]() -> bool {
                if constexpr (p.has(kw::inline_functions)) {
                    return std::forward<decltype(p(kw::inline_functions))>(p(kw::inline_functions));
                } else {
                    return true;
                }
            }();

            return std::tuple{std::move(mod_name), opt_level, fmath, i_func};
        }
    }
    explicit llvm_state(std::tuple<std::string, unsigned, bool, bool> &&);

    // Small shared helper to setup the math flags in the builder at the
    // end of a constructor.
    HEYOKA_DLL_LOCAL void ctor_setup_math_flags();

public:
    llvm_state();
    // NOTE: enable the kwargs ctor only if:
    // - there is at least 1 argument (i.e., cannot act as a def ctor),
    // - if there is only 1 argument, it cannot be of type llvm_state
    //   (so that it does not interfere with copy/move ctors).
    template <typename... KwArgs,
              std::enable_if_t<
                  (sizeof...(KwArgs) > 0u)
                      && (sizeof...(KwArgs) > 1u
                          || std::conjunction_v<std::negation<std::is_same<detail::uncvref_t<KwArgs>, llvm_state>>...>),
                  int> = 0>
    explicit llvm_state(KwArgs &&...kw_args) : llvm_state(kw_args_ctor_impl(std::forward<KwArgs>(kw_args)...))
    {
    }
    llvm_state(const llvm_state &);
    llvm_state(llvm_state &&) noexcept;
    llvm_state &operator=(const llvm_state &);
    llvm_state &operator=(llvm_state &&) noexcept;
    ~llvm_state();

    llvm::Module &module();
    ir_builder &builder();
    llvm::LLVMContext &context();
    unsigned &opt_level();
    bool &fast_math();
    bool &inline_functions();

    const std::string &module_name() const;
    const llvm::Module &module() const;
    const ir_builder &builder() const;
    const llvm::LLVMContext &context() const;
    const unsigned &opt_level() const;
    const bool &fast_math() const;
    const bool &inline_functions() const;

    std::string get_ir() const;
    void dump_object_code(const std::string &) const;
    const std::string &get_object_code() const;

    void verify_function(const std::string &);
    void verify_function(llvm::Function *);

    void optimise();

    bool is_compiled() const;

    void compile();

    std::uintptr_t jit_lookup(const std::string &);
};

} // namespace heyoka

#endif
