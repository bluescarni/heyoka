// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_LLVM_STATE_HPP
#define HEYOKA_LLVM_STATE_HPP

#include <heyoka/config.hpp>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Helper struct to signal the availability
// of certain features on the host machine.
struct target_features {
    // x86.
    bool sse2 = false;
    bool avx = false;
    bool avx2 = false;
    bool avx512f = false;
    // aarch64.
    bool aarch64 = false;
    // powerpc64.
    // NOTE: for now, in the sleef support
    // bit we need only vsx and vsx3 (not vsx2).
    bool vsx = false;
    bool vsx3 = false;
    // Recommended SIMD sizes.
    std::uint32_t simd_size_dbl = 1;
    std::uint32_t simd_size_ldbl = 1;
#if defined(HEYOKA_HAVE_REAL128)
    std::uint32_t simd_size_f128 = 1;
#endif
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
// NOTE: this flag is used to force the use of 512-bit AVX512
// registers (if the CPU supports them). At the time of this writing,
// LLVM defaults to 256-bit registers due to CPU downclocking issues
// which can lead to performance degradation. Hopefully we
// can get rid of this in the future when AVX512 implementations improve
// and LLVM learns to discriminate good and bad implementations.
IGOR_MAKE_NAMED_ARGUMENT(force_avx512);

} // namespace kw

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

template <typename T>
inline std::uint32_t recommended_simd_size()
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return 0;
}

template <>
HEYOKA_DLL_PUBLIC std::uint32_t recommended_simd_size<double>();

template <>
HEYOKA_DLL_PUBLIC std::uint32_t recommended_simd_size<long double>();

#if defined(HEYOKA_HAVE_REAL128)

template <>
HEYOKA_DLL_PUBLIC std::uint32_t recommended_simd_size<mppp::real128>();

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
HEYOKA_DLL_PUBLIC std::uint32_t recommended_simd_size<mppp::real>();

#endif

class HEYOKA_DLL_PUBLIC llvm_state
{
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

    struct jit;

    // NOTE: LLVM rules state that a context has to outlive all
    // modules associated with it.
    std::unique_ptr<jit> m_jitter;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<ir_builder> m_builder;
    unsigned m_opt_level;
    std::string m_ir_snapshot;
    std::string m_bc_snapshot;
    bool m_fast_math;
    bool m_force_avx512;
    std::string m_module_name;

    // Serialization.
    // NOTE: serialisation does not preserve the state of the builder
    // (e.g., wrt fast math flags). This needs to be documented.
    template <typename Archive>
    HEYOKA_DLL_LOCAL void save_impl(Archive &, unsigned) const;
    template <typename Archive>
    HEYOKA_DLL_LOCAL void load_impl(Archive &, unsigned);

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

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

            // Force usage of AVX512 registers (defaults to false).
            auto force_avx512 = [&p]() -> bool {
                if constexpr (p.has(kw::force_avx512)) {
                    return std::forward<decltype(p(kw::force_avx512))>(p(kw::force_avx512));
                } else {
                    return false;
                }
            }();

            return std::tuple{std::move(mod_name), opt_level, fmath, force_avx512};
        }
    }
    explicit llvm_state(std::tuple<std::string, unsigned, bool, bool> &&);

    // Small shared helper to setup the math flags in the builder at the
    // end of a constructor.
    HEYOKA_DLL_LOCAL void ctor_setup_math_flags();

    // Low-level implementation details for compilation.
    HEYOKA_DLL_LOCAL void compile_impl();
    HEYOKA_DLL_LOCAL void add_obj_trigger();

    // Meta-programming for the kwargs ctor. Enabled if:
    // - there is at least 1 argument (i.e., cannot act as a def ctor),
    // - if there is only 1 argument, it cannot be of type llvm_state
    //   (so that it does not interfere with copy/move ctors).
    template <typename... KwArgs>
    using kwargs_ctor_enabler = std::enable_if_t<
        (sizeof...(KwArgs) > 0u)
            && (sizeof...(KwArgs) > 1u
                || std::conjunction_v<std::negation<std::is_same<detail::uncvref_t<KwArgs>, llvm_state>>...>),
        int>;

public:
    llvm_state();
    template <typename... KwArgs, kwargs_ctor_enabler<KwArgs...> = 0>
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

    [[nodiscard]] const std::string &module_name() const;
    [[nodiscard]] const llvm::Module &module() const;
    [[nodiscard]] const ir_builder &builder() const;
    [[nodiscard]] const llvm::LLVMContext &context() const;
    [[nodiscard]] const unsigned &opt_level() const;
    [[nodiscard]] bool fast_math() const;
    [[nodiscard]] bool force_avx512() const;

    [[nodiscard]] std::string get_ir() const;
    [[nodiscard]] std::string get_bc() const;
    void dump_object_code(const std::string &) const;
    [[nodiscard]] const std::string &get_object_code() const;

    void verify_function(const std::string &);
    void verify_function(llvm::Function *);

    void optimise();

    [[nodiscard]] bool is_compiled() const;

    void compile();

    std::uintptr_t jit_lookup(const std::string &);

    [[nodiscard]] llvm_state make_similar() const;
};

HEYOKA_END_NAMESPACE

// Archive version changelog:
// - version 1: got rid of the inline_functions setting;
// - version 2: added the force_avx512 setting;
// - version 3: added the bitcode snapshot, compilation
//   now always triggering code generation.
BOOST_CLASS_VERSION(heyoka::llvm_state, 3)

#endif
