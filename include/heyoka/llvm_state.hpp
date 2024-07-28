// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_LLVM_STATE_HPP
#define HEYOKA_LLVM_STATE_HPP

#include <heyoka/config.hpp>

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/kw.hpp>
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
    std::uint32_t simd_size_flt = 1;
    std::uint32_t simd_size_dbl = 1;
    std::uint32_t simd_size_ldbl = 1;
#if defined(HEYOKA_HAVE_REAL128)
    std::uint32_t simd_size_f128 = 1;
#endif
};

HEYOKA_DLL_PUBLIC const target_features &get_target_features();

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);

template <typename T>
inline std::uint32_t recommended_simd_size()
{
    static_assert(detail::always_false_v<T>, "Unhandled type.");

    return 0;
}

template <>
HEYOKA_DLL_PUBLIC std::uint32_t recommended_simd_size<float>();

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

// Code model.
enum class code_model : unsigned { tiny, small, kernel, medium, large };

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, code_model);

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
    bool m_slp_vectorize;
    code_model m_c_model;
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

    // Helper to clamp the optimisation level to
    // the [0, 3] range.
    static unsigned clamp_opt_level(unsigned);

    // Helper to validate the code model enumerator,
    static void validate_code_model(code_model);

    // Implementation details for the variadic constructor.
    template <typename... KwArgs>
    static auto kw_args_ctor_impl(const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        // Module name (defaults to empty string).
        auto mod_name = [&p]() -> std::string {
            if constexpr (p.has(kw::mname)) {
                if constexpr (std::convertible_to<decltype(p(kw::mname)), std::string>) {
                    return p(kw::mname);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the 'mname' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        // Optimisation level (defaults to 3).
        auto opt_level = [&p]() -> unsigned {
            if constexpr (p.has(kw::opt_level)) {
                if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::opt_level))>>) {
                    return boost::numeric_cast<unsigned>(p(kw::opt_level));
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'opt_level' keyword argument.");
                }
            } else {
                return 3;
            }
        }();
        opt_level = clamp_opt_level(opt_level);

        // Fast math flag (defaults to false).
        auto fmath = [&p]() -> bool {
            if constexpr (p.has(kw::fast_math)) {
                if constexpr (std::convertible_to<decltype(p(kw::fast_math)), bool>) {
                    return p(kw::fast_math);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'fast_math' keyword argument.");
                }
            } else {
                return false;
            }
        }();

        // Force usage of AVX512 registers (defaults to false).
        auto force_avx512 = [&p]() -> bool {
            if constexpr (p.has(kw::force_avx512)) {
                if constexpr (std::convertible_to<decltype(p(kw::force_avx512)), bool>) {
                    return p(kw::force_avx512);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'force_avx512' keyword argument.");
                }
            } else {
                return false;
            }
        }();

        // Enable SLP vectorization (defaults to false).
        auto slp_vectorize = [&p]() -> bool {
            if constexpr (p.has(kw::slp_vectorize)) {
                if constexpr (std::convertible_to<decltype(p(kw::slp_vectorize)), bool>) {
                    return p(kw::slp_vectorize);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'slp_vectorize' keyword argument.");
                }
            } else {
                return false;
            }
        }();

        // Code model (defaults to small).
        auto c_model = [&p]() {
            if constexpr (p.has(kw::code_model)) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(p(kw::code_model))>, code_model>) {
                    return p(kw::code_model);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>,
                                  "Invalid type for the 'code_model' keyword argument.");
                }
            } else {
                return code_model::small;
            }
        }();
        validate_code_model(c_model);

        return std::tuple{std::move(mod_name), opt_level, fmath, force_avx512, slp_vectorize, c_model};
    }
    explicit llvm_state(std::tuple<std::string, unsigned, bool, bool, bool, code_model> &&);

    // Small shared helper to setup the math flags in the builder at the
    // end of a constructor.
    HEYOKA_DLL_LOCAL void ctor_setup_math_flags();

    // Low-level implementation details for compilation.
    HEYOKA_DLL_LOCAL void optimise();
    HEYOKA_DLL_LOCAL void compile_impl();
    HEYOKA_DLL_LOCAL void add_obj_trigger();

public:
    llvm_state();
    // NOTE: the constructor is enabled if:
    // - there is at least 1 argument (i.e., cannot act as a def ctor),
    // - all KwArgs are named arguments (this also prevents interference
    //   with the copy/move ctors).
    template <typename... KwArgs>
        requires(sizeof...(KwArgs) > 0u) && (!igor::has_unnamed_arguments<KwArgs...>())
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init, hicpp-member-init)
    explicit llvm_state(const KwArgs &...kw_args) : llvm_state(kw_args_ctor_impl(kw_args...))
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

    [[nodiscard]] const std::string &module_name() const;
    [[nodiscard]] const llvm::Module &module() const;
    [[nodiscard]] const ir_builder &builder() const;
    [[nodiscard]] const llvm::LLVMContext &context() const;
    [[nodiscard]] bool fast_math() const;
    [[nodiscard]] bool force_avx512() const;
    [[nodiscard]] unsigned get_opt_level() const;
    [[nodiscard]] bool get_slp_vectorize() const;
    [[nodiscard]] code_model get_code_model() const;

    [[nodiscard]] std::string get_ir() const;
    [[nodiscard]] std::string get_bc() const;
    void dump_object_code(const std::string &) const;
    [[nodiscard]] const std::string &get_object_code() const;

    void verify_function(const std::string &);
    void verify_function(llvm::Function *);

    [[nodiscard]] bool is_compiled() const;

    void compile();

    std::uintptr_t jit_lookup(const std::string &);

    [[nodiscard]] llvm_state make_similar() const;

    // Cache management.
    static std::size_t get_memcache_size();
    static std::size_t get_memcache_limit();
    static void set_memcache_limit(std::size_t);
    static void clear_memcache();
};

namespace detail
{

// The value contained in the in-memory cache.
struct llvm_mc_value {
    std::string opt_bc, opt_ir, obj;
};

// Cache lookup and insertion.
std::optional<llvm_mc_value> llvm_state_mem_cache_lookup(const std::string &, unsigned);
void llvm_state_mem_cache_try_insert(std::string, unsigned, llvm_mc_value);

} // namespace detail

HEYOKA_END_NAMESPACE

// Archive version changelog:
// - version 1: got rid of the inline_functions setting;
// - version 2: added the force_avx512 setting;
// - version 3: added the bitcode snapshot, simplified
//   compilation logic, slp_vectorize flag;
// - version 4: added the code_model option.
BOOST_CLASS_VERSION(heyoka::llvm_state, 4)

#endif
