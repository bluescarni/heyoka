// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/ranges_to.hpp>
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
HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_multi_state &);

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

HEYOKA_END_NAMESPACE

// fmt formatter for taylor_outcome, implemented on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::code_model> : fmt::ostream_formatter {
};

} // namespace fmt

HEYOKA_BEGIN_NAMESPACE

class HEYOKA_DLL_PUBLIC llvm_state
{
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_state &);
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_multi_state &);
    friend class HEYOKA_DLL_PUBLIC llvm_multi_state;

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
        const igor::parser p{kw_args...};

        // Module name (defaults to empty string).
        auto mod_name = [&p]() -> std::string {
            if constexpr (p.has(kw::mname)) {
                // NOTE: we explicitly turn the keyword argument here into a const reference in order to fortify the
                // constructor against multiple usages of the same set of keyword arguments. If we do not do this and
                // mname is passed as, say, an rvalue std::string, then using the same keyword argument to initialise a
                // second llvm_state would incur in use-after-move.
                //
                // NOTE: this also fortifies the constructor against concurrent usage of the same set of keyword
                // arguments from multiple threads.
                auto &&val = p(kw::mname);
                return std::as_const(val);
            } else {
                return {};
            }
        }();

        // Optimisation level (defaults to 3).
        const auto opt_level = clamp_opt_level(boost::numeric_cast<unsigned>(p(kw::opt_level, 3)));

        // Fast math flag (defaults to false).
        const auto fmath = p(kw::fast_math, false);

        // Force usage of AVX512 registers (defaults to false).
        const auto force_avx512 = p(kw::force_avx512, false);

        // Enable SLP vectorization (defaults to false).
        const auto slp_vectorize = p(kw::slp_vectorize, false);

        // Code model (defaults to small).
        const auto c_model = p(kw::code_model, code_model::small);
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
    // kwargs configuration.
    //
    // NOTE: this configuration (paired with the internal use of std::as_const() for mname) ensures that we can re-use
    // the set of kw_args across multiple invocations.
    static constexpr auto kw_cfg = igor::config<
        igor::descr<kw::mname, []<typename U>() { return detail::string_like<std::remove_cvref_t<U>>; }>{},
        kw::descr::integral<kw::opt_level>, kw::descr::boolean<kw::fast_math>, kw::descr::boolean<kw::force_avx512>,
        kw::descr::boolean<kw::slp_vectorize>, kw::descr::same_as<kw::code_model, code_model>>{};

    llvm_state();
    // NOTE: we require at least 1 kwarg in order to avoid competition with the default ctor.
    template <typename... KwArgs>
        requires(sizeof...(KwArgs) > 0u) && igor::validate<kw_cfg, KwArgs...>
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
    [[nodiscard]] const std::string &get_object_code() const;

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
    std::vector<std::string> opt_bc, opt_ir, obj;

    [[nodiscard]] std::size_t total_size() const;
};

// Cache lookup and insertion.
std::optional<llvm_mc_value> llvm_state_mem_cache_lookup(const std::vector<std::string> &, unsigned);
void llvm_state_mem_cache_try_insert(std::vector<std::string>, unsigned, llvm_mc_value);

// At this time, it seems like parallel compilation in lljit is buggy. It has gotten better with LLVM 20 but we still
// get occasional "Duplicate definition of symbol" errors in the CI. Thus, for the time being, let us just disable
// parallel compilation by default.
inline constexpr bool default_parjit = false;

} // namespace detail

class HEYOKA_DLL_PUBLIC llvm_multi_state
{
    friend HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const llvm_multi_state &);

    struct impl;

    std::unique_ptr<impl> m_impl;

    HEYOKA_DLL_LOCAL void compile_impl();
    HEYOKA_DLL_LOCAL void add_obj_triggers();

    // Check functions.
    HEYOKA_DLL_LOCAL void check_compiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_uncompiled(const char *) const;

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    llvm_multi_state();
    explicit llvm_multi_state(std::vector<llvm_state>, bool = detail::default_parjit);
    template <typename R>
        requires std::ranges::input_range<R>
                 && std::same_as<llvm_state, std::remove_cvref_t<std::ranges::range_reference_t<R>>>
    explicit llvm_multi_state(R &&rng, bool parjit = detail::default_parjit)
        : llvm_multi_state(detail::ranges_to<std::vector<llvm_state>>(std::forward<R>(rng)), parjit)
    {
    }
    llvm_multi_state(const llvm_multi_state &);
    llvm_multi_state(llvm_multi_state &&) noexcept;
    llvm_multi_state &operator=(const llvm_multi_state &);
    llvm_multi_state &operator=(llvm_multi_state &&) noexcept;
    ~llvm_multi_state();

    [[nodiscard]] bool is_compiled() const noexcept;

    [[nodiscard]] unsigned get_n_modules() const noexcept;

    [[nodiscard]] bool fast_math() const noexcept;
    [[nodiscard]] bool force_avx512() const noexcept;
    [[nodiscard]] unsigned get_opt_level() const noexcept;
    [[nodiscard]] bool get_slp_vectorize() const noexcept;
    [[nodiscard]] code_model get_code_model() const noexcept;
    [[nodiscard]] bool get_parjit() const noexcept;

    [[nodiscard]] std::vector<std::string> get_ir() const;
    [[nodiscard]] std::vector<std::string> get_bc() const;
    [[nodiscard]] const std::vector<std::string> &get_object_code() const;

    void compile();

    std::uintptr_t jit_lookup(const std::string &);
};

HEYOKA_END_NAMESPACE

// Archive version changelog:
// - version 1: got rid of the inline_functions setting;
// - version 2: added the force_avx512 setting;
// - version 3: added the bitcode snapshot, simplified
//   compilation logic, slp_vectorize flag;
// - version 4: added the code_model option.
BOOST_CLASS_VERSION(heyoka::llvm_state, 4)

#endif
