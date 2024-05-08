// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_EXPRESSION_HPP
#define HEYOKA_EXPRESSION_HPP

#include <heyoka/config.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/container/flat_map.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>
#include <fmt/ostream.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/func_cache.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC void swap(expression &, expression &) noexcept;

class HEYOKA_DLL_PUBLIC expression
{
    friend HEYOKA_DLL_PUBLIC void swap(expression &, expression &) noexcept;

public:
    using value_type = std::variant<number, variable, func, param>;

private:
    value_type m_value;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    expression() noexcept;

    explicit expression(float) noexcept;
    explicit expression(double) noexcept;
    explicit expression(long double) noexcept;
#if defined(HEYOKA_HAVE_REAL128)
    explicit expression(mppp::real128) noexcept;
#endif
#if defined(HEYOKA_HAVE_REAL)
    explicit expression(mppp::real);
#endif
    explicit expression(std::string);

    explicit expression(number);
    explicit expression(variable);
    explicit expression(func) noexcept;
    explicit expression(param) noexcept;

    expression(const expression &);
    expression(expression &&) noexcept;

    ~expression();

    expression &operator=(const expression &);
    expression &operator=(expression &&) noexcept;

    [[nodiscard]] const value_type &value() const noexcept;
};

HEYOKA_DLL_PUBLIC expression copy(const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> copy(const std::vector<expression> &);

inline namespace literals
{

HEYOKA_DLL_PUBLIC expression operator""_flt(long double);
HEYOKA_DLL_PUBLIC expression operator""_flt(unsigned long long);

HEYOKA_DLL_PUBLIC expression operator""_dbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_dbl(unsigned long long);

HEYOKA_DLL_PUBLIC expression operator""_ldbl(long double);
HEYOKA_DLL_PUBLIC expression operator""_ldbl(unsigned long long);

#if defined(HEYOKA_HAVE_REAL128)

template <char... Chars>
expression operator""_f128()
{
    return expression{mppp::literals::operator""_rq < Chars... > ()};
}

#endif

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

} // namespace literals

namespace detail
{

// NOTE: some member functions of func_iface_impl need to go here because
// the definition of expression must be available.
template <typename Base, typename Holder, typename T>
    requires is_udf<T>
inline void func_iface_impl<Base, Holder, T>::replace_args(std::vector<expression> new_args)
{
    return static_cast<fbase &>(getval<Holder>(this)).replace_args(std::move(new_args));
}

struct HEYOKA_DLL_PUBLIC prime_wrapper {
    std::string m_str;

    explicit prime_wrapper(std::string);
    prime_wrapper(const prime_wrapper &);
    prime_wrapper(prime_wrapper &&) noexcept;
    prime_wrapper &operator=(const prime_wrapper &);
    prime_wrapper &operator=(prime_wrapper &&) noexcept;
    ~prime_wrapper();

    // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
    std::pair<expression, expression> operator=(expression) &&;
};

} // namespace detail

HEYOKA_DLL_PUBLIC detail::prime_wrapper prime(const expression &);

inline namespace literals
{

HEYOKA_DLL_PUBLIC detail::prime_wrapper operator""_p(const char *, std::size_t);

} // namespace literals

namespace detail
{

std::size_t hash(funcptr_map<std::size_t> &, const expression &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const expression &) noexcept;

void stream_expression(std::ostringstream &, const expression &);

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const expression &);

HEYOKA_END_NAMESPACE

// fmt formatter for expression, implemented
// on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::expression> : fmt::ostream_formatter {
};

} // namespace fmt

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const expression &);
HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const std::vector<expression> &);
HEYOKA_DLL_PUBLIC expression rename_variables(const expression &, const std::unordered_map<std::string, std::string> &);
HEYOKA_DLL_PUBLIC std::vector<expression> rename_variables(const std::vector<expression> &,
                                                           const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC expression operator+(expression);
HEYOKA_DLL_PUBLIC expression operator-(const expression &);

HEYOKA_DLL_PUBLIC expression operator+(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression operator+(const expression &, float);
HEYOKA_DLL_PUBLIC expression operator+(const expression &, double);
HEYOKA_DLL_PUBLIC expression operator+(const expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator+(const expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator+(const expression &, mppp::real);
#endif
HEYOKA_DLL_PUBLIC expression operator+(float, const expression &);
HEYOKA_DLL_PUBLIC expression operator+(double, const expression &);
HEYOKA_DLL_PUBLIC expression operator+(long double, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator+(mppp::real128, const expression &);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator+(mppp::real, const expression &);
#endif

HEYOKA_DLL_PUBLIC expression operator-(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression operator-(const expression &, double);
HEYOKA_DLL_PUBLIC expression operator-(const expression &, float);
HEYOKA_DLL_PUBLIC expression operator-(const expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator-(const expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator-(const expression &, mppp::real);
#endif
HEYOKA_DLL_PUBLIC expression operator-(float, const expression &);
HEYOKA_DLL_PUBLIC expression operator-(double, const expression &);
HEYOKA_DLL_PUBLIC expression operator-(long double, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator-(mppp::real128, const expression &);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator-(mppp::real, const expression &);
#endif

HEYOKA_DLL_PUBLIC expression operator*(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression operator*(const expression &, double);
HEYOKA_DLL_PUBLIC expression operator*(const expression &, float);
HEYOKA_DLL_PUBLIC expression operator*(const expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator*(const expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator*(const expression &, mppp::real);
#endif
HEYOKA_DLL_PUBLIC expression operator*(float, const expression &);
HEYOKA_DLL_PUBLIC expression operator*(double, const expression &);
HEYOKA_DLL_PUBLIC expression operator*(long double, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator*(mppp::real128, const expression &);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator*(mppp::real, const expression &);
#endif

HEYOKA_DLL_PUBLIC expression operator/(const expression &, const expression &);
HEYOKA_DLL_PUBLIC expression operator/(const expression &, float);
HEYOKA_DLL_PUBLIC expression operator/(const expression &, double);
HEYOKA_DLL_PUBLIC expression operator/(const expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator/(const expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator/(const expression &, mppp::real);
#endif
HEYOKA_DLL_PUBLIC expression operator/(float, const expression &);
HEYOKA_DLL_PUBLIC expression operator/(double, const expression &);
HEYOKA_DLL_PUBLIC expression operator/(long double, const expression &);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression operator/(mppp::real128, const expression &);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression operator/(mppp::real, const expression &);
#endif

HEYOKA_DLL_PUBLIC expression &operator+=(expression &, const expression &);
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, float);
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression &operator+=(expression &, mppp::real);
#endif

HEYOKA_DLL_PUBLIC expression &operator-=(expression &, const expression &);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, float);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression &operator-=(expression &, mppp::real);
#endif

HEYOKA_DLL_PUBLIC expression &operator*=(expression &, const expression &);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, float);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression &operator*=(expression &, mppp::real);
#endif

HEYOKA_DLL_PUBLIC expression &operator/=(expression &, const expression &);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, float);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, double);
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, long double);
#if defined(HEYOKA_HAVE_REAL128)
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
HEYOKA_DLL_PUBLIC expression &operator/=(expression &, mppp::real);
#endif

HEYOKA_DLL_PUBLIC bool operator==(const expression &, const expression &) noexcept;
HEYOKA_DLL_PUBLIC bool operator!=(const expression &, const expression &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t get_n_nodes(const expression &);

HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<std::string, expression> &);
HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::map<expression, expression> &);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::unordered_map<std::string, expression> &);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::map<expression, expression> &);

enum class diff_args { vars, params, all };

// Fwd declaration.
class HEYOKA_DLL_PUBLIC dtens;

namespace detail
{

expression diff(funcptr_map<expression> &, const expression &, const std::string &);
expression diff(funcptr_map<expression> &, const expression &, const param &);

// NOTE: public only for testing purposes.
HEYOKA_DLL_PUBLIC std::pair<std::vector<expression>, std::vector<expression>::size_type>
diff_decompose(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC dtens diff_tensors(const std::vector<expression> &,
                                     const std::variant<diff_args, std::vector<expression>> &, std::uint32_t);

} // namespace detail

HEYOKA_DLL_PUBLIC expression diff(const expression &, const param &);
HEYOKA_DLL_PUBLIC std::vector<expression> diff(const std::vector<expression> &, const param &);
HEYOKA_DLL_PUBLIC expression diff(const expression &, const std::string &);
HEYOKA_DLL_PUBLIC std::vector<expression> diff(const std::vector<expression> &, const std::string &);
HEYOKA_DLL_PUBLIC expression diff(const expression &, const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> diff(const std::vector<expression> &, const expression &);

namespace detail
{

// Sparse structure used to index derivatives in dtens:
// - the first element of the pair is the function component index,
// - the second element is the vector of variable index/diff order pairs,
//   which is kept sorted according to the variable index, and in which no
//   diff order can be zero and no variable index can appear twice.
using dtens_sv_idx_t = std::pair<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>>;

struct dtens_sv_idx_cmp {
    [[nodiscard]] bool operator()(const dtens_sv_idx_t &, const dtens_sv_idx_t &) const;
};

using dtens_map_t = boost::container::flat_map<dtens_sv_idx_t, expression, dtens_sv_idx_cmp>;

// Utility function to check that a dtens_sv_idx_t is well-formed.
bool sv_sanity_check(const dtens_sv_idx_t &);

} // namespace detail

class HEYOKA_DLL_PUBLIC dtens
{
public:
    // Derivative indexing vector in dense form.
    using v_idx_t = std::vector<std::uint32_t>;

    // Derivative indexing vector in sparse form.
    using sv_idx_t = detail::dtens_sv_idx_t;

    using size_type = detail::dtens_map_t::size_type;

private:
    // NOTE: detail::diff_tensors() needs access to the private ctor.
    friend dtens detail::diff_tensors(const std::vector<expression> &,
                                      const std::variant<diff_args, std::vector<expression>> &, std::uint32_t);

    struct impl;

    std::unique_ptr<impl> p_impl;

    explicit HEYOKA_DLL_LOCAL dtens(impl);

    template <typename V>
    HEYOKA_DLL_LOCAL const expression &index_impl(const V &) const;

    // Serialisation.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    dtens();
    dtens(const dtens &);
    dtens(dtens &&) noexcept;
    dtens &operator=(const dtens &);
    dtens &operator=(dtens &&) noexcept;
    ~dtens();

    using iterator = detail::dtens_map_t::const_iterator;

    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;

    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] std::uint32_t get_nargs() const;
    [[nodiscard]] std::uint32_t get_nouts() const;
    [[nodiscard]] size_type size() const;
    [[nodiscard]] const std::vector<expression> &get_args() const;

    [[nodiscard]] iterator find(const v_idx_t &) const;
    [[nodiscard]] iterator find(const sv_idx_t &) const;
    [[nodiscard]] const expression &operator[](const v_idx_t &) const;
    [[nodiscard]] const expression &operator[](const sv_idx_t &) const;
    [[nodiscard]] size_type index_of(const v_idx_t &) const;
    [[nodiscard]] size_type index_of(const sv_idx_t &) const;
    [[nodiscard]] size_type index_of(const iterator &) const;

    [[nodiscard]] auto get_derivatives(std::uint32_t order) const -> decltype(std::ranges::subrange(begin(), end()));
    [[nodiscard]] auto get_derivatives(std::uint32_t component,
                                       std::uint32_t order) const -> decltype(std::ranges::subrange(begin(), end()));
    [[nodiscard]] std::vector<expression> get_gradient() const;
    [[nodiscard]] std::vector<expression> get_jacobian() const;
    [[nodiscard]] std::vector<expression> get_hessian(std::uint32_t) const;
};

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const dtens &);

HEYOKA_END_NAMESPACE

// Version changelog:
// - version 1: switched from dense to sparse
//   format for the indices vectors.
BOOST_CLASS_VERSION(heyoka::dtens::impl, 1)

// fmt formatter for dtens, implemented
// on top of the streaming operator.
namespace fmt
{

template <>
struct formatter<heyoka::dtens> : fmt::ostream_formatter {
};

} // namespace fmt

HEYOKA_BEGIN_NAMESPACE

template <typename... KwArgs>
dtens diff_tensors(const std::vector<expression> &v_ex, const std::variant<diff_args, std::vector<expression>> &d_args,
                   const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "diff_tensors() accepts only named arguments in the variadic pack.");

    // Order of derivatives. Defaults to 1.
    std::uint32_t order = 1;
    if constexpr (p.has(kw::diff_order)) {
        if constexpr (std::is_integral_v<detail::uncvref_t<decltype(p(kw::diff_order))>>) {
            order = boost::numeric_cast<std::uint32_t>(p(kw::diff_order));
        } else {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The diff_order keyword argument must be of integral type.");
        }
    }

    return detail::diff_tensors(v_ex, d_args, order);
}

template <typename... KwArgs>
dtens diff_tensors(const std::vector<expression> &v_ex, std::initializer_list<expression> d_args,
                   const KwArgs &...kw_args)
{
    return diff_tensors(v_ex, std::vector(d_args), kw_args...);
}

namespace detail
{

taylor_dc_t::size_type taylor_decompose(funcptr_map<taylor_dc_t::size_type> &, const expression &, taylor_dc_t &);

} // namespace detail

template <typename Arg0, typename... Args>
    requires std::convertible_to<const Arg0 &, std::string> && (std::convertible_to<const Args &, std::string> && ...)
auto make_vars(const Arg0 &str, const Args &...strs)
{
    if constexpr (sizeof...(Args) == 0u) {
        return expression{variable{str}};
    } else {
        return std::array{expression{variable{str}}, expression{variable{strs}}...};
    }
}

HEYOKA_DLL_PUBLIC llvm::Value *taylor_diff(llvm_state &, llvm::Type *, const expression &,
                                           const std::vector<std::uint32_t> &, const std::vector<llvm::Value *> &,
                                           llvm::Value *, llvm::Value *, std::uint32_t, std::uint32_t, std::uint32_t,
                                           std::uint32_t, bool);

HEYOKA_DLL_PUBLIC llvm::Function *taylor_c_diff_func(llvm_state &, llvm::Type *, const expression &, std::uint32_t,
                                                     std::uint32_t, bool);

HEYOKA_DLL_PUBLIC std::uint32_t get_param_size(const expression &);
HEYOKA_DLL_PUBLIC std::uint32_t get_param_size(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<expression> get_params(const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> get_params(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC bool is_time_dependent(const expression &);
HEYOKA_DLL_PUBLIC bool is_time_dependent(const std::vector<expression> &);

namespace detail
{

class HEYOKA_DLL_PUBLIC par_impl
{
public:
    expression operator[](std::uint32_t) const;
};

} // namespace detail

inline constexpr detail::par_impl par;

namespace detail
{

void verify_function_dec(const std::vector<expression> &, const std::vector<expression> &,
                         std::vector<expression>::size_type, bool = false);

std::vector<expression> function_decompose_cse(std::vector<expression> &, std::vector<expression>::size_type,
                                               std::vector<expression>::size_type);

std::vector<expression> function_sort_dc(std::vector<expression> &, std::vector<expression>::size_type,
                                         std::vector<expression>::size_type);

HEYOKA_DLL_PUBLIC std::vector<expression> split_sums_for_decompose(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<expression> split_prods_for_decompose(const std::vector<expression> &, std::uint32_t);

std::vector<expression> sums_to_sum_sqs_for_decompose(const std::vector<expression> &);

std::optional<std::vector<expression>::size_type> decompose(funcptr_map<std::vector<expression>::size_type> &,
                                                            const expression &, std::vector<expression> &);

llvm::Value *cfunc_c_load_eval(llvm_state &, llvm::Type *, llvm::Value *, llvm::Value *);

} // namespace detail

HEYOKA_DLL_PUBLIC std::vector<expression> function_decompose(const std::vector<expression> &,
                                                             const std::vector<expression> &);

namespace detail
{

template <typename>
std::vector<expression> add_cfunc(llvm_state &, const std::string &, const std::vector<expression> &,
                                  const std::vector<expression> &, std::uint32_t, bool, bool, bool, long long, bool);

// Prevent implicit instantiations.
#define HEYOKA_CFUNC_EXTERN_INST(T)                                                                                    \
    extern template std::vector<expression> add_cfunc<T>(                                                              \
        llvm_state &, const std::string &, const std::vector<expression> &, const std::vector<expression> &,           \
        std::uint32_t, bool, bool, bool, long long, bool);

HEYOKA_CFUNC_EXTERN_INST(float)
HEYOKA_CFUNC_EXTERN_INST(double)
HEYOKA_CFUNC_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CFUNC_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_CFUNC_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_CFUNC_EXTERN_INST

// Common options for the cfunc constructor and add_cfunc().
template <typename T, typename... KwArgs>
auto cfunc_common_opts(const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(),
                  "Unnamed arguments cannot be passed in the variadic pack for this function.");

    // High accuracy mode (defaults to false).
    const auto high_accuracy = [&p]() -> bool {
        if constexpr (p.has(kw::high_accuracy)) {
            if constexpr (std::convertible_to<decltype(p(kw::high_accuracy)), bool>) {
                return static_cast<bool>(p(kw::high_accuracy));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'high_accuracy' keyword argument.");
            }
        } else {
            return false;
        }
    }();

    // Compact mode (defaults to false, except for real where
    // it defaults to true).
    const auto compact_mode = [&p]() -> bool {
        if constexpr (p.has(kw::compact_mode)) {
            if constexpr (std::convertible_to<decltype(p(kw::compact_mode)), bool>) {
                return static_cast<bool>(p(kw::compact_mode));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'compact_mode' keyword argument.");
            }
        } else {
#if defined(HEYOKA_HAVE_REAL)
            return std::is_same_v<T, mppp::real>;
#else
            return false;

#endif
        }
    }();

    // Parallel mode (defaults to false).
    const auto parallel_mode = [&p]() -> bool {
        if constexpr (p.has(kw::parallel_mode)) {
            if constexpr (std::convertible_to<decltype(p(kw::parallel_mode)), bool>) {
                return static_cast<bool>(p(kw::parallel_mode));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'parallel_mode' keyword argument.");
            }
        } else {
            return false;
        }
    }();

    // Precision (defaults to zero).
    const auto prec = [&p]() -> long long {
        if constexpr (p.has(kw::prec)) {
            if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::prec))>>) {
                return boost::numeric_cast<long long>(p(kw::prec));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'prec' keyword argument.");
            }
        } else {
            return 0;
        }
    }();

    return std::make_tuple(high_accuracy, compact_mode, parallel_mode, prec);
}

} // namespace detail

template <typename T, typename... KwArgs>
std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &fn,
                                  const std::vector<expression> &vars, const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "The variadic arguments in add_cfunc() contain unnamed arguments.");

    // Batch size (defaults to 1).
    const auto batch_size = [&]() -> std::uint32_t {
        if constexpr (p.has(kw::batch_size)) {
            if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::batch_size))>>) {
                return boost::numeric_cast<std::uint32_t>(p(kw::batch_size));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'batch_size' keyword argument.");
            }
        } else {
            return 1;
        }
    }();

    // Strided mode (defaults to false).
    const auto strided = [&p]() -> bool {
        if constexpr (p.has(kw::strided)) {
            if constexpr (std::convertible_to<decltype(p(kw::strided)), bool>) {
                return static_cast<bool>(p(kw::strided));
            } else {
                static_assert(detail::always_false_v<T>, "Invalid type for the 'strided' keyword argument.");
            }
        } else {
            return false;
        }
    }();

    // Common options.
    const auto [high_accuracy, compact_mode, parallel_mode, prec] = detail::cfunc_common_opts<T>(kw_args...);

    return detail::add_cfunc<T>(s, name, fn, vars, batch_size, high_accuracy, compact_mode, parallel_mode, prec,
                                strided);
}

namespace detail
{

HEYOKA_DLL_PUBLIC bool ex_less_than(const expression &, const expression &);

// Concepts for the cfunc class.
template <typename T, typename R>
concept cfunc_out_range_1d = requires(R &r) {
    requires std::ranges::contiguous_range<R>;
    requires std::ranges::sized_range<R>;
    { std::ranges::data(r) } -> std::same_as<T *>;
    { std::ranges::size(r) } -> std::integral;
};

template <typename T, typename R>
concept cfunc_in_range_1d = requires(R &r) {
    requires std::ranges::contiguous_range<R>;
    requires std::ranges::sized_range<R>;
    { std::ranges::data(r) } -> std::convertible_to<const T *>;
    { std::ranges::size(r) } -> std::integral;
};

} // namespace detail

template <typename T>
class HEYOKA_DLL_PUBLIC_INLINE_CLASS cfunc
{
    struct HEYOKA_DLL_PUBLIC impl;

    std::unique_ptr<impl> m_impl;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    // Private implementation-detail helper and ctor.
    template <typename... KwArgs>
    static auto parse_ctor_opts(const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        // Common options.
        const auto [high_accuracy, compact_mode, parallel_mode, prec] = detail::cfunc_common_opts<T>(kw_args...);

        // Batch size: defaults to undefined.
        // NOTE: we want to handle this slightly different from add_cfunc(), thus it does
        // not go in common options.
        const auto batch_size = [&]() -> std::optional<std::uint32_t> {
            if constexpr (p.has(kw::batch_size)) {
                if constexpr (std::integral<std::remove_cvref_t<decltype(p(kw::batch_size))>>) {
                    return boost::numeric_cast<std::uint32_t>(p(kw::batch_size));
                } else {
                    static_assert(detail::always_false_v<T>, "Invalid type for the 'batch_size' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        // Precision checking for mppp::real. Defaults to true.
        const auto check_prec = [&p]() -> bool {
            if constexpr (p.has(kw::check_prec)) {
                if constexpr (std::convertible_to<decltype(p(kw::check_prec)), bool>) {
                    return static_cast<bool>(p(kw::check_prec));
                } else {
                    static_assert(detail::always_false_v<T>, "Invalid type for the 'check_prec' keyword argument.");
                }
            } else {
                return true;
            }
        }();

        // Build the template llvm_state from the keyword arguments.
        llvm_state s(kw_args...);

        return std::make_tuple(high_accuracy, compact_mode, parallel_mode, prec, batch_size, std::move(s), check_prec);
    }
    explicit cfunc(std::vector<expression>, std::vector<expression>,
                   std::tuple<bool, bool, bool, long long, std::optional<std::uint32_t>, llvm_state, bool>);

    HEYOKA_DLL_LOCAL void check_valid(const char *) const;

public:
    cfunc() noexcept;
    template <typename... KwArgs>
        requires(!igor::has_unnamed_arguments<KwArgs...>())
    explicit cfunc(std::vector<expression> fn, std::vector<expression> vars, const KwArgs &...kw_args)
        : cfunc(std::move(fn), std::move(vars), parse_ctor_opts(kw_args...))
    {
    }
    cfunc(const cfunc &);
    cfunc(cfunc &&) noexcept;
    cfunc &operator=(const cfunc &);
    cfunc &operator=(cfunc &&) noexcept;
    ~cfunc();

    // Properties getters.
    [[nodiscard]] bool is_valid() const noexcept;
    [[nodiscard]] const std::vector<expression> &get_fn() const;
    [[nodiscard]] const std::vector<expression> &get_vars() const;
    [[nodiscard]] const std::vector<expression> &get_dc() const;
    [[nodiscard]] const llvm_state &get_llvm_state_scalar() const;
    [[nodiscard]] const llvm_state &get_llvm_state_scalar_s() const;
    [[nodiscard]] const llvm_state &get_llvm_state_batch_s() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] bool get_compact_mode() const;
    [[nodiscard]] bool get_parallel_mode() const;
    [[nodiscard]] std::uint32_t get_batch_size() const;
    [[nodiscard]] std::uint32_t get_nparams() const;
    [[nodiscard]] std::uint32_t get_nvars() const;
    [[nodiscard]] std::uint32_t get_nouts() const;
    [[nodiscard]] bool is_time_dependent() const;

#if defined(HEYOKA_HAVE_REAL)

    [[nodiscard]] mpfr_prec_t get_prec() const
        requires std::same_as<T, mppp::real>;

#endif

    using in_1d = mdspan<const T, dextents<std::size_t, 1>>;
    using out_1d = mdspan<T, dextents<std::size_t, 1>>;

private:
    void single_eval(out_1d, in_1d, std::optional<in_1d>, std::optional<T>);

public:
    // NOTE: it is important to document properly the non-overlapping
    // memory requirement for the input arguments.
    template <typename Out, typename In, typename... KwArgs>
        requires(!igor::has_unnamed_arguments<KwArgs...>())
                && (detail::cfunc_out_range_1d<T, Out> || std::same_as<out_1d, std::remove_cvref_t<Out>>)
                && (detail::cfunc_in_range_1d<T, In> || std::same_as<in_1d, std::remove_cvref_t<In>>)
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    void operator()(Out &&out, In &&in, const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        out_1d oput = [&]() {
            if constexpr (std::same_as<out_1d, std::remove_cvref_t<Out>>) {
                return out;
            } else {
                return out_1d{std::ranges::data(out), boost::numeric_cast<std::size_t>(std::ranges::size(out))};
            }
        }();

        in_1d iput = [&]() {
            if constexpr (std::same_as<in_1d, std::remove_cvref_t<In>>) {
                return in;
            } else {
                return in_1d{std::ranges::data(in), boost::numeric_cast<std::size_t>(std::ranges::size(in))};
            }
        }();

        auto pars = [&]() -> std::optional<in_1d> {
            if constexpr (p.has(kw::pars)) {
                using pars_t = decltype(p(kw::pars));

                if constexpr (std::same_as<in_1d, std::remove_cvref_t<pars_t>>) {
                    return p(kw::pars);
                } else if constexpr (detail::cfunc_in_range_1d<T, pars_t>) {
                    // NOTE: as usual, we don't want to perfectly forward ranges, hence,
                    // turn it into an lvalue.
                    auto &&pars = p(kw::pars);

                    return in_1d{std::ranges::data(pars), boost::numeric_cast<std::size_t>(std::ranges::size(pars))};
                } else {
                    static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the 'pars' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        auto tm = [&]() -> std::optional<T> {
            if constexpr (p.has(kw::time)) {
                if constexpr (std::convertible_to<decltype(p(kw::time)), T>) {
                    return static_cast<T>(p(kw::time));
                } else {
                    static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the 'time' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        return single_eval(std::move(oput), std::move(iput), std::move(pars), std::move(tm));
    }

    using in_2d = mdspan<const T, dextents<std::size_t, 2>>;
    using out_2d = mdspan<T, dextents<std::size_t, 2>>;

private:
    HEYOKA_DLL_LOCAL void multi_eval_st(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>);
    HEYOKA_DLL_LOCAL void multi_eval_mt(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>);
    void multi_eval(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>);

public:
    // NOTE: it is important to document properly the non-overlapping
    // memory requirement for the input arguments.
    template <typename... KwArgs>
        requires(!igor::has_unnamed_arguments<KwArgs...>())
    void operator()(out_2d out, in_2d in, const KwArgs &...kw_args)
    {
        igor::parser p{kw_args...};

        auto pars = [&]() -> std::optional<in_2d> {
            if constexpr (p.has(kw::pars)) {
                if constexpr (std::same_as<in_2d, std::remove_cvref_t<decltype(p(kw::pars))>>) {
                    return p(kw::pars);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the 'pars' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        auto tm = [&]() -> std::optional<in_1d> {
            if constexpr (p.has(kw::time)) {
                if constexpr (std::same_as<in_1d, std::remove_cvref_t<decltype(p(kw::time))>>) {
                    return p(kw::time);
                } else {
                    static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the 'time' keyword argument.");
                }
            } else {
                return {};
            }
        }();

        multi_eval(std::move(out), std::move(in), std::move(pars), std::move(tm));
    }
};

template <typename T>
std::ostream &operator<<(std::ostream &, const cfunc<T> &);

// Prevent implicit instantiations.
#define HEYOKA_CFUNC_CLASS_EXTERN_INST(T)                                                                              \
    extern template class cfunc<T>;                                                                                    \
    extern template std::ostream &operator<<(std::ostream &, const cfunc<T> &);

HEYOKA_CFUNC_CLASS_EXTERN_INST(float)
HEYOKA_CFUNC_CLASS_EXTERN_INST(double)
HEYOKA_CFUNC_CLASS_EXTERN_INST(long double)

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_CFUNC_CLASS_EXTERN_INST(mppp::real128)

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_CFUNC_CLASS_EXTERN_INST(mppp::real)

#endif

#undef HEYOKA_CFUNC_CLASS_EXTERN_INST

HEYOKA_END_NAMESPACE

// fmt formatter for cfunc, implemented
// on top of the streaming operator.
namespace fmt
{

template <typename T>
struct formatter<heyoka::cfunc<T>> : fmt::ostream_formatter {
};

} // namespace fmt

namespace std
{

// Specialisation of std::hash for expression.
template <>
struct hash<heyoka::expression> {
    size_t operator()(const heyoka::expression &ex) const noexcept
    {
        return heyoka::detail::hash(ex);
    }
};

// Specialisation of std::less for expression.
template <>
struct less<heyoka::expression> {
    bool operator()(const heyoka::expression &ex1, const heyoka::expression &ex2) const
    {
        return heyoka::detail::ex_less_than(ex1, ex2);
    }
};

} // namespace std

#endif
