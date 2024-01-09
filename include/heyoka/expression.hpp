// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
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
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/func.hpp>
#include <heyoka/kw.hpp>
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
    expression();

    explicit expression(float);
    explicit expression(double);
    explicit expression(long double);
#if defined(HEYOKA_HAVE_REAL128)
    explicit expression(mppp::real128);
#endif
#if defined(HEYOKA_HAVE_REAL)
    explicit expression(mppp::real);
#endif
    explicit expression(std::string);

    explicit expression(number);
    explicit expression(variable);
    explicit expression(func);
    explicit expression(param);

    expression(const expression &);
    expression(expression &&) noexcept;

    ~expression();

    expression &operator=(const expression &);
    expression &operator=(expression &&) noexcept;

    [[nodiscard]] const value_type &value() const;
};

HEYOKA_DLL_PUBLIC expression copy(const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> copy(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC expression fix(expression);
HEYOKA_DLL_PUBLIC expression fix_nn(expression);
HEYOKA_DLL_PUBLIC expression unfix(const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> unfix(const std::vector<expression> &);

namespace detail
{

HEYOKA_DLL_PUBLIC bool is_fixed(const expression &);

} // namespace detail

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
inline expression operator""_f128()
{
    return expression{mppp::literals::operator""_rq < Chars... > ()};
}

#endif

HEYOKA_DLL_PUBLIC expression operator""_var(const char *, std::size_t);

} // namespace literals

namespace detail
{

// NOTE: these need to go here because
// the definition of expression must be available.
template <typename Base, typename Holder, typename T>
    requires is_udf<T>
inline expression func_iface_impl<Base, Holder, T>::diff(funcptr_map<expression> &func_map, const std::string &s) const
{
    if constexpr (func_has_diff_var<T>) {
        return this->value().diff(func_map, s);
    }

    // LCOV_EXCL_START
    assert(false);
    throw;
    // LCOV_EXCL_STOP
}

template <typename Base, typename Holder, typename T>
    requires is_udf<T>
inline expression func_iface_impl<Base, Holder, T>::diff(funcptr_map<expression> &func_map, const param &p) const
{
    if constexpr (func_has_diff_par<T>) {
        return this->value().diff(func_map, p);
    }

    // LCOV_EXCL_START
    assert(false);
    throw;
    // LCOV_EXCL_STOP
}

template <typename Base, typename Holder, typename T>
    requires is_udf<T>
inline expression func_iface_impl<Base, Holder, T>::normalise() const
{
    if constexpr (func_has_normalise<T>) {
        return this->value().normalise();
    }

    // LCOV_EXCL_START
    assert(false);
    throw;
    // LCOV_EXCL_STOP
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

} // namespace detail

namespace detail
{

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

HEYOKA_DLL_PUBLIC bool operator==(const expression &, const expression &);
HEYOKA_DLL_PUBLIC bool operator!=(const expression &, const expression &);

HEYOKA_DLL_PUBLIC std::size_t get_n_nodes(const expression &);

HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<std::string, expression> &,
                                  bool = false);
HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<expression, expression> &, bool = false);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::unordered_map<std::string, expression> &, bool = false);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::unordered_map<expression, expression> &, bool = false);

HEYOKA_DLL_PUBLIC expression normalise(const expression &);
HEYOKA_DLL_PUBLIC std::vector<expression> normalise(const std::vector<expression> &);

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
HEYOKA_DLL_PUBLIC expression diff(const expression &, const std::string &);
HEYOKA_DLL_PUBLIC expression diff(const expression &, const expression &);

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

    explicit dtens(impl);

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

    // Subrange helper class to fetch
    // ranges into dtens.
    class HEYOKA_DLL_PUBLIC subrange
    {
        friend class dtens;

        iterator m_begin, m_end;

        explicit subrange(const iterator &, const iterator &);

    public:
        subrange() = delete;
        subrange(const subrange &);
        subrange(subrange &&) noexcept;
        subrange &operator=(const subrange &);
        subrange &operator=(subrange &&) noexcept;
        ~subrange();

        [[nodiscard]] iterator begin() const;
        [[nodiscard]] iterator end() const;
    };

    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;

    [[nodiscard]] std::uint32_t get_order() const;
    [[nodiscard]] std::uint32_t get_nvars() const;
    [[nodiscard]] std::uint32_t get_nouts() const;
    [[nodiscard]] size_type size() const;
    [[nodiscard]] const std::vector<expression> &get_args() const;

    [[nodiscard]] iterator find(const v_idx_t &) const;
    [[nodiscard]] const expression &operator[](const v_idx_t &) const;
    [[nodiscard]] size_type index_of(const v_idx_t &) const;
    [[nodiscard]] size_type index_of(const iterator &) const;

    [[nodiscard]] subrange get_derivatives(std::uint32_t, std::uint32_t) const;
    [[nodiscard]] subrange get_derivatives(std::uint32_t) const;
    [[nodiscard]] std::vector<expression> get_gradient() const;
    [[nodiscard]] std::vector<expression> get_jacobian() const;
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

// NOTE: when documenting, we need to point out that the expressions
// returned by this function are optimised for evaluation. The users
// can always unfix() and normalise() these expressions if needed.
template <typename... KwArgs>
dtens diff_tensors(const std::vector<expression> &v_ex, const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    static_assert(!p.has_unnamed_arguments(), "diff_tensors() accepts only named arguments in the variadic pack.");

    // Variables and/or params wrt which the derivatives will be computed.
    // Defaults to all variables.
    std::variant<diff_args, std::vector<expression>> d_args = diff_args::vars;
    if constexpr (p.has(kw::diff_args)) {
        if constexpr (std::is_same_v<detail::uncvref_t<decltype(p(kw::diff_args))>, diff_args>) {
            d_args = p(kw::diff_args);
        } else if constexpr (std::is_constructible_v<std::vector<expression>, decltype(p(kw::diff_args))>) {
            d_args = std::vector<expression>(p(kw::diff_args));
        } else {
            static_assert(detail::always_false_v<KwArgs...>, "Invalid type for the diff_args keyword argument.");
        }
    }

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

namespace detail
{

taylor_dc_t::size_type taylor_decompose(funcptr_map<taylor_dc_t::size_type> &, const expression &, taylor_dc_t &);

} // namespace detail

template <typename... Args>
inline std::array<expression, sizeof...(Args)> make_vars(const Args &...strs)
{
    return std::array{expression{variable{strs}}...};
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

HEYOKA_DLL_PUBLIC std::pair<std::vector<expression>, std::vector<expression>::size_type>
function_decompose(const std::vector<expression> &);
HEYOKA_DLL_PUBLIC std::vector<expression> function_decompose(const std::vector<expression> &,
                                                             const std::vector<expression> &);

namespace detail
{

template <typename>
std::vector<expression> add_cfunc(llvm_state &, const std::string &, const std::vector<expression> &, std::uint32_t,
                                  bool, bool, bool, long long);

template <typename>
std::vector<expression> add_cfunc(llvm_state &, const std::string &, const std::vector<expression> &,
                                  const std::vector<expression> &, std::uint32_t, bool, bool, bool, long long);

// Prevent implicit instantiations.
#define HEYOKA_CFUNC_EXTERN_INST(T)                                                                                    \
    extern template std::vector<expression> add_cfunc<T>(llvm_state &, const std::string &,                            \
                                                         const std::vector<expression> &, std::uint32_t, bool, bool,   \
                                                         bool, long long);                                             \
    extern template std::vector<expression> add_cfunc<T>(                                                              \
        llvm_state &, const std::string &, const std::vector<expression> &, const std::vector<expression> &,           \
        std::uint32_t, bool, bool, bool, long long);

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

} // namespace detail

template <typename T, typename... KwArgs>
inline std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &fn,
                                         const KwArgs &...kw_args)
{
    igor::parser p{kw_args...};

    if constexpr (p.has_unnamed_arguments()) {
        static_assert(detail::always_false_v<KwArgs...>,
                      "The variadic arguments in add_cfunc() contain unnamed arguments.");
    } else {
        // Check if the list of variables was
        // provided explicitly.
        std::optional<std::vector<expression>> vars;
        if constexpr (p.has(kw::vars)) {
            vars = p(kw::vars);
        }

        // Batch size (defaults to 1).
        const auto batch_size = [&]() -> std::uint32_t {
            if constexpr (p.has(kw::batch_size)) {
                return p(kw::batch_size);
            } else {
                return 1;
            }
        }();

        // High accuracy mode (defaults to false).
        const auto high_accuracy = [&p]() -> bool {
            if constexpr (p.has(kw::high_accuracy)) {
                return p(kw::high_accuracy);
            } else {
                return false;
            }
        }();

        // Compact mode (defaults to false, except for real where
        // it defaults to true).
        const auto compact_mode = [&p]() -> bool {
            if constexpr (p.has(kw::compact_mode)) {
                return p(kw::compact_mode);
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
                return p(kw::parallel_mode);
            } else {
                return false;
            }
        }();

        // Precision (defaults to zero).
        const auto prec = [&p]() -> long long {
            if constexpr (p.has(kw::prec)) {
                return p(kw::prec);
            } else {
                return 0;
            }
        }();

        if (vars) {
            return detail::add_cfunc<T>(s, name, fn, *vars, batch_size, high_accuracy, compact_mode, parallel_mode,
                                        prec);
        } else {
            return detail::add_cfunc<T>(s, name, fn, batch_size, high_accuracy, compact_mode, parallel_mode, prec);
        }
    }
}

namespace detail
{

HEYOKA_DLL_PUBLIC bool ex_less_than(const expression &, const expression &);

} // namespace detail

HEYOKA_END_NAMESPACE

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
