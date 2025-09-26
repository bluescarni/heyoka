// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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

#include <heyoka/detail/ex_traversal.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/igor.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/ranges_to.hpp>
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
    return expression{mppp::literals::operator""_rq < Chars...>()};
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
    static_cast<func_base &>(getval<Holder>(this)).replace_args(std::move(new_args));
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

namespace detail
{

expression rename_variables_impl(void_ptr_map<const expression> &, sargs_ptr_map<const func_args::shared_args_t> &,
                                 const expression &, const std::unordered_map<std::string, std::string> &);

} // namespace detail

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

namespace detail
{

expression subs_impl(detail::void_ptr_map<const expression> &, detail::sargs_ptr_map<const func_args::shared_args_t> &,
                     const expression &, const std::unordered_map<std::string, expression> &);

expression subs_impl(void_ptr_map<const expression> &, sargs_ptr_map<const func_args::shared_args_t> &,
                     const expression &, const std::map<expression, expression> &);

} // namespace detail

HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::unordered_map<std::string, expression> &);
HEYOKA_DLL_PUBLIC expression subs(const expression &, const std::map<expression, expression> &);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::unordered_map<std::string, expression> &);
HEYOKA_DLL_PUBLIC std::vector<expression> subs(const std::vector<expression> &,
                                               const std::map<expression, expression> &);

// NOLINTNEXTLINE(performance-enum-size)
enum class diff_args { vars, params, all };

// Fwd declaration.
class HEYOKA_DLL_PUBLIC dtens;

namespace detail
{

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

    [[nodiscard]] auto get_derivatives(std::uint32_t) const -> decltype(std::ranges::subrange(begin(), end()));
    [[nodiscard]] auto get_derivatives(std::uint32_t, std::uint32_t) const
        -> decltype(std::ranges::subrange(begin(), end()));
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
    requires igor::validate<igor::config<kw::descr::integral<kw::diff_order>>{}, KwArgs...>
dtens diff_tensors(const std::vector<expression> &v_ex, const std::variant<diff_args, std::vector<expression>> &d_args,
                   const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Order of derivatives. Defaults to 1.
    const auto order = boost::numeric_cast<std::uint32_t>(p(kw::diff_order, 1));

    return detail::diff_tensors(v_ex, d_args, order);
}

template <typename... KwArgs>
    requires igor::validate<igor::config<kw::descr::integral<kw::diff_order>>{}, KwArgs...>
dtens diff_tensors(const std::vector<expression> &v_ex, std::initializer_list<expression> d_args,
                   const KwArgs &...kw_args)
{
    return diff_tensors(v_ex, std::vector(d_args), kw_args...);
}

namespace detail
{

HEYOKA_DLL_PUBLIC std::optional<taylor_dc_t::size_type>
taylor_decompose(void_ptr_map<const taylor_dc_t::size_type> &, sargs_ptr_map<const func_args::shared_args_t> &,
                 const expression &, taylor_dc_t &);

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

std::vector<expression> function_decompose_cse(const std::vector<expression> &, std::vector<expression>::size_type,
                                               std::vector<expression>::size_type);

std::vector<expression> function_sort_dc(const std::vector<expression> &, std::vector<expression>::size_type,
                                         std::vector<expression>::size_type);

HEYOKA_DLL_PUBLIC std::vector<expression> split_sums_for_decompose(const std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<expression> split_prods_for_decompose(const std::vector<expression> &, std::uint32_t);

std::vector<expression> sums_to_sum_sqs_for_decompose(const std::vector<expression> &);

std::optional<std::vector<expression>::size_type> decompose(void_ptr_map<const std::vector<expression>::size_type> &,
                                                            sargs_ptr_map<const func_args::shared_args_t> &,
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

// kwargs configuration for the common options of cfunc.
//
// NOTE: here we are making sure that we accept only builtin types for the kw_args. This makes the set of kw_args
// re-usable across several invocations.
inline constexpr auto cfunc_common_opts_kw_cfg
    = igor::config<kw::descr::boolean<kw::high_accuracy>, kw::descr::boolean<kw::compact_mode>,
                   kw::descr::boolean<kw::parallel_mode>, kw::descr::integral<kw::prec>>{};

// Common options for add_cfunc() and the cfunc constructor.
template <typename T, typename... KwArgs>
auto cfunc_common_opts(const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // High accuracy mode (defaults to false).
    const auto high_accuracy = p(kw::high_accuracy, false);

    // Compact mode (defaults to false, except for real where it defaults to true).
    const auto compact_mode = p(kw::compact_mode,
#if defined(HEYOKA_HAVE_REAL)
                                std::same_as<T, mppp::real>
#else
                                false
#endif
    );

    // Parallel mode (defaults to false).
    const auto parallel_mode = p(kw::parallel_mode, false);

    // Precision (defaults to zero).
    const auto prec = boost::numeric_cast<long long>(p(kw::prec, 0));

    return std::make_tuple(high_accuracy, compact_mode, parallel_mode, prec);
}

template <typename>
std::tuple<llvm_multi_state, std::vector<expression>, std::vector<std::array<std::size_t, 2>>>
make_multi_cfunc(llvm_state, const std::string &, const std::vector<expression> &, const std::vector<expression> &,
                 std::uint32_t, bool, bool, long long, bool);

// kwargs configuration for add_cfunc().
inline constexpr auto add_cfunc_kw_cfg
    = cfunc_common_opts_kw_cfg | igor::config<kw::descr::integral<kw::batch_size>, kw::descr::boolean<kw::strided>>{};

} // namespace detail

template <typename T, typename... KwArgs>
    requires igor::validate<detail::add_cfunc_kw_cfg, KwArgs...>
std::vector<expression> add_cfunc(llvm_state &s, const std::string &name, const std::vector<expression> &fn,
                                  const std::vector<expression> &vars, const KwArgs &...kw_args)
{
    const igor::parser p{kw_args...};

    // Batch size (defaults to 1).
    const auto batch_size = boost::numeric_cast<std::uint32_t>(p(kw::batch_size, 1));

    // Strided mode (defaults to false).
    const auto strided = p(kw::strided, false);

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
    struct impl;

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
        const igor::parser p{kw_args...};

        // Common options.
        const auto [high_accuracy, compact_mode, parallel_mode, prec] = detail::cfunc_common_opts<T>(kw_args...);

        // Batch size: defaults to undefined.
        //
        // NOTE: we want to handle this slightly different from add_cfunc(), thus it does not go in common options.
        const auto batch_size = [&]() -> std::optional<std::uint32_t> {
            if constexpr (p.has(kw::batch_size)) {
                return boost::numeric_cast<std::uint32_t>(p(kw::batch_size));
            } else {
                return {};
            }
        }();

        // Precision checking for mppp::real. Defaults to true.
        const auto check_prec = p(kw::check_prec, true);

        // Parallel JIT compilation.
        const auto parjit = p(kw::parjit, detail::default_parjit);

        // Build the template llvm_state from the keyword arguments.
        auto s = igor::filter_invoke<llvm_state::kw_cfg>([](const auto &...args) { return llvm_state(args...); },
                                                         kw_args...);

        return std::make_tuple(high_accuracy, compact_mode, parallel_mode, prec, batch_size, std::move(s), check_prec,
                               parjit);
    }
    explicit cfunc(std::vector<expression>, std::vector<expression>,
                   std::tuple<bool, bool, bool, long long, std::optional<std::uint32_t>, llvm_state, bool, bool>);

    HEYOKA_DLL_LOCAL void check_valid(const char *) const;

public:
    cfunc() noexcept;
    // kwargs configuration for the constructor.
    //
    // NOTE: the constraints on the keyword arguments ensure we can use them in multiple invocations.
    static constexpr auto ctor_kw_cfg
        = detail::cfunc_common_opts_kw_cfg | llvm_state::kw_cfg
          | igor::config<kw::descr::integral<kw::batch_size>, kw::descr::boolean<kw::check_prec>,
                         kw::descr::boolean<kw::parjit>>{};
    template <typename... KwArgs>
        requires igor::validate<ctor_kw_cfg, KwArgs...>
    explicit cfunc(std::vector<expression> fn, std::vector<expression> vars, const KwArgs &...kw_args)
        : cfunc(std::move(fn), std::move(vars), parse_ctor_opts(kw_args...))
    {
    }
    template <typename R1, typename R2, typename... KwArgs>
        requires igor::validate<ctor_kw_cfg, KwArgs...> && std::ranges::input_range<R1>
                 && std::constructible_from<expression, std::ranges::range_reference_t<R1>>
                 && std::ranges::input_range<R2>
                 && std::constructible_from<expression, std::ranges::range_reference_t<R2>>
    explicit cfunc(R1 &&rng1, R2 &&rng2, const KwArgs &...kw_args)
        : cfunc(detail::ranges_to<std::vector<expression>>(std::forward<R1>(rng1)),
                detail::ranges_to<std::vector<expression>>(std::forward<R2>(rng2)), kw_args...)
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
    [[nodiscard]] const std::variant<std::array<llvm_state, 3>, llvm_multi_state> &get_llvm_states() const;
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
    void single_eval(out_1d, in_1d, std::optional<in_1d>, std::optional<T>) const;

    // kwargs configuration for the call operator, single evaluation overload.
    static constexpr auto single_eval_kw_cfg
        = igor::config<igor::descr<kw::pars,
                                   []<typename U>() {
                                       return std::same_as<in_1d, std::remove_cvref_t<U>>
                                              || detail::cfunc_in_range_1d<T, U>;
                                   }>{},
                       kw::descr::convertible_to<kw::time, T>>{};

public:
    // NOTE: it is important to document properly the non-overlapping memory requirement for the input arguments.
    template <typename Out, typename In, typename... KwArgs>
        requires igor::validate<single_eval_kw_cfg, KwArgs...>
                 && (detail::cfunc_out_range_1d<T, Out> || std::same_as<out_1d, std::remove_cvref_t<Out>>)
                 && (detail::cfunc_in_range_1d<T, In> || std::same_as<in_1d, std::remove_cvref_t<In>>)
    // NOTE: accept forwarding references here to highlight that kw_args may in general be moved and that thus it is not
    // safe to re-use them.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    void operator()(Out &&out, In &&in, KwArgs &&...kw_args) const
    {
        const igor::parser p{kw_args...};

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
                } else {
                    // NOTE: as usual, we don't want to perfectly forward ranges, hence, turn it into an lvalue.
                    auto &&pars = p(kw::pars);

                    return in_1d{std::ranges::data(pars), boost::numeric_cast<std::size_t>(std::ranges::size(pars))};
                }
            } else {
                return {};
            }
        }();

        auto tm = [&]() -> std::optional<T> {
            if constexpr (p.has(kw::time)) {
                return static_cast<T>(p(kw::time));
            } else {
                return {};
            }
        }();

        return single_eval(std::move(oput), std::move(iput), std::move(pars), std::move(tm));
    }

    using in_2d = mdspan<const T, dextents<std::size_t, 2>>;
    using out_2d = mdspan<T, dextents<std::size_t, 2>>;

private:
    HEYOKA_DLL_LOCAL void multi_eval_st(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>) const;
    HEYOKA_DLL_LOCAL void multi_eval_mt(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>) const;
    void multi_eval(out_2d, in_2d, std::optional<in_2d>, std::optional<in_1d>) const;

    // kwargs configuration for the call operator, multi evaluation overload.
    static constexpr auto multi_eval_kw_cfg
        = igor::config<kw::descr::same_as<kw::pars, in_2d>, kw::descr::same_as<kw::time, in_1d>>{};

public:
    // NOTE: it is important to document properly the non-overlapping memory requirement for the input arguments.
    template <typename... KwArgs>
        requires igor::validate<multi_eval_kw_cfg, KwArgs...>
    // NOTE: accept forwarding references here to highlight that kw_args may in general be moved and that thus it is not
    // safe to re-use them.
    //
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    void operator()(out_2d out, in_2d in, KwArgs &&...kw_args) const
    {
        const igor::parser p{kw_args...};

        auto pars = [&]() -> std::optional<in_2d> {
            if constexpr (p.has(kw::pars)) {
                return p(kw::pars);
            } else {
                return {};
            }
        }();

        auto tm = [&]() -> std::optional<in_1d> {
            if constexpr (p.has(kw::time)) {
                return p(kw::time);
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

namespace detail
{

// Boost s11n class version history for the cfunc class:
//
// - 1: implemented parallel compilation for compact mode, introduced external storage for the evaluation tape.
// - 2: removed the internal tapes, rely on a global thread-safe lock-free cache instead.
inline constexpr int cfunc_s11n_version = 2;

} // namespace detail

HEYOKA_END_NAMESPACE

// Set the Boost s11n class version for the cfunc class.
BOOST_CLASS_VERSION(heyoka::cfunc<float>, heyoka::detail::cfunc_s11n_version);
BOOST_CLASS_VERSION(heyoka::cfunc<double>, heyoka::detail::cfunc_s11n_version);
BOOST_CLASS_VERSION(heyoka::cfunc<long double>, heyoka::detail::cfunc_s11n_version);

#if defined(HEYOKA_HAVE_REAL128)

BOOST_CLASS_VERSION(heyoka::cfunc<mppp::real128>, heyoka::detail::cfunc_s11n_version);

#endif

#if defined(HEYOKA_HAVE_REAL)

BOOST_CLASS_VERSION(heyoka::cfunc<mppp::real>, heyoka::detail::cfunc_s11n_version);

#endif

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
