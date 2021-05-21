// Copyright 2020, 2021 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/tpoly.hpp>
#include <heyoka/number.hpp>
#include <heyoka/param.hpp>
#include <heyoka/taylor.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace heyoka
{

namespace detail
{

tpoly_impl::tpoly_impl() : tpoly_impl(par[0], par[1]) {}

tpoly_impl::tpoly_impl(expression b, expression e)
    : func_base("tpoly", std::vector<expression>{std::move(b), std::move(e)})
{
    if (!std::holds_alternative<param>(args()[0].value())) {
        throw std::invalid_argument("Cannot construct a time polynomial from a non-param argument");
    }
    m_b_idx = std::get<param>(args()[0].value()).idx();

    if (!std::holds_alternative<param>(args()[1].value())) {
        throw std::invalid_argument("Cannot construct a time polynomial from a non-param argument");
    }
    m_e_idx = std::get<param>(args()[1].value()).idx();

    if (m_e_idx <= m_b_idx) {
        throw std::invalid_argument(
            "Cannot construct a time polynomial from param indices {} and {}: the first index is not less than the second"_format(
                m_b_idx, m_e_idx));
    }
}

void tpoly_impl::to_stream(std::ostream &os) const
{
    os << "t_poly({}, {})"_format(m_b_idx, m_e_idx);
}

namespace
{

template <typename T>
llvm::Value *taylor_diff_tpoly_impl(llvm_state &s, const tpoly_impl &tp, llvm::Value *par_ptr, llvm::Value *time_ptr,
                                    std::uint32_t order, std::uint32_t batch_size)
{
    auto &builder = s.builder();

    assert(tp.m_e_idx > tp.m_b_idx);
    const auto n = (tp.m_e_idx - tp.m_b_idx) - 1u;

    // Null retval if the diff order is larger than the
    // polynomial order.
    if (order > n) {
        return vector_splat(builder, codegen<T>(s, number{0.}), batch_size);
    }

    // Load the time value.
    auto tm = load_vector_from_memory(builder, time_ptr, batch_size);

    // Init the return value with the highest-order coefficient.
    assert(tp.m_e_idx > 0u);
    auto ret = taylor_codegen_numparam<T>(s, param{tp.m_e_idx - 1u}, par_ptr, batch_size);
    auto bc
        = boost::math::binomial_coefficient<T>(boost::numeric_cast<unsigned>(n), boost::numeric_cast<unsigned>(order));
    ret = builder.CreateFMul(ret, vector_splat(builder, codegen<T>(s, number{bc}), batch_size));

    // Horner evaluation of polynomial derivative.
    for (std::uint32_t i_ = 1; i_ <= n - order; ++i_) {
        // NOTE: need to invert i because Horner's method
        // proceeds backwards.
        const auto i = n - order - i_;

        // Compute the binomial coefficient.
        bc = boost::math::binomial_coefficient<T>(boost::numeric_cast<unsigned>(i + order),
                                                  boost::numeric_cast<unsigned>(order));

        // Load the poly coefficient from the par array and multiply it by bc.
        auto cf = taylor_codegen_numparam<T>(s, param{tp.m_b_idx + i + order}, par_ptr, batch_size);
        cf = builder.CreateFMul(cf, vector_splat(builder, codegen<T>(s, number{bc}), batch_size));

        // Horner iteration.
        ret = builder.CreateFAdd(cf, builder.CreateFMul(ret, tm));
    }

    return ret;
}

} // namespace

llvm::Value *tpoly_impl::taylor_diff_dbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                         const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                         llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                         std::uint32_t batch_size) const
{
    return taylor_diff_tpoly_impl<double>(s, *this, par_ptr, time_ptr, order, batch_size);
}

llvm::Value *tpoly_impl::taylor_diff_ldbl(llvm_state &s, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                          llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size) const
{
    return taylor_diff_tpoly_impl<long double>(s, *this, par_ptr, time_ptr, order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm::Value *tpoly_impl::taylor_diff_f128(llvm_state &s, const std::vector<std::uint32_t> &,
                                          const std::vector<llvm::Value *> &, llvm::Value *par_ptr,
                                          llvm::Value *time_ptr, std::uint32_t, std::uint32_t order, std::uint32_t,
                                          std::uint32_t batch_size) const
{
    return taylor_diff_tpoly_impl<mppp::real128>(s, *this, par_ptr, time_ptr, order, batch_size);
}

#endif

} // namespace detail

expression tpoly(expression b, expression e)
{
    return expression{func{detail::tpoly_impl{std::move(b), std::move(e)}}};
}

} // namespace heyoka
