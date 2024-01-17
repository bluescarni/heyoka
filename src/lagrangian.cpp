// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/lagrangian.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

void lagrangian_impl_sanity_checks(const expression &L, const std::vector<expression> &qs,
                                   const std::vector<expression> &qdots, const expression &D)
{
    // Sanity checks on qs and qdots.
    if (qs.size() != qdots.size()) {
        throw std::invalid_argument(fmt::format(
            "The number of generalised coordinates ({}) must be equal to the number of generalised velocities ({})",
            qs.size(), qdots.size()));
    }

    if (qs.empty()) {
        throw std::invalid_argument("Cannot define a Lagrangian without state variables");
    }

    for (const auto &q : qs) {
        if (!std::holds_alternative<variable>(q.value())) {
            throw std::invalid_argument(fmt::format(
                "The list of generalised coordinates contains the expression '{}' which is not a variable", q));
        }

        if (std::get<variable>(q.value()).name().starts_with("__")) {
            throw std::invalid_argument(
                fmt::format("The list of generalised coordinates contains a variable with the invalid name '{}': names "
                            "starting with '__' are reserved for internal use",
                            std::get<variable>(q.value()).name()));
        }
    }

    for (const auto &qdot : qdots) {
        if (!std::holds_alternative<variable>(qdot.value())) {
            throw std::invalid_argument(fmt::format(
                "The list of generalised velocities contains the expression '{}' which is not a variable", qdot));
        }

        if (std::get<variable>(qdot.value()).name().starts_with("__")) {
            throw std::invalid_argument(
                fmt::format("The list of generalised velocities contains a variable with the invalid name '{}': names "
                            "starting with '__' are reserved for internal use",
                            std::get<variable>(qdot.value()).name()));
        }
    }

    // Check for duplicates.
    const std::unordered_set<expression> qs_set{qs.begin(), qs.end()};
    const std::unordered_set<expression> qdots_set{qdots.begin(), qdots.end()};

    if (qs_set.size() != qs.size()) {
        throw std::invalid_argument("The list of generalised coordinates contains duplicates");
    }

    if (qdots_set.size() != qdots.size()) {
        throw std::invalid_argument("The list of generalised velocities contains duplicates");
    }

    for (const auto &q : qs) {
        if (qdots_set.contains(q)) {
            throw std::invalid_argument(fmt::format("The list of generalised coordinates contains the expression '{}' "
                                                    "which also appears as a generalised velocity",
                                                    q));
        }
    }

    // Sanity checks on L.
    for (const auto &v : get_variables(L)) {
        if (!qs_set.contains(expression{v}) && !qdots_set.contains(expression{v})) {
            throw std::invalid_argument(fmt::format(
                "The Lagrangian contains the variable '{}' which is not a generalised position or velocity", v));
        }
    }

    // Sanity checks on D.
    for (const auto &v : get_variables(D)) {
        if (!qdots_set.contains(expression{v})) {
            throw std::invalid_argument(fmt::format(
                "The dissipation function contains the variable '{}' which is not a generalised velocity", v));
        }
    }
}

// Matrix representing the coefficients of a linear system with n variables
// and n equations. The number of rows is n, the number of columns
// is n+1 (i.e., this is the augmented matrix with the rhs of the
// system in the last column).
struct linmat {
    std::vector<std::vector<expression>> m_rows;

    using row_idx_t = decltype(m_rows.size());
    using col_idx_t = decltype(m_rows[0].size());

    auto &operator()(row_idx_t i, col_idx_t j)
    {
        assert(i < m_rows.size());
        assert(j < m_rows[i].size());
        return m_rows[i][j];
    }

    template <typename T>
    explicit linmat(T n)
    {
        assert(n > 0);

        m_rows.resize(n);

        for (auto &row : m_rows) {
            row.resize(n + 1);
        }
    }
};

// Prepare the matrix representing the Euler-Lagrange
// equations as a linear system in the generalised
// accelerations. n_qs is the number of generalised coordinates,
// L_dt and D_dt the tensors of derivatives of the Lagrangian
// and the dissipation function respectively. qdots is the
// list of generalised velocities.
template <typename T>
linmat build_linmat(T n_qs, const dtens &L_dt, const dtens &D_dt, const std::vector<expression> &qdots)
{
    assert(n_qs > 0);
    assert(n_qs == qdots.size());

    // Prepare vectors of indices for indexing into L_dt and D_dt.
    std::vector<std::uint32_t> vidx_L;
    vidx_L.resize(1 + n_qs * 2 + 1);

    std::vector<std::uint32_t> vidx_D;
    vidx_D.resize(1 + n_qs);

    // Temporary vector to build the rhs.
    std::vector<expression> rhs_vec;

    // Init the return value.
    linmat mat(n_qs);

    // Build row-by-row.
    for (T i = 0; i < n_qs; ++i) {
        // Reset the temp vectors.
        std::ranges::fill(vidx_L, 0);
        std::ranges::fill(vidx_D, 0);
        rhs_vec.clear();

        // dL/dqi.
        vidx_L[1 + i] = 1;
        rhs_vec.push_back(fix_nn(L_dt[vidx_L]));

        // -d2L/(dqdoti dt).
        vidx_L[1 + i] = 0;
        vidx_L[1 + n_qs + i] = 1;
        vidx_L.back() = 1;
        rhs_vec.push_back(-fix_nn(L_dt[vidx_L]));

        // -dD/dqdoti
        vidx_D[1 + i] = 1;
        rhs_vec.push_back(-fix_nn(D_dt[vidx_D]));

        // Iterate over j to compute the -d2L/(dqdoti dqj) * qdotj
        // and the d2L/(dqdoti dqdotj) terms.
        auto &cur_row = mat.m_rows[i];
        for (T j = 0; j < n_qs; ++j) {
            // d2L/(dqdoti dqdotj).
            std::ranges::fill(vidx_L, 0);
            // NOTE: need to special case for i == j.
            if (i == j) {
                vidx_L[1 + n_qs + i] = 2;
            } else {
                vidx_L[1 + n_qs + i] = 1;
                vidx_L[1 + n_qs + j] = 1;
            }
            cur_row[j] = fix_nn(L_dt[vidx_L]);

            // -d2L/(dqdoti dqj) * qdotj.
            std::ranges::fill(vidx_L, 0);
            vidx_L[1 + n_qs + i] = 1;
            vidx_L[1 + j] = 1;
            rhs_vec.push_back(fix_nn(-fix_nn(L_dt[vidx_L]) * qdots[j]));
        }

        // Assemble the rhs.
        cur_row[n_qs] = sum(rhs_vec);
    }

    return mat;
}

// Helper to check if an expression is the zero constant.
bool ex_zero(const expression &ex)
{
    if (const auto *nptr = std::get_if<number>(&ex.value())) {
        return is_zero(*nptr);
    }

    return false;
}

// Solve the linear system represented by mat
// via Gaussian elimination + back-substitution.
// The solution will be stored in the last column of mat.
// Shamelessly taken from the wikipedia:
// https://en.wikipedia.org/wiki/Gaussian_elimination
void solve_linmat(linmat &mat)
{
    // Number of rows and columns.
    const auto m = mat.m_rows.size();
    assert(m > 0u);
    const auto n = mat.m_rows[0].size();
    assert(n > 0u);
    assert(n - 1u == m);

    // Init the pivot row and column.
    linmat::row_idx_t h = 0;
    linmat::col_idx_t k = 0;

    while (h < m && k < n) {
        // Find the k-th pivot.
        auto i_max = h;

        for (; i_max < m && ex_zero(mat(i_max, k)); ++i_max) {
        }

        if (i_max == m) {
            // No pivot in this column, pass to next column.
            ++k;
            continue;
        }

        // Swap rows.
        std::swap(mat.m_rows[h], mat.m_rows[i_max]);

        // Do for all rows below pivot.
        for (auto i = h + 1u; i < m; ++i) {
            const auto f = fix_nn(fix_nn(mat(i, k)) / fix_nn(mat(h, k)));

            // Fill with zeros the lower part of pivot column.
            mat(i, k) = 0_dbl;

            // Do for all remaining elements in current row.
            for (auto j = k + 1u; j < n; ++j) {
                mat(i, j) = fix_nn(fix_nn(mat(i, j)) - fix_nn(mat(h, j)) * f);
            }
        }

        ++h;
        ++k;
    }

    // Run the back-substitution.
    std::vector<expression> new_rhs;
    for (linmat::row_idx_t i_idx = 0; i_idx < m; ++i_idx) {
        // The row on which we are operating (we are moving
        // backwards).
        const auto i = m - i_idx - 1u;

        // Init the new rhs with the current one.
        new_rhs.clear();
        new_rhs.push_back(mat(i, n - 1u));

        // Move the terms to the rhs.
        for (linmat::col_idx_t j = i + 1u; j < m; ++j) {
            new_rhs.push_back(fix_nn(-mat(i, j) * mat(j, n - 1u)));
        }

        // Divide the new rhs by the coefficient of the current variable
        // and assign it.
        mat(i, n - 1u) = fix_nn(fix_nn(sum(new_rhs)) / mat(i, i));
    }
}

} // namespace

} // namespace detail

std::vector<std::pair<expression, expression>> lagrangian(const expression &L_, const std::vector<expression> &qs,
                                                          const std::vector<expression> &qdots, const expression &D)
{
    using size_type = boost::safe_numerics::safe<decltype(qs.size())>;

    // Sanity checks.
    detail::lagrangian_impl_sanity_checks(L_, qs, qdots, D);

    // Cache the number of generalised coordinates/velocities.
    const auto n_qs = size_type(qs.size());

    // Replace the time expression with a time variable.
    const auto tm_var = "__tm"_var;
    const auto L = subs(L_, {{heyoka::time, tm_var}});

    // Assemble the diff arguments.
    auto diff_args = qs;
    diff_args.insert(diff_args.end(), qdots.begin(), qdots.end());
    diff_args.push_back(tm_var);

    // NOTE: these next two bits can be run in parallel if needed.
    // Compute the tensor of derivatives of L up to order 2 wrt
    // qs, qdots and time.
    const auto L_dt = diff_tensors({L}, diff_args, kw::diff_order = 2);

    // Compute the tensor of derivatives of D up to order 1 wrt qdots.
    const auto D_dt = diff_tensors({D}, qdots, kw::diff_order = 1);

    // Build the linear system.
    auto mat = detail::build_linmat(n_qs, L_dt, D_dt, qdots);

    // Solve it.
    detail::solve_linmat(mat);

    // Start assembling the RHS of the return value.
    std::vector<expression> rhs_ret;
    rhs_ret.reserve(n_qs * 2);

    // dq/dt = qdot.
    for (size_type i = 0; i < n_qs; ++i) {
        rhs_ret.push_back(qdots[i]);
    }

    // dqdot/dt.
    for (size_type i = 0; i < n_qs; ++i) {
        rhs_ret.push_back(mat.m_rows[i].back());
    }

    // Restore the time expression.
    rhs_ret = subs(rhs_ret, {{tm_var, heyoka::time}});

    // Assemble the result.
    std::vector<std::pair<expression, expression>> ret;
    ret.reserve(n_qs * 2);

    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(qs[i], std::move(rhs_ret[i]));
    }

    for (size_type i = 0; i < n_qs; ++i) {
        ret.emplace_back(qdots[i], std::move(rhs_ret[n_qs + i]));
    }

    return ret;
}

HEYOKA_END_NAMESPACE
