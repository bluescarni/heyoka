// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/dfun.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

dfun_impl::dfun_impl() : dfun_impl("x", {}, {}) {}

namespace
{

// Helper to validate the construction arguments of a dfun and assemble
// the full function name.
auto make_dfun_name(const std::string &id_name, std::vector<expression> args,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &didx)
{
    // Init the name.
    std::string full_name = "dfun_";

    // Validate didx and build up the full name.
    if (!didx.empty()) {
        // Helper to validate an element in didx and update full_name.
        auto validate_p = [&args, &full_name](const auto &p) {
            const auto [idx, order] = p;

            if (idx >= args.size()) [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Invalid index {} detected in the indices vector passed to the constructor of a dfun: "
                                "the index must be less than the number of arguments ({})",
                                idx, args.size()));
            }

            if (order == 0u) [[unlikely]] {
                throw std::invalid_argument("Invalid zero derivative order detected in the indices vector passed to "
                                            "the constructor of a dfun: all derivative orders must be positive");
            }

            // Update full_name.
            full_name += fmt::format("{},{} ", idx, order);
        };

        // Validate the first element.
        validate_p(didx[0]);

        // Validate the remaining elements
        for (auto it = didx.begin() + 1; it != didx.end(); ++it) {
            validate_p(*it);

            // Check that didx is sorted according to the index.
            if (!(it->first > (it - 1)->first)) [[unlikely]] {
                throw std::invalid_argument("The indices in the indices vector passed to "
                                            "the constructor of a dfun must be sorted in strictly ascending order");
            }
        }
    }

    // Finalise full_name.
    full_name += "_";
    full_name += id_name;

    return std::make_tuple(std::move(full_name), std::move(args));
}

} // namespace

dfun_impl::dfun_impl(std::string id_name, std::vector<expression> args,
                     std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
    : shared_func_base(std::make_from_tuple<shared_func_base>(make_dfun_name(id_name, std::move(args), didx))),
      m_id_name(std::move(id_name)), m_didx(std::move(didx))
{
}

// NOTE: private ctor used in the implementation of gradient().
dfun_impl::dfun_impl(std::string full_name, std::string id_name, std::shared_ptr<const std::vector<expression>> args,
                     std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
    : shared_func_base(std::move(full_name), std::move(args)), m_id_name(std::move(id_name)), m_didx(std::move(didx))
{
}

const std::string &dfun_impl::get_id_name() const
{
    return m_id_name;
}

const std::vector<std::pair<std::uint32_t, std::uint32_t>> &dfun_impl::get_didx() const
{
    return m_didx;
}

void dfun_impl::to_stream(std::ostringstream &oss) const
{
    // Compute the total diff order.
    // NOTE: use unsigned long long for the calculation in order
    // to minimise the risk of overflow. If we overflow, it does not
    // matter too much as this is only the screen output.
    const auto diff_order = std::accumulate(m_didx.begin(), m_didx.end(), 0ull, [](auto acc, const auto &p) {
        return acc + static_cast<unsigned long long>(p.second);
    });

    if (diff_order == 1u) {
        oss << fmt::format("(d{})", m_id_name);
    } else {
        oss << fmt::format("(d^{} {})", diff_order, m_id_name);
    }

    if (!m_didx.empty()) {
        oss << "/(";

        for (auto it = m_didx.begin(); it != m_didx.end(); ++it) {
            const auto [idx, order] = *it;

            oss << fmt::format("da{}", idx);
            if (order > 1u) {
                oss << fmt::format("^{}", order);
            }

            if (it + 1 != m_didx.end()) {
                oss << " ";
            }
        }

        oss << ")";
    }
}

std::vector<expression> dfun_impl::gradient() const
{
    using su32_t = boost::safe_numerics::safe<std::uint32_t>;

    // Prepare the return value.
    std::vector<expression> retval;
    retval.reserve(args().size());

    for (decltype(args().size()) arg_idx = 0; arg_idx < args().size(); ++arg_idx) {
        // Prepare the new vector of indices, reserving one slot more for (potentially) an extra derivative.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> new_didx;
        new_didx.reserve(boost::safe_numerics::safe<decltype(new_didx.size())>(new_didx.size()) + 1u);

        // Prepare the new function name.
        std::string new_name = "dfun_";

        // Helper to update new_name with last pair added to new_didx.
        auto update_new_name = [&new_name, &new_didx]() {
            assert(!new_didx.empty());
            new_name += fmt::format("{},{} ", new_didx.back().first, new_didx.back().second);
        };

        // Flag to indicate we included the new derivative in new_didx.
        bool done = false;

        for (const auto &[idx, order] : m_didx) {
            assert(order > 0u);

            if (idx < arg_idx) {
                // We are before arg_idx in m_didx, copy
                // over the current derivative.
                new_didx.emplace_back(idx, order);
                update_new_name();
            } else if (idx == arg_idx) {
                // A nonzero derivative for arg_idx already exists in m_didx, bump
                // it up by one and add it to new_didx.
                new_didx.emplace_back(idx, su32_t(order) + 1);
                done = true;
                update_new_name();
            } else {
                assert(idx > arg_idx);

                // We are past arg_idx in m_didx. If we haven't added
                // the new derivative yet, do it and then mark it as done.
                if (!done) {
                    assert(new_didx.empty() || new_didx.back().first < arg_idx);
                    new_didx.emplace_back(boost::numeric_cast<std::uint32_t>(arg_idx), 1);
                    done = true;
                    update_new_name();
                }

                // Copy over the current derivative from m_didx.
                new_didx.emplace_back(idx, order);
                update_new_name();
            }
        }

        if (!done) {
            // All indices in new_didx are less than arg_idx. Add the new derivative.
            assert(new_didx.empty() || new_didx.back().first < arg_idx);
            new_didx.emplace_back(boost::numeric_cast<std::uint32_t>(arg_idx), 1);
            update_new_name();
        }

        // Finish building the new name.
        new_name += "_";
        new_name += m_id_name;

        // Build and add the gradient component.
        retval.emplace_back(func{dfun_impl{std::move(new_name), m_id_name, get_args_ptr(), std::move(new_didx)}});
    }

    return retval;
}

} // namespace detail

expression dfun(std::string id_name, std::vector<expression> args,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
{
    return expression{func{detail::dfun_impl{std::move(id_name), std::move(args), std::move(didx)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::dfun_impl)
