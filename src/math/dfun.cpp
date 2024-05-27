// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <concepts>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/func_cache.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/dfun.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

dfun_impl::dfun_impl() : dfun_impl("x", std::vector<expression>{}, {}) {}

namespace
{

// Helper to validate the construction arguments of a dfun and assemble
// the full function name.
// NOTE: Args can be either a vector of arguments or a shared pointer to
// a vector of arguments.
template <typename Args>
auto make_dfun_name(const std::string &id_name, Args args_,
                    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &didx)
{
    // Init the name.
    std::string full_name = "dfun_";

    // Fetch a reference to the arguments.
    const auto &args = [&args_]() -> const auto & {
        if constexpr (std::same_as<Args, std::vector<expression>>) {
            return args_;
        } else {
            if (args_ == nullptr) [[unlikely]] {
                throw std::invalid_argument("Cannot construct a dfun from a null shared pointer to its arguments");
            }

            return *args_;
        }
    }();

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

    return std::make_tuple(std::move(full_name), std::move(args_));
}

} // namespace

dfun_impl::dfun_impl(std::string id_name, std::vector<expression> args,
                     std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
    : shared_func_base(std::make_from_tuple<shared_func_base>(make_dfun_name(id_name, std::move(args), didx))),
      m_id_name(std::move(id_name)), m_didx(std::move(didx))
{
}

dfun_impl::dfun_impl(std::string id_name, std::shared_ptr<const std::vector<expression>> args,
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
#if !defined(NDEBUG)

    // Run some checks on m_didx in debug mode.
    if (!m_didx.empty()) {
        // Helper to validate an element in m_didx and update full_name.
        auto validate_p = [this](const auto &p) {
            const auto [idx, order] = p;
            assert(idx < this->args().size());
            assert(order != 0u);
        };

        // Validate the first element.
        validate_p(m_didx[0]);

        // Validate the remaining elements
        for (auto it = m_didx.begin() + 1; it != m_didx.end(); ++it) {
            validate_p(*it);

            // Check that m_didx is sorted according to the index.
            assert(it->first > (it - 1)->first);
        }
    }

#endif
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
        oss << fmt::format("(∂{})", m_id_name);
    } else {
        oss << fmt::format("(∂^{} {})", diff_order, m_id_name);
    }

    if (!m_didx.empty()) {
        oss << "/(";

        for (auto it = m_didx.begin(); it != m_didx.end(); ++it) {
            const auto [idx, order] = *it;

            oss << fmt::format("∂a{}", idx);
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

        // NOTE: the logic here is the following:
        // - copy over all elements from m_didx for which the index is less than idx;
        // - if m_didx contains arg_idx in its indices, bump up the derivative order by one
        //   and copy the remaining elements of m_didx; otherwise,
        // - add a first-order derivative when we identify the first index in m_didx
        //   which is greater than arg_idx, then copy the remaining elements.
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

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
bool contains_dfun_impl(funcptr_set &func_set, const expression &ex)
{
    return std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_set]<typename T>(const T &v) {
            if constexpr (std::same_as<T, func>) {
                const auto f_id = v.get_ptr();

                // Did we already determine that v is not a dfun and none of its
                // arguments contains a dfun?
                if (func_set.contains(f_id)) {
                    return false;
                }

                // Check if v is a dfun.
                if (v.template extract<dfun_impl>() != nullptr) {
                    return true;
                }

                // v is not a dfun. check if any of its arguments contains a dfun.
                for (const auto &a : v.args()) {
                    if (contains_dfun_impl(func_set, a)) {
                        return true;
                    }
                }

                // Update the cache.
                [[maybe_unused]] const auto [_, flag] = func_set.emplace(f_id);

                // An expression cannot contain itself.
                assert(flag);

                return false;
            } else {
                return false;
            }
        },
        ex.value());
}

} // namespace

// Helper to detect whether any expression in v_ex contains at least one dfun.
bool contains_dfun(const std::vector<expression> &v_ex)
{
    // NOTE: this set will contain subexpressions which are not a dfun
    // and which do not contain any dfun in the arguments.
    funcptr_set func_set;

    for (const auto &ex : v_ex) {
        if (contains_dfun_impl(func_set, ex)) {
            return true;
        }
    }

    return false;
}

namespace
{

// NOLINTNEXTLINE(misc-no-recursion)
void get_dfuns_impl(funcptr_set &func_set, std::set<expression> &retval, const expression &ex)
{
    std::visit(
        // NOLINTNEXTLINE(misc-no-recursion)
        [&func_set, &retval, &ex]<typename T>(const T &arg) {
            if constexpr (std::same_as<T, func>) {
                const auto f_id = arg.get_ptr();

                if (func_set.contains(f_id)) {
                    // We already got the dfuns from the current function, exit.
                    return;
                }

                // Get the dfuns for each function argument.
                for (const auto &farg : arg.args()) {
                    get_dfuns_impl(func_set, retval, farg);
                }

                // If the current function is a dfun, add it to retval.
                if (arg.template extract<dfun_impl>() != nullptr) {
                    retval.insert(ex);
                }

                // Add the id of f to the set.
                [[maybe_unused]] const auto [_, flag] = func_set.insert(f_id);
                // NOTE: an expression cannot contain itself.
                assert(flag);
            }
        },
        ex.value());
}

} // namespace

// Helper to fetch the set of all dfuns contained in v_ex.
std::set<expression> get_dfuns(const std::vector<expression> &v_ex)
{
    funcptr_set func_set;

    std::set<expression> retval;

    for (const auto &ex : v_ex) {
        get_dfuns_impl(func_set, retval, ex);
    }

    return retval;
}

} // namespace detail

expression dfun(std::string id_name, std::vector<expression> args,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
{
    return expression{func{detail::dfun_impl{std::move(id_name), std::move(args), std::move(didx)}}};
}

expression dfun(std::string id_name, std::shared_ptr<const std::vector<expression>> args,
                std::vector<std::pair<std::uint32_t, std::uint32_t>> didx)
{
    return expression{func{detail::dfun_impl{std::move(id_name), std::move(args), std::move(didx)}}};
}

HEYOKA_END_NAMESPACE

// NOLINTNEXTLINE(cert-err58-cpp)
HEYOKA_S11N_FUNC_EXPORT_IMPLEMENT(heyoka::detail::dfun_impl)
