// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/fast_unordered.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/func.hpp>
#include <heyoka/math/dfun.hpp>
#include <heyoka/math/time.hpp>
#include <heyoka/param.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/var_ode_sys.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

var_args operator|(var_args a1, var_args a2) noexcept
{
    return var_args{static_cast<unsigned>(a1) | static_cast<unsigned>(a2)};
}

bool operator&(var_args a1, var_args a2) noexcept
{
    return static_cast<bool>(static_cast<unsigned>(a1) & static_cast<unsigned>(a2));
}

namespace detail
{

namespace
{

void validate_var_args(var_args a)
{
    if (a == var_args{0} || a > var_args::all) [[unlikely]] {
        throw std::invalid_argument(fmt::format("Invalid var_args enumerator detected: the value of the enumerator "
                                                "must be in the [1, 7] range, but a value of {} was detected instead",
                                                static_cast<unsigned>(a)));
    }
}

} // namespace

} // namespace detail

struct var_ode_sys::impl {
    std::vector<std::pair<expression, expression>> sys;
    dtens dt;
    std::vector<expression> vargs;

    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar & sys;
        ar & dt;
        ar & vargs;
    }
};

void var_ode_sys::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_impl;
}

// NOTE: this will leave the object in the moved-from state. This needs
// to be documented properly.
void var_ode_sys::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        ar >> m_impl;
        // LCOV_EXCL_START
    } catch (...) {
        m_impl = nullptr;
        throw;
    }
    // LCOV_EXCL_STOP
}

// NOTE: this initialises into the moved-from state, this needs
// to be documented properly.
var_ode_sys::var_ode_sys() noexcept = default;

var_ode_sys::var_ode_sys(const std::vector<std::pair<expression, expression>> &sys,
                         std::initializer_list<expression> args, std::uint32_t order)
    : var_ode_sys(sys, std::vector(args), order)
{
}

var_ode_sys::var_ode_sys(const std::vector<std::pair<expression, expression>> &sys,
                         const std::variant<var_args, std::vector<expression>> &args, std::uint32_t order)
{
    // Validate input arguments.
    detail::validate_ode_sys(sys);

    if (order == 0u) [[unlikely]] {
        throw std::invalid_argument("The 'order' argument to the var_ode_sys constructor must be nonzero");
    }

    // Run a further validation on the names of the state variables:
    // names starting with "∂" are reserved to represent the variational
    // variables.
    for (const auto &[lhs, rhs] : sys) {
        if (std::get<variable>(lhs.value()).name().starts_with("∂")) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Invalid state variable '{}' detected: in a variational ODE system "
                                                    "state variable names starting with '∂' are reserved",
                                                    std::get<variable>(lhs.value()).name()));
        }
    }

    // Put the rhs into a vector.
    std::vector<expression> sys_rhs;
    sys_rhs.reserve(sys.size());
    std::ranges::transform(sys, std::back_inserter(sys_rhs), &std::pair<expression, expression>::second);

    // Build the list of arguments with respect to which
    // the variational equations will be formulated.
    // NOTE: vargs contains the actual arguments wrt we will be computing
    // the derivatives, vargs_hr is the human-readable version of the same
    // arguments. vargs_hr will be stored as a data member in this.
    std::vector<expression> vargs, vargs_hr;

    if (const auto *va_ptr = std::get_if<var_args>(&args)) {
        // Check the enumerator.
        const auto va = *va_ptr;
        detail::validate_var_args(va);

        // Are derivatives wrt the initial conditions requested?
        if (va & var_args::vars) {
            for (const auto &[lhs, rhs] : sys) {
                vargs.emplace_back(fmt::format("__{}_0", std::get<variable>(lhs.value()).name()));
                vargs_hr.push_back(lhs);
            }
        }

        // Are derivatives wrt the parameters requested?
        if (va & var_args::params) {
            // Fetch the sorted list of parameters appearing in sys.
            const auto param_list = get_params(sys_rhs);

            vargs.insert(vargs.end(), param_list.begin(), param_list.end());
            vargs_hr.insert(vargs_hr.end(), param_list.begin(), param_list.end());
        }

        // Are derivatives wrt the initial time requested?
        if (va & var_args::time) {
            // NOTE: put twice the double underscore "__", because we know
            // from validate_ode_sys() that there are no variables in the original
            // sys starting with "__", thus by prefixing another "__" we know that
            // there cannot be name collisions with the "__{}_0" variables tha may have
            // been added earlier.
            vargs.emplace_back("____t_0");
            vargs_hr.push_back(heyoka::time);
        }
    } else {
        // The user specified an explicit list of expressions wrt which
        // the variational equations are to be formulated. We need to
        // validate this list.
        const auto &va = std::get<std::vector<expression>>(args);

        if (va.empty()) [[unlikely]] {
            throw std::invalid_argument(
                "Cannot formulate the variational equations with respect to an empty list of arguments");
        }

        // Reserve the needed space.
        vargs.reserve(va.size());
        vargs_hr.reserve(va.size());

        // Fetch the list of state variables from sys and put it into
        // a set for fast lookup.
        detail::fast_uset<std::string> sv_set;
        sv_set.reserve(sys.size());
        for (const auto &[lhs, rhs] : sys) {
            assert(!sv_set.contains(std::get<variable>(lhs.value()).name()));
            sv_set.insert(std::get<variable>(lhs.value()).name());
        }

        for (const auto &ex : va) {
            // ex must be a variable, a parameter or heyoka::time.
            if (const auto *var_ptr = std::get_if<variable>(&ex.value())) {
                if (!sv_set.contains(var_ptr->name())) [[unlikely]] {
                    throw std::invalid_argument(
                        fmt::format("Cannot formulate the variational equations with respect to the "
                                    "initial conditions for the variable '{}', which is not among the state variables "
                                    "of the system",
                                    var_ptr->name()));
                }

                vargs.emplace_back(fmt::format("__{}_0", var_ptr->name()));
            } else if (const auto *par_ptr = std::get_if<param>(&ex.value())) {
                // NOTE: here we are allowing to formulate the variational equations wrt
                // parameters that might not appear in the ODE sys. We allow this for consistency
                // with the behaviour of diff_tensors(), and I do not see important drawbacks
                // at the moment.
                vargs.emplace_back(*par_ptr);
            } else if (ex == heyoka::time) {
                // NOTE: put twice the double underscore "__", because we know
                // from validate_ode_sys() that there are no variables in the original
                // sys starting with "__", thus by prefixing another "__" we know that
                // there cannot be name collisions with the "__{}_0" variables tha may have
                // been added earlier.
                vargs.emplace_back("____t_0");
            } else [[unlikely]] {
                throw std::invalid_argument(
                    fmt::format("Cannot formulate the variational equations with respect to the expression '{}': the "
                                "expression is not a variable, not a parameter and not heyoka::time",
                                ex));
            }

            // NOTE: vargs_hr ends up being just a copy of va.
            vargs_hr.push_back(ex);
        }

        // Check that there are no duplicates in vargs.
        const detail::fast_uset<expression, std::hash<expression>> vargs_set(vargs.begin(), vargs.end());
        if (vargs_set.size() != vargs.size()) [[unlikely]] {
            throw std::invalid_argument(
                fmt::format("Duplicate entries detected in the list of expressions with respect to which the "
                            "variational equations are to be formulated: {}",
                            va));
        }
    }

    assert(vargs.size() == vargs_hr.size());

    // Turn vargs into a shared_ptr for use in the dfun() API.
    auto vargs_ptr = std::make_shared<const std::vector<expression>>(std::move(vargs));

    // Create the subs map to replace the state variables with implicit functions of vargs.
    std::unordered_map<std::string, expression> subs_map;
    for (const auto &[lhs, rhs] : sys) {
        const auto &sv_name = std::get<variable>(lhs.value()).name();
        assert(!subs_map.contains(sv_name));
        subs_map.emplace(sv_name, dfun(sv_name, vargs_ptr));
    }

    // Run the substitution on sys_rhs.
    sys_rhs = subs(sys_rhs, subs_map);

    // Formulate the variational equations.
    auto dt_vareq = diff_tensors(sys_rhs, *vargs_ptr, kw::diff_order = order);

    // Now we need to replace back the dfun()s with the original and variational variables.
    std::map<expression, expression> dfun_subs_map;
    std::vector<expression> new_lhs, new_rhs;
    new_lhs.reserve(dt_vareq.size());
    new_rhs.reserve(dt_vareq.size());

    for (const auto &[key, diff_ex] : dt_vareq) {
        const auto &[sv_idx, diff_idx] = key;

        // Fetch the original state variable.
        assert(sv_idx < sys.size());
        const auto &sv = sys[sv_idx].first;
        const auto &sv_name = std::get<variable>(sv.value()).name();

        // Create the dfun() corresponding to the current state variable and diff indices.
        auto cur_dfun = dfun(sv_name, vargs_ptr, diff_idx);
        assert(!dfun_subs_map.contains(cur_dfun));

        if (diff_idx.empty()) {
            // Zero order derivative: the current dfun() must be replaced with the original
            // state variable, and in the new lhs we push the original state variable.
            new_lhs.push_back(sv);
            dfun_subs_map.emplace(std::move(cur_dfun), sv);
        } else {
            // Nonzero order derivative: we need to create a new variable to represent
            // the current dfun().
            expression variational_var{fmt::format("∂{}{}", diff_idx, sv_name)};
            new_lhs.push_back(variational_var);
            dfun_subs_map.emplace(std::move(cur_dfun), std::move(variational_var));
        }

        // Add the current variational equation.
        new_rhs.push_back(diff_ex);
    }

    // Run the substitution.
    new_rhs = subs(new_rhs, dfun_subs_map);

    // Create the new ODE sys.
    std::vector<std::pair<expression, expression>> new_sys;
    new_sys.reserve(dt_vareq.size());
    for (decltype(dt_vareq.size()) i = 0; i < dt_vareq.size(); ++i) {
        new_sys.emplace_back(std::move(new_lhs[i]), std::move(new_rhs[i]));
    }

    // Init the pimpl.
    m_impl = std::make_unique<impl>(impl{std::move(new_sys), std::move(dt_vareq), std::move(vargs_hr)});
}

var_ode_sys::var_ode_sys(const var_ode_sys &other) : m_impl{std::make_unique<impl>(*other.m_impl)} {}

var_ode_sys::var_ode_sys(var_ode_sys &&) noexcept = default;

var_ode_sys &var_ode_sys::operator=(const var_ode_sys &other)
{
    if (this != &other) {
        *this = var_ode_sys(other);
    }

    return *this;
}

var_ode_sys &var_ode_sys::operator=(var_ode_sys &&) noexcept = default;

var_ode_sys::~var_ode_sys() = default;

const std::vector<std::pair<expression, expression>> &var_ode_sys::get_sys() const noexcept
{
    return m_impl->sys;
}

const std::vector<expression> &var_ode_sys::get_vargs() const noexcept
{
    return m_impl->vargs;
}

std::uint32_t var_ode_sys::get_n_orig_sv() const noexcept
{
    return m_impl->dt.get_nouts();
}

std::uint32_t var_ode_sys::get_order() const noexcept
{
    return m_impl->dt.get_order();
}

const dtens &var_ode_sys::get_dtens() const noexcept
{
    return m_impl->dt;
}

HEYOKA_END_NAMESPACE
