// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_TAYLOR_HPP
#define HEYOKA_TAYLOR_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(expression &&,
                                                                               std::vector<expression> &);

HEYOKA_DLL_PUBLIC std::vector<expression> taylor_decompose(std::vector<expression>);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const expression &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const expression &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_dbl(llvm_state &, const expression &, std::uint32_t, const std::string &,
                                                  std::uint32_t, const std::unordered_map<std::uint32_t, number> &);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_ldbl(llvm_state &, const expression &, std::uint32_t, const std::string &,
                                                   std::uint32_t, const std::unordered_map<std::uint32_t, number> &);

namespace detail
{

template <typename T>
class HEYOKA_DLL_PUBLIC taylor_adaptive_impl
{
public:
    enum class outcome {
        success,       // Integration step was successful.
        step_limit,    // Maximum number of steps reached.
        nf_state,      // Non-finite initial state detected.
        nf_derivative, // Non-finite derivative detected.
        nan_rho        // NaN estimation of the convergence radius.
    };

private:
    std::vector<T> m_state;
    T m_time;
    const T m_rtol;
    const T m_atol;
    llvm_state m_llvm;
    std::uint32_t m_max_order;
    std::vector<T> m_jet;
    using jet_f_t = void (*)(T *, std::uint32_t);
    jet_f_t m_jet_f;

    template <bool, bool>
    HEYOKA_DLL_LOCAL std::pair<outcome, T> step_impl(T);

public:
    explicit taylor_adaptive_impl(std::vector<expression>, std::vector<T>, T, T, T);

    taylor_adaptive_impl(const taylor_adaptive_impl &) = delete;
    taylor_adaptive_impl(taylor_adaptive_impl &&) = delete;
    taylor_adaptive_impl &operator=(const taylor_adaptive_impl &) = delete;
    taylor_adaptive_impl &operator=(taylor_adaptive_impl &&) = delete;

    ~taylor_adaptive_impl();

    T get_time() const
    {
        return m_time;
    }
    const std::vector<T> &get_state() const
    {
        return m_state;
    }

    void set_state(const std::vector<T> &);
    void set_time(T);

    std::pair<outcome, T> step();
    std::pair<outcome, T> step_backward();
    // NOTE: return values:
    // - outcome,
    // - min abs(timestep),
    // - max abs(timestep),
    // - total number of steps successfully
    //   undertaken.
    // NOTE: the min/max timestep values are well-defined
    // only if at least 2 steps were taken successfully. Otherwise,
    // they are infinity and zero respectively.
    std::tuple<outcome, T, T, std::size_t> propagate_for(T, std::size_t = 0);
    std::tuple<outcome, T, T, std::size_t> propagate_until(T, std::size_t = 0);
};

} // namespace detail

class HEYOKA_DLL_PUBLIC taylor_adaptive_dbl : public detail::taylor_adaptive_impl<double>
{
public:
    using base = detail::taylor_adaptive_impl<double>;
    using base::base;
};

class HEYOKA_DLL_PUBLIC taylor_adaptive_ldbl : public detail::taylor_adaptive_impl<long double>
{
public:
    using base = detail::taylor_adaptive_impl<long double>;
    using base::base;
};

} // namespace heyoka

#endif
