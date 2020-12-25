// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_PARAM_HPP
#define HEYOKA_PARAM_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC param
{
    std::uint32_t m_index;

public:
    explicit param(std::uint32_t);

    ~param();

    const std::uint32_t &idx() const;

    std::uint32_t &idx();
};

HEYOKA_DLL_PUBLIC void swap(param &, param &) noexcept;

HEYOKA_DLL_PUBLIC std::size_t hash(const param &);

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const param &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const param &);
HEYOKA_DLL_PUBLIC void rename_variables(param &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const param &, const param &);
HEYOKA_DLL_PUBLIC bool operator!=(const param &, const param &);

HEYOKA_DLL_PUBLIC expression subs(const param &, const std::unordered_map<std::string, expression> &);

HEYOKA_DLL_PUBLIC expression diff(const param &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const param &, const std::unordered_map<std::string, double> &,
                                  const std::vector<double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const param &,
                                      const std::unordered_map<std::string, std::vector<double>> &,
                                      const std::vector<double> &);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const param &, std::size_t &);
HEYOKA_DLL_PUBLIC [[noreturn]] void update_node_values_dbl(std::vector<double> &, const param &,
                                                           const std::unordered_map<std::string, double> &,
                                                           const std::vector<std::vector<std::size_t>> &,
                                                           std::size_t &);
HEYOKA_DLL_PUBLIC [[noreturn]] void update_grad_dbl(std::unordered_map<std::string, double> &, const param &,
                                                    const std::unordered_map<std::string, double> &,
                                                    const std::vector<double> &,
                                                    const std::vector<std::vector<std::size_t>> &, std::size_t &,
                                                    double);

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(param &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_dbl(llvm_state &, const param &, const std::vector<llvm::Value *> &,
                                                 std::uint32_t);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_ldbl(llvm_state &, const param &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DLL_PUBLIC llvm::Value *taylor_u_init_f128(llvm_state &, const param &, const std::vector<llvm::Value *> &,
                                                  std::uint32_t);

#endif

template <typename T>
inline llvm::Value *taylor_u_init(llvm_state &s, const param &p, const std::vector<llvm::Value *> &arr,
                                  std::uint32_t batch_size)
{
    if constexpr (std::is_same_v<T, double>) {
        return taylor_u_init_dbl(s, p, arr, batch_size);
    } else if constexpr (std::is_same_v<T, long double>) {
        return taylor_u_init_ldbl(s, p, arr, batch_size);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return taylor_u_init_f128(s, p, arr, batch_size);
#endif
    } else {
        static_assert(detail::always_false_v<T>, "Unhandled type.");
    }
}

} // namespace heyoka

#endif
