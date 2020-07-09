// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNCTION_HPP
#define HEYOKA_FUNCTION_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC function
{
public:
    enum class type { internal, external, builtin };

    using diff_t = std::function<expression(const std::vector<expression> &, const std::string &)>;

    // Evaluation function types.
    using eval_dbl_t
        = std::function<double(const std::vector<expression> &, const std::unordered_map<std::string, double> &)>;
    using eval_batch_dbl_t = std::function<void(std::vector<double> &, const std::vector<expression> &,
                                                const std::unordered_map<std::string, std::vector<double>> &)>;
    using eval_num_dbl_t = std::function<double(const std::vector<double> &)>;
    using deval_num_dbl_t = std::function<double(const std::vector<double> &, std::vector<double>::size_type)>;

    // Taylor integration function types.
    using taylor_decompose_t
        = std::function<std::vector<expression>::size_type(function &&, std::vector<expression> &)>;
    using taylor_init_t = std::function<llvm::Value *(llvm_state &, const function &, llvm::Value *)>;
    using taylor_diff_t
        = std::function<llvm::Function *(llvm_state &, const function &, std::uint32_t, const std::string &,
                                         std::uint32_t, const std::unordered_map<std::uint32_t, number> &)>;

private:
    bool m_disable_verify = false;
    std::string m_dbl_name, m_ldbl_name, m_display_name;
    std::unique_ptr<std::vector<expression>> m_args;
    std::vector<llvm::Attribute::AttrKind> m_attributes;
    type m_ty = type::internal;

    diff_t m_diff_f;

    eval_dbl_t m_eval_dbl_f;
    eval_batch_dbl_t m_eval_batch_dbl_f;
    eval_num_dbl_t m_eval_num_dbl_f;
    deval_num_dbl_t m_deval_num_dbl_f;

    taylor_decompose_t m_taylor_decompose_f;
    taylor_init_t m_taylor_init_dbl_f;
    taylor_init_t m_taylor_init_ldbl_f;
    taylor_diff_t m_taylor_diff_dbl_f;
    taylor_diff_t m_taylor_diff_ldbl_f;

public:
    explicit function(std::vector<expression>);
    function(const function &);
    function(function &&) noexcept;
    ~function();

    function &operator=(const function &);
    function &operator=(function &&) noexcept;

    bool &disable_verify();
    std::string &dbl_name();
    std::string &ldbl_name();
    std::string &display_name();
    std::vector<expression> &args();
    std::vector<llvm::Attribute::AttrKind> &attributes();
    type &ty();
    diff_t &diff_f();
    eval_dbl_t &eval_dbl_f();
    eval_batch_dbl_t &eval_batch_dbl_f();
    eval_num_dbl_t &eval_num_dbl_f();
    deval_num_dbl_t &deval_num_dbl_f();
    taylor_decompose_t &taylor_decompose_f();
    taylor_init_t &taylor_init_dbl_f();
    taylor_init_t &taylor_init_ldbl_f();
    taylor_diff_t &taylor_diff_dbl_f();
    taylor_diff_t &taylor_diff_ldbl_f();

    const bool &disable_verify() const;
    const std::string &dbl_name() const;
    const std::string &ldbl_name() const;
    const std::string &display_name() const;
    const std::vector<expression> &args() const;
    const std::vector<llvm::Attribute::AttrKind> &attributes() const;
    const type &ty() const;
    const diff_t &diff_f() const;
    const eval_dbl_t &eval_dbl_f() const;
    const eval_batch_dbl_t &eval_batch_dbl_f() const;
    const eval_num_dbl_t &eval_num_dbl_f() const;
    const deval_num_dbl_t &deval_num_dbl_f() const;
    const taylor_decompose_t &taylor_decompose_f() const;
    const taylor_init_t &taylor_init_dbl_f() const;
    const taylor_init_t &taylor_init_ldbl_f() const;
    const taylor_diff_t &taylor_diff_dbl_f() const;
    const taylor_diff_t &taylor_diff_ldbl_f() const;
};

HEYOKA_DLL_PUBLIC void swap(function &, function &) noexcept;

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const function &);

HEYOKA_DLL_PUBLIC std::vector<std::string> get_variables(const function &);
HEYOKA_DLL_PUBLIC void rename_variables(function &, const std::unordered_map<std::string, std::string> &);

HEYOKA_DLL_PUBLIC bool operator==(const function &, const function &);
HEYOKA_DLL_PUBLIC bool operator!=(const function &, const function &);

HEYOKA_DLL_PUBLIC expression diff(const function &, const std::string &);

HEYOKA_DLL_PUBLIC double eval_dbl(const function &, const std::unordered_map<std::string, double> &);

HEYOKA_DLL_PUBLIC void eval_batch_dbl(std::vector<double> &, const function &,
                                      const std::unordered_map<std::string, std::vector<double>> &);

HEYOKA_DLL_PUBLIC double eval_num_dbl_f(const function &, const std::vector<double> &);
HEYOKA_DLL_PUBLIC double deval_num_dbl_f(const function &, const std::vector<double> &, std::vector<double>::size_type);

HEYOKA_DLL_PUBLIC void update_connections(std::vector<std::vector<std::size_t>> &, const function &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_node_values_dbl(std::vector<double> &, const function &,
                                              const std::unordered_map<std::string, double> &,
                                              const std::vector<std::vector<std::size_t>> &, std::size_t &);
HEYOKA_DLL_PUBLIC void update_grad_dbl(std::unordered_map<std::string, double> &, const function &,
                                       const std::unordered_map<std::string, double> &, const std::vector<double> &,
                                       const std::vector<std::vector<std::size_t>> &, std::size_t &, double);

HEYOKA_DLL_PUBLIC llvm::Value *codegen_dbl(llvm_state &, const function &);
HEYOKA_DLL_PUBLIC llvm::Value *codegen_ldbl(llvm_state &, const function &);

HEYOKA_DLL_PUBLIC std::vector<expression>::size_type taylor_decompose_in_place(function &&, std::vector<expression> &);

HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_dbl(llvm_state &, const function &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Value *taylor_init_ldbl(llvm_state &, const function &, llvm::Value *);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_dbl(llvm_state &, const function &, std::uint32_t, const std::string &,
                                                  std::uint32_t, const std::unordered_map<std::uint32_t, number> &);
HEYOKA_DLL_PUBLIC llvm::Function *taylor_diff_ldbl(llvm_state &, const function &, std::uint32_t, const std::string &,
                                                   std::uint32_t, const std::unordered_map<std::uint32_t, number> &);

} // namespace heyoka

#endif
