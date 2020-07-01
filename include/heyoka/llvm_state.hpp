// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_LLVM_STATE_HPP
#define HEYOKA_LLVM_STATE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>

namespace heyoka
{

class HEYOKA_DLL_PUBLIC llvm_state
{
    class jit;

    std::unique_ptr<jit> m_jitter;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>> m_builder;
    std::unique_ptr<llvm::legacy::FunctionPassManager> m_fpm;
    std::unique_ptr<llvm::legacy::PassManager> m_pm;
    std::unordered_map<std::string, llvm::Value *> m_named_values;
    bool m_verify = true;
    unsigned m_opt_level;

    HEYOKA_DLL_LOCAL void check_uncompiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_compiled(const char *) const;
    HEYOKA_DLL_LOCAL void check_add_name(const std::string &) const;

    template <typename T>
    HEYOKA_DLL_LOCAL void add_varargs_expression(const std::string &, const expression &,
                                                 const std::vector<std::string> &);
    HEYOKA_DLL_LOCAL void verify_function_impl(llvm::Function *);

public:
    explicit llvm_state(const std::string &, unsigned = 3);
    llvm_state(const llvm_state &) = delete;
    llvm_state(llvm_state &&) = delete;
    llvm_state &operator=(const llvm_state &) = delete;
    llvm_state &operator=(llvm_state &&) = delete;
    ~llvm_state();

    llvm::Module &module();
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> &builder();
    llvm::LLVMContext &context();
    bool &verify();
    std::unordered_map<std::string, llvm::Value *> &named_values();

    const llvm::Module &module() const;
    const llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> &builder() const;
    const llvm::LLVMContext &context() const;
    const bool &verify() const;
    const std::unordered_map<std::string, llvm::Value *> &named_values() const;

    void verify_function(const std::string &);

    void add_dbl(const std::string &, const expression &);

    void compile();
};

} // namespace heyoka

#endif
