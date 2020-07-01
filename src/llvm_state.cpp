// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
// #include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
// #include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
// #include <llvm/Support/Error.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Vectorize.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

std::once_flag nt_inited;

} // namespace

} // namespace detail

// Implementation of the jit class.
class llvm_state::jit
{
    llvm::orc::ExecutionSession m_es;
    llvm::orc::RTDyldObjectLinkingLayer m_object_layer;
    std::unique_ptr<llvm::orc::IRCompileLayer> m_compile_layer;
    std::unique_ptr<llvm::DataLayout> m_dl;
    // NOTE: it seems like in LLVM 11 this class was moved
    // from llvm/ExecutionEngine/Orc/Core.h to
    // llvm/ExecutionEngine/Orc/Mangling.h.
    std::unique_ptr<llvm::orc::MangleAndInterner> m_mangle;
    llvm::orc::ThreadSafeContext m_ctx;
#if LLVM_VERSION_MAJOR == 10
    llvm::orc::JITDylib &m_main_jd;
#endif

public:
    jit()
        : m_object_layer(m_es, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
          m_ctx(std::make_unique<llvm::LLVMContext>())
#if LLVM_VERSION_MAJOR == 10
          ,
          m_main_jd(m_es.createJITDylib("<main>"))
#endif
    {
        // NOTE: the native target initialization needs to be done only once
        std::call_once(detail::nt_inited, []() {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            llvm::InitializeNativeTargetAsmParser();
        });

        auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!jtmb) {
            throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
        }

        auto dlout = jtmb->getDefaultDataLayoutForTarget();
        if (!dlout) {
            throw std::invalid_argument("Error fetching the default data layout for the host system");
        }

        m_compile_layer = std::make_unique<llvm::orc::IRCompileLayer>(
            m_es, m_object_layer,
#if LLVM_VERSION_MAJOR == 10
            std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(*jtmb))
#else
            llvm::orc::ConcurrentIRCompiler(std::move(*jtmb))
#endif
        );

        m_dl = std::make_unique<llvm::DataLayout>(std::move(*dlout));

        m_mangle = std::make_unique<llvm::orc::MangleAndInterner>(m_es, *m_dl);

        auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(m_dl->getGlobalPrefix());
        if (!dlsg) {
            throw std::invalid_argument("Could not create the dynamic library search generator");
        }

#if LLVM_VERSION_MAJOR == 10
        m_main_jd.addGenerator(std::move(*dlsg));
#else
        m_es.getMainJITDylib().setGenerator(std::move(*dlsg));
#endif
    }

    jit(const jit &) = delete;
    jit(jit &&) = delete;
    jit &operator=(const jit &) = delete;
    jit &operator=(jit &&) = delete;

    ~jit() = default;

    // Accessors.
    llvm::LLVMContext &get_context()
    {
        return *m_ctx.getContext();
    }
    const llvm::LLVMContext &get_context() const
    {
        return *m_ctx.getContext();
    }
    const llvm::DataLayout &get_data_layout() const
    {
        return *m_dl;
    }

    void add_module(std::unique_ptr<llvm::Module> &&m)
    {
        auto handle = m_compile_layer->add(
#if LLVM_VERSION_MAJOR == 10
            m_main_jd,
#else
            m_es.getMainJITDylib(),
#endif
            llvm::orc::ThreadSafeModule(std::move(m), m_ctx));

        if (handle) {
            throw std::invalid_argument("The function for adding a module to the jit failed");
        }
    }

    // Symbol lookup.
    llvm::Expected<llvm::JITEvaluatedSymbol> lookup(const std::string &name)
    {
        return m_es.lookup(
#if LLVM_VERSION_MAJOR == 10
            {&m_main_jd},
#else
            {&m_es.getMainJITDylib()},
#endif
            (*m_mangle)(name));
    }
};

llvm_state::llvm_state(const std::string &name, unsigned opt_level)
    : m_jitter(std::make_unique<jit>()), m_opt_level(opt_level)
{
    static_assert(std::is_same_v<llvm::IRBuilder<>, decltype(m_builder)::element_type>,
                  "Inconsistent llvm::IRBuilder<> type.");

    // Create the module.
    m_module = std::make_unique<llvm::Module>(name, context());
    m_module->setDataLayout(m_jitter->get_data_layout());

    // Create a new builder for the module.
    m_builder = std::make_unique<llvm::IRBuilder<>>(context());

    // Set a couple of flags for faster math at the
    // price of potential change of semantics.
    llvm::FastMathFlags fmf;
    fmf.setFast();
    m_builder->setFastMathFlags(fmf);

    // Create the optimization passes.
    if (m_opt_level > 0u) {
        // Create the function pass manager.
        m_fpm = std::make_unique<llvm::legacy::FunctionPassManager>(m_module.get());
        m_fpm->add(llvm::createPromoteMemoryToRegisterPass());
        m_fpm->add(llvm::createInstructionCombiningPass());
        m_fpm->add(llvm::createReassociatePass());
        m_fpm->add(llvm::createGVNPass());
        m_fpm->add(llvm::createCFGSimplificationPass());
        m_fpm->add(llvm::createLoopVectorizePass());
        m_fpm->add(llvm::createSLPVectorizerPass());
        m_fpm->add(llvm::createLoadStoreVectorizerPass());
        m_fpm->add(llvm::createLoopUnrollPass());
        m_fpm->doInitialization();

        // The module-level optimizer. See:
        // https://stackoverflow.com/questions/48300510/llvm-api-optimisation-run
        m_pm = std::make_unique<llvm::legacy::PassManager>();
        llvm::PassManagerBuilder pm_builder;
        // See here for the defaults:
        // https://llvm.org/doxygen/PassManagerBuilder_8cpp_source.html
        pm_builder.OptLevel = m_opt_level;
        pm_builder.VerifyInput = true;
        pm_builder.VerifyOutput = true;
        pm_builder.Inliner = llvm::createFunctionInliningPass();
        if (m_opt_level >= 3u) {
            pm_builder.SLPVectorize = true;
            pm_builder.MergeFunctions = true;
        }
        pm_builder.populateModulePassManager(*m_pm);
        pm_builder.populateFunctionPassManager(*m_fpm);
    }
}

llvm_state::~llvm_state() = default;

llvm::Module &llvm_state::module()
{
    check_uncompiled(__func__);
    return *m_module;
}

llvm::IRBuilder<> &llvm_state::builder()
{
    return *m_builder;
}

llvm::LLVMContext &llvm_state::context()
{
    return m_jitter->get_context();
}

bool &llvm_state::verify()
{
    return m_verify;
}

std::unordered_map<std::string, llvm::Value *> &llvm_state::named_values()
{
    return m_named_values;
}

const llvm::Module &llvm_state::module() const
{
    check_uncompiled(__func__);
    return *m_module;
}

const llvm::IRBuilder<> &llvm_state::builder() const
{
    return *m_builder;
}

const llvm::LLVMContext &llvm_state::context() const
{
    return m_jitter->get_context();
}

const bool &llvm_state::verify() const
{
    return m_verify;
}

const std::unordered_map<std::string, llvm::Value *> &llvm_state::named_values() const
{
    return m_named_values;
}

void llvm_state::check_uncompiled(const char *f) const
{
    if (!m_module) {
        throw std::invalid_argument(std::string{"The function '"} + f
                                    + "' can be invoked only if the module has not been compiled yet");
    }
}

void llvm_state::check_compiled(const char *f) const
{
    if (!m_module) {
        throw std::invalid_argument(std::string{"The function '"} + f
                                    + "' can be invoked only after the module has been compiled");
    }
}

void llvm_state::check_add_name(const std::string &name) const
{
    assert(m_module);
    if (m_module->getNamedValue(name) != nullptr) {
        throw std::invalid_argument("The name '" + name + "' already exists in the module");
    }
}

void llvm_state::verify_function_impl(llvm::Function *f)
{
    std::string err_report;
    llvm::raw_string_ostream ostr(err_report);
    if (llvm::verifyFunction(*f, &ostr) && m_verify) {
        // Remove function before throwing.
        f->eraseFromParent();
        throw std::invalid_argument("Function verification failed. The full error message:\n" + ostr.str());
    }
}

void llvm_state::verify_function(const std::string &name)
{
    check_uncompiled(__func__);

    // Lookup the function in the module.
    auto mf = m_module->getNamedValue(name);
    if (mf == nullptr) {
        throw std::invalid_argument("The function '" + name + "' does not exist in the module");
    }

    // Try to see if it is a function.
    auto f = llvm::dyn_cast<llvm::Function>(mf);
    if (f == nullptr) {
        throw std::invalid_argument("The symbol '" + name + "' is not a function");
    }

    // Run the actual check.
    verify_function_impl(f);
}

void llvm_state::compile()
{
    check_uncompiled(__func__);

    m_jitter->add_module(std::move(m_module));
}

template <typename T>
void llvm_state::add_varargs_expression(const std::string &name, const expression &e,
                                        const std::vector<std::string> &vars)
{
    // Prepare the function prototype. First the function arguments.
    std::vector<llvm::Type *> fargs(vars.size(), detail::to_llvm_type<T>(context()));
    // Then the return type.
    auto *ft = llvm::FunctionType::get(detail::to_llvm_type<T>(context()), fargs, false);
    assert(ft != nullptr);

    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, m_module.get());
    assert(f != nullptr);
    // Set names for all arguments.
    decltype(vars.size()) idx = 0;
    for (auto &arg : f->args()) {
        arg.setName(vars[idx++]);
    }

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Record the function arguments in the m_named_values map.
    m_named_values.clear();
    for (auto &arg : f->args()) {
        m_named_values[arg.getName()] = &arg;
    }

    // Run the codegen on the expression.
    auto *ret_val = detail::invoke_codegen<T>(*this, e);
    assert(ret_val != nullptr);

    // Finish off the function.
    m_builder->CreateRet(ret_val);

    // Verify it.
    verify_function_impl(f);
}

void llvm_state::add_dbl(const std::string &name, const expression &e)
{
    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);

    add_varargs_expression<double>(name, e, vars);

    // Run the optimization pass.
    if (m_opt_level > 0u) {
        m_pm->run(*m_module);
    }
}

} // namespace heyoka
