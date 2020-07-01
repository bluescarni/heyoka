// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

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
// #include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
// #include <llvm/IR/DerivedTypes.h>
// #include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
// #include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
// #include <llvm/IR/Type.h>
// #include <llvm/IR/Value.h>
// #include <llvm/IR/Verifier.h>
// #include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
// #include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Vectorize.h>

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

llvm::LLVMContext &llvm_state::context()
{
    return m_jitter->get_context();
}

bool &llvm_state::verify()
{
    return m_verify;
}

const llvm::LLVMContext &llvm_state::context() const
{
    return m_jitter->get_context();
}

const bool &llvm_state::verify() const
{
    return m_verify;
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

void llvm_state::compile()
{
    check_uncompiled(__func__);

    m_jitter->add_module(std::move(m_module));
}

} // namespace heyoka
