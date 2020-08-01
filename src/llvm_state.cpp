// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/CommandFlags.inc>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Vectorize.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

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
struct llvm_state::jit {
    llvm::orc::ExecutionSession m_es;
    llvm::orc::RTDyldObjectLinkingLayer m_object_layer;
    std::unique_ptr<llvm::orc::IRCompileLayer> m_compile_layer;
    std::unique_ptr<llvm::DataLayout> m_dl;
    std::unique_ptr<llvm::Triple> m_triple;
    std::unique_ptr<llvm::TargetMachine> m_tm;
    // NOTE: it seems like in LLVM 11 this class was moved
    // from llvm/ExecutionEngine/Orc/Core.h to
    // llvm/ExecutionEngine/Orc/Mangling.h.
    std::unique_ptr<llvm::orc::MangleAndInterner> m_mangle;
    llvm::orc::ThreadSafeContext m_ctx;
    llvm::orc::JITDylib &m_main_jd;
    std::uint32_t m_vector_size_dbl = 0;
    std::uint32_t m_vector_size_ldbl = 0;
#if defined(HEYOKA_HAVE_REAL128)
    std::uint32_t m_vector_size_f128 = 0;
#endif
    // This is a workaround flag to
    // signal that AVX-512 is available.
    // It is used in the module optimisation
    // function to set specific function attributes.
    bool m_have_avx512 = false;

    jit()
        : m_object_layer(m_es, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
          m_ctx(std::make_unique<llvm::LLVMContext>()), m_main_jd(m_es.createJITDylib("<main>"))
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
        // Set the codegen optimisation level to aggressive.
        jtmb->setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);

        auto dlout = jtmb->getDefaultDataLayoutForTarget();
        if (!dlout) {
            throw std::invalid_argument("Error fetching the default data layout for the host system");
        }

        // Fetch the target triple.
        m_triple = std::make_unique<llvm::Triple>(jtmb->getTargetTriple());

        // Keep a target machine around to fetch various
        // properties of the host CPU.
        auto tm = jtmb->createTargetMachine();
        if (!tm) {
            throw std::invalid_argument("Error creating the target machine");
        }
        m_tm = std::move(*tm);

        m_compile_layer = std::make_unique<llvm::orc::IRCompileLayer>(
            m_es, m_object_layer, std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(*jtmb)));

        m_dl = std::make_unique<llvm::DataLayout>(std::move(*dlout));

        m_mangle = std::make_unique<llvm::orc::MangleAndInterner>(m_es, *m_dl);

        auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(m_dl->getGlobalPrefix());
        if (!dlsg) {
            throw std::invalid_argument("Could not create the dynamic library search generator");
        }

        m_main_jd.addGenerator(std::move(*dlsg));

        // Determine the vector sizes.
        const auto target_name = std::string{m_tm->getTarget().getName()};

        if (target_name == "x86-64") {
            // Look for AVX512 first, then AVX.
            const auto target_features = get_target_features();

            std::string feature = "+avx512f";

            auto it = std::search(target_features.begin(), target_features.end(),
                                  std::boyer_moore_searcher(feature.begin(), feature.end()));

            if (it != target_features.end()) {
                m_vector_size_dbl = 8;

                // Set also the flag signalling that
                // we have AVX-512.
                m_have_avx512 = true;

                return;
            }

            feature = "+avx";

            it = std::search(target_features.begin(), target_features.end(),
                             std::boyer_moore_searcher(feature.begin(), feature.end()));

            if (it != target_features.end()) {
                m_vector_size_dbl = 4;
                return;
            }

            // SSE2 is always available on x86-64.
#if !defined(NDEBUG)
            feature = "+sse2";

            it = std::search(target_features.begin(), target_features.end(),
                             std::boyer_moore_searcher(feature.begin(), feature.end()));

            assert(it != target_features.end());
#endif

            m_vector_size_dbl = 2;
        }
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
    std::string get_target_cpu() const
    {
        return m_tm->getTargetCPU();
    }
    std::string get_target_features() const
    {
        return m_tm->getTargetFeatureString();
    }
    llvm::TargetIRAnalysis get_target_ir_analysis() const
    {
        return m_tm->getTargetIRAnalysis();
    }

    void add_module(std::unique_ptr<llvm::Module> &&m)
    {
        auto handle = m_compile_layer->add(m_main_jd, llvm::orc::ThreadSafeModule(std::move(m), m_ctx));

        if (handle) {
            throw std::invalid_argument("The function for adding a module to the jit failed");
        }
    }

    // Symbol lookup.
    llvm::Expected<llvm::JITEvaluatedSymbol> lookup(const std::string &name)
    {
        return m_es.lookup({&m_main_jd}, (*m_mangle)(name));
    }

    template <typename T>
    std::uint32_t get_vector_size() const
    {
        if constexpr (std::is_same_v<T, double>) {
            return m_vector_size_dbl;
        } else if constexpr (std::is_same_v<T, long double>) {
            return m_vector_size_ldbl;
#if defined(HEYOKA_HAVE_REAL128)
        } else if constexpr (std::is_same_v<T, mppp::real128>) {
            return m_vector_size_f128;
#endif
        } else {
            static_assert(detail::always_false_v<T>, "Unhandled type.");
        }
    }
};

llvm_state::llvm_state(std::tuple<std::string, unsigned, bool> &&tup)
    : m_jitter(std::make_unique<jit>()), m_opt_level(std::get<1>(tup)), m_use_fast_math(std::get<2>(tup))
{
    // Create the module.
    m_module = std::make_unique<llvm::Module>(std::move(std::get<0>(tup)), context());
    // Setup the data layout and the target triple.
    m_module->setDataLayout(*m_jitter->m_dl);
    m_module->setTargetTriple(m_jitter->m_triple->str());

    // Create a new builder for the module.
    m_builder = std::make_unique<llvm::IRBuilder<>>(context());

    if (m_use_fast_math) {
        // Set flags for faster math at the
        // price of potential change of semantics.
        llvm::FastMathFlags fmf;
        fmf.setFast();
        m_builder->setFastMathFlags(fmf);
    }
}

// NOTE: the other kwargs will get the default values
// specified in the implementation function.
llvm_state::llvm_state() : llvm_state(kw::mname = "") {}

llvm_state::llvm_state(const llvm_state &other)
    : m_jitter(std::make_unique<jit>()), m_sig_map(other.m_sig_map), m_opt_level(other.m_opt_level),
      m_use_fast_math(other.m_use_fast_math)
{
    // Get the IR of other.
    auto other_ir = other.dump_ir();

    // Create the corresponding memory buffer.
    auto mb = llvm::MemoryBuffer::getMemBuffer(std::move(other_ir));

    // Construct a new module from the parsed IR.
    llvm::SMDiagnostic err;
    m_module = llvm::parseIR(*mb, err, context());
    if (!m_module) {
        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        err.print("", ostr);

        throw std::invalid_argument("Error parsing the IR while copying an llvm_state. The full error message:\n"
                                    + ostr.str());
    }

    // Create a new builder for the module.
    m_builder = std::make_unique<llvm::IRBuilder<>>(context());

    if (m_use_fast_math) {
        // Set flags for faster math at the
        // price of potential change of semantics.
        llvm::FastMathFlags fmf;
        fmf.setFast();
        m_builder->setFastMathFlags(fmf);
    }

    // Run the compilation if other was compiled.
    if (!other.m_module) {
        compile();
    }
}

llvm_state::llvm_state(llvm_state &&) noexcept = default;

llvm_state &llvm_state::operator=(const llvm_state &other)
{
    if (this != &other) {
        *this = llvm_state(other);
    }

    return *this;
}

llvm_state &llvm_state::operator=(llvm_state &&) noexcept = default;

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

unsigned &llvm_state::opt_level()
{
    return m_opt_level;
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

const unsigned &llvm_state::opt_level() const
{
    return m_opt_level;
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
    if (m_module) {
        throw std::invalid_argument(std::string{"The function '"} + f
                                    + "' can be invoked only after the module has been compiled");
    }
}

void llvm_state::check_add_name(const std::string &name) const
{
    assert(m_module);

    if (name.rfind("heyoka_", 0) == 0) {
        throw std::invalid_argument("Names starting with 'heyoka_' are reserved");
    }

    if (m_module->getNamedValue(name) != nullptr) {
        throw std::invalid_argument("The name '" + name + "' already exists in the module");
    }
}

void llvm_state::verify_function_impl(llvm::Function *f)
{
    assert(f != nullptr);

    std::string err_report;
    llvm::raw_string_ostream ostr(err_report);
    if (llvm::verifyFunction(*f, &ostr) && m_verify) {
        // Remove function before throwing.
        const auto fname = std::string(f->getName());
        f->eraseFromParent();
        throw std::invalid_argument("The verification of the function '" + fname + "' failed. The full error message:\n"
                                    + ostr.str());
    }
}

void llvm_state::verify_function(const std::string &name)
{
    check_uncompiled(__func__);

    // Lookup the function in the module.
    auto f = m_module->getFunction(name);
    if (f == nullptr) {
        throw std::invalid_argument("The function '" + name + "' does not exist in the module");
    }

    // Run the actual check.
    verify_function_impl(f);
}

void llvm_state::optimise()
{
    check_uncompiled(__func__);

    if (m_opt_level > 0u) {
        // NOTE: the logic here largely mimics (with a lot of simplifications)
        // the implementation of the 'opt' tool. See:
        // https://github.com/llvm/llvm-project/blob/release/10.x/llvm/tools/opt/opt.cpp

        // For every function in the module, setup its attributes
        // so that the codegen uses all the features available on
        // the host CPU.
        ::setFunctionAttributes(m_jitter->get_target_cpu(), m_jitter->get_target_features(), *m_module);

        if (m_jitter->m_have_avx512) {
            // NOTE: currently LLVM forces 256-bit vector
            // width when AVX-512 is available, due to clock
            // frequency scaling concerns. It seems like for
            // our purposes 512-bit vectors work fine,
            // thus we force their use via a specific
            // function attribute to be set on all the
            // functions in the module.
            for (auto &f : *m_module) {
                f.addFnAttr("prefer-vector-width", "512");
            }
        }

        // Init the module pass manager.
        auto module_pm = std::make_unique<llvm::legacy::PassManager>();
        // These are passes which set up target-specific info
        // that are used by successive optimisation passes.
        auto tliwp
            = std::make_unique<llvm::TargetLibraryInfoWrapperPass>(llvm::TargetLibraryInfoImpl(*m_jitter->m_triple));
        module_pm->add(tliwp.release());
        module_pm->add(llvm::createTargetTransformInfoWrapperPass(m_jitter->get_target_ir_analysis()));

        // Init the function pass manager.
        auto f_pm = std::make_unique<llvm::legacy::FunctionPassManager>(m_module.get());
        f_pm->add(llvm::createTargetTransformInfoWrapperPass(m_jitter->get_target_ir_analysis()));

        // NOTE: not sure what this does, presumably some target-specifc
        // configuration.
        module_pm->add(static_cast<llvm::LLVMTargetMachine &>(*m_jitter->m_tm).createPassConfig(*module_pm));

        // We use the helper class PassManagerBuilder to populate the module
        // pass manager with standard options.
        llvm::PassManagerBuilder pm_builder;
        // See here for the defaults:
        // https://llvm.org/doxygen/PassManagerBuilder_8cpp_source.html
        pm_builder.OptLevel = m_opt_level;
        pm_builder.SizeLevel = 0;
        pm_builder.Inliner = llvm::createFunctionInliningPass(m_opt_level, 0, false);
        if (m_opt_level >= 3u) {
            pm_builder.SLPVectorize = true;
            pm_builder.MergeFunctions = true;
        }

        m_jitter->m_tm->adjustPassManager(pm_builder);

        // Populate both the function pass manager and the module pass manager.
        pm_builder.populateFunctionPassManager(*f_pm);
        pm_builder.populateModulePassManager(*module_pm);

        // Run the function pass manager on all functions in the module.
        f_pm->doInitialization();
        for (auto &f : *m_module) {
            f_pm->run(f);
        }
        f_pm->doFinalization();

        // Run the module passes.
        module_pm->run(*m_module);
    }
}

void llvm_state::compile()
{
    check_uncompiled(__func__);

    // Store a snapshot of the IR before compiling.
    m_ir_snapshot = dump_ir();

    m_jitter->add_module(std::move(m_module));
}

namespace detail
{

namespace
{

// RAII helper to reset the verify
// flag of an LLVM state to true
// upon destruction.
struct verify_resetter {
    explicit verify_resetter(llvm_state &s) : m_s(s) {}
    ~verify_resetter()
    {
        m_s.verify() = true;
    }
    llvm_state &m_s;
};

} // namespace

} // namespace detail

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
    // NOTE: don't use the same name in vars
    // as it's not clear to me if any name
    // is allowed in the IR. Just use a simple
    // arg_n format.
    decltype(vars.size()) idx = 0;
    for (auto &arg : f->args()) {
        arg.setName("arg_" + detail::li_to_string(idx++));
    }

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Record the function arguments in the m_named_values map.
    idx = 0;
    m_named_values.clear();
    for (auto &arg : f->args()) {
        m_named_values[vars[idx++]] = &arg;
    }

    // Run the codegen on the expression.
    auto *ret_val = codegen<T>(*this, e);
    assert(ret_val != nullptr);

    // Finish off the function.
    m_builder->CreateRet(ret_val);

    // NOTE: it seems like the module-level
    // optimizer is able to figure out on its
    // own at least some useful attributes for
    // functions. Additional attributes
    // (e.g., speculatable, willreturn)
    // will also depend on the attributes
    // of function calls contained in the expression,
    // so it may be tricky to "prove" that they
    // can be added safely.

    // Verify it.
    verify_function_impl(f);

    // Add the function to m_sig_map.
    std::vector<std::type_index> sig_args(vars.size(), std::type_index(typeid(T)));
    auto sig = std::pair{std::type_index(typeid(T)), std::move(sig_args)};
    [[maybe_unused]] const auto eret = m_sig_map.emplace(name, std::move(sig));
    assert(eret.second);
}

void llvm_state::add_nary_function_dbl(const std::string &name, const expression &e)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);

    add_varargs_expression<double>(name, e, vars);

    // Run the optimization pass.
    optimise();
}

void llvm_state::add_nary_function_ldbl(const std::string &name, const expression &e)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);

    add_varargs_expression<long double>(name, e, vars);

    // Run the optimization pass.
    optimise();
}

#if defined(HEYOKA_HAVE_REAL128)

void llvm_state::add_nary_function_f128(const std::string &name, const expression &e)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);

    add_varargs_expression<mppp::real128>(name, e, vars);

    // Run the optimization pass.
    optimise();
}

#endif

template <typename T>
void llvm_state::add_vecargs_expression(const std::string &name, const expression &e)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);
    if (vars.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("The number of variables in the expression passed to add_function() is too "
                                  "large, and it results in an overflow condition");
    }

    // Setup the vecargs function. It takes in input a read-only pointer,
    // and it returns in output the value of the evaluation.
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(detail::to_llvm_type<T>(context()))};
    auto *ft = llvm::FunctionType::get(detail::to_llvm_type<T>(context()), fargs, false);
    assert(ft != nullptr);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, m_module.get());
    assert(f != nullptr);

    // Setup the properties of the pointer argument.
    auto in_ptr = f->args().begin();
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::ReadOnly);
    in_ptr->addAttr(llvm::Attribute::NoCapture);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Fill in the m_named_values map
    // with values loaded from in_ptr.
    m_named_values.clear();
    for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
        [[maybe_unused]] const auto res = m_named_values.emplace(
            vars[i], m_builder->CreateLoad(
                         m_builder->CreateInBoundsGEP(in_ptr, m_builder->getInt32(static_cast<std::uint32_t>(i)),
                                                      "in_ptr_" + detail::li_to_string(i)),
                         "var_" + detail::li_to_string(i)));
        assert(res.second);
    }

    // Create the return value from the codegen of the expression.
    m_builder->CreateRet(codegen<T>(*this, e));

    // Verify the function.
    verify_function_impl(f);

    // Add the function to m_sig_map.
    std::vector<std::type_index> sig_args{std::type_index(typeid(const T *))};
    auto sig = std::pair{std::type_index(typeid(T)), std::move(sig_args)};
    [[maybe_unused]] const auto eret = m_sig_map.emplace(name, std::move(sig));
    assert(eret.second);

    // Run the optimization pass.
    optimise();
}

void llvm_state::add_function_dbl(const std::string &name, const expression &e)
{
    add_vecargs_expression<double>(name, e);
}

void llvm_state::add_function_ldbl(const std::string &name, const expression &e)
{
    add_vecargs_expression<long double>(name, e);
}

#if defined(HEYOKA_HAVE_REAL128)

void llvm_state::add_function_f128(const std::string &name, const expression &e)
{
    add_vecargs_expression<mppp::real128>(name, e);
}

#endif

template <typename T>
void llvm_state::add_vecargs_expressions(const std::string &name, const std::vector<expression> &es)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Build the global list of variables.
    std::vector<std::string> vars;
    for (const auto &e : es) {
        auto e_vars = get_variables(e);

        vars.insert(vars.end(), std::make_move_iterator(e_vars.begin()), std::make_move_iterator(e_vars.end()));
        std::sort(vars.begin(), vars.end());
        vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    }

    if (vars.size() > std::numeric_limits<std::uint32_t>::max()
        || es.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("The number of variables/expressions passed to add_vector_function() is too "
                                  "large, and it results in an overflow condition");
    }

    // Prepare the function prototype.
    std::vector<llvm::Type *> fargs(2u, llvm::PointerType::getUnqual(detail::to_llvm_type<T>(context())));
    auto *ft = llvm::FunctionType::get(m_builder->getVoidTy(), fargs, false);
    assert(ft != nullptr);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, m_module.get());
    assert(f != nullptr);

    // Setup the properties of the pointer arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::WriteOnly);
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto in_ptr = out_ptr + 1;
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::ReadOnly);
    in_ptr->addAttr(llvm::Attribute::NoCapture);
    in_ptr->addAttr(llvm::Attribute::NoAlias);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Fill in the m_named_values map
    // with values loaded from in_ptr.
    m_named_values.clear();
    for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
        [[maybe_unused]] const auto res = m_named_values.emplace(
            vars[i], m_builder->CreateLoad(
                         m_builder->CreateInBoundsGEP(in_ptr, m_builder->getInt32(static_cast<std::uint32_t>(i)),
                                                      "in_ptr_" + detail::li_to_string(i)),
                         "var_" + detail::li_to_string(i)));
        assert(res.second);
    }

    // Run the codegen for each expression and
    // store the result of the evaluation
    // in out_ptr.
    for (decltype(es.size()) i = 0; i < es.size(); ++i) {
        m_builder->CreateStore(codegen<T>(*this, es[i]),
                               m_builder->CreateInBoundsGEP(out_ptr, m_builder->getInt32(static_cast<std::uint32_t>(i)),
                                                            "out_ptr_" + detail::li_to_string(i)));
    }

    // Create the return value.
    m_builder->CreateRetVoid();

    // Verify the function.
    verify_function_impl(f);

    // Add the function to m_sig_map.
    std::vector<std::type_index> sig_args{std::type_index(typeid(T *)), std::type_index(typeid(const T *))};
    auto sig = std::pair{std::type_index(typeid(void)), std::move(sig_args)};
    [[maybe_unused]] const auto eret = m_sig_map.emplace(name, std::move(sig));
    assert(eret.second);

    // Run the optimization pass.
    optimise();
}

void llvm_state::add_vector_function_dbl(const std::string &name, const std::vector<expression> &es)
{
    add_vecargs_expressions<double>(name, es);
}

void llvm_state::add_vector_function_ldbl(const std::string &name, const std::vector<expression> &es)
{
    add_vecargs_expressions<long double>(name, es);
}

#if defined(HEYOKA_HAVE_REAL128)

void llvm_state::add_vector_function_f128(const std::string &name, const std::vector<expression> &es)
{
    add_vecargs_expressions<mppp::real128>(name, es);
}

#endif

template <typename T>
void llvm_state::add_batch_expression_impl(const std::string &name, const expression &e, std::uint32_t batch_size)
{
    if (batch_size == 0u) {
        throw std::invalid_argument("Cannot add an expression in batch mode if the batch size is zero");
    }

    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    // Fetch the sorted list of variables in the expression.
    const auto vars = get_variables(e);
    if (vars.size() > std::numeric_limits<std::uint32_t>::max() / batch_size) {
        throw std::overflow_error("The number of variables in the expression passed to add_function_batch() is too "
                                  "large, and it results in an overflow condition");
    }

    // Setup the batch function. It takes in input a write-only pointer, a read-only pointer,
    // and it returns nothing.
    std::vector<llvm::Type *> fargs(2u, llvm::PointerType::getUnqual(detail::to_llvm_type<T>(context())));
    auto *ft = llvm::FunctionType::get(m_builder->getVoidTy(), fargs, false);
    assert(ft != nullptr);
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, m_module.get());
    assert(f != nullptr);

    // Setup the properties of the pointer arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::WriteOnly);
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto in_ptr = out_ptr + 1;
    in_ptr->setName("in_ptr");
    in_ptr->addAttr(llvm::Attribute::ReadOnly);
    in_ptr->addAttr(llvm::Attribute::NoCapture);
    in_ptr->addAttr(llvm::Attribute::NoAlias);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Clear up the variables mapping.
    m_named_values.clear();
    for (std::uint32_t b_idx = 0; b_idx < batch_size; ++b_idx) {
        // Map the variables to the values corresponding to the
        // current batch.
        for (decltype(vars.size()) i = 0; i < vars.size(); ++i) {
            m_named_values[vars[i]] = m_builder->CreateLoad(m_builder->CreateInBoundsGEP(
                in_ptr, m_builder->getInt32(static_cast<std::uint32_t>(i) * batch_size + b_idx),
                "in_ptr_" + detail::li_to_string(b_idx) + "_" + detail::li_to_string(i)));
        }

        // Do the expression codegen for the current batch, store the result
        // of the evaluation in out_ptr.
        m_builder->CreateStore(codegen<T>(*this, e), m_builder->CreateInBoundsGEP(out_ptr, m_builder->getInt32(b_idx)));
    }

    // Create the return value.
    m_builder->CreateRetVoid();

    // Verify the function.
    verify_function_impl(f);

    // Add the function to m_sig_map.
    std::vector<std::type_index> sig_args{std::type_index(typeid(T *)), std::type_index(typeid(const T *))};
    auto sig = std::pair{std::type_index(typeid(void)), std::move(sig_args)};
    [[maybe_unused]] const auto eret = m_sig_map.emplace(name, std::move(sig));
    assert(eret.second);

    // Run the optimization pass.
    optimise();
}

void llvm_state::add_function_batch_dbl(const std::string &name, const expression &e, std::uint32_t batch_size)
{
    add_batch_expression_impl<double>(name, e, batch_size);
}

void llvm_state::add_function_batch_ldbl(const std::string &name, const expression &e, std::uint32_t batch_size)
{
    add_batch_expression_impl<long double>(name, e, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

void llvm_state::add_function_batch_f128(const std::string &name, const expression &e, std::uint32_t batch_size)
{
    add_batch_expression_impl<mppp::real128>(name, e, batch_size);
}

#endif

// NOTE: this function will lookup symbol names,
// so it does not necessarily return a function
// pointer (could be, e.g., a global variable).
std::uintptr_t llvm_state::jit_lookup(const std::string &name)
{
    check_compiled(__func__);

    auto sym = m_jitter->lookup(name);
    if (!sym) {
        throw std::invalid_argument("Could not find the symbol '" + name + "' in the compiled module");
    }

    return static_cast<std::uintptr_t>((*sym).getAddress());
}

std::string llvm_state::dump_ir() const
{
    if (m_module) {
        // The module has not been compiled yet,
        // get the IR from it.
        std::string out;
        llvm::raw_string_ostream ostr(out);
        m_module->print(ostr, nullptr);
        return ostr.str();
    } else {
        // The module has been compiled.
        // Return the IR snapshot that
        // was created before the compilation.
        return m_ir_snapshot;
    }
}

std::string llvm_state::dump_function_ir(const std::string &name) const
{
    check_uncompiled(__func__);

    if (auto f = m_module->getFunction(name)) {
        std::string out;
        llvm::raw_string_ostream ostr(out);
        f->print(ostr);
        return ostr.str();
    } else {
        throw std::invalid_argument("Could not locate the function called '" + name + "'");
    }
}

void llvm_state::dump_object_code(const std::string &filename) const
{
    check_uncompiled(__func__);

    std::error_code ec;
    llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        throw std::invalid_argument("Could not open the file '" + filename
                                    + "' for dumping object code. The error message is: ec.message()");
    }

    llvm::legacy::PassManager pass;
    auto file_type = llvm::CGFT_ObjectFile;

    if (m_jitter->m_tm->addPassesToEmitFile(pass, dest, nullptr, file_type)) {
        throw std::invalid_argument("The target machine can't emit a file of this type");
    }

    pass.run(*m_module);
}

// Compute the derivative of order "order" of a state variable.
// ex is the formula for the first-order derivative of the state variable (which
// is either a u variable or a number), n_uvars the number of variables in
// the decomposition, diff_arr the array containing the derivatives of all u variables
// up to order - 1, batch_idx and batch_size the batch index and size. vector_size
// is the SIMD width.
template <typename T>
llvm::Value *llvm_state::tjb_compute_sv_diff(const expression &ex, std::uint32_t order, std::uint32_t n_uvars,
                                             llvm::Value *diff_arr, std::uint32_t batch_idx, std::uint32_t batch_size,
                                             std::uint32_t vector_size)
{
    assert(order > 0u);

    return std::visit(
        [&](const auto &v) -> llvm::Value * {
            using type = detail::uncvref_t<decltype(v)>;

            if constexpr (std::is_same_v<type, variable>) {
                // Extract the index of the u variable in the expression
                // of the first-order derivative.
                const auto u_idx = detail::uname_to_index(v.name());

                // Fetch from diff_arr the pointer to the derivative
                // of order order - 1 of the u variable at u_idx. The index is:
                // (order - 1) * n_uvars * batch_size + u_idx * batch_size + batch_idx.
                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr,
                    {m_builder->getInt32(0),
                     m_builder->getInt32((order - 1u) * n_uvars * batch_size + u_idx * batch_size + batch_idx)},
                    "sv_diff_ptr");

                // Load the value, as a scalar or vector.
                auto diff_load = (vector_size == 0u) ? m_builder->CreateLoad(diff_ptr, "sv_diff_load")
                                                     : detail::load_vector_from_memory(*m_builder, diff_ptr,
                                                                                       vector_size, "sv_diff_load");

                // We have to divide the derivative by order
                // to get the normalised derivative of the state variable.
                auto divisor = codegen<T>(*this, number(static_cast<T>(order)));

                if (vector_size > 0u) {
                    divisor = detail::create_constant_vector(*m_builder, divisor, vector_size);
                }

                return m_builder->CreateFDiv(diff_load, divisor, "sv_norm");
            } else if constexpr (std::is_same_v<type, number>) {
                // The first-order derivative is a constant.
                // If the first-order derivative is being requested,
                // do the codegen for the constant itself, otherwise
                // return 0.
                auto ret = (order == 1u) ? codegen<T>(*this, v) : codegen<T>(*this, number{0.});

                if (vector_size > 0u) {
                    ret = detail::create_constant_vector(*m_builder, ret, vector_size);
                }

                return ret;
            } else {
                assert(false);

                return nullptr;
            }
        },
        ex.value());
}

template <typename T, typename U>
auto llvm_state::add_taylor_jet_batch_impl(const std::string &name, U sys, std::uint32_t order,
                                           std::uint32_t batch_size)
{
    detail::verify_resetter vr{*this};

    check_uncompiled(__func__);
    check_add_name(name);

    if (order == 0u) {
        throw std::invalid_argument("The order of a Taylor jet cannot be zero");
    }

    if (batch_size == 0u) {
        throw std::invalid_argument("The batch size of a Taylor jet cannot be zero");
    }

    // Record the number of equations/variables.
    const auto n_eq = sys.size();

    // Decompose the system of equations.
    auto dc = taylor_decompose(std::move(sys));

    // Compute the number of u variables.
    assert(dc.size() > n_eq);
    const auto n_uvars = dc.size() - n_eq;

    // Overflow checking. We want to make sure we can do all computations
    // using uint32_t. We need to be able to:
    // - index into the jet array (size n_eq * (order + 1) * batch_size),
    // - index into the internal derivatives array (size n_uvars * order * batch_size).
    // NOTE: even though some automatic differentiation formulae have
    // sums up to i = order (and thus could formally overflow in a
    // for loop), we invoke them only up to order = order - 1.
    if (order == std::numeric_limits<std::uint32_t>::max()
        || (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || n_eq > std::numeric_limits<std::uint32_t>::max() / ((order + 1u) * batch_size)
        || n_uvars > std::numeric_limits<std::uint32_t>::max() / (order * batch_size)) {
        throw std::overflow_error(
            "An overflow condition was detected in the number of variables while adding a Taylor jet");
    }

    // Prepare the main function prototype. The only argument is a float pointer to in/out array.
    std::vector<llvm::Type *> fargs{llvm::PointerType::getUnqual(detail::to_llvm_type<T>(context()))};
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(m_builder->getVoidTy(), fargs, false);
    assert(ft != nullptr);
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, name, m_module.get());
    assert(f != nullptr);

    // Set the name of the function argument.
    auto in_out = f->args().begin();
    in_out->setName("in_out");

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context(), "entry", f);
    assert(bb != nullptr);
    m_builder->SetInsertPoint(bb);

    // Create the array of derivatives for the u variables.
    // NOTE: by allocating order rows, we are able to store derivatives
    // up to order - 1 (rather than order), because we start from order
    // 0. This is ok, because in this function we will be reading/writing from/to
    // the derivatives array only up to order - 1: the last step involves only the
    // derivatives of the state variables, which access only values at order - 1.
    auto array_type = llvm::ArrayType::get(detail::to_llvm_type<T>(context()),
                                           static_cast<std::uint64_t>(n_uvars * order * batch_size));
    assert(array_type != nullptr);
    auto diff_arr = m_builder->CreateAlloca(array_type, 0, "diff_arr");
    assert(diff_arr != nullptr);

    // Fill-in the order-0 row of the derivatives array.
    // Use a separate block for clarity.
    auto *init_bb = llvm::BasicBlock::Create(context(), "order_0_init", f);
    assert(init_bb != nullptr);
    m_builder->CreateBr(init_bb);
    m_builder->SetInsertPoint(init_bb);

    // Fetch the SIMD vector size from the JIT machinery.
    const auto vector_size = m_jitter->get_vector_size<T>();

    // Load the initial values for the state variables from in_out.
    for (std::uint32_t i = 0; i < n_eq; ++i) {
        if (vector_size == 0u) {
            // Scalar mode.

            // NOTE: do first all the loads, then all the stores.
            std::vector<llvm::Value *> values;

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                const auto arr_idx = i * batch_size + batch_idx;

                // Fetch the input pointer from in_out.
                auto in_ptr = m_builder->CreateInBoundsGEP(in_out, {m_builder->getInt32(arr_idx)},
                                                           "o0_init_ptr_" + detail::li_to_string(i) + "_"
                                                               + detail::li_to_string(batch_idx));
                assert(in_ptr != nullptr);

                // Create the load instruction from in_out.
                auto load_inst = m_builder->CreateLoad(in_ptr, "o0_init_load_" + detail::li_to_string(i) + "_"
                                                                   + detail::li_to_string(batch_idx));
                assert(load_inst != nullptr);

                values.push_back(load_inst);
            }

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                const auto arr_idx = i * batch_size + batch_idx;

                // Fetch the target pointer in diff_arr.
                auto diff_ptr = m_builder->CreateInBoundsGEP(diff_arr,
                                                             // The offsets. The first is fixed because
                                                             // diff_arr is an alloca
                                                             // and thus we need to deref it. The second
                                                             // offset is the index into the array.
                                                             {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                                                             // Name for the pointer variable.
                                                             "o0_diff_ptr_" + detail::li_to_string(i) + "_"
                                                                 + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                // Do the copy.
                m_builder->CreateStore(values[batch_idx], diff_ptr);
            }
        } else {
            // Vector mode.
            const auto n_sub_batch = batch_size / vector_size;

            for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                const auto arr_idx = i * batch_size + batch_idx;

                auto in_ptr = m_builder->CreateInBoundsGEP(in_out, {m_builder->getInt32(arr_idx)},
                                                           "o0_init_ptr_" + detail::li_to_string(i) + "_"
                                                               + detail::li_to_string(batch_idx));
                assert(in_ptr != nullptr);

                auto vec = detail::load_vector_from_memory(*m_builder, in_ptr, vector_size,
                                                           "o0_init_load_" + detail::li_to_string(i) + "_"
                                                               + detail::li_to_string(batch_idx));
                assert(vec != nullptr);

                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr, {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                    "o0_diff_ptr_" + detail::li_to_string(i) + "_" + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                detail::store_vector_to_memory(*m_builder, diff_ptr, vec, vector_size);
            }

            // NOTE: this remainder loop could be interleaved in the same way as the scalar computation
            // above. This may help the SLP vectorizer, but it is not clear at this time.
            for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                const auto arr_idx = i * batch_size + batch_idx;

                auto in_ptr = m_builder->CreateInBoundsGEP(in_out, {m_builder->getInt32(arr_idx)},
                                                           "o0_init_ptr_" + detail::li_to_string(i) + "_"
                                                               + detail::li_to_string(batch_idx));
                assert(in_ptr != nullptr);

                auto load_inst = m_builder->CreateLoad(in_ptr, "o0_init_load_" + detail::li_to_string(i) + "_"
                                                                   + detail::li_to_string(batch_idx));
                assert(load_inst != nullptr);

                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr, {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                    "o0_diff_ptr_" + detail::li_to_string(i) + "_" + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                m_builder->CreateStore(load_inst, diff_ptr);
            }
        }
    }

    // Fill in the initial values for the other u vars in the diff array.
    // These are not loaded directly from in_out, rather they are computed
    // via the taylor_init_batch machinery.
    for (auto i = n_eq; i < n_uvars; ++i) {
        const auto &u_ex = dc[i];

        if (vector_size == 0u) {
            // Scalar mode.

            // NOTE: do first all the initialisations, then all the stores.
            std::vector<llvm::Value *> values;

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // Run the initialisation.
                values.push_back(taylor_init_batch<T>(*this, u_ex, diff_arr, batch_idx, batch_size, 0));
            }

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                const auto arr_idx = static_cast<std::uint32_t>(i * batch_size + batch_idx);

                // Fetch the target pointer in diff_arr.
                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr, {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                    "o0_diff_ptr_" + detail::li_to_string(i) + "_" + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                // Store the result of the initialisation.
                m_builder->CreateStore(values[batch_idx], diff_ptr);
            }
        } else {
            // Vector mode.
            const auto n_sub_batch = batch_size / vector_size;

            for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                const auto arr_idx = static_cast<std::uint32_t>(i * batch_size + batch_idx);

                // Run the initialisation.
                auto init = taylor_init_batch<T>(*this, u_ex, diff_arr, batch_idx, batch_size, vector_size);

                // Fetch the target pointer in diff_arr.
                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr, {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                    "o0_diff_ptr_" + detail::li_to_string(i) + "_" + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                // Store the result of the initialisation.
                detail::store_vector_to_memory(*m_builder, diff_ptr, init, vector_size);
            }

            for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                const auto arr_idx = static_cast<std::uint32_t>(i * batch_size + batch_idx);

                auto init = taylor_init_batch<T>(*this, u_ex, diff_arr, batch_idx, batch_size, 0);

                auto diff_ptr = m_builder->CreateInBoundsGEP(
                    diff_arr, {m_builder->getInt32(0), m_builder->getInt32(arr_idx)},
                    "o0_diff_ptr_" + detail::li_to_string(i) + "_" + detail::li_to_string(batch_idx));
                assert(diff_ptr != nullptr);

                m_builder->CreateStore(init, diff_ptr);
            }
        }
    }

    // Establish if there are state variables whose derivatives are constants.
    // NOTE: the derivatives of the state variables
    // are at the end of the decomposition vector.
    std::unordered_map<std::uint32_t, number> cd_uvars;
    for (auto i = n_uvars; i < dc.size(); ++i) {
        std::visit(
            [&cd_uvars, i, n_uvars](const auto &v) {
                using type = detail::uncvref_t<decltype(v)>;

                if constexpr (std::is_same_v<type, number>) {
                    [[maybe_unused]] const auto res = cd_uvars.emplace(static_cast<std::uint32_t>(i - n_uvars), v);
                    assert(res.second);
                } else if constexpr (!std::is_same_v<type, variable>) {
                    // NOTE: the derivative of a state variable
                    // can only be a u variable or a number.
                    assert(false);
                }
            },
            dc[i].value());
    }

    // Compute the derivatives order by order, starting from 1 to order excluded.
    // We will compute the highest derivatives of the state variables separately
    // in the last step.
    for (std::uint32_t cur_order = 1; cur_order < order; ++cur_order) {
        // Begin with the state variables.
        // NOTE: the derivatives of the state variables
        // are at the end of the decomposition vector.
        for (auto i = n_uvars; i < dc.size(); ++i) {
            // The index of the state variable whose
            // derivative we are computing.
            const auto sv_idx = static_cast<std::uint32_t>(i - n_uvars);
            // The expression of the first-order derivative.
            const auto &ex = dc[i];

            // Place the computation in its own block for clarity.
            auto *cur_bb = llvm::BasicBlock::Create(
                context(), "block_" + detail::li_to_string(cur_order) + "_" + detail::li_to_string(sv_idx), f);
            assert(cur_bb != nullptr);
            m_builder->CreateBr(cur_bb);
            m_builder->SetInsertPoint(cur_bb);

            if (vector_size == 0u) {
                // Scalar mode.

                // NOTE: do first all the computations, then all the stores.
                std::vector<llvm::Value *> diff_values;
                for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    // Compute the derivative.
                    diff_values.push_back(tjb_compute_sv_diff<T>(ex, cur_order, static_cast<std::uint32_t>(n_uvars),
                                                                 diff_arr, batch_idx, batch_size, 0));
                }

                // Store the values from diff_values into diff_arr and in_out.
                for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    // The index in diff_arr is:
                    // cur_order * n_uvars * batch_size + sv_idx * batch_size + batch_idx.
                    m_builder->CreateStore(diff_values[batch_idx],
                                           m_builder->CreateInBoundsGEP(
                                               diff_arr,
                                               {m_builder->getInt32(0), m_builder->getInt32(static_cast<std::uint32_t>(
                                                                            cur_order * n_uvars * batch_size
                                                                            + sv_idx * batch_size + batch_idx))},
                                               "sv_" + detail::li_to_string(sv_idx) + "_diff_ptr"));

                    // The index in in_out is:
                    // cur_order * n_eq * batch_size + sv_idx * batch_size + batch_idx.
                    m_builder->CreateStore(diff_values[batch_idx],
                                           m_builder->CreateInBoundsGEP(
                                               in_out,
                                               {m_builder->getInt32(static_cast<std::uint32_t>(
                                                   cur_order * n_eq * batch_size + sv_idx * batch_size + batch_idx))},
                                               "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"));
                }
            } else {
                // Vector mode.
                const auto n_sub_batch = batch_size / vector_size;

                for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                    auto diff_val = tjb_compute_sv_diff<T>(ex, cur_order, static_cast<std::uint32_t>(n_uvars), diff_arr,
                                                           batch_idx, batch_size, vector_size);

                    detail::store_vector_to_memory(
                        *m_builder,
                        m_builder->CreateInBoundsGEP(
                            diff_arr,
                            {m_builder->getInt32(0),
                             m_builder->getInt32(static_cast<std::uint32_t>(cur_order * n_uvars * batch_size
                                                                            + sv_idx * batch_size + batch_idx))},
                            "sv_" + detail::li_to_string(sv_idx) + "_diff_ptr"),
                        diff_val, vector_size);

                    detail::store_vector_to_memory(
                        *m_builder,
                        m_builder->CreateInBoundsGEP(
                            in_out,
                            {m_builder->getInt32(static_cast<std::uint32_t>(cur_order * n_eq * batch_size
                                                                            + sv_idx * batch_size + batch_idx))},
                            "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"),
                        diff_val, vector_size);
                }

                for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                    auto diff = tjb_compute_sv_diff<T>(ex, cur_order, static_cast<std::uint32_t>(n_uvars), diff_arr,
                                                       batch_idx, batch_size, 0);

                    m_builder->CreateStore(
                        diff, m_builder->CreateInBoundsGEP(
                                  diff_arr,
                                  {m_builder->getInt32(0),
                                   m_builder->getInt32(static_cast<std::uint32_t>(cur_order * n_uvars * batch_size
                                                                                  + sv_idx * batch_size + batch_idx))},
                                  "sv_" + detail::li_to_string(sv_idx) + "_diff_ptr"));

                    m_builder->CreateStore(diff,
                                           m_builder->CreateInBoundsGEP(
                                               in_out,
                                               {m_builder->getInt32(static_cast<std::uint32_t>(
                                                   cur_order * n_eq * batch_size + sv_idx * batch_size + batch_idx))},
                                               "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"));
                }
            }
        }

        // Now the other u variables.
        for (auto i = n_eq; i < n_uvars; ++i) {
            const auto &ex = dc[i];

            auto *cur_bb = llvm::BasicBlock::Create(
                context(), "block_" + detail::li_to_string(cur_order) + "_" + detail::li_to_string(i), f);
            assert(cur_bb != nullptr);
            m_builder->CreateBr(cur_bb);
            m_builder->SetInsertPoint(cur_bb);

            if (vector_size == 0u) {
                // Scalar mode.

                // NOTE: do first all the computations, then all the stores.
                std::vector<llvm::Value *> diff_values;
                for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    diff_values.push_back(taylor_diff_batch<T>(*this, ex, static_cast<std::uint32_t>(i), cur_order,
                                                               static_cast<std::uint32_t>(n_uvars), diff_arr, batch_idx,
                                                               batch_size, 0, cd_uvars));
                }

                for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    m_builder->CreateStore(diff_values[batch_idx],
                                           m_builder->CreateInBoundsGEP(
                                               diff_arr,
                                               {m_builder->getInt32(0),
                                                m_builder->getInt32(static_cast<std::uint32_t>(
                                                    cur_order * n_uvars * batch_size + i * batch_size + batch_idx))},
                                               "uv_" + detail::li_to_string(i) + "_diff_ptr"));
                }
            } else {
                // Vector mode.
                const auto n_sub_batch = batch_size / vector_size;

                for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                    auto diff_v = taylor_diff_batch<T>(*this, ex, static_cast<std::uint32_t>(i), cur_order,
                                                       static_cast<std::uint32_t>(n_uvars), diff_arr, batch_idx,
                                                       batch_size, vector_size, cd_uvars);

                    detail::store_vector_to_memory(
                        *m_builder,
                        m_builder->CreateInBoundsGEP(
                            diff_arr,
                            {m_builder->getInt32(0),
                             m_builder->getInt32(static_cast<std::uint32_t>(cur_order * n_uvars * batch_size
                                                                            + i * batch_size + batch_idx))},
                            "uv_" + detail::li_to_string(i) + "_diff_ptr"),
                        diff_v, vector_size);
                }

                for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                    auto diff = taylor_diff_batch<T>(*this, ex, static_cast<std::uint32_t>(i), cur_order,
                                                     static_cast<std::uint32_t>(n_uvars), diff_arr, batch_idx,
                                                     batch_size, 0, cd_uvars);
                    m_builder->CreateStore(diff,
                                           m_builder->CreateInBoundsGEP(
                                               diff_arr,
                                               {m_builder->getInt32(0),
                                                m_builder->getInt32(static_cast<std::uint32_t>(
                                                    cur_order * n_uvars * batch_size + i * batch_size + batch_idx))},
                                               "uv_" + detail::li_to_string(i) + "_diff_ptr"));
                }
            }
        }
    }

    auto *final_bb = llvm::BasicBlock::Create(context(), "finalise", f);
    assert(final_bb != nullptr);
    m_builder->CreateBr(final_bb);
    m_builder->SetInsertPoint(final_bb);

    // The last step is to write the highest-order derivatives to in_out.
    for (auto i = n_uvars; i < dc.size(); ++i) {
        const auto sv_idx = static_cast<std::uint32_t>(i - n_uvars);
        const auto &ex = dc[i];

        if (vector_size == 0u) {
            // Scalar mode.

            // NOTE: do first all the computations, then all the stores.
            std::vector<llvm::Value *> diff_values;

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // Compute the derivative.
                diff_values.push_back(tjb_compute_sv_diff<T>(ex, order, static_cast<std::uint32_t>(n_uvars), diff_arr,
                                                             batch_idx, batch_size, 0));
            }

            for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                // Store the result in in_out.
                // The in_out index into which we need to write is
                // order * n_eq * batch_size + sv_idx * batch_size + batch_idx.
                m_builder->CreateStore(
                    diff_values[batch_idx],
                    m_builder->CreateInBoundsGEP(in_out,
                                                 {m_builder->getInt32(static_cast<std::uint32_t>(
                                                     order * n_eq * batch_size + sv_idx * batch_size + batch_idx))},
                                                 "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"));
            }
        } else {
            const auto n_sub_batch = batch_size / vector_size;

            for (std::uint32_t batch_idx = 0; batch_idx < n_sub_batch * vector_size; batch_idx += vector_size) {
                auto diff = tjb_compute_sv_diff<T>(ex, order, static_cast<std::uint32_t>(n_uvars), diff_arr, batch_idx,
                                                   batch_size, vector_size);

                detail::store_vector_to_memory(
                    *m_builder,
                    m_builder->CreateInBoundsGEP(in_out,
                                                 {m_builder->getInt32(static_cast<std::uint32_t>(
                                                     order * n_eq * batch_size + sv_idx * batch_size + batch_idx))},
                                                 "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"),
                    diff, vector_size);
            }

            for (std::uint32_t batch_idx = n_sub_batch * vector_size; batch_idx < batch_size; ++batch_idx) {
                auto diff = tjb_compute_sv_diff<T>(ex, order, static_cast<std::uint32_t>(n_uvars), diff_arr, batch_idx,
                                                   batch_size, 0);

                m_builder->CreateStore(diff, m_builder->CreateInBoundsGEP(
                                                 in_out,
                                                 {m_builder->getInt32(static_cast<std::uint32_t>(
                                                     order * n_eq * batch_size + sv_idx * batch_size + batch_idx))},
                                                 "sv_" + detail::li_to_string(sv_idx) + "_in_out_ptr"));
            }
        }
    }

    // Finish off the function.
    m_builder->CreateRetVoid();

    // Verify it.
    verify_function_impl(f);

    // Add the function to m_sig_map. The signature is void(T *).
    std::vector<std::type_index> sig_args{std::type_index(typeid(T *))};
    auto sig = std::pair{std::type_index(typeid(void)), std::move(sig_args)};
    [[maybe_unused]] const auto eret = m_sig_map.emplace(name, std::move(sig));
    assert(eret.second);

    // Run the optimization pass.
    optimise();

    return dc;
}

std::vector<expression> llvm_state::add_taylor_jet_batch_dbl(const std::string &name,
                                                             std::vector<std::pair<expression, expression>> sys,
                                                             std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<double>(name, std::move(sys), order, batch_size);
}

std::vector<expression> llvm_state::add_taylor_jet_batch_ldbl(const std::string &name,
                                                              std::vector<std::pair<expression, expression>> sys,
                                                              std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<long double>(name, std::move(sys), order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> llvm_state::add_taylor_jet_batch_f128(const std::string &name,
                                                              std::vector<std::pair<expression, expression>> sys,
                                                              std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<mppp::real128>(name, std::move(sys), order, batch_size);
}

#endif

std::vector<expression> llvm_state::add_taylor_jet_batch_dbl(const std::string &name, std::vector<expression> sys,
                                                             std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<double>(name, std::move(sys), order, batch_size);
}

std::vector<expression> llvm_state::add_taylor_jet_batch_ldbl(const std::string &name, std::vector<expression> sys,
                                                              std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<long double>(name, std::move(sys), order, batch_size);
}

#if defined(HEYOKA_HAVE_REAL128)

std::vector<expression> llvm_state::add_taylor_jet_batch_f128(const std::string &name, std::vector<expression> sys,
                                                              std::uint32_t order, std::uint32_t batch_size)
{
    return add_taylor_jet_batch_impl<mppp::real128>(name, std::move(sys), order, batch_size);
}

#endif

// NOTE: in the fetch_* functions, check_compiled() is run
// by jit_lookup().
llvm_state::sf_t<double> llvm_state::fetch_function_dbl(const std::string &name)
{
    return fetch_function<double>(name);
}

llvm_state::sf_t<long double> llvm_state::fetch_function_ldbl(const std::string &name)
{
    return fetch_function<long double>(name);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm_state::sf_t<mppp::real128> llvm_state::fetch_function_f128(const std::string &name)
{
    return fetch_function<mppp::real128>(name);
}

#endif

llvm_state::vf_t<double> llvm_state::fetch_vector_function_dbl(const std::string &name)
{
    return fetch_vector_function<double>(name);
}

llvm_state::vf_t<long double> llvm_state::fetch_vector_function_ldbl(const std::string &name)
{
    return fetch_vector_function<long double>(name);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm_state::vf_t<mppp::real128> llvm_state::fetch_vector_function_f128(const std::string &name)
{
    return fetch_vector_function<mppp::real128>(name);
}

#endif

llvm_state::sfb_t<double> llvm_state::fetch_function_batch_dbl(const std::string &name)
{
    return fetch_function_batch<double>(name);
}

llvm_state::sfb_t<long double> llvm_state::fetch_function_batch_ldbl(const std::string &name)
{
    return fetch_function_batch<long double>(name);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm_state::sfb_t<mppp::real128> llvm_state::fetch_function_batch_f128(const std::string &name)
{
    return fetch_function_batch<mppp::real128>(name);
}

#endif

llvm_state::tjb_t<double> llvm_state::fetch_taylor_jet_batch_dbl(const std::string &name)
{
    return fetch_taylor_jet_batch<double>(name);
}

llvm_state::tjb_t<long double> llvm_state::fetch_taylor_jet_batch_ldbl(const std::string &name)
{
    return fetch_taylor_jet_batch<long double>(name);
}

#if defined(HEYOKA_HAVE_REAL128)

llvm_state::tjb_t<mppp::real128> llvm_state::fetch_taylor_jet_batch_f128(const std::string &name)
{
    return fetch_taylor_jet_batch<mppp::real128>(name);
}

#endif

std::uint32_t llvm_state::vector_size_dbl() const
{
    return m_jitter->m_vector_size_dbl;
}

std::uint32_t llvm_state::vector_size_ldbl() const
{
    return m_jitter->m_vector_size_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

std::uint32_t llvm_state::vector_size_f128() const
{
    return m_jitter->m_vector_size_f128;
}

#endif

} // namespace heyoka
