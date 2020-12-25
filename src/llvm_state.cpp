// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Triple.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Pass.h>
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
#include <llvm/Transforms/Vectorize.h>

#if LLVM_VERSION_MAJOR == 10

#include <llvm/CodeGen/CommandFlags.inc>

#endif

#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/variable.hpp>

namespace heyoka
{

namespace detail
{

namespace
{

// Make sure our definition of ir_builder matches llvm::IRBuilder<>.
static_assert(std::is_same_v<ir_builder, llvm::IRBuilder<>>, "Inconsistent definition of the ir_builder type.");

// Helper function to detect specific features
// on the host machine via LLVM's machinery.
target_features get_target_features_impl()
{
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) {
        throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
    }

    auto tm = jtmb->createTargetMachine();
    if (!tm) {
        throw std::invalid_argument("Error creating the target machine");
    }

    target_features retval;

    const auto target_name = std::string{(*tm)->getTarget().getName()};

    if (target_name == "x86-64" || target_name == "x86") {
        const auto t_features = (*tm)->getTargetFeatureString();

        if (boost::algorithm::contains(t_features, "+avx512f")) {
            retval.avx512f = true;
        }

        if (boost::algorithm::contains(t_features, "+avx2")) {
            retval.avx2 = true;
        }

        if (boost::algorithm::contains(t_features, "+avx")) {
            retval.avx = true;
        }

        // SSE2 is always available on x86-64.
        assert(boost::algorithm::contains(t_features, "+sse2"));
        retval.sse2 = true;
    }

    return retval;
}

} // namespace

// Helper function to fetch a const ref to a global object
// containing info about the host machine.
const target_features &get_target_features()
{
    static const target_features retval{get_target_features_impl()};

    return retval;
}

namespace
{

std::once_flag nt_inited;

} // namespace

} // namespace detail

// Implementation of the jit class.
struct llvm_state::jit {
    std::unique_ptr<llvm::orc::LLJIT> m_lljit;
    std::unique_ptr<llvm::TargetMachine> m_tm;
    std::unique_ptr<llvm::orc::ThreadSafeContext> m_ctx;
#if LLVM_VERSION_MAJOR == 10
    std::unique_ptr<llvm::Triple> m_triple;
#endif

    jit()
    {
        // NOTE: the native target initialization needs to be done only once
        std::call_once(detail::nt_inited, []() {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
            llvm::InitializeNativeTargetAsmParser();
        });

        // Create the target machine builder.
        auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!jtmb) {
            throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
        }
        // Set the codegen optimisation level to aggressive.
        jtmb->setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);

        // Create the jit builder.
        llvm::orc::LLJITBuilder lljit_builder;
        // NOTE: other settable properties may
        // be of interest:
        // https://www.llvm.org/doxygen/classllvm_1_1orc_1_1LLJITBuilder.html
        lljit_builder.setJITTargetMachineBuilder(*jtmb);

        // Create the jit.
        auto lljit = lljit_builder.create();
        if (!lljit) {
            throw std::invalid_argument("Error creating an LLJIT object");
        }
        m_lljit = std::move(*lljit);

        // Setup the jit so that it can look up symbols from the current process.
        auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            m_lljit->getDataLayout().getGlobalPrefix());
        if (!dlsg) {
            throw std::invalid_argument("Could not create the dynamic library search generator");
        }
        m_lljit->getMainJITDylib().addGenerator(std::move(*dlsg));

        // Keep a target machine around to fetch various
        // properties of the host CPU.
        auto tm = jtmb->createTargetMachine();
        if (!tm) {
            throw std::invalid_argument("Error creating the target machine");
        }
        m_tm = std::move(*tm);

        // Create the context.
        m_ctx = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());

#if LLVM_VERSION_MAJOR == 10
        // NOTE: on LLVM 10, we cannot fetch the target triple
        // from the lljit class. Thus, we get it from the jtmb instead.
        m_triple = std::make_unique<llvm::Triple>(jtmb->getTargetTriple());
#endif

        // NOTE: by default, errors in the execution session are printed
        // to screen. A custom error reported can be specified, ideally
        // we would like th throw here but I am not sure whether throwing
        // here would disrupt LLVM's cleanup actions?
        // https://llvm.org/doxygen/classllvm_1_1orc_1_1ExecutionSession.html
    }

    jit(const jit &) = delete;
    jit(jit &&) = delete;
    jit &operator=(const jit &) = delete;
    jit &operator=(jit &&) = delete;

    ~jit() = default;

    // Accessors.
    llvm::LLVMContext &get_context()
    {
        return *m_ctx->getContext();
    }
    const llvm::LLVMContext &get_context() const
    {
        return *m_ctx->getContext();
    }
    std::string get_target_cpu() const
    {
        return m_tm->getTargetCPU().str();
    }
    std::string get_target_features() const
    {
        return m_tm->getTargetFeatureString().str();
    }
    llvm::TargetIRAnalysis get_target_ir_analysis() const
    {
        return m_tm->getTargetIRAnalysis();
    }
    const llvm::Triple &get_target_triple() const
    {
#if LLVM_VERSION_MAJOR == 10
        return *m_triple;
#else
        return m_lljit->getTargetTriple();
#endif
    }

    void add_module(std::unique_ptr<llvm::Module> &&m)
    {
        auto err = m_lljit->addIRModule(llvm::orc::ThreadSafeModule(std::move(m), *m_ctx));

        if (err) {
            std::string err_report;
            llvm::raw_string_ostream ostr(err_report);

            ostr << err;

            throw std::invalid_argument("The function for adding a module to the jit failed. The full error message:\n"
                                        + ostr.str());
        }
    }

    // Symbol lookup.
    llvm::Expected<llvm::JITEvaluatedSymbol> lookup(const std::string &name)
    {
        return m_lljit->lookup(name);
    }
};

llvm_state::llvm_state(std::tuple<std::string, unsigned, bool, bool, bool> &&tup)
    : m_jitter(std::make_unique<jit>()), m_opt_level(std::get<1>(tup)), m_fast_math(std::get<2>(tup)),
      m_module_name(std::move(std::get<0>(tup))), m_save_object_code(std::get<3>(tup)),
      m_inline_functions(std::get<4>(tup))
{
    // Create the module.
    m_module = std::make_unique<llvm::Module>(m_module_name, context());
    // Setup the data layout and the target triple.
    m_module->setDataLayout(m_jitter->m_lljit->getDataLayout());
    m_module->setTargetTriple(m_jitter->get_target_triple().str());

    // Create a new builder for the module.
    m_builder = std::make_unique<ir_builder>(context());

    if (m_fast_math) {
        // Set flags for faster math at the
        // price of potential change of semantics.
        llvm::FastMathFlags fmf;
        fmf.setFast();
        m_builder->setFastMathFlags(fmf);
    } else {
        // By default, allow only fp contraction.
        // NOTE: if we ever implement double-double
        // arithmetic, we must either revisit this
        // or make sure that fp contraction is off
        // for the double-double primitives.
        llvm::FastMathFlags fmf;
        fmf.setAllowContract();
        m_builder->setFastMathFlags(fmf);
    }
}

// NOTE: this will ensure that all kwargs
// are set to their default values.
llvm_state::llvm_state() : llvm_state(kw_args_ctor_impl()) {}

llvm_state::llvm_state(const llvm_state &other)
    : m_jitter(std::make_unique<jit>()), m_opt_level(other.m_opt_level), m_fast_math(other.m_fast_math),
      m_module_name(other.m_module_name), m_save_object_code(other.m_save_object_code),
      m_object_code(other.m_object_code), m_inline_functions(other.m_inline_functions)
{
    // Get the IR of other.
    auto other_ir = other.get_ir();

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
    m_builder = std::make_unique<ir_builder>(context());

    if (m_fast_math) {
        // Set flags for faster math at the
        // price of potential change of semantics.
        llvm::FastMathFlags fmf;
        fmf.setFast();
        m_builder->setFastMathFlags(fmf);
    } else {
        // By default, allow only fp contraction.
        // NOTE: if we ever implement double-double
        // arithmetic, we must either revisit this
        // or make sure that fp contraction is off
        // for the double-double primitives.
        llvm::FastMathFlags fmf;
        fmf.setAllowContract();
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

ir_builder &llvm_state::builder()
{
    check_uncompiled(__func__);
    return *m_builder;
}

llvm::LLVMContext &llvm_state::context()
{
    return m_jitter->get_context();
}

unsigned &llvm_state::opt_level()
{
    return m_opt_level;
}

bool &llvm_state::fast_math()
{
    return m_fast_math;
}

bool &llvm_state::inline_functions()
{
    return m_inline_functions;
}

const llvm::Module &llvm_state::module() const
{
    check_uncompiled(__func__);
    return *m_module;
}

const ir_builder &llvm_state::builder() const
{
    check_uncompiled(__func__);
    return *m_builder;
}

const llvm::LLVMContext &llvm_state::context() const
{
    return m_jitter->get_context();
}

const unsigned &llvm_state::opt_level() const
{
    return m_opt_level;
}

const bool &llvm_state::fast_math() const
{
    return m_fast_math;
}

const bool &llvm_state::inline_functions() const
{
    return m_inline_functions;
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

void llvm_state::verify_function(llvm::Function *f)
{
    check_uncompiled(__func__);

    if (f == nullptr) {
        throw std::invalid_argument("Cannot verify a null function pointer");
    }

    std::string err_report;
    llvm::raw_string_ostream ostr(err_report);
    if (llvm::verifyFunction(*f, &ostr)) {
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
    verify_function(f);
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
#if LLVM_VERSION_MAJOR == 10
        ::setFunctionAttributes(m_jitter->get_target_cpu(), m_jitter->get_target_features(), *m_module);
#else
        // NOTE: in LLVM > 10, the setFunctionAttributes() function is gone in favour of another
        // function in another namespace, which however does not seem to work out of the box
        // because (I think) it might be reading some non-existent command-line options. See:
        // https://llvm.org/doxygen/CommandFlags_8cpp_source.html#l00552
        // Here we are reproducing a trimmed-down version of the same function.
        const auto cpu = m_jitter->get_target_cpu();
        const auto features = m_jitter->get_target_features();

        for (auto &f : module()) {
            auto attrs = f.getAttributes();
            llvm::AttrBuilder new_attrs;

            if (!cpu.empty() && !f.hasFnAttribute("target-cpu")) {
                new_attrs.addAttribute("target-cpu", cpu);
            }

            if (!features.empty()) {
                auto old_features = f.getFnAttribute("target-features").getValueAsString();

                if (old_features.empty()) {
                    new_attrs.addAttribute("target-features", features);
                } else {
                    llvm::SmallString<256> appended(old_features);
                    appended.push_back(',');
                    appended.append(features);
                    new_attrs.addAttribute("target-features", appended);
                }
            }

            f.setAttributes(attrs.addAttributes(context(), llvm::AttributeList::FunctionIndex, new_attrs));
        }
#endif

        // NOTE: currently LLVM forces 256-bit vector
        // width when AVX-512 is available, due to clock
        // frequency scaling concerns. We used to have the following
        // code here:
        // for (auto &f : *m_module) {
        //     f.addFnAttr("prefer-vector-width", "512");
        // }
        // in order to force 512-bit vector width, but it looks
        // like this can hurt performance in scalar mode.
        // Let's keep this in mind for the future, perhaps
        // we could consider enabling 512-bit vector width
        // only in batch mode?

        // Init the module pass manager.
        auto module_pm = std::make_unique<llvm::legacy::PassManager>();
        // These are passes which set up target-specific info
        // that are used by successive optimisation passes.
        auto tliwp = std::make_unique<llvm::TargetLibraryInfoWrapperPass>(
            llvm::TargetLibraryInfoImpl(m_jitter->get_target_triple()));
        module_pm->add(tliwp.release());
        module_pm->add(llvm::createTargetTransformInfoWrapperPass(m_jitter->get_target_ir_analysis()));

        // NOTE: not sure what this does, presumably some target-specifc
        // configuration.
        module_pm->add(static_cast<llvm::LLVMTargetMachine &>(*m_jitter->m_tm).createPassConfig(*module_pm));

        // Init the function pass manager.
        auto f_pm = std::make_unique<llvm::legacy::FunctionPassManager>(m_module.get());
        f_pm->add(llvm::createTargetTransformInfoWrapperPass(m_jitter->get_target_ir_analysis()));

        // Add an initial pass to vectorize load/stores.
        // This is useful to ensure that the
        // pattern adopted in load_vector_from_memory() and
        // store_vector_to_memory() is translated to
        // vectorized store/load instructions.
        auto lsv_pass = std::unique_ptr<llvm::Pass>(llvm::createLoadStoreVectorizerPass());
        f_pm->add(lsv_pass.release());

        // We use the helper class PassManagerBuilder to populate the module
        // pass manager with standard options.
        llvm::PassManagerBuilder pm_builder;
        // See here for the defaults:
        // https://llvm.org/doxygen/PassManagerBuilder_8cpp_source.html
        // NOTE: we used to have the SLP vectorizer on here, but
        // we don't activate it any more in favour of explicit vectorization.
        // NOTE: perhaps in the future we can make the autovectorizer an
        // option like the fast math flag.
        pm_builder.OptLevel = m_opt_level;
        if (m_inline_functions) {
            // Enable function inlining if the inlining flag is enabled.
            pm_builder.Inliner = llvm::createFunctionInliningPass(m_opt_level, 0, false);
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
    m_ir_snapshot = get_ir();

    // Store also the object code, if requested.
    if (m_save_object_code) {
        // Create a name model for the llvm temporary file machinery.
        const auto model = (boost::filesystem::temp_directory_path() / "heyoka-%%-%%-%%-%%-%%.o").string();

        // Create a unique file.
        // NOTE: this will also open the file. fd is the file
        // descriptor, res_path will be the full path to the file.
        int fd;
        llvm::SmallString<128> res_path;
        const auto res = llvm::sys::fs::createUniqueFile(model, fd, res_path);

        if (res) {
            throw std::invalid_argument(
                "The function to create a unique temporary file failed. The full error message:\n" + res.message());
        }

        // RAII helper to remove the unique file that was
        // created above.
        struct file_remover {
            llvm::SmallString<128> &path;

            ~file_remover()
            {
                boost::filesystem::remove(boost::filesystem::path{path.c_str()});
            }
        } fr{res_path};

        // Create a stream from the file descriptor.
        // The 'false' parameter indicates not to close
        // the file upon destruction of dest (we will be
        // closing the file manually).
        llvm::raw_fd_ostream dest(fd, false);

        // Setup the machinery for dumping the object code.
        llvm::legacy::PassManager pass;

        if (m_jitter->m_tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile)) {
            // Make sure to close the file before throwing.
            // NOTE: the file will be removed by the fr object
            // destructor.
            dest.close();

            throw std::invalid_argument("The target machine can't emit a file of this type");
        }

        // Dump the object code.
        pass.run(*m_module);

        // Close the file.
        dest.close();

        // Re-open it for reading in binary mode.
        std::ifstream ifile(res_path.c_str(), std::ios::binary);
        if (!ifile.good()) {
            throw std::invalid_argument("Could not open the temporary file '" + std::string(res_path.c_str())
                                        + "' for writing");
        }
        // Enable exceptions on ifile.
        ifile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // Dump into a stringstream.
        std::ostringstream oss;
        oss << ifile.rdbuf();

        // Assign the read data.
        m_object_code = oss.str();
    }

    m_jitter->add_module(std::move(m_module));
}

bool llvm_state::is_compiled() const
{
    return !m_module;
}

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

std::string llvm_state::get_ir() const
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

void llvm_state::dump_object_code(const std::string &filename) const
{
    const auto compiled = !m_module;

    if (compiled && !m_save_object_code) {
        throw std::invalid_argument("Cannot dump the object code after compilation if the 'save_object_code' "
                                    "keyword argument was not set to true when constructing the llvm_state object");
    }

    std::error_code ec;
    llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);

    if (ec) {
        throw std::invalid_argument("Could not open the file '" + filename
                                    + "' for dumping object code. The full error message:\n" + ec.message());
    }

    if (compiled) {
        // The module has been compiled already, dump the saved
        // object code image.
        dest << m_object_code;
    } else {
        // The module has not been compiled yet, run the JIT
        // and dump the object code.
        llvm::legacy::PassManager pass;

        if (m_jitter->m_tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CGFT_ObjectFile)) {
            // Close and remove the file before throwing.
            dest.close();
            boost::filesystem::remove(boost::filesystem::path{filename});

            throw std::invalid_argument("The target machine can't emit a file of this type");
        }

        pass.run(*m_module);
    }
}

std::ostream &operator<<(std::ostream &os, const llvm_state &s)
{
    std::ostringstream oss;
    oss << std::boolalpha;

    oss << "Module name        : " << s.m_module_name << '\n';
    oss << "Compiled           : " << s.is_compiled() << '\n';
    oss << "Fast math          : " << s.m_fast_math << '\n';
    oss << "Optimisation level : " << s.m_opt_level << '\n';
    oss << "Inline functions   : " << s.m_inline_functions << '\n';
    oss << "Target triple      : " << s.m_jitter->get_target_triple().str() << '\n';
    oss << "Target CPU         : " << s.m_jitter->get_target_cpu() << '\n';
    oss << "Target features    : " << s.m_jitter->get_target_features() << '\n';
    oss << "IR size            : " << s.get_ir().size() << '\n';

    return os << oss.str();
}

} // namespace heyoka
