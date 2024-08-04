// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <ranges>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/format.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#if LLVM_VERSION_MAJOR >= 17

// NOTE: this header was moved in LLVM 17.
#include <llvm/TargetParser/Triple.h>

#else

#include <llvm/ADT/Triple.h>

#endif

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/variable.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Make sure our definition of ir_builder matches llvm::IRBuilder<>.
static_assert(std::is_same_v<ir_builder, llvm::IRBuilder<>>, "Inconsistent definition of the ir_builder type.");

#if defined(HEYOKA_HAVE_REAL128)

// Make sure that the size and alignment of __float128
// and mppp::real128 coincide. This is required if we
// want to be able to use mppp::real128 as an alias
// for __float128.
static_assert(sizeof(__float128) == sizeof(mppp::real128));
static_assert(alignof(__float128) == alignof(mppp::real128));

#endif

// LCOV_EXCL_START

// Regex to match the PowerPC ISA version from the
// CPU string.
// NOTE: the pattern reported by LLVM here seems to be pwrN
// (sample size of 1, on travis...).
// NOLINTNEXTLINE(cert-err58-cpp)
const std::regex ppc_regex_pattern("pwr([1-9]*)");

// Helper function to detect specific features
// on the host machine via LLVM's machinery.
target_features get_target_features_impl()
{
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmb) [[unlikely]] {
        throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
    }

    auto tm = jtmb->createTargetMachine();
    if (!tm) [[unlikely]] {
        throw std::invalid_argument("Error creating the target machine");
    }

    target_features retval;

    const auto target_name = std::string{(*tm)->getTarget().getName()};

    if (boost::starts_with(target_name, "x86")) {
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

        if (boost::algorithm::contains(t_features, "+sse2")) {
            retval.sse2 = true;
        }
    }

    if (boost::starts_with(target_name, "aarch64")) {
        retval.aarch64 = true;
    }

    if (boost::starts_with(target_name, "ppc")) {
        // On powerpc, detect the presence of the VSX
        // instruction set from the CPU string.
        const auto target_cpu = std::string{(*tm)->getTargetCPU()};

        std::cmatch m;

        if (std::regex_match(target_cpu.c_str(), m, ppc_regex_pattern)) {
            if (m.size() == 2u) {
                // The CPU name matches and contains a subgroup.
                // Extract the N from "pwrN".
                std::uint32_t pwr_idx{};
                auto ret = std::from_chars(m[1].first, m[1].second, pwr_idx);

                // NOTE: it looks like VSX3 is supported from Power9,
                // VSX from Power7.
                // https://packages.gentoo.org/useflags/cpu_flags_ppc_vsx3
                if (ret.ec == std::errc{}) {
                    if (pwr_idx >= 9) {
                        retval.vsx3 = true;
                    }

                    if (pwr_idx >= 7) {
                        retval.vsx = true;
                    }
                }
            }
        }
    }

    // Compute the recommended SIMD sizes.
    if (retval.avx512f || retval.avx2 || retval.avx) {
        // NOTE: keep the recommended SIMD size to
        // 4/8 also for AVX512 due to perf issues in early
        // implementations. Revisit this in the future, possibly
        // making it conditional on the specific CPU model
        // in use.
        retval.simd_size_flt = 8;
        retval.simd_size_dbl = 4;
    } else if (retval.sse2 || retval.aarch64 || retval.vsx || retval.vsx3) {
        retval.simd_size_flt = 4;
        retval.simd_size_dbl = 2;
    }

    return retval;
}

// LCOV_EXCL_STOP

// Machinery to initialise the native target in
// LLVM. This needs to be done only once.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
HEYOKA_CONSTINIT std::once_flag nt_inited;

void init_native_target()
{
    std::call_once(nt_inited, []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
    });
}

// Helper to create a builder for target machines.
llvm::orc::JITTargetMachineBuilder create_jit_tmb(unsigned opt_level, code_model c_model)
{
    // NOTE: codegen opt level changed in LLVM 18.
#if LLVM_VERSION_MAJOR < 18

    using cg_opt_level = llvm::CodeGenOpt::Level;

#else

    using cg_opt_level = llvm::CodeGenOptLevel;

#endif

    // Try creating the target machine builder.
    auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    // LCOV_EXCL_START
    if (!jtmb) [[unlikely]] {
        throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
    }
    // LCOV_EXCL_STOP

    // Set the codegen optimisation level.
    switch (opt_level) {
        case 0u:
            jtmb->setCodeGenOptLevel(cg_opt_level::None);
            break;
        case 1u:
            jtmb->setCodeGenOptLevel(cg_opt_level::Less);
            break;
        case 2u:
            jtmb->setCodeGenOptLevel(cg_opt_level::Default);
            break;
        default:
            assert(opt_level == 3u);
            jtmb->setCodeGenOptLevel(cg_opt_level::Aggressive);
    }

    // NOTE: not all code models are supported on all archs. We make an effort
    // here to prevent unsupported code models to be requested, as that will
    // result in the termination of the program.
    constexpr code_model supported_code_models[] = {
#if defined(HEYOKA_ARCH_X86)
        code_model::small, code_model::kernel, code_model::medium, code_model::large
#elif defined(HEYOKA_ARCH_ARM)
        code_model::tiny, code_model::small, code_model::large
#elif defined(HEYOKA_ARCH_PPC)
        code_model::small, code_model::medium, code_model::large
#else
        // NOTE: by default we assume only small and large are supported.
        code_model::small, code_model::large
#endif
    };

    if (std::ranges::find(supported_code_models, c_model) == std::ranges::end(supported_code_models)) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The code model '{}' is not supported on the current architecture", c_model));
    }

    // LCOV_EXCL_START

#if LLVM_VERSION_MAJOR >= 17

    // NOTE: the code model setup is working only on LLVM>=19 (or at least
    // LLVM 18 + patches, as in the conda-forge LLVM package), due to this bug:
    //
    // https://github.com/llvm/llvm-project/issues/88115
    //
    // Additionally, there are indications from our CI that attempting to set
    // the code model before LLVM 17 might just be buggy, as we see widespread
    // ASAN failures all over the place. Thus, let us not do anything with the code
    // model setting before LLVM 17.

    // Setup the code model.
    switch (c_model) {
        case code_model::tiny:
            jtmb->setCodeModel(llvm::CodeModel::Tiny);
            break;
        case code_model::small:
            jtmb->setCodeModel(llvm::CodeModel::Small);
            break;
        case code_model::kernel:
            jtmb->setCodeModel(llvm::CodeModel::Kernel);
            break;
        case code_model::medium:
            jtmb->setCodeModel(llvm::CodeModel::Medium);
            break;
        case code_model::large:
            jtmb->setCodeModel(llvm::CodeModel::Large);
            break;
        default:
            // NOTE: we should never end up here.
            assert(false);
            ;
    }

#endif

    //  LCOV_EXCL_STOP

    return std::move(*jtmb);
}

// Helper to optimise the input module M. Implemented here for re-use.
// NOTE: this may end up being invoked concurrently from multiple threads.
// If that is the case, we make sure before invocation to construct a different
// TargetMachine per thread, so that we are sure no data races are possible.
void optimise_module(llvm::Module &M, llvm::TargetMachine &tm, unsigned opt_level, bool force_avx512,
                     bool slp_vectorize)
{
    // NOTE: don't run any optimisation pass at O0.
    if (opt_level == 0u) {
        return;
    }

    // NOTE: the logic here largely mimics (with a lot of simplifications)
    // the implementation of the 'opt' tool. See:
    // https://github.com/llvm/llvm-project/blob/release/10.x/llvm/tools/opt/opt.cpp

    // For every function in the module, setup its attributes
    // so that the codegen uses all the features available on
    // the host CPU.
    const auto cpu = tm.getTargetCPU().str();
    const auto features = tm.getTargetFeatureString().str();

    // Fetch the module's context.
    auto &ctx = M.getContext();

    for (auto &f : M) {
        auto attrs = f.getAttributes();

        llvm::AttrBuilder new_attrs(ctx);

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

        // Let new_attrs override attrs.
        f.setAttributes(attrs.addFnAttributes(ctx, new_attrs));
    }

    // Force usage of AVX512 registers, if requested.
    if (force_avx512 && get_target_features().avx512f) {
        for (auto &f : M) {
            f.addFnAttr("prefer-vector-width", "512");
        }
    }

    // NOTE: adapted from here:
    // https://llvm.org/docs/NewPassManager.html

    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    // NOTE: in the new pass manager, this seems to be the way to
    // set the target library info bits. See:
    // https://github.com/llvm/llvm-project/blob/b7fd30eac3183993806cc218b6deb39eb625c083/llvm/tools/opt/NewPMDriver.cpp#L408
    // Not sure if this matters, but we did it in the old pass manager
    // and opt does it too.
    llvm::TargetLibraryInfoImpl TLII(tm.getTargetTriple());
    FAM.registerPass([&] { return llvm::TargetLibraryAnalysis(TLII); });

    // Create the new pass manager builder, passing the supplied target machine.
    // NOTE: if requested, we turn manually on the SLP vectoriser here, which is off
    // by default. Not sure why it is off, the LLVM docs imply this
    // is on by default at nonzero optimisation levels for clang and opt.
    // NOTE: the reason for this inconsistency is that opt uses PB.parsePassPipeline()
    // (instead of PB.buildPerModuleDefaultPipeline()) to set up the optimisation
    // pipeline. Indeed, if we replace PB.buildPerModuleDefaultPipeline(ol) with
    // PB.parsePassPipeline(MPM, "default<O3>") (which corresponds to invoking
    // "opt -passes='default<O3>'"), we do NOT need to set SLP vectorization on
    // here to get the SLP vectorizer. Not sure if we should consider switching to this
    // alternative way of setting up the optimisation pipeline in the future.
    llvm::PipelineTuningOptions pto;
    pto.SLPVectorization = slp_vectorize;
    llvm::PassBuilder PB(&tm, pto);

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Construct the optimisation level.
    llvm::OptimizationLevel ol{};

    switch (opt_level) {
        case 1u:
            ol = llvm::OptimizationLevel::O1;
            break;
        case 2u:
            ol = llvm::OptimizationLevel::O2;
            break;
        default:
            assert(opt_level == 3u);
            ol = llvm::OptimizationLevel::O3;
    }

    // Create the module pass manager.
    auto MPM = PB.buildPerModuleDefaultPipeline(ol);

    // Optimize the IR.
    MPM.run(M, MAM);
}

// Helper to add a module to an lljt, throwing on error.
void add_module_to_lljit(llvm::orc::LLJIT &lljit, std::unique_ptr<llvm::Module> m, llvm::orc::ThreadSafeContext ctx)
{
    auto err = lljit.addIRModule(llvm::orc::ThreadSafeModule(std::move(m), std::move(ctx)));

    // LCOV_EXCL_START
    if (err) {
        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(
            fmt::format("The function for adding a module to the jit failed. The full error message:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP
}

// Helper to fetch the bitcode from a module.
std::string bc_from_module(llvm::Module &m)
{
    std::string out;
    llvm::raw_string_ostream ostr(out);

    llvm::WriteBitcodeToFile(m, ostr);

    return std::move(ostr.str());
}

// Helper to fetch the textual IR from a module.
std::string ir_from_module(llvm::Module &m)
{
    std::string out;
    llvm::raw_string_ostream ostr(out);

    m.print(ostr, nullptr);

    return std::move(ostr.str());
}

// An implementation of llvm::MemoryBuffer offering a view over a std::string.
class string_view_mem_buffer final : public llvm::MemoryBuffer
{
public:
    explicit string_view_mem_buffer(const std::string &s)
    {
        // NOTE: the important bit here is from the LLVM docs:
        //
        // """
        // In addition to basic access to the characters in the file, this interface
        // guarantees you can read one character past the end of the file, and that
        // this character will read as '\0'.
        // """
        //
        // This is exactly the guarantee given by std::string:
        //
        // https://en.cppreference.com/w/cpp/string/basic_string/data
        //
        // Not sure about the third parameter to this function though, it does not
        // seem to have any influence apart from debug checking:
        //
        // https://llvm.org/doxygen/MemoryBuffer_8cpp_source.html
        this->init(s.data(), s.data() + s.size(), true);
    }
    // LCOV_EXCL_START
    llvm::MemoryBuffer::BufferKind getBufferKind() const final
    {
        // Hopefully std::string is not memory-mapped...
        return llvm::MemoryBuffer::BufferKind::MemoryBuffer_Malloc;
    }
    // LCOV_EXCL_STOP
};

// Helper to add an object file to the jit, throwing in case of errors.
void add_obj_to_lljit(llvm::orc::LLJIT &lljit, const std::string &obj)
{
    // NOTE: an empty obj can happen when we are copying a compiled
    // llvm_multi_state. In such case, the object files of the individual
    // states have all be empty-inited. We then need to avoid adding
    // obj to the jit because that will result in an error.
    if (obj.empty()) {
        return;
    }

    // Add the object file.
    auto err = lljit.addObjectFile(std::make_unique<string_view_mem_buffer>(obj));

    // LCOV_EXCL_START
    if (err) {
        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(fmt::format(
            "The function for adding an object file to an lljit failed. The full error message:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP
}

// Helper to verify a module, throwing if verification fails.
void verify_module(const llvm::Module &m)
{
    std::string out;
    llvm::raw_string_ostream ostr(out);

    if (llvm::verifyModule(m, &ostr)) {
        // LCOV_EXCL_START
        throw std::runtime_error(fmt::format("The verification of the module '{}' produced an error:\n{}",
                                             m.getModuleIdentifier(), ostr.str()));
        // LCOV_EXCL_STOP
    }
}

} // namespace

// Helper function to fetch a const ref to a global object
// containing info about the host machine.
const target_features &get_target_features()
{
    static const target_features retval = []() {
        // NOTE: need to init the native target
        // in order to get its features.
        init_native_target();

        return get_target_features_impl();
    }();

    return retval;
}

} // namespace detail

template <>
std::uint32_t recommended_simd_size<float>()
{
    return detail::get_target_features().simd_size_flt;
}

template <>
std::uint32_t recommended_simd_size<double>()
{
    return detail::get_target_features().simd_size_dbl;
}

template <>
std::uint32_t recommended_simd_size<long double>()
{
    return detail::get_target_features().simd_size_ldbl;
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::uint32_t recommended_simd_size<mppp::real128>()
{
    return detail::get_target_features().simd_size_f128;
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
std::uint32_t recommended_simd_size<mppp::real>()
{
    return 1;
}

#endif

// Implementation of the jit class.
struct llvm_state::jit {
    std::unique_ptr<llvm::orc::LLJIT> m_lljit;
    std::unique_ptr<llvm::TargetMachine> m_tm;
    std::unique_ptr<llvm::orc::ThreadSafeContext> m_ctx;
    std::optional<std::string> m_object_file;

    // NOTE: make sure to coordinate changes in this constructor with multi_jit.
    explicit jit(unsigned opt_level, code_model c_model)
    {
        // NOTE: we assume here that the input arguments have
        // been validated already.
        assert(opt_level <= 3u);
        assert(c_model >= code_model::tiny && c_model <= code_model::large);

        // Ensure the native target is inited.
        detail::init_native_target();

        // Create the target machine builder.
        auto jtmb = detail::create_jit_tmb(opt_level, c_model);

        // Create the jit builder.
        llvm::orc::LLJITBuilder lljit_builder;
        // NOTE: other settable properties may
        // be of interest:
        // https://www.llvm.org/doxygen/classllvm_1_1orc_1_1LLJITBuilder.html
        lljit_builder.setJITTargetMachineBuilder(jtmb);

        // Create the jit.
        auto lljit = lljit_builder.create();
        // LCOV_EXCL_START
        if (!lljit) {
            auto err = lljit.takeError();

            std::string err_report;
            llvm::raw_string_ostream ostr(err_report);

            ostr << err;

            throw std::invalid_argument(
                fmt::format("Could not create an LLJIT object. The full error message is:\n{}", ostr.str()));
        }
        // LCOV_EXCL_STOP
        m_lljit = std::move(*lljit);

        // Setup the machinery to store the module's binary code
        // when it is generated.
        m_lljit->getObjTransformLayer().setTransform([this](std::unique_ptr<llvm::MemoryBuffer> obj_buffer) {
            assert(obj_buffer);

            // NOTE: this callback will be invoked the first time a jit lookup is performed,
            // even if the object code was manually injected via llvm_state_add_obj_to_jit()
            // (e.g., during copy, des11n, etc.). In such a case, m_object_file has already been set up properly and we
            // just sanity check in debug mode that the content of m_object_file matches the content of obj_buffer.
            if (m_object_file) {
                assert(obj_buffer->getBufferSize() == m_object_file->size());
                assert(std::equal(obj_buffer->getBufferStart(), obj_buffer->getBufferEnd(), m_object_file->begin()));
            } else {
                // Copy obj_buffer to the local m_object_file member.
                m_object_file.emplace(obj_buffer->getBufferStart(), obj_buffer->getBufferEnd());
            }

            return llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>(std::move(obj_buffer));
        });

        // Setup the jit so that it can look up symbols from the current process.
        auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            m_lljit->getDataLayout().getGlobalPrefix());
        // LCOV_EXCL_START
        if (!dlsg) {
            throw std::invalid_argument("Could not create the dynamic library search generator");
        }
        // LCOV_EXCL_STOP
        m_lljit->getMainJITDylib().addGenerator(std::move(*dlsg));

        // Keep a target machine around to fetch various
        // properties of the host CPU.
        auto tm = jtmb.createTargetMachine();
        // LCOV_EXCL_START
        if (!tm) {
            throw std::invalid_argument("Error creating the target machine");
        }
        // LCOV_EXCL_STOP
        m_tm = std::move(*tm);

        // Create the context.
        m_ctx = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());

        // NOTE: by default, errors in the execution session are printed
        // to screen. A custom error reported can be specified, ideally
        // we would like th throw here but I am not sure whether throwing
        // here would disrupt LLVM's cleanup actions?
        // https://llvm.org/doxygen/classllvm_1_1orc_1_1ExecutionSession.html

#if defined(HEYOKA_HAVE_REAL) && !defined(NDEBUG)

        // Run several checks to ensure that real_t matches the layout of mppp::real/mpfr_struct_t.
        // NOTE: these checks need access to the data layout, so we put them here for convenience.
        const auto &dl = m_lljit->getDataLayout();
        auto *real_t = llvm::cast<llvm::StructType>(detail::to_llvm_type<mppp::real>(*m_ctx->getContext()));
        const auto *slo = dl.getStructLayout(real_t);
        assert(slo->getSizeInBytes() == sizeof(mppp::real));
        assert(slo->getAlignment().value() == alignof(mppp::real));
        assert(slo->getElementOffset(0) == offsetof(mppp::mpfr_struct_t, _mpfr_prec));
        assert(slo->getElementOffset(1) == offsetof(mppp::mpfr_struct_t, _mpfr_sign));
        assert(slo->getElementOffset(2) == offsetof(mppp::mpfr_struct_t, _mpfr_exp));
        assert(slo->getElementOffset(3) == offsetof(mppp::mpfr_struct_t, _mpfr_d));
        assert(slo->getMemberOffsets().size() == 4u);

#endif
    }

    jit(const jit &) = delete;
    jit(jit &&) = delete;
    jit &operator=(const jit &) = delete;
    jit &operator=(jit &&) = delete;

    ~jit() = default;

    // Accessors.
    [[nodiscard]] llvm::LLVMContext &get_context() const
    {
        return *m_ctx->getContext();
    }
    [[nodiscard]] std::string get_target_cpu() const
    {
        return m_tm->getTargetCPU().str();
    }
    [[nodiscard]] std::string get_target_features() const
    {
        return m_tm->getTargetFeatureString().str();
    }
    [[nodiscard]] const llvm::Triple &get_target_triple() const
    {
        return m_lljit->getTargetTriple();
    }

    void add_module(std::unique_ptr<llvm::Module> m) const
    {
        detail::add_module_to_lljit(*m_lljit, std::move(m), *m_ctx);
    }

    // Symbol lookup.
    auto lookup(const std::string &name) const
    {
        return m_lljit->lookup(name);
    }
};

// Small shared helper to setup the math flags in the builder at the
// end of a constructor or a deserialization.
void llvm_state::ctor_setup_math_flags()
{
    assert(m_builder);

    llvm::FastMathFlags fmf;

    if (m_fast_math) {
        // Set flags for faster math at the
        // price of potential change of semantics.
        fmf.setFast();
    } else {
        // By default, allow only fp contraction.
        fmf.setAllowContract();
    }

    m_builder->setFastMathFlags(fmf);
}

namespace detail
{

namespace
{

// Helper to load object code into the jit of an llvm_state.
template <typename Jit>
void llvm_state_add_obj_to_jit(Jit &j, std::string obj)
{
    // Add the object code to the lljit.
    add_obj_to_lljit(*j.m_lljit, obj);

    // Add the object code also to the
    // m_object_file member.
    // NOTE: this function at the moment is used when m_object_file
    // is supposed to be empty.
    assert(!j.m_object_file);
    j.m_object_file.emplace(std::move(obj));
}

// Helper to create an LLVM module from bitcode.
// NOTE: the module name needs to be passed explicitly (although it is already
// contained in the bitcode) because apparently llvm::parseBitcodeFile() discards the module
// name when parsing.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto bc_to_module(const std::string &module_name, const std::string &bc, llvm::LLVMContext &ctx)
{
    // Create the corresponding memory buffer view on bc.
    auto mb = std::make_unique<string_view_mem_buffer>(bc);

    // Parse the bitcode.
    auto ret = llvm::parseBitcodeFile(mb->getMemBufferRef(), ctx);

    // LCOV_EXCL_START
    if (!ret) {
        const auto err = ret.takeError();
        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(
            fmt::format("LLVM bitcode parsing failed. The full error message:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP

    // Set the module name.
    ret.get()->setModuleIdentifier(module_name);

    return std::move(ret.get());
}

} // namespace

} // namespace detail

std::ostream &operator<<(std::ostream &os, code_model c_model)
{
    switch (c_model) {
        case code_model::tiny:
            os << "tiny";
            break;
        case code_model::small:
            os << "small";
            break;
        case code_model::kernel:
            os << "kernel";
            break;
        case code_model::medium:
            os << "medium";
            break;
        case code_model::large:
            os << "large";
            break;
        default:
            os << "invalid";
    }

    return os;
}

void llvm_state::validate_code_model(code_model c_model)
{
    if (c_model < code_model::tiny || c_model > code_model::large) [[unlikely]] {
        throw std::invalid_argument(fmt::format(
            "An invalid code model enumerator with a value of {} was passed to the constructor of an llvm_state",
            static_cast<unsigned>(c_model)));
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
llvm_state::llvm_state(std::tuple<std::string, unsigned, bool, bool, bool, code_model> &&tup)
    : m_jitter(std::make_unique<jit>(std::get<1>(tup), std::get<5>(tup))), m_opt_level(std::get<1>(tup)),
      m_fast_math(std::get<2>(tup)), m_force_avx512(std::get<3>(tup)), m_slp_vectorize(std::get<4>(tup)),
      m_c_model(std::get<5>(tup)), m_module_name(std::move(std::get<0>(tup)))
{
    // Create the module.
    m_module = std::make_unique<llvm::Module>(m_module_name, context());
    // Setup the data layout and the target triple.
    m_module->setDataLayout(m_jitter->m_lljit->getDataLayout());
    m_module->setTargetTriple(m_jitter->get_target_triple().str());

    // Create a new builder for the module.
    m_builder = std::make_unique<ir_builder>(context());

    // Setup the math flags in the builder.
    ctor_setup_math_flags();
}

// NOTE: this will ensure that all kwargs
// are set to their default values.
llvm_state::llvm_state() : llvm_state(kw_args_ctor_impl()) {}

llvm_state::llvm_state(const llvm_state &other)
    // NOTE: start off by:
    // - creating a new jit,
    // - copying over the options from other.
    : m_jitter(std::make_unique<jit>(other.m_opt_level, other.m_c_model)), m_opt_level(other.m_opt_level),
      m_fast_math(other.m_fast_math), m_force_avx512(other.m_force_avx512), m_slp_vectorize(other.m_slp_vectorize),
      m_c_model(other.m_c_model), m_module_name(other.m_module_name)
{
    if (other.is_compiled()) {
        // 'other' was compiled.
        // We leave module and builder empty, copy over the
        // IR/bitcode snapshots and add the compiled module
        // to the jit.
        m_ir_snapshot = other.m_ir_snapshot;
        m_bc_snapshot = other.m_bc_snapshot;

        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        detail::llvm_state_add_obj_to_jit(*m_jitter, *other.m_jitter->m_object_file);
    } else {
        // 'other' has not been compiled yet.
        // We will fetch its bitcode and reconstruct
        // module and builder. The IR/bitcode snapshots
        // are left in their default-constructed (empty)
        // state.
        m_module = detail::bc_to_module(m_module_name, other.get_bc(), context());

        // Create a new builder for the module.
        m_builder = std::make_unique<ir_builder>(context());

        // Setup the math flags in the builder.
        ctor_setup_math_flags();
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

// NOTE: this cannot be defaulted because the moving of the LLVM objects
// needs to be done in a different order (specifically, we need to
// ensure that the LLVM objects in this are destroyed in a specific
// order).
llvm_state &llvm_state::operator=(llvm_state &&other) noexcept
{
    if (this != &other) {
        // The LLVM bits.
        m_builder = std::move(other.m_builder);
        m_module = std::move(other.m_module);
        m_jitter = std::move(other.m_jitter);

        // The remaining bits.
        m_opt_level = other.m_opt_level;
        m_ir_snapshot = std::move(other.m_ir_snapshot);
        m_bc_snapshot = std::move(other.m_bc_snapshot);
        m_fast_math = other.m_fast_math;
        m_force_avx512 = other.m_force_avx512;
        m_slp_vectorize = other.m_slp_vectorize;
        m_c_model = other.m_c_model;
        m_module_name = std::move(other.m_module_name);
    }

    return *this;
}

// NOTE: we used to have debug sanity checks here. However, in certain rare corner cases,
// an invalid llvm_state could end up being destroyed, thus triggering assertion errors
// in debug mode (this could happen for instance when resetting an llvm_state to the
// def-cted state after an exception had been thrown during compilation). Thus, just
// do not run the debug checks.
llvm_state::~llvm_state() = default;

template <typename Archive>
void llvm_state::save_impl(Archive &ar, unsigned) const
{
    // Start by establishing if the state is compiled.
    const auto cmp = is_compiled();
    ar << cmp;

    // Store the config options.
    ar << m_opt_level;
    ar << m_fast_math;
    ar << m_force_avx512;
    ar << m_slp_vectorize;
    ar << m_c_model;
    ar << m_module_name;

    // Store the bitcode.
    // NOTE: avoid get_bc() if the module has been compiled,
    // and use the snapshot directly, so that we don't make
    // a useless copy.
    if (cmp) {
        ar << m_bc_snapshot;
    } else {
        ar << get_bc();
    }

    if (cmp) {
        // Save the object file.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        ar << *m_jitter->m_object_file;
    }

    // Save a copy of the IR snapshot if the state
    // is compiled.
    // NOTE: we want this because otherwise we would
    // need to re-parse the bitcode during des11n to
    // restore the IR snapshot.
    if (cmp) {
        ar << m_ir_snapshot;
    }
}

// NOTE: currently loading from an archive won't interact with the
// memory cache - that is, if the archive contains a compiled module
// not in the cache *before* loading, it won't have been inserted in the cache
// *after* loading. I don't think this is an issue at the moment, but if needed
// we can always implement the feature at a later stage.
template <typename Archive>
void llvm_state::load_impl(Archive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<llvm_state>::type::value)) {
        throw std::invalid_argument(fmt::format("Unable to load an llvm_state object: "
                                                "the archive version ({}) is too old",
                                                version));
    }
    // LCOV_EXCL_STOP

    // NOTE: all serialised objects in the archive
    // are primitive types, no need to reset the
    // addresses.

    // Load the compiled status flag from the archive.
    // NOTE: not sure why clang-tidy wants cmp to be
    // const here, as clearly ar >> cmp is going to
    // write something into it. Perhaps some const_cast
    // shenanigangs in Boost.Serialization?
    // NOLINTNEXTLINE(misc-const-correctness)
    bool cmp{};
    ar >> cmp;

    // Load the config options.
    // NOLINTNEXTLINE(misc-const-correctness)
    unsigned opt_level{};
    ar >> opt_level;

    // NOLINTNEXTLINE(misc-const-correctness)
    bool fast_math{};
    ar >> fast_math;

    // NOLINTNEXTLINE(misc-const-correctness)
    bool force_avx512{};
    ar >> force_avx512;

    // NOLINTNEXTLINE(misc-const-correctness)
    bool slp_vectorize{};
    ar >> slp_vectorize;

    // NOLINTNEXTLINE(misc-const-correctness)
    code_model c_model{};
    ar >> c_model;

    // NOLINTNEXTLINE(misc-const-correctness)
    std::string module_name;
    ar >> module_name;

    // Load the bitcode.
    std::string bc_snapshot;
    ar >> bc_snapshot;

    // Recover the object file, if available.
    std::optional<std::string> obj_file;
    if (cmp) {
        obj_file.emplace();
        ar >> *obj_file;
    }

    // Recover the IR snapshot, if available.
    std::string ir_snapshot;
    if (cmp) {
        ar >> ir_snapshot;
    }

    try {
        // Set the config options.
        m_opt_level = opt_level;
        m_fast_math = fast_math;
        m_force_avx512 = force_avx512;
        m_slp_vectorize = slp_vectorize;
        m_c_model = c_model;
        m_module_name = module_name;

        // Reset module and builder to the def-cted state.
        m_module.reset();
        m_builder.reset();

        // Reset the jit with a new one.
        m_jitter = std::make_unique<jit>(opt_level, c_model);

        if (cmp) {
            // Assign the snapshots.
            m_ir_snapshot = std::move(ir_snapshot);
            m_bc_snapshot = std::move(bc_snapshot);

            // Add the object code to the jit.
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            detail::llvm_state_add_obj_to_jit(*m_jitter, std::move(*obj_file));
        } else {
            // Clear the existing snapshots.
            m_ir_snapshot.clear();
            m_bc_snapshot.clear();

            // Create the module from the bitcode.
            m_module = detail::bc_to_module(m_module_name, bc_snapshot, context());

            // Create a new builder for the module.
            m_builder = std::make_unique<ir_builder>(context());

            // Setup the math flags in the builder.
            ctor_setup_math_flags();
        }
        // LCOV_EXCL_START
    } catch (...) {
        // Reset to a def-cted state in case of error,
        // as it looks like there's no way of recovering.
        *this = []() noexcept { return llvm_state{}; }();

        throw;
        // LCOV_EXCL_STOP
    }
}

void llvm_state::save(boost::archive::binary_oarchive &ar, unsigned v) const
{
    save_impl(ar, v);
}

void llvm_state::load(boost::archive::binary_iarchive &ar, unsigned v)
{
    load_impl(ar, v);
}

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

unsigned llvm_state::get_opt_level() const
{
    return m_opt_level;
}

bool llvm_state::fast_math() const
{
    return m_fast_math;
}

bool llvm_state::force_avx512() const
{
    return m_force_avx512;
}

bool llvm_state::get_slp_vectorize() const
{
    return m_slp_vectorize;
}

code_model llvm_state::get_code_model() const
{
    return m_c_model;
}

unsigned llvm_state::clamp_opt_level(unsigned opt_level)
{
    return std::min<unsigned>(opt_level, 3u);
}

void llvm_state::check_uncompiled(const char *f) const
{
    if (!m_module) {
        throw std::invalid_argument(
            fmt::format("The function '{}' can be invoked only if the module has not been compiled yet", f));
    }
}

void llvm_state::check_compiled(const char *f) const
{
    if (m_module) {
        throw std::invalid_argument(
            fmt::format("The function '{}' can be invoked only after the module has been compiled", f));
    }
}

void llvm_state::optimise()
{
    // NOTE: we used to fetch the target triple from the lljit object,
    // but recently we switched to asking the target triple directly
    // from the target machine. Assert equality between the two for a while,
    // just in case.
    assert(m_jitter->m_lljit->getTargetTriple() == m_jitter->m_tm->getTargetTriple());
    // NOTE: the target triple is also available in the module.
    assert(m_jitter->m_lljit->getTargetTriple().str() == module().getTargetTriple());

    detail::optimise_module(module(), *m_jitter->m_tm, m_opt_level, m_force_avx512, m_slp_vectorize);
}

namespace detail
{

namespace
{

// The name of the function used to trigger the
// materialisation of object code.
constexpr auto obj_trigger_name = "heyoka.obj_trigger";

} // namespace

} // namespace detail

// NOTE: this adds a public no-op function to the state which is
// used to trigger the generation of object code after compilation.
void llvm_state::add_obj_trigger()
{
    auto &bld = builder();

    auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {}, false);
    assert(ft != nullptr);
    auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, detail::obj_trigger_name, &module());
    assert(f != nullptr);

    bld.SetInsertPoint(llvm::BasicBlock::Create(context(), "entry", f));
    bld.CreateRetVoid();
}

// NOTE: this function is NOT exception-safe, proper cleanup
// needs to be done externally if needed.
void llvm_state::compile_impl()
{
    // Preconditions.
    assert(m_module);
    assert(m_builder);
    assert(m_ir_snapshot.empty());
    assert(m_bc_snapshot.empty());

    // Store a snapshot of the current IR and bitcode.
    m_ir_snapshot = get_ir();
    m_bc_snapshot = get_bc();

    // Add the module to the jit (this will clear out m_module).
    m_jitter->add_module(std::move(m_module));

    // Clear out the builder, which won't be usable any more.
    m_builder.reset();

    // Trigger object code materialisation via lookup.
    jit_lookup(detail::obj_trigger_name);

    assert(m_jitter->m_object_file);
}

namespace detail
{

namespace
{

// Combine opt_level, force_avx512, slp_vectorize and c_model into a single flag.
// NOTE: here we need:
//
// - 2 bits for opt_level,
// - 1 bit for force_avx512 and slp_vectorize each,
// - 3 bits for c_model,
//
// for a total of 7 bits.
unsigned assemble_comp_flag(unsigned opt_level, bool force_avx512, bool slp_vectorize, code_model c_model)
{
    assert(opt_level <= 3u);
    assert(static_cast<unsigned>(c_model) <= 7u);
    static_assert(std::numeric_limits<unsigned>::digits >= 7u);

    return opt_level + (static_cast<unsigned>(force_avx512) << 2) + (static_cast<unsigned>(slp_vectorize) << 3)
           + (static_cast<unsigned>(c_model) << 4);
}

} // namespace

} // namespace detail

// NOTE: we need to emphasise in the docs that compilation
// triggers an optimisation pass.
void llvm_state::compile()
{
    check_uncompiled(__func__);

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Run a verification on the module before compiling.
    detail::verify_module(*m_module);

    logger->trace("module verification runtime: {}", sw);

    // Add the object materialisation trigger function.
    // NOTE: do it **after** verification, on the assumption
    // that add_obj_trigger() is implemented correctly. Like this,
    // if module verification fails, the user still has the option
    // to fix the module and re-attempt compilation without having
    // altered the module and without having already added the trigger
    // function.
    // NOTE: this function does its own cleanup, no need to
    // start the try catch block yet.
    add_obj_trigger();

    try {
        // Fetch the bitcode *before* optimisation.
        auto orig_bc = get_bc();
        std::vector<std::string> obc;
        obc.push_back(std::move(orig_bc));

        // Assemble the compilation flag.
        const auto comp_flag = detail::assemble_comp_flag(m_opt_level, m_force_avx512, m_slp_vectorize, m_c_model);

        // Lookup in the cache.
        if (auto cached_data = detail::llvm_state_mem_cache_lookup(obc, comp_flag)) {
            // Cache hit.

            // Assign the optimised snapshots.
            assert(cached_data->opt_ir.size() == 1u);
            assert(cached_data->opt_bc.size() == 1u);
            assert(cached_data->obj.size() == 1u);
            m_ir_snapshot = std::move(cached_data->opt_ir[0]);
            m_bc_snapshot = std::move(cached_data->opt_bc[0]);

            // Clear out module and builder.
            m_module.reset();
            m_builder.reset();

            // Assign the object file.
            detail::llvm_state_add_obj_to_jit(*m_jitter, std::move(cached_data->obj[0]));

            // Look up the trigger.
            jit_lookup(detail::obj_trigger_name);
        } else {
            // Cache miss.

            sw.reset();

            // Run the optimisation pass.
            optimise();

            logger->trace("optimisation runtime: {}", sw);

            sw.reset();

            // Run the compilation.
            compile_impl();

            logger->trace("materialisation runtime: {}", sw);

            // Try to insert obc into the cache.
            detail::llvm_state_mem_cache_try_insert(
                std::move(obc), comp_flag,
                // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                {.opt_bc = {m_bc_snapshot}, .opt_ir = {m_ir_snapshot}, .obj = {*m_jitter->m_object_file}});
        }
        // LCOV_EXCL_START
    } catch (...) {
        // Reset to a def-cted state in case of error,
        // as it looks like there's no way of recovering.
        *this = []() noexcept { return llvm_state{}; }();

        throw;
        // LCOV_EXCL_STOP
    }
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
        throw std::invalid_argument(fmt::format("Could not find the symbol '{}' in the compiled module", name));
    }

    return static_cast<std::uintptr_t>((*sym).getValue());
}

std::string llvm_state::get_ir() const
{
    if (m_module) {
        // The module has not been compiled yet,
        // get the IR from it.
        return detail::ir_from_module(*m_module);
    } else {
        // The module has been compiled.
        // Return the IR snapshot that
        // was created before the compilation.
        return m_ir_snapshot;
    }
}

std::string llvm_state::get_bc() const
{
    if (m_module) {
        // The module has not been compiled yet,
        // get the bitcode from it.
        return detail::bc_from_module(*m_module);
    } else {
        // The module has been compiled.
        // Return the bitcode snapshot that
        // was created before the compilation.
        return m_bc_snapshot;
    }
}

const std::string &llvm_state::get_object_code() const
{
    if (!is_compiled()) {
        throw std::invalid_argument(
            "Cannot extract the object code from an llvm_state which has not been compiled yet");
    }

    assert(m_jitter->m_object_file);

    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *m_jitter->m_object_file;
}

const std::string &llvm_state::module_name() const
{
    return m_module_name;
}

// A helper that returns a new llvm_state configured in the same
// way as this (i.e., same module name, opt level, fast math flags, etc.),
// but with no code defined in it.
llvm_state llvm_state::make_similar() const
{
    return llvm_state(kw::mname = m_module_name, kw::opt_level = m_opt_level, kw::fast_math = m_fast_math,
                      kw::force_avx512 = m_force_avx512, kw::slp_vectorize = m_slp_vectorize,
                      kw::code_model = m_c_model);
}

std::ostream &operator<<(std::ostream &os, const llvm_state &s)
{
    std::ostringstream oss;
    oss << std::boolalpha;

    oss << "Module name       : " << s.m_module_name << '\n';
    oss << "Compiled          : " << s.is_compiled() << '\n';
    oss << "Fast math         : " << s.m_fast_math << '\n';
    oss << "Force AVX512      : " << s.m_force_avx512 << '\n';
    oss << "SLP vectorization : " << s.m_slp_vectorize << '\n';
    oss << "Code model        : " << s.m_c_model << '\n';
    oss << "Optimisation level: " << s.m_opt_level << '\n';
    oss << "Data layout       : " << s.m_jitter->m_lljit->getDataLayout().getStringRepresentation() << '\n';
    oss << "Target triple     : " << s.m_jitter->get_target_triple().str() << '\n';
    oss << "Target CPU        : " << s.m_jitter->get_target_cpu() << '\n';
    oss << "Target features   : " << s.m_jitter->get_target_features() << '\n';
    oss << "Bitcode size      : ";

    if (s.is_compiled()) {
        oss << s.m_bc_snapshot.size() << '\n';
    } else {
        oss << s.get_bc().size() << '\n';
    }

    return os << oss.str();
}

namespace detail
{

namespace
{

// NOTE: this is a class similar in spirit to llvm_state, but set up for parallel
// compilation of multiple modules.
struct multi_jit {
    // NOTE: this is the total number of modules, including
    // the master module.
    const unsigned m_n_modules = 0;
    // NOTE: enumerate the LLVM members here in the same order
    // as llvm_state, as this is important to ensure proper
    // destruction order.
    std::unique_ptr<llvm::orc::LLJIT> m_lljit;
    std::unique_ptr<llvm::orc::ThreadSafeContext> m_ctx;
    std::unique_ptr<llvm::Module> m_module;
    std::unique_ptr<ir_builder> m_builder;
    // Object files.
    // NOTE: these may be modified concurrently during compilation,
    // protect with mutex.
    std::mutex m_object_files_mutex;
    std::vector<std::string> m_object_files;
    // IR and bc optimised snapshots.
    // NOTE: these may be modified concurrently during compilation,
    // protect with mutex.
    std::mutex m_ir_bc_mutex;
    std::vector<std::string> m_ir_snapshots;
    std::vector<std::string> m_bc_snapshots;

    explicit multi_jit(unsigned, unsigned, code_model, bool, bool);
    multi_jit(const multi_jit &) = delete;
    multi_jit(multi_jit &&) noexcept = delete;
    llvm_multi_state &operator=(const multi_jit &) = delete;
    llvm_multi_state &operator=(multi_jit &&) noexcept = delete;
    ~multi_jit() = default;

    // Helper to fetch the context from its thread-safe counterpart.
    [[nodiscard]] llvm::LLVMContext &context() const noexcept
    {
        return *m_ctx->getContext();
    }
};

#if 0

// A task dispatcher class built on top of TBB's task group.
class tbb_task_dispatcher : public llvm::orc::TaskDispatcher
{
    oneapi::tbb::task_group m_tg;

public:
    void dispatch(std::unique_ptr<llvm::orc::Task> T) override
    {
        m_tg.run([T = std::move(T)]() { T->run(); });
    }
    void shutdown() override
    {
        m_tg.wait();
    }
    ~tbb_task_dispatcher() noexcept
    {
        m_tg.wait();
    }
};

#endif

// Reserved identifier for the master module in an llvm_multi_state.
constexpr auto master_module_name = "heyoka.master";

// NOTE: this largely replicates the logic from the constructors of llvm_state and llvm_state::jit.
// NOTE: make sure to coordinate changes in this constructor with llvm_state::jit.
multi_jit::multi_jit(unsigned n_modules, unsigned opt_level, code_model c_model, bool force_avx512, bool slp_vectorize)
    : m_n_modules(n_modules)
{
    assert(n_modules >= 2u);

    // NOTE: we assume here that the input arguments have
    // been validated already.
    assert(opt_level <= 3u);
    assert(c_model >= code_model::tiny && c_model <= code_model::large);

    // Ensure the native target is inited.
    init_native_target();

    // Create the target machine builder.
    auto jtmb = create_jit_tmb(opt_level, c_model);

    // Create the jit builder.
    llvm::orc::LLJITBuilder lljit_builder;
    // NOTE: other settable properties may
    // be of interest:
    // https://www.llvm.org/doxygen/classllvm_1_1orc_1_1LLJITBuilder.html
    lljit_builder.setJITTargetMachineBuilder(jtmb);

#if 0
    // Create a task dispatcher.
    auto tdisp = std::make_unique<tbb_task_dispatcher>();

    // Create an ExecutorProcessControl.
    auto epc = llvm::orc::SelfExecutorProcessControl::Create(nullptr, std::move(tdisp));
    // LCOV_EXCL_START
    if (!epc) {
        auto err = epc.takeError();

        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(
            fmt::format("Could not create a SelfExecutorProcessControl. The full error message is:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP

    // Set it in the lljit builder.
    lljit_builder.setExecutorProcessControl(std::move(*epc));
#else

    // Set the number of compilation threads.
    lljit_builder.setNumCompileThreads(std::thread::hardware_concurrency());

#endif

    // Create the jit.
    auto lljit = lljit_builder.create();
    // LCOV_EXCL_START
    if (!lljit) {
        auto err = lljit.takeError();

        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(
            fmt::format("Could not create an LLJIT object. The full error message is:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP
    m_lljit = std::move(*lljit);

    // Setup the machinery to store the modules' binary code
    // when it is generated.
    m_lljit->getObjTransformLayer().setTransform([this](std::unique_ptr<llvm::MemoryBuffer> obj_buffer) {
        assert(obj_buffer);

        // Lock down for access to m_object_files.
        std::lock_guard lock{m_object_files_mutex};

        assert(m_object_files.size() <= m_n_modules);

        // NOTE: this callback will be invoked the first time a jit lookup is performed,
        // even if the object code was manually injected. In such a case, m_object_files
        // has already been set up properly and we just sanity check in debug mode that
        // one object file matches the content of obj_buffer.
        if (m_object_files.size() < m_n_modules) {
            // Add obj_buffer.
            m_object_files.push_back(std::string(obj_buffer->getBufferStart(), obj_buffer->getBufferEnd()));
        } else {
            // Check that at least one buffer in m_object_files is exactly
            // identical to obj_buffer.
            assert(std::ranges::any_of(m_object_files, [&obj_buffer](const auto &cur) {
                return obj_buffer->getBufferSize() == cur.size()
                       && std::equal(obj_buffer->getBufferStart(), obj_buffer->getBufferEnd(), cur.begin());
                ;
            }));
        }

        return llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>(std::move(obj_buffer));
    });

    // Setup the machinery to run the optimisation passes on the modules.
    m_lljit->getIRTransformLayer().setTransform(
        [this, opt_level, force_avx512, slp_vectorize, c_model](llvm::orc::ThreadSafeModule TSM,
                                                                llvm::orc::MaterializationResponsibility &) {
            // See here for an explanation of what withModuleDo() entails:
            //
            // https://groups.google.com/g/llvm-dev/c/QauU4L_bHac
            //
            // In our case, the locking/thread safety aspect is not important as we are not sharing
            // contexts between threads. More references from discord:
            //
            // https://discord.com/channels/636084430946959380/687692371038830597/1252428080648163328
            // https://discord.com/channels/636084430946959380/687692371038830597/1252118666187640892
            TSM.withModuleDo([this, opt_level, force_avx512, slp_vectorize, c_model](llvm::Module &M) {
                // NOTE: don't run any optimisation on the master module.
                if (M.getModuleIdentifier() != master_module_name) {
                    // NOTE: running the optimisation passes requires mutable access to a target
                    // machine. Thus, we create a new target machine per thread in order to avoid likely data races
                    // with a shared target machine.

                    // Fetch a target machine builder.
                    auto jtmb = detail::create_jit_tmb(opt_level, c_model);

                    // Try creating the target machine.
                    auto tm = jtmb.createTargetMachine();
                    // LCOV_EXCL_START
                    if (!tm) [[unlikely]] {
                        throw std::invalid_argument("Error creating the target machine");
                    }
                    // LCOV_EXCL_STOP

                    // NOTE: we used to fetch the target triple from the lljit object,
                    // but recently we switched to asking the target triple directly
                    // from the target machine. Assert equality between the two for a while,
                    // just in case.
                    // NOTE: lljit.getTargetTriple() just returns a const ref to an internal
                    // object, it should be ok with concurrent invocation.
                    assert(m_lljit->getTargetTriple() == (*tm)->getTargetTriple());
                    // NOTE: the target triple is also available in the module.
                    assert(m_lljit->getTargetTriple().str() == M.getTargetTriple());

                    // Optimise the module.
                    detail::optimise_module(M, **tm, opt_level, force_avx512, slp_vectorize);
                } else {
                    ;
                }

                // Store the optimised bitcode/IR for this module.
                auto bc_snap = detail::bc_from_module(M);
                auto ir_snap = detail::ir_from_module(M);

                // NOTE: protect for multi-threaded access.
                std::lock_guard lock{m_ir_bc_mutex};

                m_bc_snapshots.push_back(std::move(bc_snap));
                m_ir_snapshots.push_back(std::move(ir_snap));
            });

            return llvm::Expected<llvm::orc::ThreadSafeModule>(std::move(TSM));
        });

    // Setup the jit so that it can look up symbols from the current process.
    auto dlsg
        = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(m_lljit->getDataLayout().getGlobalPrefix());
    // LCOV_EXCL_START
    if (!dlsg) {
        throw std::invalid_argument("Could not create the dynamic library search generator");
    }
    // LCOV_EXCL_STOP
    m_lljit->getMainJITDylib().addGenerator(std::move(*dlsg));

    // Create the master context.
    m_ctx = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());

    // Create the master module.
    m_module = std::make_unique<llvm::Module>(master_module_name, context());
    // Setup the data layout and the target triple.
    m_module->setDataLayout(m_lljit->getDataLayout());
    m_module->setTargetTriple(m_lljit->getTargetTriple().str());

    // Create a new builder for the master module.
    // NOTE: no need to mess around with fast math flags for this builder.
    m_builder = std::make_unique<ir_builder>(context());
}

} // namespace

} // namespace detail

struct llvm_multi_state::impl {
    std::vector<llvm_state> m_states;
    std::unique_ptr<detail::multi_jit> m_jit;
};

llvm_multi_state::llvm_multi_state() = default;

llvm_multi_state::llvm_multi_state(std::vector<llvm_state> states_)
{
    // Fetch a const ref, as we want to make extra sure we do not modify
    // states_ until we move it to construct the impl.
    const auto &states = states_;

    // We need at least 1 state.
    if (states.empty()) [[unlikely]] {
        throw std::invalid_argument("At least 1 llvm_state object is needed to construct an llvm_multi_state");
    }

    // All states must be uncompiled.
    if (std::ranges::any_of(states, &llvm_state::is_compiled)) [[unlikely]] {
        throw std::invalid_argument("An llvm_multi_state can be constructed only from uncompiled llvm_state objects");
    }

    // Module names must not collide with master_module_name.
    if (std::ranges::any_of(states, [](const auto &s) { return s.module_name() == detail::master_module_name; }))
        [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("An invalid llvm_state was passed to the constructor of an llvm_multi_state: the module name "
                        "'{}' is reserved for internal use by llvm_multi_state",
                        detail::master_module_name));
    }

    // Settings in all states must be consistent.
    auto states_differ = [](const llvm_state &s1, const llvm_state &s2) {
        if (s1.get_opt_level() != s2.get_opt_level()) {
            return true;
        }

        if (s1.fast_math() != s2.fast_math()) {
            return true;
        }

        if (s1.force_avx512() != s2.force_avx512()) {
            return true;
        }

        if (s1.get_slp_vectorize() != s2.get_slp_vectorize()) {
            return true;
        }

        if (s1.get_code_model() != s2.get_code_model()) {
            return true;
        }

        // NOTE: bit of paranoia here.
        assert(s1.m_jitter->m_lljit->getDataLayout() == s2.m_jitter->m_lljit->getDataLayout());
        assert(s1.m_jitter->get_target_triple() == s2.m_jitter->get_target_triple());
        assert(s1.m_jitter->get_target_cpu() == s2.m_jitter->get_target_cpu());
        assert(s1.m_jitter->get_target_features() == s2.m_jitter->get_target_features());

        return false;
    };

    if (std::ranges::adjacent_find(states, states_differ) != states.end()) [[unlikely]] {
        throw std::invalid_argument(
            "Inconsistent llvm_state settings detected in the constructor of an llvm_multi_state");
    }

    // Fetch settings from the first state.
    const auto opt_level = states[0].get_opt_level();
    const auto c_model = states[0].get_code_model();
    const auto force_avx512 = states[0].force_avx512();
    const auto slp_vectorize = states[0].get_slp_vectorize();

    // Create the multi_jit.
    auto jit = std::make_unique<detail::multi_jit>(boost::safe_numerics::safe<unsigned>(states.size()) + 1, opt_level,
                                                   c_model, force_avx512, slp_vectorize);

    // Build and assign the implementation.
    impl imp{.m_states = std::move(states_), .m_jit = std::move(jit)};
    m_impl = std::make_unique<impl>(std::move(imp));
}

llvm_multi_state::llvm_multi_state(const llvm_multi_state &other)
{
    // NOTE: start off by creating a new jit and copying the states.
    // This will work regardless of whether other is compiled or not.
    // No need to do any validation on the states are they are coming
    // from a llvm_multi_state and they have been checked already.
    impl imp{.m_states = other.m_impl->m_states,
             .m_jit = std::make_unique<detail::multi_jit>(other.m_impl->m_jit->m_n_modules, other.get_opt_level(),
                                                          other.get_code_model(), other.force_avx512(),
                                                          other.get_slp_vectorize())};
    m_impl = std::make_unique<impl>(std::move(imp));

    if (other.is_compiled()) {
        // 'other' was compiled.

        // Reset builder and module.
        m_impl->m_jit->m_module.reset();
        m_impl->m_jit->m_builder.reset();

        // Copy over the snapshots and the object files,
        m_impl->m_jit->m_object_files = other.m_impl->m_jit->m_object_files;
        m_impl->m_jit->m_ir_snapshots = other.m_impl->m_jit->m_ir_snapshots;
        m_impl->m_jit->m_bc_snapshots = other.m_impl->m_jit->m_bc_snapshots;

        // Add the files to the jit.
        for (const auto &obj : m_impl->m_jit->m_object_files) {
            detail::add_obj_to_lljit(*m_impl->m_jit->m_lljit, obj);
        }
    } else {
        // If 'other' was not compiled, we do not need to do anything - the
        // copy construction of the states takes care of everything. I.e., this
        // is basically the same as construction from a list of states.
        // NOTE: regarding the master module: this is always created empty
        // and it remains empty until compilation, thus we do not need to care
        // about it if other is uncompiled - the new empty master module constructed
        // with the jit is ok.
        assert(other.m_impl->m_jit->m_object_files.empty());
        assert(other.m_impl->m_jit->m_ir_snapshots.empty());
        assert(other.m_impl->m_jit->m_bc_snapshots.empty());
    }
}

llvm_multi_state::llvm_multi_state(llvm_multi_state &&) noexcept = default;

llvm_multi_state &llvm_multi_state::operator=(const llvm_multi_state &other)
{
    if (this != &other) {
        *this = llvm_multi_state(other);
    }

    return *this;
}

llvm_multi_state &llvm_multi_state::operator=(llvm_multi_state &&) noexcept = default;

llvm_multi_state::~llvm_multi_state() = default;

void llvm_multi_state::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    // Start by establishing if the state is compiled.
    const auto cmp = is_compiled();
    ar << cmp;

    // Store the states.
    ar << m_impl->m_states;

    // Store the object files and the snapshots. These may be empty.
    ar << m_impl->m_jit->m_object_files;
    ar << m_impl->m_jit->m_ir_snapshots;
    ar << m_impl->m_jit->m_bc_snapshots;

    // NOTE: no need to explicitly store the bitcode of the master
    // module: if this is compiled, the master module is in the snapshots.
    // Otherwise, the master module is empty and there's no need to
    // store anything.
}

void llvm_multi_state::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        // Load the compiled status flag from the archive.
        // NOLINTNEXTLINE(misc-const-correctness)
        bool cmp{};
        ar >> cmp;

        // Load the states.
        ar >> m_impl->m_states;

        // Reset the jit with a new one.
        m_impl->m_jit = std::make_unique<detail::multi_jit>(
            boost::safe_numerics::safe<unsigned>(m_impl->m_states.size()) + 1, get_opt_level(), get_code_model(),
            force_avx512(), get_slp_vectorize());

        // Load the object files and the snapshots.
        ar >> m_impl->m_jit->m_object_files;
        ar >> m_impl->m_jit->m_ir_snapshots;
        ar >> m_impl->m_jit->m_bc_snapshots;

        if (cmp) {
            // If the stored state was compiled, we need to reset
            // master builder and module. Otherwise, the empty default-constructed
            // master module is ok (the master module remains empty until compilation
            // is triggered).
            m_impl->m_jit->m_module.reset();
            m_impl->m_jit->m_builder.reset();

            // We also need to add all the object files to the jit.
            for (const auto &obj : m_impl->m_jit->m_object_files) {
                detail::add_obj_to_lljit(*m_impl->m_jit->m_lljit, obj);
            }
        }

        // Debug checks.
        assert((m_impl->m_jit->m_object_files.empty() && !cmp)
               || m_impl->m_jit->m_object_files.size() == m_impl->m_jit->m_n_modules);
        assert((m_impl->m_jit->m_object_files.empty() && !cmp)
               || m_impl->m_jit->m_ir_snapshots.size() == m_impl->m_jit->m_n_modules);
        assert((m_impl->m_jit->m_object_files.empty() && !cmp)
               || m_impl->m_jit->m_bc_snapshots.size() == m_impl->m_jit->m_n_modules);

        // LCOV_EXCL_START
    } catch (...) {
        m_impl.reset();

        throw;
    }
    // LCOV_EXCL_STOP
}

void llvm_multi_state::add_obj_triggers()
{
    // NOTE: the idea here is that we add one trigger function per module, and then
    // we invoke all the trigger functions from a trigger function in the master module.
    // Like this, we ensure materialisation of all modules when we lookup the
    // master trigger.

    // Implement the per-module triggers.
    for (decltype(m_impl->m_states.size()) i = 0; i < m_impl->m_states.size(); ++i) {
        // Fetch builder/module/context for the current state.
        auto &bld = m_impl->m_states[i].builder();
        auto &md = m_impl->m_states[i].module();
        auto &ctx = m_impl->m_states[i].context();

        // The function name.
        const auto fname = fmt::format("{}_{}", detail::obj_trigger_name, i);

        auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {}, false);
        assert(ft != nullptr);
        auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, fname.c_str(), &md);
        assert(f != nullptr);

        bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));
        bld.CreateRetVoid();
    }

    // Fetch the master builder/module/context.
    auto &bld = *m_impl->m_jit->m_builder;
    auto &md = *m_impl->m_jit->m_module;
    auto &ctx = m_impl->m_jit->context();

    // Add the prototypes of all per-module trigger functions to the master module.
    std::vector<llvm::Function *> callees;
    callees.reserve(m_impl->m_states.size());
    for (decltype(m_impl->m_states.size()) i = 0; i < m_impl->m_states.size(); ++i) {
        // The function name.
        const auto fname = fmt::format("{}_{}", detail::obj_trigger_name, i);

        auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {}, false);
        assert(ft != nullptr);
        auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, fname.c_str(), &md);
        assert(f != nullptr);

        callees.push_back(f);
    }

    // Create the master trigger function.
    auto *ft = llvm::FunctionType::get(bld.getVoidTy(), {}, false);
    assert(ft != nullptr);
    auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, detail::obj_trigger_name, &md);
    assert(f != nullptr);

    bld.SetInsertPoint(llvm::BasicBlock::Create(ctx, "entry", f));

    // Invoke all the triggers.
    for (auto *tf : callees) {
        bld.CreateCall(tf, {});
    }

    // Return.
    bld.CreateRetVoid();
}

void llvm_multi_state::check_compiled(const char *f) const
{
    if (m_impl->m_jit->m_module) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}' can be invoked only after the llvm_multi_state has been compiled", f));
    }
}

void llvm_multi_state::check_uncompiled(const char *f) const
{
    if (!m_impl->m_jit->m_module) [[unlikely]] {
        throw std::invalid_argument(
            fmt::format("The function '{}' can be invoked only if the llvm_multi_state has not been compiled yet", f));
    }
}

unsigned llvm_multi_state::get_n_modules() const noexcept
{
    return m_impl->m_jit->m_n_modules;
}

unsigned llvm_multi_state::get_opt_level() const noexcept
{
    return m_impl->m_states[0].get_opt_level();
}

bool llvm_multi_state::fast_math() const noexcept
{
    return m_impl->m_states[0].fast_math();
}

bool llvm_multi_state::force_avx512() const noexcept
{
    return m_impl->m_states[0].force_avx512();
}

bool llvm_multi_state::get_slp_vectorize() const noexcept
{
    return m_impl->m_states[0].get_slp_vectorize();
}

code_model llvm_multi_state::get_code_model() const noexcept
{
    return m_impl->m_states[0].get_code_model();
}

bool llvm_multi_state::is_compiled() const noexcept
{
    return !m_impl->m_jit->m_module;
}

std::vector<std::string> llvm_multi_state::get_ir() const
{
    if (is_compiled()) {
        return m_impl->m_jit->m_ir_snapshots;
    } else {
        std::vector<std::string> retval;
        retval.reserve(m_impl->m_jit->m_n_modules);

        for (const auto &s : m_impl->m_states) {
            retval.push_back(s.get_ir());
        }

        // Add the IR from the master module.
        retval.push_back(detail::ir_from_module(*m_impl->m_jit->m_module));

        return retval;
    }
}

std::vector<std::string> llvm_multi_state::get_bc() const
{
    if (is_compiled()) {
        return m_impl->m_jit->m_bc_snapshots;
    } else {
        std::vector<std::string> retval;
        retval.reserve(m_impl->m_jit->m_n_modules);

        for (const auto &s : m_impl->m_states) {
            retval.push_back(s.get_bc());
        }

        // Add the bitcode from the master module.
        retval.push_back(detail::bc_from_module(*m_impl->m_jit->m_module));

        return retval;
    }
}

const std::vector<std::string> &llvm_multi_state::get_object_code() const
{
    check_compiled(__func__);

    return m_impl->m_jit->m_object_files;
}

// NOTE: this function is NOT exception-safe, proper cleanup
// needs to be done externally if needed.
void llvm_multi_state::compile_impl()
{
    // Add all the modules from the states.
    for (auto &s : m_impl->m_states) {
        detail::add_module_to_lljit(*m_impl->m_jit->m_lljit, std::move(s.m_module), *s.m_jitter->m_ctx);

        // Clear out the builder.
        s.m_builder.reset();

        // NOTE: need to manually construct the object file, as this would
        // normally be done by the invocation of s.compile() (which we do not do).
        s.m_jitter->m_object_file.emplace();
    }

    // Add the master module.
    detail::add_module_to_lljit(*m_impl->m_jit->m_lljit, std::move(m_impl->m_jit->m_module), *m_impl->m_jit->m_ctx);

    // Clear out the master builder.
    m_impl->m_jit->m_builder.reset();

    // Trigger optimisation and object code materialisation via lookup.
    jit_lookup(detail::obj_trigger_name);

    // Sanity checks.
    assert(m_impl->m_jit->m_bc_snapshots.size() == m_impl->m_jit->m_n_modules);
    assert(m_impl->m_jit->m_ir_snapshots.size() == m_impl->m_jit->m_n_modules);
    assert(m_impl->m_jit->m_object_files.size() == m_impl->m_jit->m_n_modules);
}

void llvm_multi_state::compile()
{
    check_uncompiled(__func__);

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Verify the modules before compiling.
    // NOTE: probably this can be parallelised if needed.
    for (decltype(m_impl->m_states.size()) i = 0; i < m_impl->m_states.size(); ++i) {
        detail::verify_module(*m_impl->m_states[i].m_module);
    }

    logger->trace("llvm_multi_state module verification runtime: {}", sw);

    try {
        // Add the object materialisation trigger functions.
        // NOTE: contrary to llvm_state::add_obj_trigger(), add_obj_triggers()
        // does not implement any automatic cleanup in case of errors. Thus, we fold
        // it into the try/catch block in order to avoid leaving the
        // llvm_multi_state in a half-baked state.
        add_obj_triggers();

        // Fetch the bitcode *before* optimisation.
        std::vector<std::string> obc;
        obc.reserve(boost::safe_numerics::safe<decltype(obc.size())>(m_impl->m_states.size()) + 1u);
        for (const auto &s : m_impl->m_states) {
            obc.push_back(s.get_bc());
        }
        // Add the master bitcode.
        obc.push_back(detail::bc_from_module(*m_impl->m_jit->m_module));

        // Assemble the compilation flag.
        const auto comp_flag
            = detail::assemble_comp_flag(get_opt_level(), force_avx512(), get_slp_vectorize(), get_code_model());

        // Lookup in the cache.
        if (auto cached_data = detail::llvm_state_mem_cache_lookup(obc, comp_flag)) {
            // Cache hit.

            // Assign the optimised snapshots.
            assert(cached_data->opt_ir.size() == m_impl->m_jit->m_n_modules);
            assert(cached_data->opt_bc.size() == m_impl->m_jit->m_n_modules);
            assert(cached_data->obj.size() == m_impl->m_jit->m_n_modules);
            assert(m_impl->m_jit->m_ir_snapshots.empty());
            assert(m_impl->m_jit->m_bc_snapshots.empty());
            m_impl->m_jit->m_ir_snapshots = std::move(cached_data->opt_ir);
            m_impl->m_jit->m_bc_snapshots = std::move(cached_data->opt_bc);

            // NOTE: here it is important that we replicate the logic happening
            // in llvm_state::compile(): clear out module/builder, construct
            // the object file. The snapshots can be left empty.
            for (auto &s : m_impl->m_states) {
                s.m_module.reset();
                s.m_builder.reset();
                s.m_jitter->m_object_file.emplace();
            }

            // Clear out master module and builder.
            m_impl->m_jit->m_module.reset();
            m_impl->m_jit->m_builder.reset();

            // Add and assign the object files.
            for (const auto &obj : cached_data->obj) {
                detail::add_obj_to_lljit(*m_impl->m_jit->m_lljit, obj);
            }

            // Assign the compiled objects.
            assert(m_impl->m_jit->m_object_files.empty());
            m_impl->m_jit->m_object_files = std::move(cached_data->obj);

            // Lookup the trigger.
            jit_lookup(detail::obj_trigger_name);
        } else {
            // Cache miss.

            sw.reset();

            // Run the compilation.
            compile_impl();

            logger->trace("optimisation + materialisation runtime: {}", sw);

            // NOTE: at this point, m_ir_snapshots, m_bc_snapshots and m_object_files
            // have all been constructed in random order because of multithreading.
            // Sort them so that we provided deterministic behaviour. Probably
            // not strictly needed, but let's try to avoid nondeterminism.
            // All of this can be parallelised if needed.
            std::ranges::sort(m_impl->m_jit->m_ir_snapshots);
            std::ranges::sort(m_impl->m_jit->m_bc_snapshots);
            std::ranges::sort(m_impl->m_jit->m_object_files);

            // Try to insert obc into the cache.
            detail::llvm_state_mem_cache_try_insert(std::move(obc), comp_flag,
                                                    {.opt_bc = m_impl->m_jit->m_bc_snapshots,
                                                     .opt_ir = m_impl->m_jit->m_ir_snapshots,
                                                     .obj = m_impl->m_jit->m_object_files});
            // LCOV_EXCL_START
        }
    } catch (...) {
        // Reset to a def-cted state in case of error,
        // as it looks like there's no way of recovering.
        m_impl.reset();

        throw;
    }
    // LCOV_EXCL_STOP
}

std::uintptr_t llvm_multi_state::jit_lookup(const std::string &name)
{
    check_compiled(__func__);

    auto sym = m_impl->m_jit->m_lljit->lookup(name);
    if (!sym) {
        throw std::invalid_argument(fmt::format("Could not find the symbol '{}' in an llvm_multi_state", name));
    }

    return static_cast<std::uintptr_t>((*sym).getValue());
}

std::ostream &operator<<(std::ostream &os, const llvm_multi_state &s)
{
    std::ostringstream oss;
    oss << std::boolalpha;

    oss << "N of modules      : " << s.get_n_modules() << '\n';
    oss << "Compiled          : " << s.is_compiled() << '\n';
    oss << "Fast math         : " << s.fast_math() << '\n';
    oss << "Force AVX512      : " << s.force_avx512() << '\n';
    oss << "SLP vectorization : " << s.get_slp_vectorize() << '\n';
    oss << "Code model        : " << s.get_code_model() << '\n';
    oss << "Optimisation level: " << s.get_opt_level() << '\n';
    oss << "Data layout       : " << s.m_impl->m_states[0].m_jitter->m_lljit->getDataLayout().getStringRepresentation()
        << '\n';
    oss << "Target triple     : " << s.m_impl->m_states[0].m_jitter->get_target_triple().str() << '\n';
    oss << "Target CPU        : " << s.m_impl->m_states[0].m_jitter->get_target_cpu() << '\n';
    oss << "Target features   : " << s.m_impl->m_states[0].m_jitter->get_target_features() << '\n';

    return os << oss.str();
}

HEYOKA_END_NAMESPACE
