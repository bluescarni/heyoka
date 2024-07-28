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
#include <fstream>
#include <initializer_list>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
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
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>

#if LLVM_VERSION_MAJOR < 14

// NOTE: this header was moved in LLVM 14.
#include <llvm/Support/TargetRegistry.h>

#else

#include <llvm/MC/TargetRegistry.h>

#endif

// NOTE: new pass manager API.
// NOTE: this is available since LLVM 13, but in that
// version it seems like auto-vectorization with
// vector-function-abi-variant is not working
// properly with the new pass manager. Hence, we
// enable it from LLVM 14.
#if LLVM_VERSION_MAJOR >= 14

#define HEYOKA_USE_NEW_LLVM_PASS_MANAGER

#endif

#if defined(HEYOKA_USE_NEW_LLVM_PASS_MANAGER)

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>

#if LLVM_VERSION_MAJOR >= 14

// NOTE: this header is available since LLVM 14.
#include <llvm/Passes/OptimizationLevel.h>

#endif

#else

#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#endif

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

// NOTE: logging here lhames' instructions on how to set up LLJIT
// for parallel compilation of multiple modules.
//
//   auto J = LLJITBuilder()
//              .setNumCompileThreads(<N>)
//              .create();
//   if (!J) { /* bail on error */ }
//   (*J)->getIRTransformLayer().setTransform(
//     [](ThreadSafeModule TSM, MaterializationResponsibility &R) -> Expected<ThreadSafeModule> {
//       TSM.withModuleDo([](Module &M) {
//         /* Apply your IR optimizations here */
//       });
//       return std::move(TSM);
//     });
//
// Note that the optimisation passes in this approach are moved into the
// transform layer. References:
// https://discord.com/channels/636084430946959380/687692371038830597/1252428080648163328
// https://discord.com/channels/636084430946959380/687692371038830597/1252118666187640892

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
    if (!jtmb) {
        throw std::invalid_argument("Error creating a JITTargetMachineBuilder for the host system");
    }

    auto tm = jtmb->createTargetMachine();
    if (!tm) {
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

    explicit jit(unsigned opt_level, code_model c_model)
    {
        // NOTE: we assume here the opt level has already been clamped
        // from the outside.
        assert(opt_level <= 3u);

        // Ensure the native target is inited.
        detail::init_native_target();

        // NOTE: codegen opt level changed in LLVM 18.
#if LLVM_VERSION_MAJOR < 18

        using cg_opt_level = llvm::CodeGenOpt::Level;

#else

        using cg_opt_level = llvm::CodeGenOptLevel;

#endif

        // Create the target machine builder.
        auto jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
        // LCOV_EXCL_START
        if (!jtmb) {
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
                // LCOV_EXCL_START
                // NOTE: we should never end up here.
                assert(false);
                ;
                // LCOV_EXCL_STOP
        }

        // Create the jit builder.
        llvm::orc::LLJITBuilder lljit_builder;
        // NOTE: other settable properties may
        // be of interest:
        // https://www.llvm.org/doxygen/classllvm_1_1orc_1_1LLJITBuilder.html
        lljit_builder.setJITTargetMachineBuilder(*jtmb);

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
        auto tm = jtmb->createTargetMachine();
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
    [[nodiscard]] llvm::TargetIRAnalysis get_target_ir_analysis() const
    {
        return m_tm->getTargetIRAnalysis();
    }
    [[nodiscard]] const llvm::Triple &get_target_triple() const
    {
        return m_lljit->getTargetTriple();
    }

    void add_module(std::unique_ptr<llvm::Module> m) const
    {
        auto err = m_lljit->addIRModule(llvm::orc::ThreadSafeModule(std::move(m), *m_ctx));

        // LCOV_EXCL_START
        if (err) {
            std::string err_report;
            llvm::raw_string_ostream ostr(err_report);

            ostr << err;

            throw std::invalid_argument(fmt::format(
                "The function for adding a module to the jit failed. The full error message:\n{}", ostr.str()));
        }
        // LCOV_EXCL_STOP
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

// Helper to load object code into a jit.
template <typename Jit>
void llvm_state_add_obj_to_jit(Jit &j, std::string obj)
{
    llvm::SmallVector<char, 0> buffer(obj.begin(), obj.end());
    auto err = j.m_lljit->addObjectFile(std::make_unique<llvm::SmallVectorMemoryBuffer>(std::move(buffer)));

    // LCOV_EXCL_START
    if (err) {
        std::string err_report;
        llvm::raw_string_ostream ostr(err_report);

        ostr << err;

        throw std::invalid_argument(fmt::format(
            "The function for adding a compiled module to the jit failed. The full error message:\n{}", ostr.str()));
    }
    // LCOV_EXCL_STOP

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
auto llvm_state_bc_to_module(const std::string &module_name, const std::string &bc, llvm::LLVMContext &ctx)
{
    // Create the corresponding memory buffer.
    auto mb = llvm::MemoryBuffer::getMemBuffer(bc);
    assert(mb);

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
        m_module = detail::llvm_state_bc_to_module(m_module_name, other.get_bc(), context());

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

llvm_state::~llvm_state()
{
    // Sanity checks in debug mode.
    if (m_jitter) {
        if (is_compiled()) {
            assert(m_jitter->m_object_file);
            assert(!m_builder);
        } else {
            assert(!m_jitter->m_object_file);
            assert(m_builder);
            assert(m_ir_snapshot.empty());
            assert(m_bc_snapshot.empty());
        }
    }

    assert(m_opt_level <= 3u);
}

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
            m_module = detail::llvm_state_bc_to_module(m_module_name, bc_snapshot, context());

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

        throw std::invalid_argument(fmt::format(
            "The verification of the function '{}' failed. The full error message:\n{}", fname, ostr.str()));
    }
}

void llvm_state::verify_function(const std::string &name)
{
    check_uncompiled(__func__);

    // Lookup the function in the module.
    auto *f = m_module->getFunction(name);
    if (f == nullptr) {
        throw std::invalid_argument(fmt::format("The function '{}' does not exist in the module", name));
    }

    // Run the actual check.
    verify_function(f);
}

void llvm_state::optimise()
{
    check_uncompiled(__func__);

    // NOTE: don't run any optimisation pass at O0.
    if (m_opt_level == 0u) {
        return;
    }

    // NOTE: the logic here largely mimics (with a lot of simplifications)
    // the implementation of the 'opt' tool. See:
    // https://github.com/llvm/llvm-project/blob/release/10.x/llvm/tools/opt/opt.cpp

    // For every function in the module, setup its attributes
    // so that the codegen uses all the features available on
    // the host CPU.
    const auto cpu = m_jitter->get_target_cpu();
    const auto features = m_jitter->get_target_features();

    auto &ctx = context();

    for (auto &f : module()) {
        auto attrs = f.getAttributes();

        llvm::AttrBuilder
#if LLVM_VERSION_MAJOR < 14
            new_attrs
#else
            new_attrs(ctx)
#endif
            ;

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
#if LLVM_VERSION_MAJOR < 14
        f.setAttributes(attrs.addAttributes(ctx, llvm::AttributeList::FunctionIndex, new_attrs));
#else
        f.setAttributes(attrs.addFnAttributes(ctx, new_attrs));
#endif
    }

    // Force usage of AVX512 registers, if requested.
    if (m_force_avx512 && detail::get_target_features().avx512f) {
        for (auto &f : module()) {
            f.addFnAttr("prefer-vector-width", "512");
        }
    }

#if defined(HEYOKA_USE_NEW_LLVM_PASS_MANAGER)

    // NOTE: adapted from here:
    // https://llvm.org/docs/NewPassManager.html

    // Optimisation level for the module pass manager.
    // NOTE: the OptimizationLevel class has changed location
    // since LLVM 14.
#if LLVM_VERSION_MAJOR >= 14
    using olevel = llvm::OptimizationLevel;
#else
    using olevel = llvm::PassBuilder::OptimizationLevel;
#endif

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
    llvm::TargetLibraryInfoImpl TLII(m_jitter->get_target_triple());
    FAM.registerPass([&] { return llvm::TargetLibraryAnalysis(TLII); });

    // Create the new pass manager builder, passing
    // the native target machine from the JIT class.
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
    pto.SLPVectorization = m_slp_vectorize;
    llvm::PassBuilder PB(m_jitter->m_tm.get(), pto);

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Construct the optimisation level.
    olevel ol{};

    switch (m_opt_level) {
        case 1u:
            ol = olevel::O1;
            break;
        case 2u:
            ol = olevel::O2;
            break;
        default:
            assert(m_opt_level == 3u);
            ol = olevel::O3;
    }

    // Create the module pass manager.
    auto MPM = PB.buildPerModuleDefaultPipeline(ol);

    // Optimize the IR.
    MPM.run(*m_module, MAM);

#else

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

    // We use the helper class PassManagerBuilder to populate the module
    // pass manager with standard options.
    llvm::PassManagerBuilder pm_builder;
    // See here for the defaults:
    // https://llvm.org/doxygen/PassManagerBuilder_8cpp_source.html
    pm_builder.OptLevel = m_opt_level;
    // Enable function inlining.
    pm_builder.Inliner = llvm::createFunctionInliningPass(m_opt_level, 0, false);
    // NOTE: if requested, we turn manually on the SLP vectoriser here, which is off
    // by default. Not sure why it is off, the LLVM docs imply this
    // is on by default at nonzero optimisation levels for clang and opt.
    pm_builder.SLPVectorize = m_slp_vectorize;

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

#endif
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

// NOTE: we need to emphasise in the docs that compilation
// triggers an optimisation pass.
void llvm_state::compile()
{
    check_uncompiled(__func__);

    // Log runtime in trace mode.
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Run a verification on the module before compiling.
    {
        std::string out;
        llvm::raw_string_ostream ostr(out);

        if (llvm::verifyModule(*m_module, &ostr)) {
            // LCOV_EXCL_START
            throw std::runtime_error(
                fmt::format("The verification of the module '{}' produced an error:\n{}", m_module_name, ostr.str()));
            // LCOV_EXCL_STOP
        }
    }

    logger->trace("module verification runtime: {}", sw);

    // Add the object materialisation trigger function.
    // NOTE: do it **after** verification, on the assumption
    // that add_obj_trigger() is implemented correctly. Like this,
    // if module verification fails, the user still has the option
    // to fix the module and re-attempt compilation without having
    // altered the module and without having already added the trigger
    // function.
    add_obj_trigger();

    try {
        // Fetch the bitcode *before* optimisation.
        auto orig_bc = get_bc();

        // Combine m_opt_level, m_force_avx512, m_slp_vectorize and m_c_model into a single value,
        // as they all affect codegen.
        // NOTE: here we need:
        // - 2 bits for m_opt_level,
        // - 1 bit for m_force_avx512 and m_slp_vectorize each,
        // - 3 bits for m_c_model,
        // for a total of 7 bits.
        assert(m_opt_level <= 3u);
        assert(static_cast<unsigned>(m_c_model) <= 7u);
        static_assert(std::numeric_limits<unsigned>::digits >= 7u);
        const auto olevel = m_opt_level + (static_cast<unsigned>(m_force_avx512) << 2)
                            + (static_cast<unsigned>(m_slp_vectorize) << 3) + (static_cast<unsigned>(m_c_model) << 4);

        if (auto cached_data = detail::llvm_state_mem_cache_lookup(orig_bc, olevel)) {
            // Cache hit.

            // Assign the snapshots.
            m_ir_snapshot = std::move(cached_data->opt_ir);
            m_bc_snapshot = std::move(cached_data->opt_bc);

            // Clear out module and builder.
            m_module.reset();
            m_builder.reset();

            // Assign the object file.
            detail::llvm_state_add_obj_to_jit(*m_jitter, std::move(cached_data->obj));
        } else {
            sw.reset();

            // Run the optimisation pass.
            optimise();

            logger->trace("optimisation runtime: {}", sw);

            sw.reset();

            // Run the compilation.
            compile_impl();

            logger->trace("materialisation runtime: {}", sw);

            // Try to insert orig_bc into the cache.
            detail::llvm_state_mem_cache_try_insert(std::move(orig_bc), olevel,
                                                    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
                                                    {m_bc_snapshot, m_ir_snapshot, *m_jitter->m_object_file});
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

#if LLVM_VERSION_MAJOR >= 15
    return static_cast<std::uintptr_t>((*sym).getValue());
#else
    return static_cast<std::uintptr_t>((*sym).getAddress());
#endif
}

std::string llvm_state::get_ir() const
{
    if (m_module) {
        // The module has not been compiled yet,
        // get the IR from it.
        std::string out;
        llvm::raw_string_ostream ostr(out);

        m_module->print(ostr, nullptr);

        return std::move(ostr.str());
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
        std::string out;
        llvm::raw_string_ostream ostr(out);

        llvm::WriteBitcodeToFile(*m_module, ostr);

        return std::move(ostr.str());
    } else {
        // The module has been compiled.
        // Return the bitcode snapshot that
        // was created before the compilation.
        return m_bc_snapshot;
    }
}

// LCOV_EXCL_START

void llvm_state::dump_object_code(const std::string &filename) const
{
    const auto &oc = get_object_code();

    std::ofstream ofs;
    // NOTE: turn on exceptions, and overwrite any existing content.
    ofs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    ofs.open(filename, std::ios_base::out | std::ios::trunc);

    // Write out the binary data to ofs.
    ofs.write(oc.data(), boost::numeric_cast<std::streamsize>(oc.size()));
}

// LCOV_EXCL_STOP

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

HEYOKA_END_NAMESPACE
