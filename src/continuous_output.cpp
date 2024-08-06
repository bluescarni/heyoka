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
#include <cmath>
#include <cstdint>
#include <ios>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <boost/core/demangle.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/continuous_output.hpp>
#include <heyoka/detail/dfloat.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

HEYOKA_BEGIN_NAMESPACE

// NOTE: there are situations (e.g., ensemble simulations) in which
// we may end up recompiling over and over the same code for the computation
// of continuous output. Perhaps we should consider some caching of llvm states
// containing continuous output functions.
template <typename T>
void continuous_output<T>::add_c_out_function(std::uint32_t order, std::uint32_t dim, bool high_accuracy)
{
#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        // Double check that the initialisation of the continuous_output
        // object in the integrator code set up everything
        // with consistent precisions.
        assert(!m_output.empty());
        assert(std::all_of(m_tcs.begin(), m_tcs.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_times_hi.begin(), m_times_hi.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_times_lo.begin(), m_times_lo.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_output.begin(), m_output.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
    }

#endif

    // Overflow check: we want to be able to index into the arrays of
    // times and Taylor coefficients using 32-bit ints.
    // LCOV_EXCL_START
    if (m_tcs.size() > std::numeric_limits<std::uint32_t>::max()
        || m_times_hi.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Overflow detected while adding continuous output to a Taylor integrator");
    }
    // LCOV_EXCL_STOP

    auto &md = m_llvm_state.module();
    auto &builder = m_llvm_state.builder();
    auto &context = m_llvm_state.context();

    // Fetch the internal floating-point type.
    auto *fp_t = detail::internal_llvm_type_like(m_llvm_state, m_output[0]);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // Add the function for the computation of the dense output.
    // NOTE: the dense output function operates on data in external format.
    detail::taylor_add_d_out_function(m_llvm_state, fp_t, dim, order, 1, high_accuracy, false);

    // Fetch it.
    auto *d_out_f = md.getFunction("d_out_f");
    assert(d_out_f != nullptr); // LCOV_EXCL_LINE

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // Establish the time direction.
    const detail::dfloat<T> df_t_start(m_times_hi[0], m_times_lo[0]), df_t_end(m_times_hi.back(), m_times_lo.back());
    const auto dir = df_t_start < df_t_end;

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the pointer to the time value (read/write: after the time value
    //   is read, the pointer will be re-used to store the h value
    //   that needs to be passed to the dense output function),
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the hi times (read-only),
    // - the pointer to the lo times (read-only).
    // No overlap is allowed. All pointers are external.
    auto *ext_fp_t = detail::to_external_llvm_type<T>(context);
    auto *ptr_t = llvm::PointerType::getUnqual(ext_fp_t);
    const std::vector<llvm::Type *> fargs(5u, ptr_t);
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, "c_out", &md);

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tm_ptr = f->args().begin() + 1;
    tm_ptr->setName("tm_ptr");
    tm_ptr->addAttr(llvm::Attribute::NoCapture);
    tm_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tc_ptr = f->args().begin() + 2;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *times_ptr_hi = f->args().begin() + 3;
    times_ptr_hi->setName("times_ptr_hi");
    times_ptr_hi->addAttr(llvm::Attribute::NoCapture);
    times_ptr_hi->addAttr(llvm::Attribute::NoAlias);
    times_ptr_hi->addAttr(llvm::Attribute::ReadOnly);

    auto *times_ptr_lo = f->args().begin() + 4;
    times_ptr_lo->setName("times_ptr_lo");
    times_ptr_lo->addAttr(llvm::Attribute::NoCapture);
    times_ptr_lo->addAttr(llvm::Attribute::NoAlias);
    times_ptr_lo->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Load the time value from tm_ptr.
    auto *tm = detail::ext_load_vector_from_memory(m_llvm_state, fp_t, tm_ptr, 1);

    // Look for the index in the times vector corresponding to
    // a time greater than tm (less than tm in backwards integration).
    // This is essentially an implementation of std::upper_bound:
    // https://en.cppreference.com/w/cpp/algorithm/upper_bound
    auto *tidx = builder.CreateAlloca(builder.getInt32Ty());
    auto *count = builder.CreateAlloca(builder.getInt32Ty());
    auto *step = builder.CreateAlloca(builder.getInt32Ty());
    auto *first = builder.CreateAlloca(builder.getInt32Ty());

    // count is inited with the size of the range.
    builder.CreateStore(builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size())), count);
    // first is inited to zero.
    builder.CreateStore(builder.getInt32(0), first);

    detail::llvm_while_loop(
        m_llvm_state,
        [&]() { return builder.CreateICmpNE(builder.CreateLoad(builder.getInt32Ty(), count), builder.getInt32(0)); },
        [&]() {
            // tidx = first.
            builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), first), tidx);
            // step = count / 2.
            builder.CreateStore(
                builder.CreateUDiv(builder.CreateLoad(builder.getInt32Ty(), count), builder.getInt32(2)), step);
            // tidx = tidx + step.
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), tidx),
                                                  builder.CreateLoad(builder.getInt32Ty(), step)),
                                tidx);

            // Logical condition:
            // - !(tm < *tidx), if integrating forward,
            // - !(tm > *tidx), if integrating backward.
            auto *tidx_val_hi = detail::ext_load_vector_from_memory(
                m_llvm_state, fp_t,
                builder.CreateInBoundsGEP(ext_fp_t, times_ptr_hi, builder.CreateLoad(builder.getInt32Ty(), tidx)), 1);
            auto *tidx_val_lo = detail::ext_load_vector_from_memory(
                m_llvm_state, fp_t,
                builder.CreateInBoundsGEP(ext_fp_t, times_ptr_lo, builder.CreateLoad(builder.getInt32Ty(), tidx)), 1);
            auto *zero_val = detail::llvm_constantfp(m_llvm_state, fp_t, 0.);
            auto *cond = dir ? detail::llvm_dl_lt(m_llvm_state, tm, zero_val, tidx_val_hi, tidx_val_lo)
                             : detail::llvm_dl_gt(m_llvm_state, tm, zero_val, tidx_val_hi, tidx_val_lo);
            cond = builder.CreateNot(cond);

            detail::llvm_if_then_else(
                m_llvm_state, cond,
                [&]() {
                    // ++tidx.
                    builder.CreateStore(
                        builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), tidx), builder.getInt32(1)), tidx);
                    // first = tidx.
                    builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), tidx), first);
                    // count = count - (step + 1).
                    builder.CreateStore(
                        builder.CreateSub(
                            builder.CreateLoad(builder.getInt32Ty(), count),
                            builder.CreateAdd(builder.CreateLoad(builder.getInt32Ty(), step), builder.getInt32(1))),
                        count);
                },
                [&]() {
                    // count = step.
                    builder.CreateStore(builder.CreateLoad(builder.getInt32Ty(), step), count);
                });
        });

    // NOTE: the output of the std::upper_bound algorithm
    // is in the 'first' variable.
    llvm::Value *tc_idx = builder.CreateLoad(builder.getInt32Ty(), first);

    // Normally, the TC index should be first - 1. The exceptions are:
    // - first == 0, in which case TC index is also 0,
    // - first == range size, in which case TC index is first - 2.
    // These two exceptions arise when tm is outside the range of validity
    // for the continuous output. In such cases, we will use either the first
    // or the last possible set of TCs.
    detail::llvm_if_then_else(
        m_llvm_state, builder.CreateICmpEQ(tc_idx, builder.getInt32(0)),
        []() {
            // first == 0, do nothing.
        },
        [&]() {
            detail::llvm_if_then_else(
                m_llvm_state,
                builder.CreateICmpEQ(tc_idx, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size()))),
                [&]() {
                    // first == range size.
                    builder.CreateStore(builder.CreateSub(tc_idx, builder.getInt32(2)), first);
                },
                [&]() {
                    // The normal path.
                    builder.CreateStore(builder.CreateSub(tc_idx, builder.getInt32(1)), first);
                });
        });

    // Reload tc_idx.
    tc_idx = builder.CreateLoad(builder.getInt32Ty(), first);

    // Load the time corresponding to tc_idx.
    auto *start_tm_hi = detail::ext_load_vector_from_memory(
        m_llvm_state, fp_t, builder.CreateInBoundsGEP(ext_fp_t, times_ptr_hi, tc_idx), 1);
    auto *start_tm_lo = detail::ext_load_vector_from_memory(
        m_llvm_state, fp_t, builder.CreateInBoundsGEP(ext_fp_t, times_ptr_lo, tc_idx), 1);

    // Compute and store the value of h = tm - start_tm into tm_ptr.
    auto [h_hi, h_lo] = detail::llvm_dl_add(m_llvm_state, tm, detail::llvm_constantfp(m_llvm_state, fp_t, 0.),
                                            detail::llvm_fneg(m_llvm_state, start_tm_hi),
                                            detail::llvm_fneg(m_llvm_state, start_tm_lo));
    detail::ext_store_vector_to_memory(m_llvm_state, tm_ptr, h_hi);

    // Compute the index into the Taylor coefficients array.
    tc_idx = builder.CreateMul(tc_idx, builder.getInt32(dim * (order + 1u)));

    // Invoke the d_out function.
    builder.CreateCall(d_out_f, {out_ptr, builder.CreateInBoundsGEP(ext_fp_t, tc_ptr, tc_idx), tm_ptr});

    // Create the return value.
    builder.CreateRetVoid();

    // Compile.
    m_llvm_state.compile();

    // Fetch the function pointer and assign it.
    m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
}

template <typename T>
continuous_output<T>::continuous_output() = default;

template <typename T>
continuous_output<T>::continuous_output(llvm_state &&s) : m_llvm_state(std::move(s))
{
}

template <typename T>
continuous_output<T>::continuous_output(const continuous_output &o)
    : m_llvm_state(o.m_llvm_state), m_tcs(o.m_tcs), m_times_hi(o.m_times_hi), m_times_lo(o.m_times_lo),
      m_output(o.m_output)
{
    // If o is valid, fetch the function pointer from the copied state.
    // Otherwise, m_f_ptr will remain null.
    if (o.m_f_ptr != nullptr) {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
continuous_output<T>::continuous_output(continuous_output &&) noexcept = default;

template <typename T>
continuous_output<T>::~continuous_output() = default;

template <typename T>
continuous_output<T> &continuous_output<T>::operator=(const continuous_output &o)
{
    if (this != &o) {
        *this = continuous_output(o);
    }

    return *this;
}

template <typename T>
continuous_output<T> &continuous_output<T>::operator=(continuous_output &&) noexcept = default;

// NOTE: pass by copy so that we are sure t does not
// alias other data.
template <typename T>
void continuous_output<T>::call_impl(T t)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)

    // m_output must not be empty.
    assert(!m_output.empty());
    // Need at least 2 time points.
    assert(m_times_hi.size() >= 2u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        // All data must have the same precision
        // (inferred from the first element of m_output).
        assert(std::all_of(m_tcs.begin(), m_tcs.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_times_hi.begin(), m_times_hi.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_times_lo.begin(), m_times_lo.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
        assert(std::all_of(m_output.begin(), m_output.end(),
                           [&](const auto &x) { return x.get_prec() == m_output[0].get_prec(); }));
    }

#endif

#endif
    // LCOV_EXCL_STOP

    if (!isfinite(t)) {
        throw std::invalid_argument(
            fmt::format("Cannot compute the continuous output at the non-finite time {}", detail::fp_to_string(t)));
    }

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        t.prec_round(m_output[0].get_prec());
    }

#endif

    m_f_ptr(m_output.data(), &t, m_tcs.data(), m_times_hi.data(), m_times_lo.data());
}

template <typename T>
const llvm_state &continuous_output<T>::get_llvm_state() const
{
    return m_llvm_state;
}

template <typename T>
const std::vector<T> &continuous_output<T>::operator()(T tm)
{
    call_impl(std::move(tm));
    return m_output;
}

template <typename T>
const std::vector<T> &continuous_output<T>::get_output() const
{
    return m_output;
}

template <typename T>
const std::vector<T> &continuous_output<T>::get_times() const
{
    return m_times_hi;
}

template <typename T>
const std::vector<T> &continuous_output<T>::get_tcs() const
{
    return m_tcs;
}

template <typename T>
void continuous_output<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_llvm_state;
    ar << m_tcs;
    ar << m_times_hi;
    ar << m_times_lo;
    ar << m_output;
}

template <typename T>
void continuous_output<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_llvm_state;
    ar >> m_tcs;
    ar >> m_times_hi;
    ar >> m_times_lo;
    ar >> m_output;

    // NOTE: if m_output is not empty, it means the archived
    // object had been initialised.
    if (m_output.empty()) {
        m_f_ptr = nullptr;
    } else {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
std::pair<T, T> continuous_output<T>::get_bounds() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    return {m_times_hi[0], m_times_hi.back()};
}

template <typename T>
std::size_t continuous_output<T>::get_n_steps() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output object");
    }

    return boost::numeric_cast<std::size_t>(m_times_hi.size() - 1u);
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC continuous_output<float>;
template class HEYOKA_DLL_PUBLIC continuous_output<double>;
template class HEYOKA_DLL_PUBLIC continuous_output<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class HEYOKA_DLL_PUBLIC continuous_output<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class HEYOKA_DLL_PUBLIC continuous_output<mppp::real>;

#endif

namespace detail
{

template <typename T>
std::ostream &c_out_stream_impl(std::ostream &os, const continuous_output<T> &co)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "C++ datatype: " << boost::core::demangle(typeid(T).name()) << '\n';

    if (co.get_output().empty()) {
        oss << "Default-constructed continuous_output";
    } else {
        const detail::dfloat<T> df_t_start(co.m_times_hi[0], co.m_times_lo[0]),
            df_t_end(co.m_times_hi.back(), co.m_times_lo.back());
        const auto dir = df_t_start < df_t_end;
        oss << "Direction   : " << (dir ? "forward" : "backward") << '\n';
        oss << "Time range  : "
            << (dir ? fmt::format("[{}, {})", fp_to_string(co.m_times_hi[0]), fp_to_string(co.m_times_hi.back()))
                    : fmt::format("({}, {}]", fp_to_string(co.m_times_hi.back()), fp_to_string(co.m_times_hi[0])))
            << '\n';
        oss << "N of steps  : " << (co.m_times_hi.size() - 1u) << '\n';
    }

    return os << oss.str();
}

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<float> &co)
{
    return detail::c_out_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<double> &co)
{
    return detail::c_out_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<long double> &co)
{
    return detail::c_out_stream_impl(os, co);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<mppp::real128> &co)
{
    return detail::c_out_stream_impl(os, co);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output<mppp::real> &co)
{
    return detail::c_out_stream_impl(os, co);
}

#endif

HEYOKA_END_NAMESPACE

#if !defined(NDEBUG)

extern "C" {

// Function to check, in debug mode, the indexing of the Taylor coefficients
// in the batch mode continuous output implementation.
HEYOKA_DLL_PUBLIC void heyoka_continuous_output_batch_tc_idx_debug(const std::uint32_t *tc_idx,
                                                                   std::uint32_t times_size,
                                                                   std::uint32_t batch_size) noexcept
{
    // LCOV_EXCL_START
    assert(batch_size != 0u);
    assert(times_size % batch_size == 0u);
    assert(times_size / batch_size >= 3u);
    // LCOV_EXCL_STOP

    for (std::uint32_t i = 0; i < batch_size; ++i) {
        assert(tc_idx[i] < times_size / batch_size - 2u); // LCOV_EXCL_LINE
    }
}
}

#endif

HEYOKA_BEGIN_NAMESPACE

// Continuous output for the batch integrator.
template <typename T>
void continuous_output_batch<T>::add_c_out_function(std::uint32_t order, std::uint32_t dim, bool high_accuracy)
{
    // Overflow check: we want to be able to index into the arrays of
    // times and Taylor coefficients using 32-bit ints.
    // LCOV_EXCL_START
    if (m_tcs.size() > std::numeric_limits<std::uint32_t>::max()
        || m_times_hi.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "Overflow detected while adding continuous output to a Taylor integrator in batch mode");
    }
    // LCOV_EXCL_STOP

    auto &md = m_llvm_state.module();
    auto &builder = m_llvm_state.builder();
    auto &context = m_llvm_state.context();

    // The function arguments:
    // - the output pointer (read/write, used also for accumulation),
    // - the pointer to the target time values (read-only),
    // - the pointer to the Taylor coefficients (read-only),
    // - the pointer to the hi times (read-only),
    // - the pointer to the lo times (read-only).
    // No overlap is allowed.
    auto fp_t = detail::to_external_llvm_type<T>(context);
    auto fp_vec_t = detail::make_vector_type(fp_t, m_batch_size);
    auto ptr_t = llvm::PointerType::getUnqual(fp_t);
    const std::vector<llvm::Type *> fargs(5, ptr_t);
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = detail::llvm_func_create(ft, llvm::Function::ExternalLinkage, "c_out", &md);

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *tm_ptr = f->args().begin() + 1;
    tm_ptr->setName("tm_ptr");
    tm_ptr->addAttr(llvm::Attribute::NoCapture);
    tm_ptr->addAttr(llvm::Attribute::NoAlias);
    tm_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *tc_ptr = f->args().begin() + 2;
    tc_ptr->setName("tc_ptr");
    tc_ptr->addAttr(llvm::Attribute::NoCapture);
    tc_ptr->addAttr(llvm::Attribute::NoAlias);
    tc_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *times_ptr_hi = f->args().begin() + 3;
    times_ptr_hi->setName("times_ptr_hi");
    times_ptr_hi->addAttr(llvm::Attribute::NoCapture);
    times_ptr_hi->addAttr(llvm::Attribute::NoAlias);
    times_ptr_hi->addAttr(llvm::Attribute::ReadOnly);

    auto *times_ptr_lo = f->args().begin() + 4;
    times_ptr_lo->setName("times_ptr_lo");
    times_ptr_lo->addAttr(llvm::Attribute::NoCapture);
    times_ptr_lo->addAttr(llvm::Attribute::NoAlias);
    times_ptr_lo->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Establish the time directions.
    auto *bool_vector_t = detail::make_vector_type(builder.getInt1Ty(), m_batch_size);
    assert(bool_vector_t != nullptr); // LCOV_EXCL_LINE
    llvm::Value *dir_vec{};
    if (m_batch_size == 1u) {
        // In scalar mode, the direction is a single value.
        const detail::dfloat<T> df_t_start(m_times_hi[0], m_times_lo[0]),
            // NOTE: we load from the padding values here.
            df_t_end(m_times_hi.back(), m_times_lo.back());
        const auto dir = df_t_start < df_t_end;

        dir_vec = builder.getInt1(dir);
    } else {
        dir_vec = llvm::UndefValue::get(bool_vector_t);
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            const detail::dfloat<T> df_t_start(m_times_hi[i], m_times_lo[i]),
                // NOTE: we load from the padding values here.
                df_t_end(m_times_hi[m_times_hi.size() - m_batch_size + i],
                         m_times_lo[m_times_lo.size() - m_batch_size + i]);
            const auto dir = df_t_start < df_t_end;

            dir_vec = builder.CreateInsertElement(dir_vec, builder.getInt1(dir), i);
        }
    }

    // Look for the index in the times vector corresponding to
    // a time greater than tm (less than tm in backwards integration).
    // This is essentially an implementation of std::upper_bound:
    // https://en.cppreference.com/w/cpp/algorithm/upper_bound
    auto *int32_vec_t = detail::make_vector_type(builder.getInt32Ty(), m_batch_size);
    auto *tidx = builder.CreateAlloca(int32_vec_t);
    auto *count = builder.CreateAlloca(int32_vec_t);
    auto *step = builder.CreateAlloca(int32_vec_t);
    auto *first = builder.CreateAlloca(int32_vec_t);

    // count is inited with the size of the range.
    // NOTE: count includes the padding.
    builder.CreateStore(
        detail::vector_splat(builder, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size()) / m_batch_size),
                             m_batch_size),
        count);

    // first is inited to zero.
    auto *zero_vec_i32 = detail::vector_splat(builder, builder.getInt32(0), m_batch_size);
    builder.CreateStore(zero_vec_i32, first);

    // Load the time value from tm_ptr.
    auto tm = detail::load_vector_from_memory(builder, fp_t, tm_ptr, m_batch_size);

    // This is the vector [0, 1, 2, ..., (batch_size - 1)].
    llvm::Value *batch_offset{};
    if (m_batch_size == 1u) {
        // In scalar mode, use a single value.
        batch_offset = builder.getInt32(0);
    } else {
        batch_offset = llvm::UndefValue::get(int32_vec_t);
        for (std::uint32_t i = 0; i < m_batch_size; ++i) {
            batch_offset = builder.CreateInsertElement(batch_offset, builder.getInt32(i), i);
        }
    }

    // Splatted version of the batch size.
    auto *batch_splat = detail::vector_splat(builder, builder.getInt32(m_batch_size), m_batch_size);

    // Splatted versions of the base pointers for the time data.
    auto *times_ptr_hi_vec = detail::vector_splat(builder, times_ptr_hi, m_batch_size);
    auto *times_ptr_lo_vec = detail::vector_splat(builder, times_ptr_lo, m_batch_size);

    // fp vector of zeroes.
    auto *zero_vec_fp = detail::llvm_constantfp(m_llvm_state, fp_vec_t, 0.);

    // Vector of i32 ones.
    auto *one_vec_i32 = detail::vector_splat(builder, builder.getInt32(1), m_batch_size);

    detail::llvm_while_loop(
        m_llvm_state,
        [&]() -> llvm::Value * {
            // NOTE: the condition here is that any value in count is not zero.
            auto *cmp = builder.CreateICmpNE(builder.CreateLoad(int32_vec_t, count), zero_vec_i32);

            // NOTE: in scalar mode, no reduction is needed.
            return (m_batch_size == 1u) ? cmp : builder.CreateOrReduce(cmp);
        },
        [&]() {
            // tidx = first.
            builder.CreateStore(builder.CreateLoad(int32_vec_t, first), tidx);
            // step = count / 2.
            auto *two_vec_i32 = detail::vector_splat(builder, builder.getInt32(2), m_batch_size);
            builder.CreateStore(builder.CreateUDiv(builder.CreateLoad(int32_vec_t, count), two_vec_i32), step);
            // tidx = tidx + step.
            builder.CreateStore(
                builder.CreateAdd(builder.CreateLoad(int32_vec_t, tidx), builder.CreateLoad(int32_vec_t, step)), tidx);

            // Compute the indices for loading the times from the pointers.
            auto *tl_idx = builder.CreateAdd(builder.CreateMul(builder.CreateLoad(int32_vec_t, tidx), batch_splat),
                                             batch_offset);

            // Compute the pointers for loading the time data.
            auto tptr_hi = builder.CreateInBoundsGEP(fp_t, times_ptr_hi_vec, tl_idx);
            auto tptr_lo = builder.CreateInBoundsGEP(fp_t, times_ptr_lo_vec, tl_idx);

            // Gather the hi/lo values.
            auto tidx_val_hi = detail::gather_vector_from_memory(builder, fp_vec_t, tptr_hi);
            auto tidx_val_lo = detail::gather_vector_from_memory(builder, fp_vec_t, tptr_lo);

            // Compute the two conditions !(tm < *tidx) and !(tm > *tidx).
            auto cmp_lt
                = builder.CreateNot(detail::llvm_dl_lt(m_llvm_state, tm, zero_vec_fp, tidx_val_hi, tidx_val_lo));
            auto cmp_gt
                = builder.CreateNot(detail::llvm_dl_gt(m_llvm_state, tm, zero_vec_fp, tidx_val_hi, tidx_val_lo));

            // Select cmp_lt if integrating forward, cmp_gt when integrating backward.
            auto cond = builder.CreateSelect(dir_vec, cmp_lt, cmp_gt);

            // tidx += (1 or 0).
            builder.CreateStore(builder.CreateAdd(builder.CreateLoad(int32_vec_t, tidx),
                                                  builder.CreateSelect(cond, one_vec_i32, zero_vec_i32)),
                                tidx);

            // first = (tidx or first).
            builder.CreateStore(builder.CreateSelect(cond, builder.CreateLoad(int32_vec_t, tidx),
                                                     builder.CreateLoad(int32_vec_t, first)),
                                first);

            // count = count - (step or count).
            auto *old_count = builder.CreateLoad(int32_vec_t, count);
            auto new_count = builder.CreateSub(
                old_count, builder.CreateSelect(cond, builder.CreateLoad(int32_vec_t, step), old_count));

            // count = count + (-1 or step).
            new_count = builder.CreateAdd(new_count, builder.CreateSelect(cond, builder.CreateNeg(one_vec_i32),
                                                                          builder.CreateLoad(int32_vec_t, step)));
            builder.CreateStore(new_count, count);
        });

    // NOTE: the output of the std::upper_bound algorithm
    // is in the 'first' variable.
    llvm::Value *tc_idx = builder.CreateLoad(int32_vec_t, first);

    // Normally, the TC index should be first - 1. The exceptions are:
    // - first == 0, in which case TC index is also 0,
    // - first == (range size - 1), in which case TC index is first - 2.
    // These two exceptions arise when tm is outside the range of validity
    // for the continuous output. In such cases, we will use either the first
    // or the last possible set of TCs.
    // NOTE: the second check is range size - 1 (rather than just range size
    // like in the scalar case) due to padding.
    // In order to vectorise the check, we compute:
    // tc_idx = tc_idx - (tc_idx != 0) - (tc_idx == range size - 1).
    auto *tc_idx_cmp1 = builder.CreateZExt(builder.CreateICmpNE(tc_idx, zero_vec_i32), int32_vec_t);
    auto *tc_idx_cmp2 = builder.CreateZExt(
        builder.CreateICmpEQ(
            tc_idx, detail::vector_splat(
                        builder, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size() / m_batch_size - 1u)),
                        m_batch_size)),
        int32_vec_t);
    tc_idx = builder.CreateSub(tc_idx, tc_idx_cmp1);
    tc_idx = builder.CreateSub(tc_idx, tc_idx_cmp2);

#if !defined(NDEBUG)

    {
        // In debug mode, invoke the index checking function.
        auto *array_t = llvm::ArrayType::get(builder.getInt32Ty(), m_batch_size);
        auto *tc_idx_debug_ptr = builder.CreateInBoundsGEP(array_t, builder.CreateAlloca(array_t),
                                                           {builder.getInt32(0), builder.getInt32(0)});
        detail::store_vector_to_memory(builder, tc_idx_debug_ptr, tc_idx);
        detail::llvm_invoke_external(m_llvm_state, "heyoka_continuous_output_batch_tc_idx_debug", builder.getVoidTy(),
                                     {tc_idx_debug_ptr, builder.getInt32(static_cast<std::uint32_t>(m_times_hi.size())),
                                      builder.getInt32(m_batch_size)});
    }

#endif

    // Convert tc_idx into an index for loading from the time vectors.
    auto *tc_l_idx = builder.CreateAdd(builder.CreateMul(tc_idx, batch_splat), batch_offset);

    // Load the times corresponding to tc_idx.
    auto start_tm_hi = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                         builder.CreateInBoundsGEP(fp_t, times_ptr_hi_vec, tc_l_idx));
    auto start_tm_lo = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                         builder.CreateInBoundsGEP(fp_t, times_ptr_lo_vec, tc_l_idx));

    // Compute the value of h = tm - start_tm.
    auto h = detail::llvm_dl_add(m_llvm_state, tm, zero_vec_fp, detail::llvm_fneg(m_llvm_state, start_tm_hi),
                                 detail::llvm_fneg(m_llvm_state, start_tm_lo))
                 .first;

    // Compute the base pointers in the array of TC for the computation
    // of Horner's scheme.
    tc_idx = builder.CreateAdd(
        builder.CreateMul(
            tc_idx, detail::vector_splat(builder, builder.getInt32(dim * (order + 1u) * m_batch_size), m_batch_size)),
        batch_offset);
    // NOTE: each pointer in tc_ptrs points to the Taylor coefficient of
    // order 0 for the first state variable in the timestep data block selected
    // for that batch index.
    auto tc_ptrs = builder.CreateInBoundsGEP(fp_t, tc_ptr, tc_idx);

    // Run the Horner scheme.
    if (high_accuracy) {
        // Create the array for storing the running compensations.
        auto array_type = llvm::ArrayType::get(fp_vec_t, dim);
        auto comp_arr = builder.CreateInBoundsGEP(array_type, builder.CreateAlloca(array_type),
                                                  {builder.getInt32(0), builder.getInt32(0)});

        // Start by writing into out_ptr the zero-order coefficients
        // and by filling with zeroes the running compensations.
        detail::llvm_loop_u32(m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptrs. The index is:
            // m_batch_size * (order + 1u) * cur_var_idx.
            auto *load_idx = builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx);
            auto *tcs = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                          builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

            // Store it in out_ptr. The index is:
            // m_batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
            detail::store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx), tcs);

            // Zero-init the element in comp_arr.
            builder.CreateStore(zero_vec_fp, builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx));
        });

        // Init the running updater for the powers of h.
        auto *cur_h = builder.CreateAlloca(fp_vec_t);
        builder.CreateStore(h, cur_h);

        // Run the evaluation.
        detail::llvm_loop_u32(
            m_llvm_state, builder.getInt32(1), builder.getInt32(order + 1u), [&](llvm::Value *cur_order) {
                // Load the current power of h.
                auto *cur_h_val = builder.CreateLoad(fp_vec_t, cur_h);

                detail::llvm_loop_u32(
                    m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
                        // Load the coefficient from tc_ptrs. The index is:
                        // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * cur_order.
                        auto *load_idx = builder.CreateAdd(
                            builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                            builder.CreateMul(builder.getInt32(m_batch_size), cur_order));
                        auto *cf = detail::gather_vector_from_memory(
                            builder, fp_vec_t, builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));
                        auto *tmp = detail::llvm_fmul(m_llvm_state, cf, cur_h_val);

                        // Compute the quantities for the compensation.
                        auto *comp_ptr = builder.CreateInBoundsGEP(fp_vec_t, comp_arr, cur_var_idx);
                        auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
                        auto *res_ptr = builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx);
                        auto *y = detail::llvm_fsub(m_llvm_state, tmp, builder.CreateLoad(fp_vec_t, comp_ptr));
                        auto *cur_res = detail::load_vector_from_memory(builder, fp_t, res_ptr, m_batch_size);
                        auto *t = detail::llvm_fadd(m_llvm_state, cur_res, y);

                        // Update the compensation and the return value.
                        builder.CreateStore(
                            detail::llvm_fsub(m_llvm_state, detail::llvm_fsub(m_llvm_state, t, cur_res), y), comp_ptr);
                        detail::store_vector_to_memory(builder, res_ptr, t);
                    });

                // Update the value of h.
                builder.CreateStore(detail::llvm_fmul(m_llvm_state, cur_h_val, h), cur_h);
            });
    } else {
        // Start by writing into out_ptr the coefficients of the highest-degree
        // monomial in each polynomial.
        detail::llvm_loop_u32(m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
            // Load the coefficient from tc_ptrs. The index is:
            // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * order.
            auto *load_idx
                = builder.CreateAdd(builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                                    builder.getInt32(m_batch_size * order));
            auto *tcs = detail::gather_vector_from_memory(builder, fp_vec_t,
                                                          builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

            // Store it in out_ptr. The index is:
            // m_batch_size * cur_var_idx.
            auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
            detail::store_vector_to_memory(builder, builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx), tcs);
        });

        // Now let's run the Horner scheme.
        detail::llvm_loop_u32(
            m_llvm_state, builder.getInt32(1), builder.CreateAdd(builder.getInt32(order), builder.getInt32(1)),
            [&](llvm::Value *cur_order) {
                detail::llvm_loop_u32(
                    m_llvm_state, builder.getInt32(0), builder.getInt32(dim), [&](llvm::Value *cur_var_idx) {
                        // Load the current Taylor coefficients from tc_ptrs.
                        // NOTE: we are loading the coefficients backwards wrt the order, hence
                        // we specify order - cur_order.
                        // NOTE: the index is:
                        // m_batch_size * (order + 1u) * cur_var_idx + m_batch_size * (order - cur_order).
                        auto *load_idx = builder.CreateAdd(
                            builder.CreateMul(builder.getInt32(m_batch_size * (order + 1u)), cur_var_idx),
                            builder.CreateMul(builder.getInt32(m_batch_size),
                                              builder.CreateSub(builder.getInt32(order), cur_order)));
                        auto *tcs = detail::gather_vector_from_memory(
                            builder, fp_vec_t, builder.CreateInBoundsGEP(fp_t, tc_ptrs, load_idx));

                        // Accumulate in out_ptr. The index is:
                        // m_batch_size * cur_var_idx.
                        auto *out_idx = builder.CreateMul(builder.getInt32(m_batch_size), cur_var_idx);
                        auto *out_p = builder.CreateInBoundsGEP(fp_t, out_ptr, out_idx);
                        auto *cur_out = detail::load_vector_from_memory(builder, fp_t, out_p, m_batch_size);
                        detail::store_vector_to_memory(
                            builder, out_p,
                            detail::llvm_fadd(m_llvm_state, tcs, detail::llvm_fmul(m_llvm_state, cur_out, h)));
                    });
            });
    }

    // Create the return value.
    builder.CreateRetVoid();

    // Compile.
    m_llvm_state.compile();

    // Fetch the function pointer and assign it.
    m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch() = default;

template <typename T>
continuous_output_batch<T>::continuous_output_batch(llvm_state &&s) : m_llvm_state(std::move(s))
{
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch(const continuous_output_batch &o)
    : m_batch_size(o.m_batch_size), m_llvm_state(o.m_llvm_state), m_tcs(o.m_tcs), m_times_hi(o.m_times_hi),
      m_times_lo(o.m_times_lo), m_output(o.m_output), m_tmp_tm(o.m_tmp_tm)
{
    // If o is valid, fetch the function pointer from the copied state.
    // Otherwise, m_f_ptr will remain null.
    if (o.m_f_ptr != nullptr) {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
continuous_output_batch<T>::continuous_output_batch(continuous_output_batch &&) noexcept = default;

template <typename T>
continuous_output_batch<T>::~continuous_output_batch() = default;

template <typename T>
continuous_output_batch<T> &continuous_output_batch<T>::operator=(const continuous_output_batch &o)
{
    if (this != &o) {
        *this = continuous_output_batch(o);
    }

    return *this;
}

template <typename T>
continuous_output_batch<T> &continuous_output_batch<T>::operator=(continuous_output_batch &&) noexcept = default;

template <typename T>
void continuous_output_batch<T>::call_impl(const T *t)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)
    // The batch size must not be zero.
    assert(m_batch_size != 0u);
    // m_batch_size must divide m_output exactly.
    assert(m_output.size() % m_batch_size == 0u);
    // m_tmp_tm must be of size m_batch_size.
    assert(m_tmp_tm.size() == m_batch_size);
    // m_batch_size must divide the time and tcs vectors exactly.
    assert(m_times_hi.size() % m_batch_size == 0u);
    assert(m_tcs.size() % m_batch_size == 0u);
    // Need at least 3 time points (2 + 1 for padding).
    assert(m_times_hi.size() / m_batch_size >= 3u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());
#endif
    // LCOV_EXCL_STOP

    // Copy over the times to the temp buffer and check that they are finite.
    // NOTE: this copy ensures we avoid aliasing issues with the
    // other data members.
    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        if (!isfinite(t[i])) {
            throw std::invalid_argument(fmt::format("Cannot compute the continuous output in batch mode "
                                                    "for the batch index {} at the non-finite time {}",
                                                    i, detail::fp_to_string(t[i])));
        }

        m_tmp_tm[i] = t[i];
    }

    m_f_ptr(m_output.data(), m_tmp_tm.data(), m_tcs.data(), m_times_hi.data(), m_times_lo.data());
}

template <typename T>
const std::vector<T> &continuous_output_batch<T>::operator()(const std::vector<T> &tm)
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    if (tm.size() != m_batch_size) {
        throw std::invalid_argument(
            fmt::format("An invalid time vector was passed to the call operator of continuous_output_batch: the "
                        "vector size is {}, but a size of {} was expected instead",
                        tm.size(), m_batch_size));
    }

    return (*this)(tm.data());
}

// NOTE: there's some overlap with the call_impl() code here.
template <typename T>
const std::vector<T> &continuous_output_batch<T>::operator()(T tm)
{
    using std::isfinite;

    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: run the assertions only after ensuring this
    // is a valid object.

    // LCOV_EXCL_START
#if !defined(NDEBUG)
    // The batch size must not be zero.
    assert(m_batch_size != 0u);
    // m_batch_size must divide m_output exactly.
    assert(m_output.size() % m_batch_size == 0u);
    // m_tmp_tm must be of size m_batch_size.
    assert(m_tmp_tm.size() == m_batch_size);
    // m_batch_size must divide the time and tcs vectors exactly.
    assert(m_times_hi.size() % m_batch_size == 0u);
    assert(m_tcs.size() % m_batch_size == 0u);
    // Need at least 3 time points (2 + 1 for padding).
    assert(m_times_hi.size() / m_batch_size >= 3u);
    // hi/lo parts of times must have the same sizes.
    assert(m_times_hi.size() == m_times_lo.size());
#endif
    // LCOV_EXCL_STOP

    if (!isfinite(tm)) {
        throw std::invalid_argument(fmt::format("Cannot compute the continuous output in batch mode "
                                                "at the non-finite time {}",
                                                detail::fp_to_string(tm)));
    }

    // Copy over the time to the temp buffer.
    std::fill(m_tmp_tm.begin(), m_tmp_tm.end(), tm);

    m_f_ptr(m_output.data(), m_tmp_tm.data(), m_tcs.data(), m_times_hi.data(), m_times_lo.data());

    return m_output;
}

template <typename T>
const llvm_state &continuous_output_batch<T>::get_llvm_state() const
{
    return m_llvm_state;
}

template <typename T>
const std::vector<T> &continuous_output_batch<T>::operator()(const T *tm)
{
    call_impl(tm);
    return m_output;
}

template <typename T>
const std::vector<T> &continuous_output_batch<T>::get_output() const
{
    return m_output;
}

// NOTE: when documenting this function,
// we need to warn about the padding.
template <typename T>
const std::vector<T> &continuous_output_batch<T>::get_times() const
{
    return m_times_hi;
}

template <typename T>
const std::vector<T> &continuous_output_batch<T>::get_tcs() const
{
    return m_tcs;
}

template <typename T>
std::uint32_t continuous_output_batch<T>::get_batch_size() const
{
    return m_batch_size;
}

template <typename T>
void continuous_output_batch<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_batch_size;
    ar << m_llvm_state;
    ar << m_tcs;
    ar << m_times_hi;
    ar << m_times_lo;
    ar << m_output;
    ar << m_tmp_tm;
}

template <typename T>
void continuous_output_batch<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_batch_size;
    ar >> m_llvm_state;
    ar >> m_tcs;
    ar >> m_times_hi;
    ar >> m_times_lo;
    ar >> m_output;
    ar >> m_tmp_tm;

    // NOTE: if m_output is not empty, it means the archived
    // object had been initialised.
    if (m_output.empty()) {
        m_f_ptr = nullptr;
    } else {
        m_f_ptr = reinterpret_cast<fptr_t>(m_llvm_state.jit_lookup("c_out"));
    }
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> continuous_output_batch<T>::get_bounds() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    std::vector<T> lb, ub;
    lb.resize(boost::numeric_cast<decltype(lb.size())>(m_batch_size));
    ub.resize(boost::numeric_cast<decltype(ub.size())>(m_batch_size));

    for (std::uint32_t i = 0; i < m_batch_size; ++i) {
        lb[i] = m_times_hi[i];
        // NOTE: take into account the padding.
        ub[i] = m_times_hi[m_times_hi.size() - 2u * m_batch_size + i];
    }

    return std::make_pair(std::move(lb), std::move(ub));
}

template <typename T>
std::size_t continuous_output_batch<T>::get_n_steps() const
{
    if (m_f_ptr == nullptr) {
        throw std::invalid_argument("Cannot use a default-constructed continuous_output_batch object");
    }

    // NOTE: account for padding.
    return boost::numeric_cast<std::size_t>(m_times_hi.size() / m_batch_size - 2u);
}

// Explicit instantiations.
template class HEYOKA_DLL_PUBLIC continuous_output_batch<float>;
template class HEYOKA_DLL_PUBLIC continuous_output_batch<double>;
template class HEYOKA_DLL_PUBLIC continuous_output_batch<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class HEYOKA_DLL_PUBLIC continuous_output_batch<mppp::real128>;

#endif

namespace detail
{

template <typename T>
std::ostream &c_out_batch_stream_impl(std::ostream &os, const continuous_output_batch<T> &co)
{
    std::ostringstream oss;
    oss.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    oss.imbue(std::locale::classic());
    oss << std::showpoint;
    oss.precision(std::numeric_limits<T>::max_digits10);

    oss << "C++ datatype: " << boost::core::demangle(typeid(T).name()) << '\n';

    if (co.get_output().empty()) {
        oss << "Default-constructed continuous_output_batch";
    } else {
        const auto batch_size = co.m_batch_size;

        oss << "Directions  : [";
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const detail::dfloat<T> df_t_start(co.m_times_hi[i], co.m_times_lo[i]),
                df_t_end(co.m_times_hi[co.m_times_hi.size() - 2u * batch_size + i],
                         co.m_times_lo[co.m_times_lo.size() - 2u * batch_size + i]);
            const auto dir = df_t_start < df_t_end;

            oss << (dir ? "forward" : "backward");

            if (i != batch_size - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";

        oss << "Time ranges : [";
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const detail::dfloat<T> df_t_start(co.m_times_hi[i], co.m_times_lo[i]),
                df_t_end(co.m_times_hi[co.m_times_hi.size() - 2u * batch_size + i],
                         co.m_times_lo[co.m_times_lo.size() - 2u * batch_size + i]);
            const auto dir = df_t_start < df_t_end;
            oss << (dir ? fmt::format("[{}, {})", fp_to_string(df_t_start.hi), fp_to_string(df_t_end.hi))
                        : fmt::format("({}, {}]", fp_to_string(df_t_end.hi), fp_to_string(df_t_start.hi)));

            if (i != batch_size - 1u) {
                oss << ", ";
            }
        }
        oss << "]\n";

        oss << "N of steps  : " << co.get_n_steps() << '\n';
    }

    return os << oss.str();
}

} // namespace detail

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<float> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<double> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<long double> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
std::ostream &operator<<(std::ostream &os, const continuous_output_batch<mppp::real128> &co)
{
    return detail::c_out_batch_stream_impl(os, co);
}

#endif

HEYOKA_END_NAMESPACE
