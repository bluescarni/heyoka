// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/poly_root_finding/frpr_workspace_impl.hpp>
#include <heyoka/detail/poly_root_finding/vector_cache.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/find_real_poly_roots.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Helper to add the JIT-compiled function for the conversion of an input scalar polynomial to the Bernstein basis with
// interval coefficients. In addition to the Bernstein polynomial, the function will also return the rescaled interval
// version of the original polynomial.
//
// fp_t is the scalar internal type to be used for the computations. n is the polynomial degree. 'name' is the name of
// the function that will be added.
void frpr_workspace_add_bernstein(const std::string &name, llvm_state &s, llvm::Type *const fp_t, const std::uint32_t n)
{
    assert(fp_t != nullptr);
    assert(!fp_t->isVectorTy());

    // Overflow check: 2*(n+1), which is the total number of scalar values in a polynomial of degree n with interval
    // coefficients, needs to be representable as a 32-bit unsigned int.
    try {
        using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
        static_cast<void>(static_cast<std::uint32_t>((safe_uint32_t(n) + 1) * 2));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in frpr_workspace_add_bernstein()");
    }
    // LCOV_EXCL_STOP

    auto &bld = s.builder();
    auto &context = s.context();

    // Fetch the external type corresponding to fp_t and the pointer type.
    auto *const ext_fp_t = make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    // Remember and restore the current insertion point.
    const llvm::IRBuilderBase::InsertPointGuard ipg(bld);

    // Define the function type. 4 arguments:
    //
    // - the Bernstein output (ptr),
    // - the rescaled polynomial output (ptr),
    // - the input poly coefficients (const ptr),
    // - the value of h (const ptr).
    //
    // No aliasing is allowed among the pointers. The output coefficients are stored in the usual lo-negated, flattened
    // out format.
    auto *const ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, ptr_t, ptr_t}, false);

    // Create the function.
    auto *const f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &s.module());

    // Fetch the arguments.
    auto *const out_ptr_bernstein = f->getArg(0);
    out_ptr_bernstein->setName("out_ptr_bernstein");
    llvm_add_no_capture_argattr(s, out_ptr_bernstein);
    out_ptr_bernstein->addAttr(llvm::Attribute::NoAlias);

    auto *const out_ptr_rescaled = f->getArg(1);
    out_ptr_rescaled->setName("out_ptr_rescaled");
    llvm_add_no_capture_argattr(s, out_ptr_rescaled);
    out_ptr_rescaled->addAttr(llvm::Attribute::NoAlias);

    auto *const cfs_ptr = f->getArg(2);
    cfs_ptr->setName("cfs_ptr");
    llvm_add_no_capture_argattr(s, cfs_ptr);
    cfs_ptr->addAttr(llvm::Attribute::NoAlias);
    cfs_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *const h_ptr = f->getArg(3);
    h_ptr->setName("h_ptr");
    llvm_add_no_capture_argattr(s, h_ptr);
    h_ptr->addAttr(llvm::Attribute::NoAlias);
    h_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *const bb = llvm::BasicBlock::Create(context, "entry", f);
    bld.SetInsertPoint(bb);

    // Setup the interval arithmetic guard.
    const auto ivg = llvm_ival_guard(s, f, fp_t);

    // Load h.
    auto *const h = ext_load_vector_from_memory(s, fp_t, h_ptr, 1);

    // Load the poly coefficients.
    std::vector<llvm::Value *> cfs;
    cfs.reserve(n + 1u);
    for (std::uint32_t i = 0; i <= n; ++i) {
        auto *const ptr = bld.CreateInBoundsGEP(ext_fp_t, cfs_ptr, bld.getInt32(i));
        cfs.push_back(ext_load_vector_from_memory(s, fp_t, ptr, 1));
    }

    // Run the Bernstein conversion.
    const auto [bern_cfs, rescaled_cfs] = llvm_ed_ival_poly_bernstein(ivg, cfs, h);

    // Store the results.
    for (std::uint32_t i = 0; i <= n; ++i) {
        // Bernstein first.
        auto *out_ptr_mlo = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_bernstein, bld.getInt32(2u * i));
        auto *out_ptr_hi = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_bernstein, bld.getInt32((2u * i) + 1u));
        auto [mlo, hi] = ivg.ival_extract(bern_cfs[i]);
        ext_store_vector_to_memory(s, out_ptr_mlo, mlo);
        ext_store_vector_to_memory(s, out_ptr_hi, hi);

        // Rescaled second.
        out_ptr_mlo = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_rescaled, bld.getInt32(2u * i));
        out_ptr_hi = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_rescaled, bld.getInt32((2u * i) + 1u));
        std::tie(mlo, hi) = ivg.ival_extract(rescaled_cfs[i]);
        ext_store_vector_to_memory(s, out_ptr_mlo, mlo);
        ext_store_vector_to_memory(s, out_ptr_hi, hi);
    }

    // Finalise the guard before emitting the return statement.
    ivg.finalise();

    // Exit.
    bld.CreateRetVoid();
}

// Helper to add the JIT-compiled function for the de Casteljau split of an input interval Bernstein polynomial.
//
// fp_t is the scalar internal type to be used for the computations. n is the polynomial degree. 'name' is the name of
// the function that will be added.
void frpr_workspace_add_de_casteljau(const std::string &name, llvm_state &s, llvm::Type *const fp_t,
                                     const std::uint32_t n)
{
    assert(fp_t != nullptr);
    assert(!fp_t->isVectorTy());

    // Overflow check: 2*(n+1), which is the total number of scalar values in a polynomial of degree n with interval
    // coefficients, needs to be representable as a 32-bit unsigned int.
    try {
        using safe_uint32_t = boost::safe_numerics::safe<std::uint32_t>;
        static_cast<void>(static_cast<std::uint32_t>((safe_uint32_t(n) + 1) * 2));
        // LCOV_EXCL_START
    } catch (...) {
        throw std::overflow_error("Overflow detected in frpr_workspace_add_de_casteljau()");
    }
    // LCOV_EXCL_STOP

    auto &bld = s.builder();
    auto &context = s.context();

    // Fetch the external type corresponding to fp_t and the pointer type.
    auto *const ext_fp_t = make_external_llvm_type(fp_t);
    auto *const ptr_t = llvm::PointerType::getUnqual(context);

    // Remember and restore the current insertion point.
    const llvm::IRBuilderBase::InsertPointGuard ipg(bld);

    // Define the function type. 3 arguments:
    //
    // - the left output (ptr),
    // - the right output (ptr),
    // - the input poly coefficients (const ptr).
    //
    // No aliasing is allowed among the pointers. All coefficients are stored in the usual lo-negated, flattened out
    // format.
    auto *const ft = llvm::FunctionType::get(bld.getVoidTy(), {ptr_t, ptr_t, ptr_t}, false);

    // Create the function.
    auto *const f = llvm_func_create(ft, llvm::Function::ExternalLinkage, name, &s.module());

    // Fetch the arguments.
    auto *const out_ptr_left = f->getArg(0);
    out_ptr_left->setName("out_ptr_left");
    llvm_add_no_capture_argattr(s, out_ptr_left);
    out_ptr_left->addAttr(llvm::Attribute::NoAlias);

    auto *const out_ptr_right = f->getArg(1);
    out_ptr_right->setName("out_ptr_right");
    llvm_add_no_capture_argattr(s, out_ptr_right);
    out_ptr_right->addAttr(llvm::Attribute::NoAlias);

    auto *const cfs_ptr = f->getArg(2);
    cfs_ptr->setName("cfs_ptr");
    llvm_add_no_capture_argattr(s, cfs_ptr);
    cfs_ptr->addAttr(llvm::Attribute::NoAlias);
    cfs_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *const bb = llvm::BasicBlock::Create(context, "entry", f);
    bld.SetInsertPoint(bb);

    // Setup the interval arithmetic guard.
    const auto ivg = llvm_ival_guard(s, f, fp_t);

    // Load the poly coefficients.
    std::vector<llvm::Value *> cfs;
    cfs.reserve(n + 1u);
    for (std::uint32_t i = 0; i <= n; ++i) {
        auto *const ptr_mlo = bld.CreateInBoundsGEP(ext_fp_t, cfs_ptr, bld.getInt32(2u * i));
        auto *const ptr_hi = bld.CreateInBoundsGEP(ext_fp_t, cfs_ptr, bld.getInt32((2u * i) + 1u));

        cfs.push_back(ivg.ival_construct(ext_load_vector_from_memory(s, fp_t, ptr_mlo, 1),
                                         ext_load_vector_from_memory(s, fp_t, ptr_hi, 1)));
    }

    // Run the de Casteljau split.
    const auto [left_cfs, right_cfs] = llvm_ed_ival_de_casteljau_half(ivg, cfs);

    // Store the results.
    for (std::uint32_t i = 0; i <= n; ++i) {
        // Left.
        auto *out_ptr_mlo = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_left, bld.getInt32(2u * i));
        auto *out_ptr_hi = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_left, bld.getInt32((2u * i) + 1u));
        auto [mlo, hi] = ivg.ival_extract(left_cfs[i]);
        ext_store_vector_to_memory(s, out_ptr_mlo, mlo);
        ext_store_vector_to_memory(s, out_ptr_hi, hi);

        // Right.
        out_ptr_mlo = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_right, bld.getInt32(2u * i));
        out_ptr_hi = bld.CreateInBoundsGEP(ext_fp_t, out_ptr_right, bld.getInt32((2u * i) + 1u));
        std::tie(mlo, hi) = ivg.ival_extract(right_cfs[i]);
        ext_store_vector_to_memory(s, out_ptr_mlo, mlo);
        ext_store_vector_to_memory(s, out_ptr_hi, hi);
    }

    // Finalise the guard before emitting the return statement.
    ivg.finalise();

    // Exit.
    bld.CreateRetVoid();
}

} // namespace

template <typename T>
void frpr_workspace_jit_data<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_state;
}

template <typename T>
void frpr_workspace_jit_data<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_state;

    // Fetch the function pointers from the LLVM state.
    m_bernstein_func = reinterpret_cast<bernstein_func_t>(m_state.jit_lookup("bernstein"));
    m_decasteljau_func = reinterpret_cast<decasteljau_func_t>(m_state.jit_lookup("decasteljau"));
}

template <typename T>
frpr_workspace_jit_data<T>::frpr_workspace_jit_data() = default;

template <typename T>
frpr_workspace_jit_data<T>::frpr_workspace_jit_data(llvm_state s) : m_state(std::move(s))
{
}

template <typename T>
frpr_workspace_jit_data<T>::~frpr_workspace_jit_data() = default;

// Explicit instantiations.
#define HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(F) template struct frpr_workspace_jit_data<F>

HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(float);
HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(double);
HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST(mppp::real);

#endif

#undef HEYOKA_DETAIL_FRPR_WORKSPACE_JIT_DATA_INST

template <typename T>
void frpr_workspace_impl<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_prec;
    ar << m_degree;
    ar << m_jit_data;
    // NOTE: m_intervals is *not* saved on purpose.
    ar << m_vector_cache;
    // NOTE: m_wlist is *not* saved on purpose.
}

// NOTE: the load() behaviour matches the copy constructor behaviour - the scratch buffers are restored in an empty
// state.
template <typename T>
void frpr_workspace_impl<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    // NOTE: before deserialisation, make sure to clear out m_wlist. It may contain cached vectors referring to the
    // current vector cache, which will be soon replaced below.
    m_wlist.clear();

    ar >> m_prec;
    ar >> m_degree;
    ar >> m_jit_data;
    // NOTE: not strictly necessary, but this makes sure that the deserialised object starts off in a known state.
    m_intervals.clear();
    ar >> m_vector_cache;
}

template <typename T>
frpr_workspace_impl<T>::frpr_workspace_impl() = default;

namespace
{

// Helper to construct a vector_cache instance in the ctor of frpr_workspace_impl.
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
vector_cache<T> frpr_workspace_impl_make_vector_cache(const long long prec, const std::uint32_t degree)
{
    // Setup the fill value for the cache.
    T fill_value = 0;

#if defined(HEYOKA_HAVE_REAL)

    // NOTE: for mppp::real, validate the prec argument and make sure the fill value has the correct precision.
    if constexpr (std::same_as<T, mppp::real>) {
        if (prec < mppp::real_prec_min() || prec > mppp::real_prec_max()) [[unlikely]] {
            throw std::invalid_argument(fmt::format("Invalid precision value {} specified to the constructor of a "
                                                    "frpr_workspace: the value must be within the [{}, {}] bounds",
                                                    prec, mppp::real_prec_min(), mppp::real_prec_max()));
        }

        fill_value.prec_round(static_cast<mpfr_prec_t>(prec));
    }

#endif

    // Construct and return the cache.
    using safe_width_t = boost::safe_numerics::safe<typename vector_cache<T>::width_type>;
    return vector_cache<T>(std::move(fill_value), (safe_width_t(degree) + 1) * 2);
}

} // namespace

// Main constructor.
//
// The precision argument is relevant only for multiprecision computations. The degree argument represents the
// polynomial degree. The optional s argument is a template llvm_state: if provided, the internal llvm_state will be
// constructed with the same settings as s, otherwise a default-constructed llvm_state will be used.
template <typename T>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
frpr_workspace_impl<T>::frpr_workspace_impl(const long long prec, const std::uint32_t degree, const llvm_state *const s)
    : m_prec(prec), m_degree(degree),
      m_jit_data(std::make_shared<frpr_workspace_jit_data<T>>((s == nullptr) ? llvm_state{} : s->make_similar())),
      m_vector_cache(frpr_workspace_impl_make_vector_cache<T>(prec, degree))
{
    // JIT setup.

    // Fetch the internal scalar type corresponding to T.
    auto *fp_t = to_internal_llvm_type<T>(m_jit_data->m_state, prec);

    // Add the JIT functions.
    frpr_workspace_add_bernstein("bernstein", m_jit_data->m_state, fp_t, m_degree);
    frpr_workspace_add_de_casteljau("decasteljau", m_jit_data->m_state, fp_t, m_degree);

    // Compile and fetch the pointers.
    m_jit_data->m_state.compile();
    m_jit_data->m_bernstein_func
        = reinterpret_cast<frpr_workspace_jit_data<T>::bernstein_func_t>(m_jit_data->m_state.jit_lookup("bernstein"));
    m_jit_data->m_decasteljau_func = reinterpret_cast<frpr_workspace_jit_data<T>::decasteljau_func_t>(
        m_jit_data->m_state.jit_lookup("decasteljau"));
}

// Copy constructor.
//
// For the scratch data, we adopt the policy of a fresh init - empty m_intervals, empty m_wlist and an empty vector
// cache inited with the width and fill value from other.
template <typename T>
frpr_workspace_impl<T>::frpr_workspace_impl(const frpr_workspace_impl &other)
    : m_prec(other.m_prec), m_degree(other.m_degree), m_jit_data(other.m_jit_data),
      // NOTE: the cache copy ctor will only copy the fill value and the width, not the content.
      m_vector_cache(other.m_vector_cache)
{
}

template <typename T>
frpr_workspace_impl<T>::~frpr_workspace_impl() = default;

#if defined(HEYOKA_HAVE_REAL)

// Getter for the precision member, relevant only for multiprecision computations.
template <typename T>
mpfr_prec_t frpr_workspace_impl<T>::get_prec() const noexcept
    requires std::same_as<mppp::real, T>
{
    // NOTE: this is safe as we checked the precision range upon construction.
    return static_cast<mpfr_prec_t>(m_prec);
}

#endif

// Getter for the polynomial degree.
template <typename T>
std::uint32_t frpr_workspace_impl<T>::get_degree() const noexcept
{
    return m_degree;
}

// Explicit instantiations.
#define HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(F) template struct frpr_workspace_impl<F>

HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(float);
HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(double);
HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(long double);

#if defined(HEYOKA_HAVE_REAL128)

HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(mppp::real128);

#endif

#if defined(HEYOKA_HAVE_REAL)

HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST(mppp::real);

#endif

#undef HEYOKA_DETAIL_FRPR_WORKSPACE_IMPL_INST

} // namespace detail

HEYOKA_END_NAMESPACE
