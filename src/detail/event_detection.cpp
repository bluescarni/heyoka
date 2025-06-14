// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/numeric/conversion/cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#include <boost/math/special_functions/sign.hpp>
#include <boost/math/tools/roots.hpp>

#endif

#include <fmt/format.h>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/ed_data.hpp>
#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_func_create.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/logging_impl.hpp>
#include <heyoka/detail/num_utils.hpp>
#include <heyoka/detail/optional_s11n.hpp>
#include <heyoka/detail/safe_integer.hpp>
#include <heyoka/detail/string_conv.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/number.hpp>
#include <heyoka/s11n.hpp>
#include <heyoka/taylor.hpp>

#if defined(HEYOKA_HAVE_REAL)

namespace boost::math
{

// NOTE: specialise the Boost sign() function for mppp::real,
// as this is needed in the bisection algorithm.
template <>
int sign(const mppp::real &z)
{
    // NOTE: avoid throwing if z is nan,
    // and return zero for consistency with
    // the branchless sign function.
    if (z.nan_p()) {
        return 0;
    }

    const auto ret = z.sgn();

    // NOLINTNEXTLINE(readability-avoid-nested-conditional-operator)
    return ret == 0 ? 0 : (ret < 0 ? -1 : 1);
}

} // namespace boost::math

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Given an input polynomial a(x), substitute
// x with x_1 * scal and write to ret the resulting
// polynomial in the new variable x_1. Requires
// random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt, typename T>
void poly_rescale(OutputIt ret, InputIt a, const T &scal, std::uint32_t n)
{
#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        assert(std::all_of(ret, ret + n + 1u, [&](const auto &x) { return x.get_prec() == scal.get_prec(); }));
        assert(std::all_of(a, a + n + 1u, [&](const auto &x) { return x.get_prec() == scal.get_prec(); }));
    }

#endif

    auto cur_f = num_one_like(scal);

    for (std::uint32_t i = 0; i <= n; ++i) {
        // NOTE: possible optimisation for mppp::real here:
        // don't assign separately, do the ternary mul instead.
        ret[i] = a[i];
        ret[i] *= cur_f;
        cur_f *= scal;
    }
}

// Transform the polynomial a(x) into 2**n * a(x / 2).
// Requires random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt>
void poly_rescale_p2(OutputIt ret, InputIt a, std::uint32_t n)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

#if defined(HEYOKA_HAVE_REAL)
    if constexpr (std::is_same_v<value_type, mppp::real>) {
        assert(std::all_of(ret, ret + n + 1u, [&](const auto &x) { return x.get_prec() == ret[0].get_prec(); }));
        assert(std::all_of(a, a + n + 1u, [&](const auto &x) { return x.get_prec() == ret[0].get_prec(); }));

        for (std::uint32_t i = 0; i <= n; ++i) {
            mppp::mul_2ui(ret[n - i], a[n - i], boost::numeric_cast<unsigned long>(i));
        }
    } else {
#endif
        value_type cur_f(1);

        for (std::uint32_t i = 0; i <= n; ++i) {
            ret[n - i] = cur_f * a[n - i];
            cur_f *= 2;
        }
#if defined(HEYOKA_HAVE_REAL)
    }
#endif
}

// Generic branchless sign function.
template <typename T>
int sgn(T val)
{
    return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
}

#if defined(HEYOKA_HAVE_REAL)

int sgn(const mppp::real &val)
{
    // NOTE: this is only used when val is guaranteed
    // to be finite, thus this should never throw.
    assert(!val.nan_p());
    const auto ret = val.sgn();

    // NOTE: for consistency with the other implementation,
    // don't return ret directly (as it could be different from
    // 0, -1 or 1).
    return sgn(ret);
}

#endif

// Evaluate the first derivative of a polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval_1(InputIt a, const T &x, std::uint32_t n)
{
    assert(n >= 2u); // LCOV_EXCL_LINE

    // Init the return value.
    auto ret1 = a[n] * static_cast<T>(n);

    for (std::uint32_t i = 1; i < n; ++i) {
        // NOTE: possible optimisation for mppp::real here:
        // use fmma() directly, once exposed in mp++.
        ret1 = a[n - i] * static_cast<T>(n - i) + std::move(ret1) * x;
    }

    return ret1;
}

// Evaluate polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval(InputIt a, const T &x, std::uint32_t n)
{
    auto ret = a[n];

    for (std::uint32_t i = 1; i <= n; ++i) {
        // NOTE: possible optimisation for mppp::real here:
        // use fma() directly, at least in case of MPFR 4 and later.
        ret = a[n - i] + std::move(ret) * x;
    }

    return ret;
}

#if defined(HEYOKA_HAVE_REAL)

// Custom implementation of the tolerance class
// for mppp::real, for use in the bisection algorithm.
// This is copied/adapted from the default implementation
// of boost::math::tools::eps_tolerance.
class real_eps_tolerance
{
public:
    explicit real_eps_tolerance(const mppp::real &x) : eps(4 * num_eps_like(x)) {}
    bool operator()(const mppp::real &a, const mppp::real &b)
    {
        using std::abs;

        return abs(a - b) <= (eps * std::min(abs(a), abs(b)));
    }

private:
    mppp::real eps;
};

#endif

// Find the only existing root for the polynomial poly of the given order
// existing in [lb, ub).
template <typename T>
std::tuple<T, int> bracketed_root_find(const T *poly, std::uint32_t order, T lb, T ub)
{
    using std::isfinite;
    using std::nextafter;

    // NOTE: the Boost root finding routine searches in a closed interval,
    // but the goal here is to find a root in [lb, ub). Thus, we move ub
    // one position down so that it is not considered in the root finding routine.
    if (isfinite(lb) && isfinite(ub) && ub > lb) {
        ub = nextafter(ub, lb);
    }

    // NOTE: iter limit will be derived from the number of binary digits
    // in the significand.
    const boost::uintmax_t iter_limit = [&]() {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            // NOTE: we use lb here, but any of lb, ub or the poly
            // coefficients should be guaranteed to have at least the
            // working precision of the root finding scheme.
            // NOTE: since we use bisection for mppp::real, we need to allow
            // for more iterations than the number of digits.
            return boost::safe_numerics::safe<boost::uintmax_t>(lb.get_prec()) * 2;
        } else {
#endif
            return boost::numeric_cast<boost::uintmax_t>(std::numeric_limits<T>::digits);
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }();

    auto max_iter = iter_limit;

    // Ensure that root finding does not throw on error,
    // rather it will write something to errno instead.
    // https://www.boost.org/doc/libs/1_75_0/libs/math/doc/html/math_toolkit/pol_tutorial/namespace_policies.html
    using boost::math::policies::domain_error;
    using boost::math::policies::errno_on_error;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::pole_error;
    using boost::math::policies::policy;

    using pol = policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>,
                       evaluation_error<errno_on_error>>;

    // Clear out errno before running the root finding.
    errno = 0;

    // Run the root finder.
    auto p = [&]() {
#if defined(HEYOKA_HAVE_REAL)
        if constexpr (std::is_same_v<T, mppp::real>) {
            return boost::math::tools::bisect([poly, order](const auto &x) { return poly_eval(poly, x, order); }, lb,
                                              ub, real_eps_tolerance(lb), max_iter, pol{});
        } else {
#endif
            return boost::math::tools::toms748_solve([poly, order](T x) { return poly_eval(poly, x, order); }, lb, ub,
                                                     boost::math::tools::eps_tolerance<T>(), max_iter, pol{});
#if defined(HEYOKA_HAVE_REAL)
        }
#endif
    }();

    auto ret = std::move(p.first) / 2 + std::move(p.second) / 2;

    SPDLOG_LOGGER_DEBUG(get_logger(), "root finding iterations: {}", max_iter);

    if (errno > 0) {
        // Some error condition arose during root finding,
        // return zero and errno.
        return std::tuple{static_cast<T>(0), errno};
    }

    if (max_iter < iter_limit) {
        // Root finding terminated within the
        // iteration limit, return ret and success.
        return std::tuple{std::move(ret), 0};
    } else {
        // LCOV_EXCL_START
        // Root finding needed too many iterations,
        // return the (possibly wrong) result
        // and flag -1.
        return std::tuple{std::move(ret), -1};
        // LCOV_EXCL_STOP
    }
}

// Helper to detect events of terminal type.
template <typename>
struct is_terminal_event : std::false_type {
};

template <typename T, bool B>
struct is_terminal_event<t_event_impl<T, B>> : std::true_type {
};

template <typename T>
constexpr bool is_terminal_event_v = is_terminal_event<T>::value;

} // namespace

// Helper to add a polynomial translation function
// to the state 's'.
// NOTE: these event-detection-related LLVM functions are currently not mangled in any way.
llvm::Function *add_poly_translator_1(llvm_state &s, llvm::Type *fp_t, std::uint32_t order, std::uint32_t batch_size)
{
    assert(order > 0u); // LCOV_EXCL_LINE

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (order == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (order + 1u)) {
        throw std::overflow_error("Overflow detected while adding a polynomial translation function");
    }
    // LCOV_EXCL_STOP

    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type corresponding to fp_t.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Helper to fetch the (i, j) binomial coefficient from
    // a precomputed global array. The returned value is already
    // splatted.
    auto get_bc = [&, bc_ptr = llvm_add_bc_array(s, fp_t, order)](llvm::Value *i, llvm::Value *j) {
        auto *idx = builder.CreateMul(i, builder.getInt32(order + 1u));
        idx = builder.CreateAdd(idx, j);

        auto *val = builder.CreateLoad(fp_t, builder.CreateInBoundsGEP(fp_t, bc_ptr, idx));

        return vector_splat(builder, val, batch_size);
    };

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - the output pointer,
    // - the pointer to the poly coefficients (read-only).
    // No overlap is allowed, all pointers are to external types.
    const std::vector<llvm::Type *> fargs(2, llvm::PointerType::getUnqual(ext_fp_t));
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, "poly_translate_1", &s.module());

    // Set the names/attributes of the function arguments.
    auto *out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto *cf_ptr = f->args().begin() + 1;
    cf_ptr->setName("cf_ptr");
    cf_ptr->addAttr(llvm::Attribute::NoCapture);
    cf_ptr->addAttr(llvm::Attribute::NoAlias);
    cf_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Init the return values as zeroes.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(order + 1u), [&](llvm::Value *i) {
        auto *ptr = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.CreateMul(i, builder.getInt32(batch_size)));
        ext_store_vector_to_memory(s, ptr, vector_splat(builder, llvm_constantfp(s, fp_t, 0.), batch_size));
    });

    // Do the translation.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(order + 1u), [&](llvm::Value *i) {
        auto *ai = ext_load_vector_from_memory(
            s, fp_t, builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, builder.CreateMul(i, builder.getInt32(batch_size))),
            batch_size);

        llvm_loop_u32(s, builder.getInt32(0), builder.CreateAdd(i, builder.getInt32(1)), [&](llvm::Value *k) {
            auto *tmp = llvm_fmul(s, ai, get_bc(i, k));

            auto *ptr
                = builder.CreateInBoundsGEP(ext_fp_t, out_ptr, builder.CreateMul(k, builder.getInt32(batch_size)));
            auto *new_val = llvm_fadd(s, ext_load_vector_from_memory(s, fp_t, ptr, batch_size), tmp);
            ext_store_vector_to_memory(s, ptr, new_val);
        });
    });

    // Create the return value.
    builder.CreateRetVoid();

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // NOTE: the optimisation pass will be run outside.
    return f;
}

namespace
{

// Helper to automatically deduce the cooldown
// for a terminal event. g_eps is the maximum
// absolute error on the Taylor series of the event
// equation (which depends on the integrator tolerance
// and the infinity norm of the state vector/event equations),
// abs_der is the absolute value of the time derivative of the
// event equation at the zero.
template <typename T>
T taylor_deduce_cooldown_impl(T g_eps, T abs_der)
{
    using std::isfinite;

    // LCOV_EXCL_START
    assert(isfinite(g_eps));
    assert(isfinite(abs_der));
    assert(g_eps >= 0);
    assert(abs_der >= 0);
    // LCOV_EXCL_STOP

    // NOTE: the * 10 is a safety factor composed of:
    // - 2 is the original factor from theoretical considerations,
    // - 2 factor to deal with very small values of the derivative,
    // - 2 factor to deal with the common case of event equation
    //   flipping around after the event (e.g., for collisions).
    // The rest is additional safety.
    auto ret = std::move(g_eps) / std::move(abs_der) * 10;

    if (isfinite(ret)) {
        return ret;
    } else {
        // LCOV_EXCL_START
        get_logger()->warn("deducing a cooldown of zero for a terminal event because the automatic deduction "
                           "heuristic produced a non-finite value of {}",
                           fp_to_string(ret));

        return 0;
        // LCOV_EXCL_STOP
    }
}

} // namespace

template <>
float taylor_deduce_cooldown(float g_eps, float abs_der)
{
    return taylor_deduce_cooldown_impl(g_eps, abs_der);
}

template <>
double taylor_deduce_cooldown(double g_eps, double abs_der)
{
    return taylor_deduce_cooldown_impl(g_eps, abs_der);
}

template <>
long double taylor_deduce_cooldown(long double g_eps, long double abs_der)
{
    return taylor_deduce_cooldown_impl(g_eps, abs_der);
}

#if defined(HEYOKA_HAVE_REAL128)

template <>
mppp::real128 taylor_deduce_cooldown(mppp::real128 g_eps, mppp::real128 abs_der)
{
    return taylor_deduce_cooldown_impl(g_eps, abs_der);
}

#endif

#if defined(HEYOKA_HAVE_REAL)

template <>
mppp::real taylor_deduce_cooldown(mppp::real g_eps, mppp::real abs_der)
{
    return taylor_deduce_cooldown_impl(std::move(g_eps), std::move(abs_der));
}

#endif

// Add a function that, given an input polynomial of order n represented
// as an array of coefficients:
// - reverses it,
// - translates it by 1,
// - counts the sign changes in the coefficients
//   of the resulting polynomial.
llvm::Function *llvm_add_poly_rtscc(llvm_state &s, llvm::Type *fp_t, std::uint32_t n, std::uint32_t batch_size)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding an rtscc function");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Add the translator and the sign changes counting function.
    auto *pt = add_poly_translator_1(s, fp_t, n, batch_size);
    auto *scc = llvm_add_csc(s, fp_t, n, batch_size);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - two poly coefficients output pointers,
    // - the output pointer to the number of sign changes (write-only),
    // - the input pointer to the original poly coefficients (read-only).
    // No overlap is allowed. The coefficient pointers are to external types.
    auto *ext_fp_ptr_t = llvm::PointerType::getUnqual(ext_fp_t);
    const std::vector<llvm::Type *> fargs{ext_fp_ptr_t, ext_fp_ptr_t,
                                          llvm::PointerType::getUnqual(builder.getInt32Ty()), ext_fp_ptr_t};
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, "poly_rtscc", &md);

    // Set the names/attributes of the function arguments.
    // NOTE: out_ptr1/2 are used both in read and write mode,
    // even though this function never actually reads from them
    // (they are just forwarded to other functions reading from them).
    // Because I am not 100% sure about the writeonly attribute
    // in this case, let's err on the side of caution and do not
    // mark them as writeonly.
    auto *out_ptr1 = f->args().begin();
    out_ptr1->setName("out_ptr1");
    out_ptr1->addAttr(llvm::Attribute::NoCapture);
    out_ptr1->addAttr(llvm::Attribute::NoAlias);

    auto *out_ptr2 = f->args().begin() + 1;
    out_ptr2->setName("out_ptr2");
    out_ptr2->addAttr(llvm::Attribute::NoCapture);
    out_ptr2->addAttr(llvm::Attribute::NoAlias);

    auto *n_sc_ptr = f->args().begin() + 2;
    n_sc_ptr->setName("n_sc_ptr");
    n_sc_ptr->addAttr(llvm::Attribute::NoCapture);
    n_sc_ptr->addAttr(llvm::Attribute::NoAlias);
    n_sc_ptr->addAttr(llvm::Attribute::WriteOnly);

    auto *cf_ptr = f->args().begin() + 3;
    cf_ptr->setName("cf_ptr");
    cf_ptr->addAttr(llvm::Attribute::NoCapture);
    cf_ptr->addAttr(llvm::Attribute::NoAlias);
    cf_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Do the reversion into out_ptr1.
    llvm_loop_u32(s, builder.getInt32(0), builder.getInt32(n + 1u), [&](llvm::Value *i) {
        auto *load_idx = builder.CreateMul(builder.CreateSub(builder.getInt32(n), i), builder.getInt32(batch_size));
        auto *store_idx = builder.CreateMul(i, builder.getInt32(batch_size));

        auto *cur_cf
            = ext_load_vector_from_memory(s, fp_t, builder.CreateInBoundsGEP(ext_fp_t, cf_ptr, load_idx), batch_size);
        ext_store_vector_to_memory(s, builder.CreateInBoundsGEP(ext_fp_t, out_ptr1, store_idx), cur_cf);
    });

    // Translate out_ptr1 into out_ptr2.
    builder.CreateCall(pt, {out_ptr2, out_ptr1});

    // Count the sign changes in out_ptr2.
    builder.CreateCall(scc, {n_sc_ptr, out_ptr2});

    // Return.
    builder.CreateRetVoid();

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // NOTE: the optimisation pass will be run outside.
    return f;
}

// Add a function implementing fast event exclusion check via the computation
// of the enclosure of the event equation's Taylor polynomial. The enclosure is computed
// either via the Cargo-Shisha algorithm (use_cs == true) or
// via Horner's scheme using interval arithmetic (use_cs == false). The default is the
// interval arithmetics implementation.
llvm::Function *llvm_add_fex_check(llvm_state &s, llvm::Type *fp_t, std::uint32_t n, std::uint32_t batch_size,
                                   bool use_cs)
{
    assert(batch_size > 0u); // LCOV_EXCL_LINE

    // Overflow check: we need to be able to index
    // into the array of coefficients.
    // LCOV_EXCL_START
    if (n == std::numeric_limits<std::uint32_t>::max()
        || batch_size > std::numeric_limits<std::uint32_t>::max() / (n + 1u)) {
        throw std::overflow_error("Overflow detected while adding a fex_check function");
    }
    // LCOV_EXCL_STOP

    auto &md = s.module();
    auto &builder = s.builder();
    auto &context = s.context();

    // Fetch the external type.
    auto *ext_fp_t = make_external_llvm_type(fp_t);

    // Fetch the current insertion block.
    auto *orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - pointer to the array of poly coefficients (read-only),
    // - pointer to the timestep value (s) (read-only),
    // - pointer to the int32 flag(s) to signal integration backwards in time (read-only),
    // - output pointer (write-only).
    // No overlap is allowed. All floating-point pointers are to external types.
    auto *ext_fp_ptr_t = llvm::PointerType::getUnqual(ext_fp_t);
    auto *int32_ptr_t = llvm::PointerType::getUnqual(builder.getInt32Ty());
    const std::vector<llvm::Type *> fargs{ext_fp_ptr_t, ext_fp_ptr_t, int32_ptr_t, int32_ptr_t};
    // The function does not return anything.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm_func_create(ft, llvm::Function::ExternalLinkage, "fex_check", &md);

    // Set the names/attributes of the function arguments.
    auto *cf_ptr = f->args().begin();
    cf_ptr->setName("cf_ptr");
    cf_ptr->addAttr(llvm::Attribute::NoCapture);
    cf_ptr->addAttr(llvm::Attribute::NoAlias);
    cf_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *h_ptr = f->args().begin() + 1;
    h_ptr->setName("h_ptr");
    h_ptr->addAttr(llvm::Attribute::NoCapture);
    h_ptr->addAttr(llvm::Attribute::NoAlias);
    h_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *back_flag_ptr = f->args().begin() + 2;
    back_flag_ptr->setName("back_flag_ptr");
    back_flag_ptr->addAttr(llvm::Attribute::NoCapture);
    back_flag_ptr->addAttr(llvm::Attribute::NoAlias);
    back_flag_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto *out_ptr = f->args().begin() + 3;
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);
    out_ptr->addAttr(llvm::Attribute::WriteOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Load the timestep.
    auto *h = ext_load_vector_from_memory(s, fp_t, h_ptr, batch_size);

    // Init the extrema of the enclosure.
    llvm::Value *enc_lo = nullptr, *enc_hi = nullptr;

    if (use_cs) {
        // Compute the enclosure of the polynomial.
        std::tie(enc_lo, enc_hi) = llvm_penc_cargo_shisha(s, fp_t, cf_ptr, n, h, batch_size);
    } else {
        // Load back_flag and convert it to a boolean vector.
        auto *back_flag
            = builder.CreateTrunc(load_vector_from_memory(builder, builder.getInt32Ty(), back_flag_ptr, batch_size),
                                  make_vector_type(builder.getInt1Ty(), batch_size));

        // Compute the components of the interval version of h. If we are integrating
        // forward, the components are (0, h), otherwise they are (h, 0).
        auto *h_lo = builder.CreateSelect(back_flag, h, llvm_constantfp(s, h->getType(), 0.));
        auto *h_hi = builder.CreateSelect(back_flag, llvm_constantfp(s, h->getType(), 0.), h);

        // Compute the enclosure of the polynomial.
        std::tie(enc_lo, enc_hi) = llvm_penc_interval(s, fp_t, cf_ptr, n, h_lo, h_hi, batch_size);
    }

    // Compute the sign of the components of the accumulator.
    auto *s_lo = llvm_sgn(s, enc_lo);
    auto *s_hi = llvm_sgn(s, enc_hi);

    // Check if the signs are equal and the low sign is nonzero.
    auto *cmp1 = builder.CreateICmpEQ(s_lo, s_hi);
    auto *cmp2 = builder.CreateICmpNE(s_lo, llvm::ConstantInt::get(s_lo->getType(), 0u));
    auto *cmp = builder.CreateLogicalAnd(cmp1, cmp2);
    // Extend cmp to int32_t.
    auto *retval = builder.CreateZExt(cmp, make_vector_type(builder.getInt32Ty(), batch_size));

    // Store the result in out_ptr.
    ext_store_vector_to_memory(s, out_ptr, retval);

    // Return.
    builder.CreateRetVoid();

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);

    // NOTE: the optimisation pass will be run outside.
    return f;
}

// A RAII helper to extract polys from a cache and
// return them to the cache upon destruction.
template <typename T>
class taylor_pwrap
{
    auto get_poly_from_cache(std::uint32_t n, const T &poly_init)
    {
        if (pc->empty()) {
            // No polynomials are available, create a new one.
            return std::vector<T>(boost::numeric_cast<typename std::vector<T>::size_type>(n + 1u), poly_init);
        } else {
            // Extract an existing polynomial from the cache.
            auto retval = std::move(pc->back());
            pc->pop_back();

            assert(retval.size() == n + 1u);

            return retval;
        }
    }

    void back_to_cache()
    {
        // NOTE: v will be empty when this has been
        // moved-from. In that case, we do not want
        // to send v back to the cache.
        if (!v.empty()) {
            assert(pc->empty() || (*pc)[0].size() == v.size());

            // Move v into the cache.
            pc->push_back(std::move(v));
        }
    }

public:
    explicit taylor_pwrap(taylor_poly_cache<T> &cache, std::uint32_t n, const T &poly_init)
        : pc(&cache), v(get_poly_from_cache(n, poly_init))
    {
    }

    taylor_pwrap(taylor_pwrap &&other) noexcept : pc(other.pc), v(std::move(other.v))
    {
        // Make sure we moved from a valid taylor_pwrap.
        assert(!v.empty()); // LCOV_EXCL_LINE

        // NOTE: we must ensure that other.v is cleared out, because
        // otherwise, when other is destructed, we could end up
        // returning to the cache a polynomial of the wrong size.
        //
        // In basically every existing implementation, moving a std::vector
        // will leave the original object empty, but technically this
        // does not seem to be guaranteed by the standard, so, better
        // safe than sorry. Quick checks on godbolt indicate that compilers
        // are anyway able to elide this clearing out of the vector.
        other.v.clear();
    }
    // NOTE: this does not support self-move, and requires that
    // the cache of other is the same as the cache of this.
    taylor_pwrap &operator=(taylor_pwrap &&other) noexcept
    {
        // Disallow self move.
        assert(this != &other); // LCOV_EXCL_LINE

        // Make sure the polyomial caches match.
        assert(pc == other.pc); // LCOV_EXCL_LINE

        // Make sure we are not moving from a
        // moved-from taylor_pwrap.
        assert(!other.v.empty()); // LCOV_EXCL_LINE

        // Put the current v in the cache.
        back_to_cache();

        // Do the move-assignment.
        v = std::move(other.v);

        // NOTE: we must ensure that other.v is cleared out, because
        // otherwise, when other is destructed, we could end up
        // returning to the cache a polynomial of the wrong size.
        //
        // In basically every existing implementation, moving a std::vector
        // will leave the original object empty, but technically this
        // does not seem to be guaranteed by the standard, so, better
        // safe than sorry. Quick checks on godbolt indicate that compilers
        // are anyway able to elide this clearing out of the vector.
        other.v.clear();

        return *this;
    }

    // Delete copy semantics.
    taylor_pwrap(const taylor_pwrap &) = delete;
    taylor_pwrap &operator=(const taylor_pwrap &) = delete;

    ~taylor_pwrap()
    {
#if !defined(NDEBUG)

        // Run consistency checks on the cache in debug mode.
        // The cache must not contain empty vectors
        // and all vectors in the cache must have the same size.
        if (!pc->empty()) {
            const auto op1 = (*pc)[0].size();

            for (const auto &vec : *pc) {
                assert(!vec.empty());
                assert(vec.size() == op1);
            }
        }

#endif

        // Put the current v in the cache.
        back_to_cache();
    }

    taylor_poly_cache<T> *pc;
    std::vector<T> v;
};

} // namespace detail

// NOTE: the def ctor is used only for serialisation purposes.
template <typename T>
taylor_adaptive<T>::ed_data::ed_data() = default;

template <typename T>
taylor_adaptive<T>::ed_data::ed_data(llvm_state s, std::vector<t_event_t> tes, std::vector<nt_event_t> ntes,
                                     std::uint32_t order, std::uint32_t dim, const T &s0)
    : m_tes(std::move(tes)), m_ntes(std::move(ntes)), m_state(std::move(s))
{
    assert(!m_tes.empty() || !m_ntes.empty()); // LCOV_EXCL_LINE

    // Fetch the scalar FP type.
    // NOTE: s0 is the first value in the state vector of the integrator,
    // from which the internal floating-point type is deduced.
    auto *fp_t = detail::internal_llvm_type_like(m_state, s0);

    // NOTE: the numeric cast will also ensure that we can
    // index into the events using 32-bit ints.
    const auto n_tes = boost::numeric_cast<std::uint32_t>(m_tes.size());
    const auto n_ntes = boost::numeric_cast<std::uint32_t>(m_ntes.size());

    // Setup m_ev_jet.
    // NOTE: check that we can represent
    // the requested size for m_ev_jet using
    // both its size type and std::uint32_t.
    // LCOV_EXCL_START
    if (n_tes > std::numeric_limits<std::uint32_t>::max() - n_ntes || order == std::numeric_limits<std::uint32_t>::max()
        || dim > std::numeric_limits<std::uint32_t>::max() - (n_tes + n_ntes)
        || dim + (n_tes + n_ntes) > std::numeric_limits<std::uint32_t>::max() / (order + 1u)
        || dim + (n_tes + n_ntes) > std::numeric_limits<decltype(m_ev_jet.size())>::max() / (order + 1u)) {
        throw std::overflow_error("Overflow detected in the initialisation of an adaptive Taylor integrator: the order "
                                  "or the state size is too large");
    }
    // LCOV_EXCL_STOP

    m_ev_jet.resize((dim + (n_tes + n_ntes)) * (order + 1u), detail::num_zero_like(s0));

    // Setup the vector of cooldowns.
    m_te_cooldowns.resize(boost::numeric_cast<decltype(m_te_cooldowns.size())>(m_tes.size()));

    // Setup the JIT-compiled functions.

    // Add the rtscc function. This will also indirectly
    // add the translator function.
    detail::llvm_add_poly_rtscc(m_state, fp_t, order, 1);

    // Add the function for the fast exclusion check.
    detail::llvm_add_fex_check(m_state, fp_t, order, 1);

    // Compile.
    m_state.compile();

    // Fetch the function pointers.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));
}

template <typename T>
taylor_adaptive<T>::ed_data::ed_data(const ed_data &o)
    : m_tes(o.m_tes), m_ntes(o.m_ntes), m_ev_jet(o.m_ev_jet), m_te_cooldowns(o.m_te_cooldowns), m_state(o.m_state),
      m_poly_cache(o.m_poly_cache)
{
    // For the vectors of detected events, just reserve the same amount of space.
    // These vectors are cleared out anyway during event detection.
    m_d_tes.reserve(o.m_d_tes.capacity());
    m_d_ntes.reserve(o.m_d_ntes.capacity());

    // Fetch the function pointers from the copied LLVM state.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));

    // Reserve space in m_wlist and m_isol.
    m_wlist.reserve(o.m_wlist.capacity());
    m_isol.reserve(o.m_isol.capacity());
}

template <typename T>
taylor_adaptive<T>::ed_data::~ed_data() = default;

template <typename T>
void taylor_adaptive<T>::ed_data::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_tes;
    ar << m_ntes;
    ar << m_ev_jet;
    ar << m_te_cooldowns;
    ar << m_state;
    ar << m_poly_cache;

    // Save the capacities of the vectors of detected events and
    // the root finding structures.
    ar << m_d_tes.capacity();
    ar << m_d_ntes.capacity();
    ar << m_wlist.capacity();
    ar << m_isol.capacity();
}

template <typename T>
void taylor_adaptive<T>::ed_data::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_tes;
    ar >> m_ntes;
    ar >> m_ev_jet;
    ar >> m_te_cooldowns;
    ar >> m_state;
    ar >> m_poly_cache;

    // Fetch the capacities.
    decltype(m_d_tes.capacity()) d_tes_cap{};
    ar >> d_tes_cap;
    decltype(m_d_ntes.capacity()) d_ntes_cap{};
    ar >> d_ntes_cap;
    decltype(m_wlist.capacity()) wlist_cap{};
    ar >> wlist_cap;
    decltype(m_isol.capacity()) isol_cap{};
    ar >> isol_cap;

    // Clear and reserve the capacities.
    m_d_tes.clear();
    m_d_tes.reserve(d_tes_cap);
    m_d_ntes.clear();
    m_d_ntes.reserve(d_ntes_cap);
    m_wlist.clear();
    m_wlist.reserve(wlist_cap);
    m_isol.clear();
    m_isol.reserve(isol_cap);

    // Fetch the function pointers from the LLVM state.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));
}

// Implementation of event detection.
template <typename T>
void taylor_adaptive<T>::ed_data::detect_events(const T &h, std::uint32_t order, std::uint32_t dim, const T &g_eps)
{
    using std::abs;
    using std::isfinite;

#if defined(HEYOKA_HAVE_REAL)

    if constexpr (std::is_same_v<T, mppp::real>) {
        assert(h.get_prec() == g_eps.get_prec());
    }

#endif

    // Clear the vectors of detected events.
    // NOTE: do it here as this is always necessary,
    // regardless of issues with h/g_eps.
    m_d_tes.clear();
    m_d_ntes.clear();

    // LCOV_EXCL_START
    if (!isfinite(h)) {
        detail::get_logger()->warn("event detection skipped due to an invalid timestep value of {}",
                                   detail::fp_to_string(h));
        return;
    }
    if (!isfinite(g_eps)) {
        detail::get_logger()->warn(
            "event detection skipped due to an invalid value of {} for the maximum error on the Taylor "
            "series of the event equations",
            detail::fp_to_string(g_eps));
        return;
    }
    // LCOV_EXCL_STOP

    if (h == 0) {
        // If the timestep is zero, skip event detection.
        return;
    }

    assert(order >= 2u); // LCOV_EXCL_LINE

    // The value that will be used to initialise the coefficients
    // of newly-created polynomials in the caches.
    const auto poly_init = detail::num_zero_like(h);

    // Temporary polynomials used in the bisection loop.
    detail::taylor_pwrap<T> tmp1(m_poly_cache, order, poly_init), tmp2(m_poly_cache, order, poly_init),
        tmp(m_poly_cache, order, poly_init);

    // Determine if we are integrating backwards in time.
    const std::uint32_t back_int = h < 0;

    // Helper to run event detection on a vector of events
    // (terminal or not). 'out' is the vector of detected
    // events, 'ev_vec' the input vector of events to detect.
    auto run_detection = [&](auto &out, const auto &ev_vec) {
        // Fetch the event type.
        using ev_type = typename detail::uncvref_t<decltype(ev_vec)>::value_type;

        for (std::uint32_t i = 0; i < ev_vec.size(); ++i) {
            // Extract the pointer to the Taylor polynomial for the
            // current event.
            const auto ptr = m_ev_jet.data()
                             + (i + dim + (detail::is_terminal_event_v<ev_type> ? 0u : m_tes.size())) * (order + 1u);

            // Run the fast exclusion check.
            // NOTE: in case of non-finite values in the Taylor
            // coefficients of the event equation, the worst that
            // can happen here is that we end up skipping event
            // detection altogether without a warning. This is ok,
            // and non-finite Taylor coefficients will be caught in the
            // step() implementations anyway.
            std::uint32_t fex_check_result{};
            m_fex_check(ptr, &h, &back_int, &fex_check_result);
            if (fex_check_result) {
                continue;
            }

            // Clear out the list of isolating intervals.
            m_isol.clear();

            // Reset the working list.
            m_wlist.clear();

            // Helper to add a detected event to out.
            // NOTE: the root here is expected to be already rescaled
            // to the [0, h) range.
            auto add_d_event = [&](T root) {
                // NOTE: we do one last check on the root in order to
                // avoid non-finite event times. This guarantees that
                // sorting the events by time is safe.
                if (!isfinite(root)) {
                    // LCOV_EXCL_START
                    detail::get_logger()->warn(
                        "polynomial root finding produced a non-finite root of {} - skipping the event",
                        detail::fp_to_string(root));
                    return;
                    // LCOV_EXCL_STOP
                }

                if (abs(root) >= abs(h)) {
                    // LCOV_EXCL_START
                    // NOTE: although event detection nominally
                    // happens in the [0, h) range, due to floating-point
                    // rounding we may end up detecting a abs(root) >= abs(h) in
                    // corner cases. If that happens, let us clamp
                    // root to be strictly < h in absolute value
                    // in order to respect
                    // the guarantee that events are always detected
                    // in the [0, h) range.
                    // NOTE: because throughout the various iterations
                    // of root finding the lower bound always remains exactly zero,
                    // it should not be possible for root to exit the [0, h)
                    // range from the other side.
                    detail::get_logger()->warn(
                        "polynomial root finding produced the root {} which is not smaller, in absolute "
                        "value, than the integration timestep {}",
                        detail::fp_to_string(root), detail::fp_to_string(h));

                    using std::nextafter;
                    root = nextafter(h, static_cast<T>(0));
                    // LCOV_EXCL_STOP
                }

                // Evaluate the derivative and its absolute value.
                const auto der = detail::poly_eval_1(ptr, root, order);
                auto abs_der = abs(der);

                // Check it before proceeding.
                if (!isfinite(der)) {
                    // LCOV_EXCL_START
                    detail::get_logger()->warn(
                        "polynomial root finding produced the root {} with nonfinite derivative {} - "
                        "skipping the event",
                        detail::fp_to_string(root), detail::fp_to_string(der));
                    return;
                    // LCOV_EXCL_STOP
                }

                // Compute the sign of the derivative.
                const auto d_sgn = detail::sgn(der);

                // Fetch and cache the desired event direction.
                const auto dir = ev_vec[i].get_direction();

                if (dir == event_direction::any) {
                    // If the event direction does not
                    // matter, just add it.
                    if constexpr (detail::is_terminal_event_v<ev_type>) {
                        out.emplace_back(i, std::move(root), d_sgn, std::move(abs_der));
                    } else {
                        out.emplace_back(i, std::move(root), d_sgn);
                    }
                } else {
                    // Otherwise, we need to record the event only if its direction
                    // matches the sign of the derivative.
                    if (static_cast<event_direction>(d_sgn) == dir) {
                        if constexpr (detail::is_terminal_event_v<ev_type>) {
                            out.emplace_back(i, std::move(root), d_sgn, std::move(abs_der));
                        } else {
                            out.emplace_back(i, std::move(root), d_sgn);
                        }
                    }
                }
            };

            // NOTE: if we are dealing with a terminal event on cooldown,
            // we will need to ignore roots within the cooldown period.
            // lb_offset is the value in the original [0, 1) range corresponding
            // to the end of the cooldown.
            const auto lb_offset = [&]() {
                if constexpr (detail::is_terminal_event_v<ev_type>) {
                    if (m_te_cooldowns[i]) {
                        // NOTE: need to distinguish between forward
                        // and backward integration.
                        // NOTE: the division by abs(h) will ensure that, in case
                        // of mppp::real, the result is computed in a precision
                        // no less than h's.
                        if (h >= 0) {
                            return (m_te_cooldowns[i]->second - m_te_cooldowns[i]->first) / abs(h);
                        } else {
                            return (m_te_cooldowns[i]->second + m_te_cooldowns[i]->first) / abs(h);
                        }
                    }
                }

                // NOTE: we end up here if the event is not terminal
                // or not on cooldown.
                // NOTE: ensure this is inited properly, precision-wise.
                return detail::num_zero_like(h);
            }();

            if (lb_offset >= 1) {
                // LCOV_EXCL_START
                // NOTE: the whole integration range is in the cooldown range,
                // move to the next event.
                SPDLOG_LOGGER_DEBUG(
                    detail::get_logger(),
                    "the integration timestep falls within the cooldown range for the terminal event {}, skipping", i);
                continue;
                // LCOV_EXCL_STOP
            }

            // Rescale the event polynomial so that the range [0, h)
            // becomes [0, 1), and write the resulting polynomial into tmp.
            // NOTE: at the first iteration (i.e., for the first event),
            // tmp has been constructed correctly outside this function.
            // Below, tmp will first be moved into m_wlist (thus rendering
            // it invalid) but it will immediately be revived at the
            // first iteration of the do/while loop. Thus, when we get
            // here again, tmp will be again in a well-formed state.
            assert(!tmp.v.empty());             // LCOV_EXCL_LINE
            assert(tmp.v.size() - 1u == order); // LCOV_EXCL_LINE
            detail::poly_rescale(tmp.v.data(), ptr, h, order);

            // Place the first element in the working list.
            // NOTE: it's important that the initial bounds
            // are created with the appropriate precision, in
            // case of mppp::real. Otherwise, we risk of running
            // the bisection at 32 bits of precision.
            m_wlist.emplace_back(detail::num_zero_like(h), detail::num_one_like(h), std::move(tmp));

#if !defined(NDEBUG)
            auto max_wl_size = m_wlist.size();
            auto max_isol_size = m_isol.size();
#endif

            // Flag to signal that the do-while loop below failed.
            bool loop_failed = false;

            // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
            do {
                // Fetch the current interval and polynomial from the working list.
                // NOTE: from now on, tmp contains the polynomial referred
                // to as q(x) in the real-root isolation wikipedia page.
                // NOTE: q(x) is the transformed polynomial whose roots in the x range [0, 1) we will
                // be looking for. lb and ub represent what 0 and 1 correspond to in the *original*
                // [0, 1) range.
                auto lb = std::move(std::get<0>(m_wlist.back()));
                auto ub = std::move(std::get<1>(m_wlist.back()));
                // NOTE: this will either revive an invalid tmp (first iteration),
                // or it will replace it with one of the bisecting polynomials.
                tmp = std::move(std::get<2>(m_wlist.back()));
                m_wlist.pop_back();

                // Check for an event at the lower bound, which occurs
                // if the constant term of the polynomial is zero. We also
                // check for finiteness of all the other coefficients, otherwise
                // we cannot really claim to have detected an event.
                // When we do proper root finding below, the
                // algorithm should be able to detect non-finite
                // polynomials.
                // NOTE: it's not 100% clear to me why we have to special-case
                // this, but if we don't, the root is not detected. Note that
                // the wikipedia algorithm also has a special case for this.
                if (tmp.v[0] == 0 // LCOV_EXCL_LINE
                    && std::all_of(tmp.v.data() + 1, tmp.v.data() + 1 + order,
                                   [](const auto &x) { return isfinite(x); })) {
                    // NOTE: we will have to skip the event if we are dealing
                    // with a terminal event on cooldown and the lower bound
                    // falls within the cooldown time.
                    std::conditional_t<detail::is_terminal_event_v<ev_type>, bool, const bool> skip_event = false;
                    if constexpr (detail::is_terminal_event_v<ev_type>) {
                        if (lb < lb_offset) {
                            SPDLOG_LOGGER_DEBUG(detail::get_logger(),
                                                "terminal event {} detected at the beginning of an isolating interval "
                                                "is subject to cooldown, ignoring",
                                                i);
                            skip_event = true;
                        }
                    }

                    if (!skip_event) {
                        // NOTE: the original range had been rescaled wrt to h.
                        // Thus, we need to rescale back when adding the detected
                        // event.
                        add_d_event(lb * h);
                    }
                }

                // Reverse tmp into tmp1, translate tmp1 by 1 with output
                // in tmp2, and count the sign changes in tmp2.
                std::uint32_t n_sc{};
                m_rtscc(tmp1.v.data(), tmp2.v.data(), &n_sc, tmp.v.data());

                if (n_sc == 1u) {
                    // Found isolating interval, add it to isol.
                    m_isol.emplace_back(std::move(lb), std::move(ub));
                } else if (n_sc > 1u) {
                    // No isolating interval found, bisect.

                    // First we transform q into 2**n * q(x/2) and store the result
                    // into tmp1.
                    detail::poly_rescale_p2(tmp1.v.data(), tmp.v.data(), order);
                    // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
                    m_pt(tmp2.v.data(), tmp1.v.data());

                    // Finally we add tmp1 and tmp2 to the working list.
                    auto mid = lb / 2 + ub / 2;
                    // NOTE: don't add the lower range if it falls
                    // entirely within the cooldown range.
                    if (lb_offset < mid) {
                        m_wlist.emplace_back(std::move(lb), mid, std::move(tmp1));

                        // Revive tmp1.
                        tmp1 = detail::taylor_pwrap<T>(m_poly_cache, order, poly_init);
                    } else {
                        // LCOV_EXCL_START
                        SPDLOG_LOGGER_DEBUG(
                            detail::get_logger(),
                            "ignoring lower interval in a bisection that would fall entirely in the cooldown period");
                        // LCOV_EXCL_STOP
                    }
                    m_wlist.emplace_back(std::move(mid), std::move(ub), std::move(tmp2));

                    // Revive tmp2.
                    tmp2 = detail::taylor_pwrap<T>(m_poly_cache, order, poly_init);
                }

#if !defined(NDEBUG)
                max_wl_size = std::max(max_wl_size, m_wlist.size());
                max_isol_size = std::max(max_isol_size, m_isol.size());
#endif

                // LCOV_EXCL_START
                // We want to put limits in order to avoid an endless loop when the algorithm fails.
                // The first check is on the working list size and it is based
                // on heuristic observation of the algorithm's behaviour in pathological
                // cases. The second check is that we cannot possibly find more isolating
                // intervals than the degree of the polynomial.
                if (m_wlist.size() > 250u || m_isol.size() > order) {
                    detail::get_logger()->warn(
                        "the polynomial root isolation algorithm failed during event detection: the working "
                        "list size is {} and the number of isolating intervals is {}",
                        m_wlist.size(), m_isol.size());

                    loop_failed = true;

                    break;
                }
                // LCOV_EXCL_STOP

            } while (!m_wlist.empty());

#if !defined(NDEBUG)
            SPDLOG_LOGGER_DEBUG(detail::get_logger(), "max working list size: {}", max_wl_size);
            SPDLOG_LOGGER_DEBUG(detail::get_logger(), "max isol list size   : {}", max_isol_size);
#endif

            if (m_isol.empty() || loop_failed) {
                // Don't do root finding for this event if the loop failed,
                // or if the list of isolating intervals is empty. Just
                // move to the next event.
                continue;
            }

            // Reconstruct a version of the original event polynomial
            // in which the range [0, h) is rescaled to [0, 1). We need
            // to do root finding on the rescaled polynomial because the
            // isolating intervals are also rescaled to [0, 1).
            // NOTE: tmp1 was either created with the correct size outside this
            // function, or it was re-created in the bisection above.
            detail::poly_rescale(tmp1.v.data(), ptr, h, order);

            // Run the root finding in the isolating intervals.
            for (auto &[lb, ub] : m_isol) {
                if constexpr (detail::is_terminal_event_v<ev_type>) {
                    // NOTE: if we are dealing with a terminal event
                    // subject to cooldown, we need to ensure that
                    // we don't look for roots before the cooldown has expired.
                    if (lb < lb_offset) {
                        // Make sure we move lb past the cooldown.
                        lb = lb_offset;

                        // NOTE: this should be ensured by the fact that
                        // we ensure above (lb_offset < mid) that we don't
                        // end up with an invalid interval.
                        assert(lb < ub); // LCOV_EXCL_LINE

                        // Check if the interval still contains a zero.
                        auto f_lb = detail::poly_eval(tmp1.v.data(), lb, order);
                        auto f_ub = detail::poly_eval(tmp1.v.data(), ub, order);

                        if (!(std::move(f_lb) * std::move(f_ub) < 0)) {
                            SPDLOG_LOGGER_DEBUG(detail::get_logger(),
                                                "terminal event {} is subject to cooldown, ignoring", i);
                            continue;
                        }
                    }
                }

                // Run the root finding.
                const auto [root, cflag] = detail::bracketed_root_find(tmp1.v.data(), order, lb, ub);

                if (cflag == 0) {
                    // Root finding finished successfully, record the event.
                    // The found root needs to be rescaled by h.
                    add_d_event(root * h);
                } else {
                    // LCOV_EXCL_START
                    // Root finding encountered some issue. Ignore the
                    // event and log the issue.
                    if (cflag == -1) {
                        detail::get_logger()->warn(
                            "polynomial root finding during event detection failed due to too many iterations");
                    } else {
                        detail::get_logger()->warn("polynomial root finding during event detection returned a nonzero "
                                                   "errno with error code {}",
                                                   cflag);
                    }
                    // LCOV_EXCL_STOP
                }
            }
        }
    };

    run_detection(m_d_tes, m_tes);
    run_detection(m_d_ntes, m_ntes);
}

// Explicit instantiation of the book-keeping structures for event detection
// in the scalar integrator.
template struct taylor_adaptive<float>::ed_data;
template struct taylor_adaptive<double>::ed_data;
template struct taylor_adaptive<long double>::ed_data;

#if defined(HEYOKA_HAVE_REAL128)

template struct taylor_adaptive<mppp::real128>::ed_data;

#endif

#if defined(HEYOKA_HAVE_REAL)

template struct taylor_adaptive<mppp::real>::ed_data;

#endif

// NOTE: the def ctor is used only for serialisation purposes.
template <typename T>
taylor_adaptive_batch<T>::ed_data::ed_data() = default;

template <typename T>
taylor_adaptive_batch<T>::ed_data::ed_data(llvm_state s, std::vector<t_event_t> tes, std::vector<nt_event_t> ntes,
                                           std::uint32_t order, std::uint32_t dim, std::uint32_t batch_size)
    : m_tes(std::move(tes)), m_ntes(std::move(ntes)), m_state(std::move(s))
{
    assert(!m_tes.empty() || !m_ntes.empty()); // LCOV_EXCL_LINE
    assert(batch_size != 0u);                  // LCOV_EXCL_LINE

    // Fetch the scalar FP type.
    auto *fp_t = detail::to_external_llvm_type<T>(m_state.context());

    // NOTE: the numeric cast will also ensure that we can
    // index into the events using 32-bit ints.
    const auto n_tes = boost::numeric_cast<std::uint32_t>(m_tes.size());
    const auto n_ntes = boost::numeric_cast<std::uint32_t>(m_ntes.size());

    // Setup m_ev_jet.
    // NOTE: check that we can represent
    // the requested size for m_ev_jet using
    // both its size type and std::uint32_t.
    // LCOV_EXCL_START
    if (n_tes > std::numeric_limits<std::uint32_t>::max() - n_ntes || order == std::numeric_limits<std::uint32_t>::max()
        || dim > std::numeric_limits<std::uint32_t>::max() - (n_tes + n_ntes)
        || dim + (n_tes + n_ntes) > std::numeric_limits<std::uint32_t>::max() / (order + 1u)
        || (dim + (n_tes + n_ntes)) * (order + 1u) > std::numeric_limits<std::uint32_t>::max() / batch_size
        || (dim + (n_tes + n_ntes)) * (order + 1u)
               > std::numeric_limits<decltype(m_ev_jet.size())>::max() / batch_size) {
        throw std::overflow_error(
            "Overflow detected in the initialisation of an adaptive Taylor integrator in batch mode: the order "
            "or the state size is too large");
    }
    // LCOV_EXCL_STOP
    m_ev_jet.resize((dim + (n_tes + n_ntes)) * (order + 1u) * batch_size);

    // Prepare m_max_abs_state.
    m_max_abs_state.resize(batch_size);

    // Prepare m_g_eps.
    m_g_eps.resize(batch_size);

    // Prepare m_d_tes.
    m_d_tes.resize(boost::numeric_cast<decltype(m_d_tes.size())>(batch_size));

    // Setup the vector of cooldowns.
    m_te_cooldowns.resize(boost::numeric_cast<decltype(m_te_cooldowns.size())>(batch_size));
    for (auto &v : m_te_cooldowns) {
        v.resize(boost::numeric_cast<decltype(v.size())>(m_tes.size()));
    }

    // Prepare m_d_ntes.
    m_d_ntes.resize(boost::numeric_cast<decltype(m_d_ntes.size())>(batch_size));

    // Prepare m_back_int.
    m_back_int.resize(boost::numeric_cast<decltype(m_back_int.size())>(batch_size));

    // Prepare m_fex_check_res.
    m_fex_check_res.resize(batch_size);

    // Setup the JIT-compiled functions.

    // Add the rtscc function. This will also indirectly
    // add the translator function.
    // NOTE: keep batch size to 1 because the real-root
    // isolation is scalarised.
    detail::llvm_add_poly_rtscc(m_state, fp_t, order, 1);

    // Add the function for the fast exclusion check.
    // NOTE: the fast exclusion check is vectorised.
    detail::llvm_add_fex_check(m_state, fp_t, order, batch_size);

    // Compile.
    m_state.compile();

    // Fetch the function pointers.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));
}

template <typename T>
taylor_adaptive_batch<T>::ed_data::ed_data(const ed_data &o)
    : m_tes(o.m_tes), m_ntes(o.m_ntes), m_ev_jet(o.m_ev_jet), m_max_abs_state(o.m_max_abs_state), m_g_eps(o.m_g_eps),
      m_te_cooldowns(o.m_te_cooldowns), m_state(o.m_state), m_back_int(o.m_back_int),
      m_fex_check_res(o.m_fex_check_res), m_poly_cache(o.m_poly_cache)
{
    // Fetch the batch size.
    const auto batch_size = static_cast<std::uint32_t>(o.m_d_tes.size());

    // Prepare m_d_tes with the correct size and capacities.
    m_d_tes.resize(batch_size);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        m_d_tes[i].reserve(o.m_d_tes[i].capacity());
    }

    // Prepare m_d_ntes with the correct size and capacities.
    m_d_ntes.resize(batch_size);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        m_d_ntes[i].reserve(o.m_d_ntes[i].capacity());
    }

    // Fetch the function pointers from the copied LLVM state.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));

    // Reserve space in m_wlist and m_isol.
    m_wlist.reserve(o.m_wlist.capacity());
    m_isol.reserve(o.m_isol.capacity());
}

template <typename T>
taylor_adaptive_batch<T>::ed_data::~ed_data() = default;

template <typename T>
void taylor_adaptive_batch<T>::ed_data::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_tes;
    ar << m_ntes;
    ar << m_ev_jet;
    ar << m_max_abs_state;
    ar << m_g_eps;
    ar << m_te_cooldowns;
    ar << m_state;
    ar << m_back_int;
    ar << m_fex_check_res;
    ar << m_poly_cache;

    // Save the batch size.
    ar << static_cast<std::uint32_t>(m_d_tes.size());

    // Save the capacities of m_d_tes.
    for (const auto &v : m_d_tes) {
        ar << v.capacity();
    }

    // Save the capacities of m_d_ntes.
    for (const auto &v : m_d_ntes) {
        ar << v.capacity();
    }

    // Save the capacities of m_wlist and m_isol.
    ar << m_wlist.capacity();
    ar << m_isol.capacity();
}

template <typename T>
void taylor_adaptive_batch<T>::ed_data::load(boost::archive::binary_iarchive &ar, unsigned)
{
    ar >> m_tes;
    ar >> m_ntes;
    ar >> m_ev_jet;
    ar >> m_max_abs_state;
    ar >> m_g_eps;
    ar >> m_te_cooldowns;
    ar >> m_state;
    ar >> m_back_int;
    ar >> m_fex_check_res;
    ar >> m_poly_cache;

    // Recover the batch size.
    std::uint32_t batch_size{};
    ar >> batch_size;

    // Recover m_d_tes.
    m_d_tes.resize(batch_size);
    for (auto &v : m_d_tes) {
        decltype(v.capacity()) cap{};
        ar >> cap;
        v.clear();
        v.reserve(cap);
    }

    // Recover m_d_mtes.
    m_d_ntes.resize(batch_size);
    for (auto &v : m_d_ntes) {
        decltype(v.capacity()) cap{};
        ar >> cap;
        v.clear();
        v.reserve(cap);
    }

    // Recover the capacities of m_wlist and m_isol.
    decltype(m_wlist.capacity()) wlist_cap{};
    ar >> wlist_cap;
    decltype(m_isol.capacity()) isol_cap{};
    ar >> isol_cap;

    // Clear and reserve the capacities.
    m_wlist.clear();
    m_wlist.reserve(wlist_cap);
    m_isol.clear();
    m_isol.reserve(isol_cap);

    // Fetch the function pointers from the LLVM state.
    m_pt = reinterpret_cast<pt_t>(m_state.jit_lookup("poly_translate_1"));
    m_rtscc = reinterpret_cast<rtscc_t>(m_state.jit_lookup("poly_rtscc"));
    m_fex_check = reinterpret_cast<fex_check_t>(m_state.jit_lookup("fex_check"));
}

// Implementation of event detection.
template <typename T>
void taylor_adaptive_batch<T>::ed_data::detect_events(const T *h_ptr, std::uint32_t order, std::uint32_t dim,
                                                      std::uint32_t batch_size)
{
    using std::abs;
    using std::isfinite;

    // Clear the vectors of detected events, and determine if we are integrating
    // backwards in time.
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        m_d_tes[i].clear();
        m_d_ntes[i].clear();
        m_back_int[i] = h_ptr[i] < 0;
    }

    assert(order >= 2u); // LCOV_EXCL_LINE

    // The value that will be used to initialise the coefficients
    // of newly-created polynomials in the caches.
    const T poly_init = 0;

    // Temporary polynomials used in the bisection loop.
    detail::taylor_pwrap<T> tmp1(m_poly_cache, order, poly_init), tmp2(m_poly_cache, order, poly_init),
        tmp(m_poly_cache, order, poly_init);
    // The temporary polynomial used when extracting a specific batch element
    // from a polynomial of batches.
    detail::taylor_pwrap<T> scal_poly(m_poly_cache, order, poly_init);

    // Helper to run event detection on a vector of events
    // (terminal or not). 'out_vec' is the vector of detected
    // events, 'ev_vec' the input vector of events to detect.
    auto run_detection = [&](auto &out_vec, const auto &ev_vec) {
        // Fetch the event type.
        using ev_type = typename detail::uncvref_t<decltype(ev_vec)>::value_type;

        for (std::uint32_t i = 0; i < ev_vec.size(); ++i) {
            // Extract the pointer to the Taylor polynomial for the
            // current event.
            const auto batch_ptr
                = m_ev_jet.data()
                  + (i + dim + (detail::is_terminal_event_v<ev_type> ? 0u : m_tes.size())) * (order + 1u) * batch_size;

            // Run the fast exclusion check.
            // NOTE: in case of non-finite values in the Taylor
            // coefficients of the event equation, the worst that
            // can happen here is that we end up skipping event
            // detection altogether without a warning. This is ok,
            // and non-finite Taylor coefficients will be caught in the
            // step() implementations anyway.
            m_fex_check(batch_ptr, h_ptr, m_back_int.data(), m_fex_check_res.data());
            // NOTE: remove this check, or does this provide any performance
            // benefit wrt just checking in the next for loop?
            if (std::all_of(m_fex_check_res.begin(), m_fex_check_res.end(),
                            [](auto f) { return static_cast<bool>(f); })) {
                continue;
            }

            // Run event detection on all the batch elements.
            for (std::uint32_t j = 0; j < batch_size; ++j) {
                // See if the fast exclusion check was positive
                // for the current batch element.
                if (m_fex_check_res[j]) {
                    continue;
                }

                // Start by running the checks that are
                // run at the very beginning of the scalar event detection
                // function.
                const auto h = h_ptr[j];
                const auto g_eps = m_g_eps[j];

                // LCOV_EXCL_START
                if (!isfinite(h)) {
                    detail::get_logger()->warn(
                        "event detection skipped due to an invalid timestep value of {} at the batch index {}",
                        detail::fp_to_string(h), j);
                    continue;
                }
                if (!isfinite(g_eps)) {
                    detail::get_logger()->warn(
                        "event detection skipped due to an invalid value of {} for the maximum error on the Taylor "
                        "series of the event equations at the batch index {}",
                        detail::fp_to_string(g_eps), j);
                    continue;
                }
                // LCOV_EXCL_STOP

                if (h == 0) {
                    // If the timestep is zero, skip event detection.
                    continue;
                }

                // Clear out the list of isolating intervals.
                m_isol.clear();

                // Reset the working list.
                m_wlist.clear();

                // Fetch a reference to the vector of detected events.
                auto &out = out_vec[j];

                // Copy the polynomial coefficients from batch_ptr to scal_poly.
                for (std::uint32_t k = 0; k <= order; ++k) {
                    scal_poly.v[k] = *(batch_ptr + j + k * batch_size);
                }
                const auto ptr = std::as_const(scal_poly.v).data();

                // Helper to add a detected event to out.
                // NOTE: the root here is expected to be already rescaled
                // to the [0, h) range.
                auto add_d_event = [&](T root) {
                    // NOTE: we do one last check on the root in order to
                    // avoid non-finite event times. This guarantees that
                    // sorting the events by time is safe.
                    if (!isfinite(root)) {
                        // LCOV_EXCL_START
                        detail::get_logger()->warn(
                            "polynomial root finding produced a non-finite root of {} at the batch "
                            "index {} - skipping the event",
                            detail::fp_to_string(root), j);
                        return;
                        // LCOV_EXCL_STOP
                    }

                    if (abs(root) >= abs(h)) {
                        // LCOV_EXCL_START
                        // NOTE: although event detection nominally
                        // happens in the [0, h) range, due to floating-point
                        // rounding we may end up detecting a abs(root) >= abs(h) in
                        // corner cases. If that happens, let us clamp
                        // root to be strictly < h in absolute value
                        // in order to respect
                        // the guarantee that events are always detected
                        // in the [0, h) range.
                        // NOTE: because throughout the various iterations
                        // of root finding the lower bound always remains exactly zero,
                        // it should not be possible for root to exit the [0, h)
                        // range from the other side.
                        detail::get_logger()->warn(
                            "polynomial root finding produced the root {} which is not smaller, in absolute "
                            "value, than the integration timestep {}",
                            detail::fp_to_string(root), detail::fp_to_string(h));

                        using std::nextafter;
                        root = nextafter(h, static_cast<T>(0));
                        // LCOV_EXCL_STOP
                    }

                    // Evaluate the derivative and its absolute value.
                    const auto der = detail::poly_eval_1(ptr, root, order);
                    const auto abs_der = abs(der);

                    // Check it before proceeding.
                    if (!isfinite(der)) {
                        // LCOV_EXCL_START
                        detail::get_logger()->warn(
                            "polynomial root finding produced the root {} with nonfinite derivative {} "
                            "at the batch index {} - skipping the event",
                            detail::fp_to_string(root), detail::fp_to_string(der), j);
                        return;
                        // LCOV_EXCL_STOP
                    }

                    // Compute sign of the derivative.
                    const auto d_sgn = detail::sgn(der);

                    // Fetch and cache the desired event direction.
                    const auto dir = ev_vec[i].get_direction();

                    if (dir == event_direction::any) {
                        // If the event direction does not
                        // matter, just add it.
                        if constexpr (detail::is_terminal_event_v<ev_type>) {
                            out.emplace_back(i, root, d_sgn, abs_der);
                        } else {
                            out.emplace_back(i, root, d_sgn);
                        }
                    } else {
                        // Otherwise, we need to record the event only if its direction
                        // matches the sign of the derivative.
                        if (static_cast<event_direction>(d_sgn) == dir) {
                            if constexpr (detail::is_terminal_event_v<ev_type>) {
                                out.emplace_back(i, root, d_sgn, abs_der);
                            } else {
                                out.emplace_back(i, root, d_sgn);
                            }
                        }
                    }
                };

                // NOTE: if we are dealing with a terminal event on cooldown,
                // we will need to ignore roots within the cooldown period.
                // lb_offset is the value in the original [0, 1) range corresponding
                // to the end of the cooldown.
                const auto lb_offset = [&]() {
                    if constexpr (detail::is_terminal_event_v<ev_type>) {
                        if (m_te_cooldowns[j][i]) {
                            // NOTE: need to distinguish between forward
                            // and backward integration.
                            if (h >= 0) {
                                return (m_te_cooldowns[j][i]->second - m_te_cooldowns[j][i]->first) / abs(h);
                            } else {
                                return (m_te_cooldowns[j][i]->second + m_te_cooldowns[j][i]->first) / abs(h);
                            }
                        }
                    }

                    // NOTE: we end up here if the event is not terminal
                    // or not on cooldown.
                    return static_cast<T>(0);
                }();

                if (lb_offset >= 1) {
                    // LCOV_EXCL_START
                    // NOTE: the whole integration range is in the cooldown range,
                    // move to the next event.
                    SPDLOG_LOGGER_DEBUG(detail::get_logger(),
                                        "the integration timestep falls within the cooldown range for the terminal "
                                        "event {} at the batch index {}, skipping",
                                        i, j);
                    continue;
                    // LCOV_EXCL_STOP
                }

                // Rescale the event polynomial so that the range [0, h)
                // becomes [0, 1), and write the resulting polynomial into tmp.
                // NOTE: at the first iteration (i.e., for the first event),
                // tmp has been constructed correctly outside this function.
                // Below, tmp will first be moved into m_wlist (thus rendering
                // it invalid) but it will immediately be revived at the
                // first iteration of the do/while loop. Thus, when we get
                // here again, tmp will be again in a well-formed state.
                assert(!tmp.v.empty());             // LCOV_EXCL_LINE
                assert(tmp.v.size() - 1u == order); // LCOV_EXCL_LINE
                detail::poly_rescale(tmp.v.data(), ptr, h, order);

                // Place the first element in the working list.
                m_wlist.emplace_back(0, 1, std::move(tmp));

#if !defined(NDEBUG)
                auto max_wl_size = m_wlist.size();
                auto max_isol_size = m_isol.size();
#endif

                // Flag to signal that the do-while loop below failed.
                bool loop_failed = false;

                // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
                do {
                    // Fetch the current interval and polynomial from the working list.
                    // NOTE: from now on, tmp contains the polynomial referred
                    // to as q(x) in the real-root isolation wikipedia page.
                    // NOTE: q(x) is the transformed polynomial whose roots in the x range [0, 1) we will
                    // be looking for. lb and ub represent what 0 and 1 correspond to in the *original*
                    // [0, 1) range.
                    auto lb = std::get<0>(m_wlist.back());
                    auto ub = std::get<1>(m_wlist.back());
                    // NOTE: this will either revive an invalid tmp (first iteration),
                    // or it will replace it with one of the bisecting polynomials.
                    tmp = std::move(std::get<2>(m_wlist.back()));
                    m_wlist.pop_back();

                    // Check for an event at the lower bound, which occurs
                    // if the constant term of the polynomial is zero. We also
                    // check for finiteness of all the other coefficients, otherwise
                    // we cannot really claim to have detected an event.
                    // When we do proper root finding below, the
                    // algorithm should be able to detect non-finite
                    // polynomials.
                    if (tmp.v[0] == 0 // LCOV_EXCL_LINE
                        && std::all_of(tmp.v.data() + 1, tmp.v.data() + 1 + order,
                                       [](const auto &x) { return isfinite(x); })) {
                        // NOTE: we will have to skip the event if we are dealing
                        // with a terminal event on cooldown and the lower bound
                        // falls within the cooldown time.
                        std::conditional_t<detail::is_terminal_event_v<ev_type>, bool, const bool> skip_event = false;
                        if constexpr (detail::is_terminal_event_v<ev_type>) {
                            if (lb < lb_offset) {
                                SPDLOG_LOGGER_DEBUG(detail::get_logger(),
                                                    "terminal event {} detected at the beginning of an isolating "
                                                    "interval at the batch index {} "
                                                    "is subject to cooldown, ignoring",
                                                    i, j);
                                skip_event = true;
                            }
                        }

                        if (!skip_event) {
                            // NOTE: the original range had been rescaled wrt to h.
                            // Thus, we need to rescale back when adding the detected
                            // event.
                            add_d_event(lb * h);
                        }
                    }

                    // Reverse tmp into tmp1, translate tmp1 by 1 with output
                    // in tmp2, and count the sign changes in tmp2.
                    std::uint32_t n_sc{};
                    m_rtscc(tmp1.v.data(), tmp2.v.data(), &n_sc, tmp.v.data());

                    if (n_sc == 1u) {
                        // Found isolating interval, add it to isol.
                        m_isol.emplace_back(lb, ub);
                    } else if (n_sc > 1u) {
                        // No isolating interval found, bisect.

                        // First we transform q into 2**n * q(x/2) and store the result
                        // into tmp1.
                        detail::poly_rescale_p2(tmp1.v.data(), tmp.v.data(), order);
                        // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
                        m_pt(tmp2.v.data(), tmp1.v.data());

                        // Finally we add tmp1 and tmp2 to the working list.
                        // NOTE: not sure why this is not picked up by the code
                        // coverage tool.
                        const auto mid = lb / 2 + ub / 2; // LCOV_EXCL_LINE
                        // NOTE: don't add the lower range if it falls
                        // entirely within the cooldown range.
                        if (lb_offset < mid) {
                            m_wlist.emplace_back(lb, mid, std::move(tmp1));

                            // Revive tmp1.
                            tmp1 = detail::taylor_pwrap<T>(m_poly_cache, order, poly_init);
                        } else {
                            // LCOV_EXCL_START
                            SPDLOG_LOGGER_DEBUG(detail::get_logger(),
                                                "ignoring lower interval in a bisection that would fall "
                                                "entirely in the cooldown period at the batch index {}",
                                                j);
                            // LCOV_EXCL_STOP
                        }
                        m_wlist.emplace_back(mid, ub, std::move(tmp2));

                        // Revive tmp2.
                        tmp2 = detail::taylor_pwrap<T>(m_poly_cache, order, poly_init);
                    }

#if !defined(NDEBUG)
                    max_wl_size = std::max(max_wl_size, m_wlist.size());
                    max_isol_size = std::max(max_isol_size, m_isol.size());
#endif

                    // We want to put limits in order to avoid an endless loop when the algorithm fails.
                    // The first check is on the working list size and it is based
                    // on heuristic observation of the algorithm's behaviour in pathological
                    // cases. The second check is that we cannot possibly find more isolating
                    // intervals than the degree of the polynomial.
                    // LCOV_EXCL_START
                    if (m_wlist.size() > 250u || m_isol.size() > order) {
                        detail::get_logger()->warn(
                            "the polynomial root isolation algorithm failed during event detection at "
                            "the batch index {}: the working "
                            "list size is {} and the number of isolating intervals is {}",
                            j, m_wlist.size(), m_isol.size());

                        loop_failed = true;

                        break;
                    }
                    // LCOV_EXCL_STOP

                } while (!m_wlist.empty());

#if !defined(NDEBUG)
                SPDLOG_LOGGER_DEBUG(detail::get_logger(), "max working list size at the batch index {}: {}", j,
                                    max_wl_size);
                SPDLOG_LOGGER_DEBUG(detail::get_logger(), "max isol list size at the batch index {}   : {}", j,
                                    max_isol_size);
#endif

                if (m_isol.empty() || loop_failed) {
                    // Don't do root finding for this event if the loop failed,
                    // or if the list of isolating intervals is empty. Just
                    // move to the next event.
                    continue;
                }

                // Reconstruct a version of the original event polynomial
                // in which the range [0, h) is rescaled to [0, 1). We need
                // to do root finding on the rescaled polynomial because the
                // isolating intervals are also rescaled to [0, 1).
                // NOTE: tmp1 was either created with the correct size outside this
                // function, or it was re-created in the bisection above.
                detail::poly_rescale(tmp1.v.data(), ptr, h, order);

                // Run the root finding in the isolating intervals.
                for (auto &[lb, ub] : m_isol) {
                    if constexpr (detail::is_terminal_event_v<ev_type>) {
                        // NOTE: if we are dealing with a terminal event
                        // subject to cooldown, we need to ensure that
                        // we don't look for roots before the cooldown has expired.
                        if (lb < lb_offset) {
                            // Make sure we move lb past the cooldown.
                            lb = lb_offset;

                            // NOTE: this should be ensured by the fact that
                            // we ensure above (lb_offset < mid) that we don't
                            // end up with an invalid interval.
                            assert(lb < ub); // LCOV_EXCL_LINE

                            // Check if the interval still contains a zero.
                            const auto f_lb = detail::poly_eval(tmp1.v.data(), lb, order);
                            const auto f_ub = detail::poly_eval(tmp1.v.data(), ub, order);

                            if (!(f_lb * f_ub < 0)) {
                                SPDLOG_LOGGER_DEBUG(
                                    detail::get_logger(),
                                    "terminal event {} at the batch index {} is subject to cooldown, ignoring", i, j);
                                continue;
                            }
                        }
                    }

                    // Run the root finding.
                    const auto [root, cflag] = detail::bracketed_root_find(tmp1.v.data(), order, lb, ub);

                    if (cflag == 0) {
                        // Root finding finished successfully, record the event.
                        // The found root needs to be rescaled by h.
                        add_d_event(root * h);
                    } else {
                        // LCOV_EXCL_START
                        // Root finding encountered some issue. Ignore the
                        // event and log the issue.
                        if (cflag == -1) {
                            detail::get_logger()->warn(
                                "polynomial root finding during event detection failed due to too many "
                                "iterations at the batch index {}",
                                j);
                        } else {
                            detail::get_logger()->warn(
                                "polynomial root finding during event detection at the batch index {} "
                                "returned a nonzero errno with error code {}",
                                j, cflag);
                        }
                        // LCOV_EXCL_STOP
                    }
                }
            }
        }
    };

    run_detection(m_d_tes, m_tes);
    run_detection(m_d_ntes, m_ntes);
}

// Explicit instantiation of the book-keeping structures for event detection
// in the batch integrator.
template struct taylor_adaptive_batch<float>::ed_data;
template struct taylor_adaptive_batch<double>::ed_data;
template struct taylor_adaptive_batch<long double>::ed_data;

#if defined(HEYOKA_HAVE_REAL128)

template struct taylor_adaptive_batch<mppp::real128>::ed_data;

#endif

HEYOKA_END_NAMESPACE
