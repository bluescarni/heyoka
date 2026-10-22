#include <heyoka/config.hpp>

#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/mdspan.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

namespace heyoka_test
{

#if defined(HEYOKA_HAVE_REAL)

const mppp::real approximately<mppp::real>::default_tol(100);

approximately<mppp::real>::approximately(mppp::real x, mppp::real eps_mul)
    : m_value(std::move(x)), m_eps_mul(std::move(eps_mul), m_value.get_prec())
{
}

template <>
bool operator== <mppp::real>(const mppp::real &cmp, const approximately<mppp::real> &a)
{
    using std::abs;

    assert(a.m_value.get_prec() == cmp.get_prec());

    // NOTE: for consistency with the epsilons returned for the other
    // types, we return here 2**-(prec - 1). See:
    // https://en.wikipedia.org/wiki/Machine_epsilon
    const auto tol = mppp::real{1ul, -(a.m_value.get_prec() - 1), a.m_value.get_prec()} * a.m_eps_mul;

    if (abs(cmp) < tol) {
        return abs(cmp - a.m_value) <= tol;
    } else {
        return abs((cmp - a.m_value) / cmp) <= tol;
    }
}

template <>
std::ostream &operator<< <mppp::real>(std::ostream &os, const approximately<mppp::real> &a)
{
    return os << a.m_value.to_string();
}

#endif

template <typename T>
std::vector<T> tc_to_jet(const heyoka::taylor_adaptive<T> &ta)
{
    const auto dim = ta.get_dim();
    const auto order = ta.get_order();

    std::vector<T> retval;
    retval.reserve((order + 1u) * dim);

    using tc_span_t = heyoka::mdspan<const T, heyoka::dextents<unsigned, 2>>;
    tc_span_t tc_span(ta.get_tc().data(), dim, order + 1u);

    for (auto j = 0u; j < order + 1u; ++j) {
        for (auto i = 0u; i < dim; ++i) {
            retval.push_back(tc_span(i, j));
        }
    }

    return retval;
}

template <typename T>
std::vector<T> tc_to_jet(const heyoka::taylor_adaptive_batch<T> &ta)
{
    const auto batch_size = ta.get_batch_size();
    const auto dim = ta.get_dim();
    const auto order = ta.get_order();

    std::vector<T> retval;
    retval.reserve((order + 1u) * dim * batch_size);

    using tc_span_t = heyoka::mdspan<const T, heyoka::dextents<unsigned, 3>>;
    tc_span_t tc_span(ta.get_tc().data(), dim, order + 1u, batch_size);

    for (auto j = 0u; j < order + 1u; ++j) {
        for (auto i = 0u; i < dim; ++i) {
            for (auto k = 0u; k < batch_size; ++k) {
                retval.push_back(tc_span(i, j, k));
            }
        }
    }

    return retval;
}

template std::vector<float> tc_to_jet(const heyoka::taylor_adaptive<float> &);
template std::vector<double> tc_to_jet(const heyoka::taylor_adaptive<double> &);
template std::vector<long double> tc_to_jet(const heyoka::taylor_adaptive<long double> &);

template std::vector<float> tc_to_jet(const heyoka::taylor_adaptive_batch<float> &);
template std::vector<double> tc_to_jet(const heyoka::taylor_adaptive_batch<double> &);
template std::vector<long double> tc_to_jet(const heyoka::taylor_adaptive_batch<long double> &);

#if defined(HEYOKA_HAVE_REAL128)

template std::vector<mppp::real128> tc_to_jet(const heyoka::taylor_adaptive<mppp::real128> &);
template std::vector<mppp::real128> tc_to_jet(const heyoka::taylor_adaptive_batch<mppp::real128> &);

#endif

#if defined(HEYOKA_HAVE_REAL)

template std::vector<mppp::real> tc_to_jet(const heyoka::taylor_adaptive<mppp::real> &);

#endif

template <typename T>
void compare_batch_scalar(const std::vector<std::pair<heyoka::expression, heyoka::expression>> &sys, unsigned opt_level,
                          bool high_accuracy, bool compact_mode, std::mt19937 &rng, float lb, float ub, T tol)
{
    namespace kw = heyoka::kw;

    const auto dim = sys.size();

    std::uniform_real_distribution<float> dist(lb, ub);

    for (auto batch_size : {2u, 4u, 8u, 5u}) {
        // Randomly-generate the batch initial state.
        std::vector<T> orig_batch_state(batch_size * dim);
        std::ranges::generate(orig_batch_state, [&dist, &rng]() { return T{dist(rng)}; });

        // Randomly-generate the batch initial time.
        std::vector<T> orig_batch_time(batch_size);
        std::ranges::generate(orig_batch_time, [&dist, &rng]() { return T{dist(rng)}; });

        auto ta = heyoka::taylor_adaptive<T>{sys,
                                             std::vector<T>(dim),
                                             kw::tol = .1,
                                             kw::high_accuracy = high_accuracy,
                                             kw::compact_mode = compact_mode,
                                             kw::opt_level = opt_level};

        auto ta_batch = heyoka::taylor_adaptive_batch<T>{sys,
                                                         orig_batch_state,
                                                         batch_size,
                                                         kw::tol = .1,
                                                         kw::high_accuracy = high_accuracy,
                                                         kw::compact_mode = compact_mode,
                                                         kw::opt_level = opt_level,
                                                         kw::time = orig_batch_time};

        // Take the batch step.
        ta_batch.step(true);

        // Fetch the Taylor coefficients.
        const auto jet_batch = tc_to_jet(ta_batch);

        for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
            for (auto i = 0ul; i < dim; ++i) {
                ta.get_state_data()[i] = orig_batch_state[i * batch_size + batch_idx];
            }

            ta.set_time(orig_batch_time[batch_idx]);

            ta.step(true);

            const auto jet_scalar = tc_to_jet(ta);

            for (auto i = 0ul; i < dim * 4ul; ++i) {
                REQUIRE(jet_scalar[i] == approximately(jet_batch[i * batch_size + batch_idx], tol));
            }
        }
    }
}

template void compare_batch_scalar<float>(const std::vector<std::pair<heyoka::expression, heyoka::expression>> &,
                                          unsigned, bool, bool, std::mt19937 &, float, float, float);
template void compare_batch_scalar<double>(const std::vector<std::pair<heyoka::expression, heyoka::expression>> &,
                                           unsigned, bool, bool, std::mt19937 &, float, float, double);
template void compare_batch_scalar<long double>(const std::vector<std::pair<heyoka::expression, heyoka::expression>> &,
                                                unsigned, bool, bool, std::mt19937 &, float, float, long double);

#if defined(HEYOKA_HAVE_REAL128)

template void
compare_batch_scalar<mppp::real128>(const std::vector<std::pair<heyoka::expression, heyoka::expression>> &, unsigned,
                                    bool, bool, std::mt19937 &, float, float, mppp::real128);

#endif

// Convert the finite value x of type T exactly to a rational.
//
// x is split into a sum of doubles, each of which is converted exactly. Each subtraction is exact by Sterbenz's lemma,
// as d (x rounded to double) is within a factor 2 of x. This requires x and the remainders to be within the normal
// range of double.
template <typename T>
boost::multiprecision::cpp_rational to_rat(T x)
{
    using rat = boost::multiprecision::cpp_rational;

    rat ret = 0;

    while (x != 0) {
        const auto d = static_cast<double>(x);
        ret += rat(d);
        x -= static_cast<T>(d);
    }

    return ret;
}

template boost::multiprecision::cpp_rational to_rat<float>(float x);
template boost::multiprecision::cpp_rational to_rat<double>(double x);
template boost::multiprecision::cpp_rational to_rat<long double>(long double x);

#if defined(HEYOKA_HAVE_REAL128)

template boost::multiprecision::cpp_rational to_rat<mppp::real128>(mppp::real128 x);

#endif

#if defined(HEYOKA_HAVE_REAL)

template boost::multiprecision::cpp_rational to_rat<mppp::real>(mppp::real x);

#endif

namespace
{

// Add a and b.
template <typename T>
T test_add(T a, T b)
{
    return a + b;
}

// Multiply a and b.
template <typename T>
T test_mul(T a, T b)
{
    return a * b;
}

} // namespace

// Check whether addition in the type T honours the rounding mode set via std::fesetround(). Software-emulated
// quadruple-precision arithmetic may not honour it.
template <typename T>
bool add_honours_rounding()
{
    // NOTE: call through a volatile function pointer, so that the compiler can neither evaluate the addition at compile
    // time nor move it across the changes of rounding mode.
    T (*volatile f)(T, T) = &test_add<T>;

    const T eps = std::numeric_limits<T>::epsilon();

    REQUIRE(std::fesetround(FE_UPWARD) == 0);
    const T res = f(T(1), eps / 4);
    REQUIRE(std::fesetround(FE_TONEAREST) == 0);

    // The exact sum is 1 + eps / 4: rounding up gives 1 + eps, rounding to nearest 1.
    return res > 1;
}

// Check whether multiplication in the type T honours the rounding mode set via std::fesetround(). This is not the case
// for the software-emulated quadruple precision of LLVM's compiler-rt (used, e.g., on FreeBSD), whose multiplication
// always rounds to nearest.
template <typename T>
bool mul_honours_rounding()
{
    // NOTE: call through a volatile function pointer, so that the compiler can neither evaluate the multiplication at
    // compile time nor move it across the changes of rounding mode.
    T (*volatile f)(T, T) = &test_mul<T>;

    const T eps = std::numeric_limits<T>::epsilon();
    const T x = 1 + eps;

    REQUIRE(std::fesetround(FE_UPWARD) == 0);
    const T res = f(x, x);
    REQUIRE(std::fesetround(FE_TONEAREST) == 0);

    // The exact product is 1 + 2 * eps + eps * eps: rounding up gives 1 + 3 * eps, rounding to nearest 1 + 2 * eps.
    return res > 1 + 2 * eps;
}

// Generate a random value of type T approximately uniformly distributed in [lo, hi]. The value is the sum of two
// doubles, thus for the types wider than double it can have up to about 106 significant bits.
template <typename T>
T rand_uniform(std::mt19937 &rng, double lo, double hi)
{
    std::uniform_real_distribution<double> dist(lo, hi);

    // NOTE: the second double is scaled by 2**-53, so that its bits extend those of the first one.
    return static_cast<T>(dist(rng)) + static_cast<T>(dist(rng)) * static_cast<T>(0x1p-53);
}

template bool add_honours_rounding<float>();
template bool add_honours_rounding<double>();
template bool add_honours_rounding<long double>();

template bool mul_honours_rounding<float>();
template bool mul_honours_rounding<double>();
template bool mul_honours_rounding<long double>();

template float rand_uniform<float>(std::mt19937 &, double, double);
template double rand_uniform<double>(std::mt19937 &, double, double);
template long double rand_uniform<long double>(std::mt19937 &, double, double);

#if defined(HEYOKA_HAVE_REAL128)

template bool add_honours_rounding<mppp::real128>();
template bool mul_honours_rounding<mppp::real128>();
template mppp::real128 rand_uniform<mppp::real128>(std::mt19937 &, double, double);

#endif

// The largest finite value of the type T, as a rational.
//
// NOTE: the largest finite value is (2**p - 1) * 2**(e - p), where p is the number of significand bits and e is the
// maximum exponent, as defined by std::numeric_limits.
template <typename T>
boost::multiprecision::cpp_rational max_rat()
{
    using boost::multiprecision::cpp_int;

    constexpr auto p = static_cast<unsigned>(std::numeric_limits<T>::digits);
    constexpr auto e = static_cast<unsigned>(std::numeric_limits<T>::max_exponent);

    return boost::multiprecision::cpp_rational(((cpp_int(1) << p) - 1) << (e - p));
}

template boost::multiprecision::cpp_rational max_rat<float>();
template boost::multiprecision::cpp_rational max_rat<double>();
template boost::multiprecision::cpp_rational max_rat<long double>();

#if defined(HEYOKA_HAVE_REAL128)

template boost::multiprecision::cpp_rational max_rat<mppp::real128>();

#endif

} // namespace heyoka_test
