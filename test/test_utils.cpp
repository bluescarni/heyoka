#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <utility>

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

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
bool operator==<mppp::real>(const mppp::real &cmp, const approximately<mppp::real> &a)
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

#endif

} // namespace heyoka_test
