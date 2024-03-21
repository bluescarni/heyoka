#include <heyoka/config.hpp>

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/mdspan.hpp>
#include <heyoka/taylor.hpp>

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

} // namespace heyoka_test
