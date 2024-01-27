// Copyright 2020, 2021, 2022, 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <heyoka/config.hpp>
#include <heyoka/detail/dtens_impl.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

// Implementation of dtens_sv_idx_cmp::operator().
bool dtens_sv_idx_cmp_impl(const dtens_sv_idx_t &v1, const dtens_sv_idx_t &v2)
{
    // Sanity checks on the inputs.
    assert(sv_sanity_check(v1));
    assert(sv_sanity_check(v2));

    // Compute the total derivative order for both v1 and v2.
    // NOTE: here we have to use safe_numerics because this comparison operator
    // might end up being invoked on a user-supplied dtens_sv_idx_t, whose total degree
    // may overflow. The dtens_sv_idx_t in dtens, by contrast, are guaranteed to never
    // overflow when computing the total degree.
    using su32 = boost::safe_numerics::safe<std::uint32_t>;

    // The accumulator.
    auto acc = [](const auto &val, const auto &p) { return val + p.second; };

    const auto deg1 = std::accumulate(v1.second.begin(), v1.second.end(), su32(0), acc);
    const auto deg2 = std::accumulate(v2.second.begin(), v2.second.end(), su32(0), acc);

    if (deg1 < deg2) {
        return true;
    }

    if (deg1 > deg2) {
        return false;
    }

    // The total derivative order is the same, look at
    // the component index next.
    if (v1.first < v2.first) {
        return true;
    }

    if (v1.first > v2.first) {
        return false;
    }

    // Component and total derivative order are the same,
    // resort to reverse lexicographical compare on the
    // derivative orders.
    auto it1 = v1.second.begin(), it2 = v2.second.begin();
    const auto end1 = v1.second.end(), end2 = v2.second.end();
    for (; it1 != end1 && it2 != end2; ++it1, ++it2) {
        const auto [idx1, n1] = *it1;
        const auto [idx2, n2] = *it2;

        if (idx2 > idx1) {
            return true;
        }

        if (idx1 > idx2) {
            return false;
        }

        if (n1 > n2) {
            return true;
        }

        if (n2 > n1) {
            return false;
        }

        assert(std::equal(v1.second.begin(), it1 + 1, v2.second.begin()));
    }

    // NOTE: if we end up here, it means that:
    // - component and diff order are the same, and
    // - the two index/order lists share a common
    //   (possibly empty) initial sequence.
    // It follows then that both it1 and it2 must be
    // end iterators, because if either index/order list
    // had additional terms, then the diff orders
    // could not possibly be equal.
    assert(it1 == end1 && it2 == end2);
    assert(v1.second == v2.second);

    return false;
}

#if !defined(NDEBUG)

// Same comparison as the previous function, but in dense format.
// Used only for debug.
bool dtens_v_idx_cmp_impl(const dtens::v_idx_t &v1, const dtens::v_idx_t &v2)
{
    assert(v1.size() == v2.size());
    assert(!v1.empty());

    // Compute the total derivative order for both
    // vectors.
    boost::safe_numerics::safe<std::uint32_t> deg1 = 0, deg2 = 0;
    const auto size = v1.size();
    for (decltype(v1.size()) i = 1; i < size; ++i) {
        deg1 += v1[i];
        deg2 += v2[i];
    }

    if (deg1 < deg2) {
        return true;
    }

    if (deg1 > deg2) {
        return false;
    }

    // The total derivative order is the same, look at
    // the component index next.
    if (v1[0] < v2[0]) {
        return true;
    }

    if (v1[0] > v2[0]) {
        return false;
    }

    // Component and total derivative order are the same,
    // resort to reverse lexicographical compare on the
    // derivative orders.
    return std::lexicographical_compare(v1.begin() + 1, v1.end(), v2.begin() + 1, v2.end(), std::greater{});
}

#endif

} // namespace

bool dtens_sv_idx_cmp::operator()(const dtens_sv_idx_t &v1, const dtens_sv_idx_t &v2) const
{
    auto ret = dtens_sv_idx_cmp_impl(v1, v2);

#if !defined(NDEBUG)

    // Convert to dense and re-run the same comparison.
    auto to_dense = [](const dtens_sv_idx_t &v) {
        dtens::v_idx_t dv{v.first};

        std::uint32_t cur_d_idx = 0;
        for (auto it = v.second.begin(); it != v.second.end(); ++cur_d_idx) {
            if (cur_d_idx == it->first) {
                dv.push_back(it->second);
                ++it;
            } else {
                dv.push_back(0);
            }
        }

        return dv;
    }; // LCOV_EXCL_LINE

    auto dv1 = to_dense(v1);
    auto dv2 = to_dense(v2);
    dv1.resize(std::max(dv1.size(), dv2.size()));
    dv2.resize(std::max(dv1.size(), dv2.size()));

    assert(ret == dtens_v_idx_cmp_impl(dv1, dv2));

#endif

    return ret;
}

} // namespace detail

// Serialisation.
void dtens::impl::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    // NOTE: this is essentially a manual implementation of serialisation
    // for flat_map, which is currently missing. See:
    // https://stackoverflow.com/questions/69492511/boost-serialize-writing-a-general-map-serialization-function

    // Serialise the size.
    const auto size = m_map.size();
    ar << size;

    // Serialise the elements.
    for (const auto &p : m_map) {
        ar << p;
    }

    // Serialise m_args.
    ar << m_args;
}

// NOTE: as usual, we assume here that the archive contains
// a correctly-serialised instance. In particular, we are assuming
// that the elements in ar are sorted correctly.
void dtens::impl::load(boost::archive::binary_iarchive &ar, unsigned version)
{
    // LCOV_EXCL_START
    if (version < static_cast<unsigned>(boost::serialization::version<dtens>::type::value)) {
        throw std::invalid_argument(
            fmt::format("Unable to load a dtens object: the archive version ({}) is too old", version));
    }
    // LCOV_EXCL_STOP

    try {
        // Reset m_map.
        m_map.clear();

        // Read the size.
        size_type size = 0;
        ar >> size;

        // Reserve space.
        // NOTE: this is important as it ensures that
        // the addresses of the inserted elements
        // do not change as we insert more elements.
        m_map.reserve(size);

        // Read the elements.
        for (size_type i = 0; i < size; ++i) {
            detail::dtens_map_t::value_type tmp_val;
            ar >> tmp_val;
            const auto it = m_map.insert(m_map.end(), std::move(tmp_val));
            assert(it == m_map.end() - 1);

            // Reset the object address.
            // NOLINTNEXTLINE(bugprone-use-after-move,hicpp-invalid-access-moved)
            ar.reset_object_address(std::addressof(*it), &tmp_val);
        }

        assert(m_map.size() == size);

        // Deserialise m_args.
        ar >> m_args;

        // LCOV_EXCL_START
    } catch (...) {
        *this = impl{};
        throw;
    }
    // LCOV_EXCL_STOP
}

dtens::dtens(impl x) : p_impl(std::make_unique<impl>(std::move(x))) {}

dtens::dtens() : dtens(impl{}) {}

dtens::dtens(const dtens &other) : dtens(*other.p_impl) {}

dtens::dtens(dtens &&) noexcept = default;

dtens &dtens::operator=(const dtens &other)
{
    if (&other != this) {
        *this = dtens(other);
    }

    return *this;
}

dtens &dtens::operator=(dtens &&) noexcept = default;

dtens::~dtens() = default;

dtens::iterator dtens::begin() const
{
    return p_impl->m_map.begin();
}

dtens::iterator dtens::end() const
{
    return p_impl->m_map.end();
}

std::uint32_t dtens::get_order() const
{
    // First we handle the empty case.
    if (p_impl->m_map.empty()) {
        return 0;
    }

    // We can fetch the total derivative
    // order from the last derivative
    // in the map (specifically, it is
    // the last element in the indices
    // vector of the last derivative).
    const auto &sv = (end() - 1)->first.second;
    if (sv.empty()) {
        // NOTE: an empty index/order vector
        // at the end means that the maximum
        // diff order is zero and that we are
        // only storing the original function
        // components in the dtens object.

#if !defined(NDEBUG)

        for (const auto &[v, _] : *this) {
            assert(v.first == 0u);
            assert(v.second.empty());
        }

#endif

        return 0;
    } else {
        assert(sv.size() == 1u);

        return sv.back().second;
    }
}

dtens::iterator dtens::find(const v_idx_t &vidx) const
{
    // NOTE: run sanity checks on vidx. If the checks fail,
    // return end().

    // vidx must at least contain the function component index.
    if (vidx.empty()) {
        return end();
    }

    // The size of vidx must be consistent with the number
    // of diff args.
    if (vidx.size() - 1u != get_nargs()) {
        return end();
    }

    // Turn vidx into sparse format.
    detail::dtens_sv_idx_t s_vidx{vidx[0], {}};
    for (decltype(vidx.size()) i = 1; i < vidx.size(); ++i) {
        if (vidx[i] != 0u) {
            s_vidx.second.emplace_back(boost::numeric_cast<std::uint32_t>(i - 1u), vidx[i]);
        }
    }

    // Lookup.
    return p_impl->m_map.find(s_vidx);
}

dtens::iterator dtens::find(const sv_idx_t &vidx) const
{
    // NOTE: run sanity checks on vidx. If the checks fail,
    // return end().
    if (!detail::sv_sanity_check(vidx)) {
        return end();
    }

    return p_impl->m_map.find(vidx);
}

template <typename V>
const expression &dtens::index_impl(const V &vidx) const
{
    const auto it = find(vidx);

    if (it == end()) {
        throw std::out_of_range(
            fmt::format("Cannot locate the derivative corresponding to the indices vector {}", vidx));
    }

    return it->second;
}

const expression &dtens::operator[](const v_idx_t &vidx) const
{
    return index_impl(vidx);
}

const expression &dtens::operator[](const sv_idx_t &vidx) const
{
    return index_impl(vidx);
}

dtens::size_type dtens::index_of(const v_idx_t &vidx) const
{
    return index_of(find(vidx));
}

dtens::size_type dtens::index_of(const sv_idx_t &vidx) const
{
    return index_of(find(vidx));
}

dtens::size_type dtens::index_of(const iterator &it) const
{
    return p_impl->m_map.index_of(it);
}

// Get a range containing all derivatives of the given order for all components.
auto dtens::get_derivatives(std::uint32_t order) const -> decltype(std::ranges::subrange(begin(), end()))
{
    // First we handle the empty case. This will return
    // an empty range.
    if (p_impl->m_map.empty()) {
        return {begin(), end()};
    }

    // Create the indices vector corresponding to the first derivative
    // of component 0 for the given order in the map.
    detail::dtens_sv_idx_t s_vidx{0, {}};
    if (order != 0u) {
        s_vidx.second.emplace_back(0, order);
    }

    // Locate the corresponding derivative in the map.
    // NOTE: this could be end() for invalid order.
    const auto b = p_impl->m_map.find(s_vidx);

#if !defined(NDEBUG)

    if (order <= get_order()) {
        assert(b != end());
    } else {
        assert(b == end());
    }

#endif

    // Modify s_vidx so that it now refers to the last derivative
    // for the last component at the given order in the map.
    // NOTE: get_nouts() can return zero only if the internal
    // map is empty, and we handled this corner case earlier.
    assert(get_nouts() > 0u);
    s_vidx.first = get_nouts() - 1u;
    if (order != 0u) {
        assert(get_nargs() > 0u);
        s_vidx.second[0].first = get_nargs() - 1u;
    }

    // NOTE: this could be end() for invalid order.
    auto e = p_impl->m_map.find(s_vidx);

#if !defined(NDEBUG)

    if (order <= get_order()) {
        assert(e != end());
    } else {
        assert(e == end());
    }

#endif

    // Need to move 1 past, if possible,
    // to produce a half-open range.
    if (e != end()) {
        ++e;
    }

    return {b, e};
}

// Get a range containing all derivatives of the given order for a component.
auto dtens::get_derivatives(std::uint32_t component, std::uint32_t order) const
    -> decltype(std::ranges::subrange(begin(), end()))
{
    // First we handle the empty case. This will return
    // an empty range.
    if (p_impl->m_map.empty()) {
        return {begin(), end()};
    }

    // Create the indices vector corresponding to the first derivative
    // for the given order and component in the map.
    detail::dtens_sv_idx_t s_vidx{component, {}};
    if (order != 0u) {
        s_vidx.second.emplace_back(0, order);
    }

    // Locate the corresponding derivative in the map.
    // NOTE: this could be end() for invalid component/order.
    const auto b = p_impl->m_map.find(s_vidx);

#if !defined(NDEBUG)

    if (component < get_nouts() && order <= get_order()) {
        assert(b != end());
    } else {
        assert(b == end());
    }

#endif

    // Modify vidx so that it now refers to the last derivative
    // for the given order and component in the map.
    assert(get_nargs() > 0u);
    if (order != 0u) {
        s_vidx.second[0].first = get_nargs() - 1u;
    }

    // NOTE: this could be end() for invalid component/order.
    auto e = p_impl->m_map.find(s_vidx);

#if !defined(NDEBUG)

    if (component < get_nouts() && order <= get_order()) {
        assert(e != end());
    } else {
        assert(e == end());
    }

#endif

    // Need to move 1 past, if possible,
    // to produce a half-open range.
    if (e != end()) {
        ++e;
    }

    return {b, e};
}

std::vector<expression> dtens::get_gradient() const
{
    if (get_nouts() != 1u) {
        throw std::invalid_argument(fmt::format("The gradient can be requested only for a function with a single "
                                                "output, but the number of outputs is instead {}",
                                                get_nouts()));
    }

    if (get_order() == 0u) {
        throw std::invalid_argument("First-order derivatives are not available");
    }

    const auto sr = get_derivatives(0, 1);
    std::vector<expression> retval;
    retval.reserve(get_nargs());
    std::transform(sr.begin(), sr.end(), std::back_inserter(retval), [](const auto &p) { return p.second; });

    assert(retval.size() == get_nargs());

    return retval;
}

std::vector<expression> dtens::get_jacobian() const
{
    if (get_nouts() == 0u) {
        throw std::invalid_argument("Cannot return the Jacobian of a function with no outputs");
    }

    if (get_order() == 0u) {
        throw std::invalid_argument("First-order derivatives are not available");
    }

    const auto sr = get_derivatives(1);
    std::vector<expression> retval;
    retval.reserve(boost::safe_numerics::safe<decltype(retval.size())>(get_nargs()) * get_nouts());
    std::transform(sr.begin(), sr.end(), std::back_inserter(retval), [](const auto &p) { return p.second; });

    assert(retval.size() == boost::safe_numerics::safe<decltype(retval.size())>(get_nargs()) * get_nouts());

    return retval;
}

std::uint32_t dtens::get_nargs() const
{
    // NOTE: we ensure in the diff_tensors() implementation
    // that the number of diff variables is representable
    // by std::uint32_t.
    auto ret = static_cast<std::uint32_t>(get_args().size());

#if !defined(NDEBUG)

    if (p_impl->m_map.empty()) {
        assert(ret == 0u);
    }

#endif

    return ret;
}

namespace detail
{

namespace
{

// The indices vector corresponding
// to the first derivative of order 1
// of the first component.
// NOLINTNEXTLINE(cert-err58-cpp)
const dtens_sv_idx_t s_vidx_001{0, {{0, 1}}};

} // namespace

} // namespace detail

std::uint32_t dtens::get_nouts() const
{
    if (p_impl->m_map.empty()) {
        return 0;
    }

    // Try to find in the map the indices vector corresponding
    // to the first derivative of order 1 of the first component.
    const auto it = p_impl->m_map.find(detail::s_vidx_001);

    // NOTE: the number of outputs is always representable by
    // std::uint32_t, otherwise we could not index the function
    // components via std::uint32_t.
    if (it == end()) {
        // There are no derivatives in the map, which
        // means that the order must be zero and that the
        // size of the map gives directly the number of components.
        assert(get_order() == 0u);
        return static_cast<std::uint32_t>(p_impl->m_map.size());
    } else {
        assert(get_order() > 0u);
        return static_cast<std::uint32_t>(p_impl->m_map.index_of(it));
    }
}

dtens::size_type dtens::size() const
{
    return p_impl->m_map.size();
}

const std::vector<expression> &dtens::get_args() const
{
    return p_impl->m_args;
}

void dtens::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << p_impl;
}

void dtens::load(boost::archive::binary_iarchive &ar, unsigned)
{
    try {
        ar >> p_impl;
        // LCOV_EXCL_START
    } catch (...) {
        *this = dtens{};
        throw;
    }
    // LCOV_EXCL_STOP
}

std::ostream &operator<<(std::ostream &os, const dtens &dt)
{
    os << "Highest diff order: " << dt.get_order() << '\n';
    os << "Number of outputs : " << dt.get_nouts() << '\n';
    os << "Diff arguments    : " << fmt::format("{}", dt.get_args()) << '\n';

    return os;
}

HEYOKA_END_NAMESPACE
