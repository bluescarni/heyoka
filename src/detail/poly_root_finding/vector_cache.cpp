// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cassert>
#include <span>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/poly_root_finding/vector_cache.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

template <typename T>
void vector_cache<T>::save(boost::archive::binary_oarchive &ar, unsigned) const
{
    ar << m_fill_value;
    ar << m_width;
    // NOTE: we do *not* save the cache content on purpose.
}

// NOTE: deserialisation will produce an empty cache, consistently with the copy constructor.
template <typename T>
void vector_cache<T>::load(boost::archive::binary_iarchive &ar, unsigned)
{
    // NOTE: loading into an existing cache may change its width and fill value. Thus, any cached_vector referring to
    // the pre-des11n cache must be destroyed before deserialising the cache.

    // NOTE: make sure to drop the current content.
    m_cache.clear();

    // Restore fill value and width.
    ar >> m_fill_value;
    ar >> m_width;
}

// The default constructor initialises a cache for zero-sized vectors, in which the fill value is a value-initialised
// instance of T.
template <typename T>
vector_cache<T>::vector_cache() = default;

// Main constructor.
//
// It will initialise a cache for vectors of size w with the provided fill_value.
template <typename T>
vector_cache<T>::vector_cache(T fill_value, const width_type w) : m_fill_value(std::move(fill_value)), m_width(w)
{
}

// Copy constructor.
//
// The copy constructor will copy the fill value and the width, but not the content: the new cache will start out empty.
template <typename T>
vector_cache<T>::vector_cache(const vector_cache &other) : m_fill_value(other.m_fill_value), m_width(other.m_width)
{
}

template <typename T>
vector_cache<T>::~vector_cache() = default;

// Explicit instantiations.
template class vector_cache<float>;
template class vector_cache<double>;
template class vector_cache<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class vector_cache<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class vector_cache<mppp::real>;

#endif

namespace
{

// Helper to implement the main ctor of cached_vector.
//
// It will either fetch and return a vector from the input cache, or initialise and return a new vector of the specified
// size with the provided fill value.
template <typename T>
std::vector<T> cached_vector_init_from_cache(const T &fill_value, std::vector<std::vector<T>> &c,
                                             const typename std::vector<T>::size_type width)
{
    if (c.empty()) {
        return std::vector<T>(width, fill_value);
    } else {
        auto ret = std::move(c.back());
        assert(ret.size() == width);
        c.pop_back();
        return ret;
    }
}

} // namespace

// Main constructor.
//
// This will create a cached_vector from the input vector_cache. The size of the cached vector is determined by the
// input vector_cache. If a new vector needs to be created because the cache is empty, the cached_vector will be filled
// with the fill value of the cache.
template <typename T>
cached_vector<T>::cached_vector(vector_cache<T> &c)
    : m_vc(&c), m_vector(cached_vector_init_from_cache(c.m_fill_value, c.m_cache, c.m_width))
{
}

// Move ctor.
//
// After a move, the other cached vector is left in a state which supports only destruction.
template <typename T>
cached_vector<T>::cached_vector(cached_vector &&other) noexcept : m_vc(other.m_vc), m_vector(std::move(other.m_vector))
{
    // NOTE: make sure to flag the other object into the moved-from state.
    //
    // NOTE: the state of the vector does not matter (it will most likely be empty).
    other.m_vc = nullptr;
}

// Dtor.
//
// The destructor will return the internal vector to the cache.
template <typename T>
cached_vector<T>::~cached_vector()
{
    // If this was not moved-from, return the vector to the cache.
    if (m_vc != nullptr) {
        // NOTE: push_back() can in principle throw, ignore the exceptions in such a case. This is still correct, the
        // only pessimisation is that the vector won't be returned to the cache.
        try {
            m_vc->m_cache.push_back(std::move(m_vector));
            // LCOV_EXCL_START
        } catch (...) {
            ;
        }
        // LCOV_EXCL_STOP
    }
}

// Helpers to get const/mutable access to the internal vector data.
template <typename T>
std::span<const T> cached_vector<T>::get_span() const
{
    assert(m_vc != nullptr);
    assert(m_vector.size() == m_vc->m_width);

    const auto span_size = boost::numeric_cast<typename std::span<const T>::size_type>(m_vector.size());

    return std::span<const T>(m_vector.data(), span_size);
}

template <typename T>
std::span<T> cached_vector<T>::get_span()
{
    assert(m_vc != nullptr);
    assert(m_vector.size() == m_vc->m_width);

    const auto span_size = boost::numeric_cast<typename std::span<T>::size_type>(m_vector.size());

    return std::span<T>(m_vector.data(), span_size);
}

// Explicit instantiations.
template class cached_vector<float>;
template class cached_vector<double>;
template class cached_vector<long double>;

#if defined(HEYOKA_HAVE_REAL128)

template class cached_vector<mppp::real128>;

#endif

#if defined(HEYOKA_HAVE_REAL)

template class cached_vector<mppp::real>;

#endif

} // namespace detail

HEYOKA_END_NAMESPACE
