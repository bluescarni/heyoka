// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_POLY_ROOT_FINDING_VECTOR_CACHE_HPP
#define HEYOKA_DETAIL_POLY_ROOT_FINDING_VECTOR_CACHE_HPP

#if !defined(HEYOKA_BUILD_LIBRARY)

#error This header can be included only when building heyoka.

#endif

#include <span>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

// Fwd declaration.
template <typename T>
class cached_vector;

// Cache of vectors.
//
// This opaque class internally manages a cache of vectors. It is used as a data source by cached_vector.
//
// The cache is guaranteed to contain vectors of the same size, specified on construction. When new vectors need to be
// created, they are filled with a fill value also specified on construction.
template <typename T>
class vector_cache
{
    friend class cached_vector<T>;

public:
    using width_type = std::vector<T>::size_type;

private:
    T m_fill_value{};
    width_type m_width = 0;
    std::vector<std::vector<T>> m_cache;

    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    vector_cache();
    explicit vector_cache(T, width_type);
    vector_cache(const vector_cache &);
    vector_cache(vector_cache &&) noexcept = delete;
    vector_cache &operator=(const vector_cache &) = delete;
    vector_cache &operator=(vector_cache &&) noexcept = delete;
    ~vector_cache();
};

// Cached vector.
//
// This class stores internally a std::vector<T> instance which is managed by a vector_cache<T>. That is, on
// construction the std::vector<T> instance is popped from the cache (if possible, otherwise the vector is created
// ex-novo), whereas on destruction the std::vector<T> instance is returned to the cache.
//
// Cached vectors are required to never outlive the vector_cache instance they were created from. Failing to do so will
// result in undefined behaviour.
template <typename T>
class cached_vector
{
    vector_cache<T> *m_vc = nullptr;
    std::vector<T> m_vector;

public:
    explicit cached_vector(vector_cache<T> &);
    cached_vector(const cached_vector &) = delete;
    cached_vector(cached_vector &&) noexcept;
    cached_vector &operator=(const cached_vector &) = delete;
    cached_vector &operator=(cached_vector &&) noexcept = delete;
    ~cached_vector();

    // Data access.
    [[nodiscard]] std::span<const T> get_span() const;
    [[nodiscard]] std::span<T> get_span();
};

} // namespace detail

HEYOKA_END_NAMESPACE

#endif
