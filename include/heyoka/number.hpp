// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_NUMBER_HPP
#define HEYOKA_NUMBER_HPP

#include <heyoka/config.hpp>

#include <cstddef>
#include <functional>
#include <ostream>
#include <variant>

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/llvm_fwd.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

HEYOKA_DLL_PUBLIC void swap(number &, number &) noexcept;

class HEYOKA_DLL_PUBLIC number
{
    friend HEYOKA_DLL_PUBLIC void swap(number &, number &) noexcept;

public:
    using value_type = std::variant<float, double, long double
#if defined(HEYOKA_HAVE_REAL128)
                                    ,
                                    mppp::real128
#endif
#if defined(HEYOKA_HAVE_REAL)
                                    ,
                                    mppp::real
#endif
                                    >;

private:
    value_type m_value;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    number() noexcept;
    explicit number(float) noexcept;
    explicit number(double) noexcept;
    explicit number(long double) noexcept;
#if defined(HEYOKA_HAVE_REAL128)
    explicit number(mppp::real128) noexcept;
#endif
#if defined(HEYOKA_HAVE_REAL)
    explicit number(mppp::real);
#endif
    number(const number &);
    number(number &&) noexcept;
    ~number();

    number &operator=(const number &);
    number &operator=(number &&) noexcept;

    [[nodiscard]] const value_type &value() const noexcept;
};

namespace detail
{

HEYOKA_DLL_PUBLIC std::size_t hash(const number &) noexcept;

} // namespace detail

HEYOKA_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const number &);

HEYOKA_DLL_PUBLIC bool is_zero(const number &);
HEYOKA_DLL_PUBLIC bool is_one(const number &);
HEYOKA_DLL_PUBLIC bool is_negative_one(const number &);
HEYOKA_DLL_PUBLIC bool is_negative(const number &);
HEYOKA_DLL_PUBLIC bool is_integer(const number &);

HEYOKA_DLL_PUBLIC number operator+(number);
HEYOKA_DLL_PUBLIC number operator-(const number &);

HEYOKA_DLL_PUBLIC number operator+(const number &, const number &);
HEYOKA_DLL_PUBLIC number operator-(const number &, const number &);
HEYOKA_DLL_PUBLIC number operator*(const number &, const number &);
HEYOKA_DLL_PUBLIC number operator/(const number &, const number &);

HEYOKA_DLL_PUBLIC bool operator==(const number &, const number &) noexcept;
HEYOKA_DLL_PUBLIC bool operator!=(const number &, const number &) noexcept;
HEYOKA_DLL_PUBLIC bool operator<(const number &, const number &) noexcept;

HEYOKA_DLL_PUBLIC number exp(const number &);
HEYOKA_DLL_PUBLIC number binomial(const number &, const number &);
HEYOKA_DLL_PUBLIC number nextafter(const number &, const number &);
HEYOKA_DLL_PUBLIC number sqrt(const number &);

HEYOKA_DLL_PUBLIC llvm::Value *llvm_codegen(llvm_state &, llvm::Type *, const number &);

namespace detail
{

HEYOKA_DLL_PUBLIC number number_like(llvm_state &, llvm::Type *, double);

} // namespace detail

HEYOKA_END_NAMESPACE

namespace std
{

template <>
struct hash<heyoka::number> {
    size_t operator()(const heyoka::number &n) const noexcept
    {
        return heyoka::detail::hash(n);
    }
};

} // namespace std

#endif
