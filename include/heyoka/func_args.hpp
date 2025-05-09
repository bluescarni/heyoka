// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_FUNC_ARGS_HPP
#define HEYOKA_FUNC_ARGS_HPP

#include <memory>
#include <variant>
#include <vector>

#include <heyoka/config.hpp>
#include <heyoka/detail/fwd_decl.hpp>
#include <heyoka/detail/visibility.hpp>
#include <heyoka/s11n.hpp>

HEYOKA_BEGIN_NAMESPACE

class HEYOKA_DLL_PUBLIC func_args
{
public:
    using shared_args_t = std::shared_ptr<const std::vector<expression>>;

private:
    std::variant<std::vector<expression>, shared_args_t> m_args;

    // Serialization.
    friend class boost::serialization::access;
    void save(boost::archive::binary_oarchive &, unsigned) const;
    void load(boost::archive::binary_iarchive &, unsigned);
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    func_args();
    explicit func_args(std::vector<expression>, bool = false);
    explicit func_args(shared_args_t);
    func_args(const func_args &);
    func_args(func_args &&) noexcept;
    func_args &operator=(const func_args &);
    func_args &operator=(func_args &&) noexcept;
    ~func_args();

    [[nodiscard]] const std::vector<expression> &get_args() const noexcept;
    [[nodiscard]] shared_args_t get_shared_args() const noexcept;
};

HEYOKA_END_NAMESPACE

#endif
