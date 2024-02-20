// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_S11N_HPP
#define HEYOKA_S11N_HPP

#include <optional>

// NOTE: workaround for a GCC bug when including the Boost.Serialization
// support for std::shared_ptr:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84075

#if defined(__GNUC__) && __GNUC__ >= 7

namespace boost::serialization
{

struct U {
};

} // namespace boost::serialization

#endif

#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>

// NOTE: we used to have polymorphic
// archives here instead, but apparently
// those do not support long double and thus
// lead to compilation errors when trying
// to (de)serialize numbers.
// NOTE: we also had text archives here, but
// they have issues with the infinite time
// values in the batch integrator.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#endif
