// Copyright 2020, 2021, 2022, 2023, 2024 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include <heyoka/config.hpp>
#include <heyoka/detail/get_dl_path.hpp>

#if __has_include(<dlfcn.h>)

#define HEYOKA_DETAIL_GET_DL_PATH_DLFCN

#include <dlfcn.h>

#endif

HEYOKA_BEGIN_NAMESPACE

namespace detail
{

namespace
{

#if defined(HEYOKA_DETAIL_GET_DL_PATH_DLFCN)

// NOTE: need a dummy variable to take the address of.
const int dummy = 42;

// Implementation of make_dl_path() based on dladdr(), from the dlfcn.h header.
std::string make_dl_path_dlfcn()
{
    // Invoke dladdr().
    ::Dl_info dl_info;
    const auto ret = ::dladdr(static_cast<const void *>(&dummy), &dl_info);

    // NOTE: in case of any failure, we will fall through the
    // "return {};" statement and produce an empty string.
    if (ret != 0 && dl_info.dli_fname != nullptr) {
        try {
            return std::filesystem::canonical(std::filesystem::path(dl_info.dli_fname)).string();
            // LCOV_EXCL_START
        } catch (const std::exception &e) {
            std::cerr << "WARNING - exception raised while trying to determine the path of the heyoka library: "
                      << e.what() << std::endl;
        } catch (...) {
            std::cerr << "WARNING - exception raised while trying to determine the path of the heyoka library"
                      << std::endl;
        }
    }

    return {};
    // LCOV_EXCL_STOP
}

#endif

// Factory function for dl_path.
std::string make_dl_path()
{
#if defined(HEYOKA_DETAIL_GET_DL_PATH_DLFCN)

    return make_dl_path_dlfcn();

#else

    return {};

#endif
}

// The path to the heyoka shared library.
const std::string dl_path = make_dl_path();

} // namespace

// This function is meant to return the full canonicalised path to the heyoka shared library.
//
// If, for any reason, the path cannot be determined, an empty
// string will be returned instead.
const std::string &get_dl_path()
{
    return detail::dl_path;
}

} // namespace detail

HEYOKA_END_NAMESPACE
