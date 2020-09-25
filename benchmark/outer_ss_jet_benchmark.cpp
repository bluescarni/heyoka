// Copyright 2020 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <heyoka/llvm_state.hpp>
#include <heyoka/nbody.hpp>
#include <heyoka/taylor.hpp>

using namespace heyoka;

int main(int argc, char *argv[])
{
    auto batch_size = 1u;

    if (argc > 1) {
        auto bs = std::stoi(argv[1]);
        if (bs <= 0) {
            throw std::invalid_argument("The batch size must be positive, but it is " + std::string(argv[1])
                                        + " instead");
        }
        batch_size = static_cast<unsigned>(bs);
    }

    auto masses = std::vector{1.00000597682, 1. / 1047.355, 1. / 3501.6, 1. / 22869., 1. / 19314., 7.4074074e-09};

    const auto G = 0.01720209895 * 0.01720209895;

    auto sys = make_nbody_sys(6, kw::masses = masses, kw::Gconst = G);

    llvm_state s;

    const auto order = 20u;

    auto start = std::chrono::high_resolution_clock::now();

    taylor_add_jet<double>(s, "jet", std::move(sys), order, batch_size, false);

    // std::cout << s.get_ir() << '\n';
    // s.dump_object_code("tjb.o");

    s.compile();

    auto jet_ptr = reinterpret_cast<void (*)(double *)>(s.jit_lookup("jet"));

    auto elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Construction time: " << elapsed << "μs\n";

    std::vector<double> jet(36u * (order + 1u) * batch_size);
    auto ic = {// Sun.
               -4.06428567034226e-3, -6.08813756435987e-3, -1.66162304225834e-6, +6.69048890636161e-6,
               -6.33922479583593e-6, -3.13202145590767e-9,
               // Jupiter.
               +3.40546614227466e+0, +3.62978190075864e+0, +3.42386261766577e-2, -5.59797969310664e-3,
               +5.51815399480116e-3, -2.66711392865591e-6,
               // Saturn.
               +6.60801554403466e+0, +6.38084674585064e+0, -1.36145963724542e-1, -4.17354020307064e-3,
               +3.99723751748116e-3, +1.67206320571441e-5,
               // Uranus.
               +1.11636331405597e+1, +1.60373479057256e+1, +3.61783279369958e-1, -3.25884806151064e-3,
               +2.06438412905916e-3, -2.17699042180559e-5,
               // Neptune.
               -3.01777243405203e+1, +1.91155314998064e+0, -1.53887595621042e-1, -2.17471785045538e-4,
               -3.11361111025884e-3, +3.58344705491441e-5,
               // Pluto.
               -2.13858977531573e+1, +3.20719104739886e+1, +2.49245689556096e+0, -1.76936577252484e-3,
               -2.06720938381724e-3, +6.58091931493844e-4};
    std::copy(ic.begin(), ic.end(), jet.begin());

    auto ptr = jet.data();

    // Warm up.
    jet_ptr(ptr);

    start = std::chrono::high_resolution_clock::now();

    // Do 400 evaluations.
    for (auto i = 0; i < 40; ++i) {
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
        jet_ptr(ptr);
    }

    elapsed = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start)
            .count());

    std::cout << "Order " << order << ", batch size " << batch_size << ": " << elapsed / 400 << "ns\n";

    return 0;
}
