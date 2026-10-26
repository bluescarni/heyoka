// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <heyoka/config.hpp>

#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/predef/architecture.h>

#if BOOST_ARCH_X86_64

#include <xmmintrin.h>

#endif

#if defined(HEYOKA_HAVE_REAL128)

#include <mp++/real128.hpp>

#endif

#if defined(HEYOKA_HAVE_REAL)

#include <mp++/real.hpp>

#endif

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/detail/type_traits.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/llvm_state.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

// Tests for llvm_ed_add_fex_check().
//
// llvm_ed_add_fex_check() computes an interval [lo, hi] enclosing the values of a polynomial over the time range [0, h]
// (or [h, 0] if h < 0). The tests check that [lo, hi] contains the exact enclosure, i.e., the interval computed by the
// textbook interval Horner scheme in exact arithmetic.
//
// The test cases are random polynomials and timesteps. Their exact enclosures were computed offline with rational
// arithmetic, and they are stored rounded outward (the lower bound down, the upper bound up). An implementation which
// does not round all operations up fails a large fraction of the test cases.

using namespace heyoka;
using namespace heyoka_test;

// The floating-point types tested.
const auto fp_types = std::tuple<float, double
#if !defined(HEYOKA_ARCH_PPC)
                                 ,
                                 long double
#endif
#if defined(HEYOKA_HAVE_REAL128)
                                 ,
                                 mppp::real128
#endif
                                 >{};

// A test case: the timestep h, the polynomial coefficients cf (from degree 0 upwards), and the bounds of the exact
// enclosure rounded outward. All values are hexadecimal floating-point strings, exactly representable in the
// floating-point type.
struct test_case {
    const char *h;
    std::vector<const char *> cf;
    const char *exact_lo;
    const char *exact_hi;
};

// The test cases, indexed by the number of significand bits of the floating-point type (24 for float, 53 for double, 64
// or 113 for long double depending on the platform, 113 for mppp::real128).
const std::map<int, std::vector<test_case>> test_cases = {
    {24,
     {
         {.h = "0x540273p-22",
          .cf = {"0x4c205p-17", "-0x30fec7p-21"},
          .exact_lo = "0x2f402bp-23",
          .exact_hi = "0x4c205p-17"},
         {.h = "-0x1f748fp-23",
          .cf = {"-0x5f679fp-22", "-0x77d807p-25"},
          .exact_lo = "-0x5f679fp-22",
          .exact_hi = "-0xb77261p-23"},
         {.h = "0xf61ad3p-23",
          .cf = {"0x85d225p-20", "-0x5c139dp-18"},
          .exact_lo = "-0x8f9475p-18",
          .exact_hi = "0x85d225p-20"},
         {.h = "-0x6e0c01p-22",
          .cf = {"0xab2e09p-27", "0x1d0777p-21"},
          .exact_lo = "-0xbcf639p-23",
          .exact_hi = "0xab2e09p-27"},
         {.h = "0x578907p-22",
          .cf = {"0x5a96fdp-26", "0xe9436bp-22"},
          .exact_lo = "0x5a96fdp-26",
          .exact_hi = "0x512d21p-20"},
         {.h = "-0xfa7a35p-24",
          .cf = {"-0x1ca7b1p-17", "-0xc93571p-25"},
          .exact_lo = "-0x1ca7b1p-17",
          .exact_hi = "-0x6f8b4bp-19"},
         {.h = "0xed5bedp-24",
          .cf = {"0x6d5a6fp-25", "-0x23e209p-18"},
          .exact_lo = "-0x81a9c1p-20",
          .exact_hi = "0x6d5a6fp-25"},
         {.h = "-0xc8db9p-23",
          .cf = {"-0xe31f29p-23", "0x3cc7a1p-25"},
          .exact_lo = "-0x724e55p-22",
          .exact_hi = "-0xe31f29p-23"},
         {.h = "0x49ce3fp-23",
          .cf = {"-0xa0589dp-23", "-0x6a7221p-23", "0x75a065p-20"},
          .exact_lo = "-0x1bb727p-20",
          .exact_hi = "0xb6478fp-24"},
         {.h = "-0xb8bc99p-23",
          .cf = {"-0xe19a35p-20", "-0xdb90d3p-22", "0x9614cfp-22"},
          .exact_lo = "-0xe19a35p-20",
          .exact_hi = "-0x88738dp-21"},
         {.h = "0xb35c09p-25",
          .cf = {"-0x9cc285p-26", "0xb89d7dp-24", "-0x884507p-20"},
          .exact_lo = "-0x790a69p-23",
          .exact_hi = "0x32f73bp-25"},
         {.h = "-0xccdaffp-24",
          .cf = {"0x20967p-19", "0xe9252bp-20", "0x1903cbp-20"},
          .exact_lo = "-0xb67e43p-20",
          .exact_hi = "0x20967p-19"},
         {.h = "0xfbf4d1p-24",
          .cf = {"-0xab9465p-21", "0xc005a7p-20", "-0x1d3097p-18"},
          .exact_lo = "-0xab9465p-21",
          .exact_hi = "0xce65f5p-21"},
         {.h = "-0xc23173p-26",
          .cf = {"-0x8df7f3p-26", "-0x7b0bd7p-18", "-0x807b9fp-20"},
          .exact_lo = "-0x8df7f3p-26",
          .exact_hi = "0x2d8f6fp-19"},
         {.h = "0x898d63p-26",
          .cf = {"-0x6a5147p-20", "-0x5f4babp-18", "0xa6637p-17"},
          .exact_lo = "-0x13b0adp-17",
          .exact_hi = "-0x6a5147p-20"},
         {.h = "-0xc32521p-27",
          .cf = {"-0x86a57dp-27", "0x6684a7p-20", "-0x9aa3f1p-25"},
          .exact_lo = "-0xadd43bp-24",
          .exact_hi = "-0x86a57dp-27"},
         {.h = "0xc3ca29p-26",
          .cf = {"0x4dde91p-22", "0xd903b7p-23", "-0x9dc9p-15", "0x3a62a5p-20", "0xce4737p-19", "0x5c144fp-24"},
          .exact_lo = "0x4dde91p-22",
          .exact_hi = "0xc726c7p-23"},
         {.h = "-0x18c453p-24",
          .cf = {"-0x92084bp-22", "0x761fcdp-20", "0xc93c4dp-22", "-0x78bc7bp-24", "-0x617ed5p-25", "-0xa103ebp-27"},
          .exact_lo = "-0xbfbe93p-22",
          .exact_hi = "-0x92084bp-22"},
         {.h = "0xeb6efp-19",
          .cf = {"0xb2f35p-21", "-0x59b65fp-19", "0xafcaf1p-24", "0x4da04bp-21", "0x777ccbp-25", "0xc641dfp-26"},
          .exact_lo = "-0xa236c7p-19",
          .exact_hi = "0xf8d9cbp-22"},
         {.h = "-0x7ef93dp-26",
          .cf = {"-0x57e143p-20", "-0x8058bdp-26", "-0x4f235p-15", "0x4cbfbp-14", "0xddabe9p-24", "0xc28183p-24"},
          .exact_lo = "-0x2d5337p-19",
          .exact_hi = "-0x2bd0cdp-19"},
         {.h = "0x4f3ecfp-23",
          .cf = {"-0x9a83b9p-26", "0xefdc9bp-22", "0xe63691p-22", "-0xea1357p-20", "-0xfe56d7p-20", "-0x464eedp-23"},
          .exact_lo = "-0x49e12dp-21",
          .exact_hi = "0x1c6295p-19"},
         {.h = "-0x8a5ea9p-26",
          .cf = {"0x2d26f1p-18", "0x67eda1p-26", "-0xf8ceddp-27", "0xdafe1bp-21", "-0x5a2f2dp-25", "0xa06ee7p-23"},
          .exact_lo = "0xb414dfp-20",
          .exact_hi = "0x2d26f1p-18"},
         {.h = "0x79d7e3p-23",
          .cf = {"0xe9a39bp-24", "0xf0372bp-26", "0x3074f9p-19", "0xcd5f71p-22", "-0xa4deb7p-26", "-0x93cc19p-21"},
          .exact_lo = "0xe9a39bp-24",
          .exact_hi = "0x4b2345p-19"},
         {.h = "-0xd8b1adp-25",
          .cf = {"0x8accbp-18", "0xdcb1c5p-19", "-0xa1d585p-27", "-0x38bc11p-24", "0x7b3e51p-20", "-0x4adf9p-15"},
          .exact_lo = "-0x9855f9p-20",
          .exact_hi = "0x8accbp-18"},
         {.h = "0xd52351p-23",
          .cf = {"-0xb55db1p-27", "0xcc3131p-19", "-0x476e6bp-22", "0x728aa3p-26",  "-0x4d88e5p-21", "0x852221p-27",
                 "-0xc81cebp-22", "0xf2cbf1p-19", "0x7df189p-26",  "0xb1aad7p-23",  "-0x86d693p-27", "-0xeec6a9p-22",
                 "0xf21c7p-19",   "0xf33215p-23", "0xd09cc1p-25",  "-0xcb1665p-25", "-0xe53679p-25", "-0x16d555p-22",
                 "-0xad0599p-21", "0x7abc0bp-21", "0xcc43f1p-23"},
          .exact_lo = "-0xd27929p-8",
          .exact_hi = "0xc68debp-8"},
         {.h = "-0xb71a8fp-27",
          .cf = {"0x222417p-24", "-0x67e03fp-19", "-0xf20dafp-25", "0x8d4f8bp-26",  "0x436831p-19",  "-0x2412abp-25",
                 "0x4c4137p-20", "0x78bbabp-22",  "0x47a5d3p-22",  "-0x71df57p-22", "-0x1a460dp-24", "0xb5f167p-24",
                 "0x9ec8bp-15",  "-0x812ad9p-20", "-0xdb4097p-26", "0x5b64ffp-25",  "0x7b1fd7p-22",  "0x496751p-18",
                 "0x672b4bp-26", "0xbb115bp-23",  "-0xc40f8bp-27"},
          .exact_lo = "0x222417p-24",
          .exact_hi = "0x296a8dp-21"},
         {.h = "0xd96afdp-27",
          .cf = {"-0x489e21p-25", "-0x8cf201p-19", "-0x1806a5p-22", "0x71b87dp-26", "0xb32d85p-27",  "0x4fe775p-23",
                 "0x1cfa9fp-19",  "0x66b2d1p-25",  "-0x45fda3p-19", "0x9d122fp-26", "-0xced23dp-25", "-0x76fc67p-24",
                 "0x6021b1p-24",  "-0xd66789p-27", "0xa006c1p-21",  "0x86bb71p-21", "-0xde115bp-20", "0x5e9365p-22",
                 "0x5ac219p-22",  "-0xcb5215p-19", "-0x785323p-22"},
          .exact_lo = "-0x810d21p-22",
          .exact_hi = "-0x489e21p-25"},
         {.h = "-0x33cc47p-21",
          .cf = {"0x37b395p-18", "0x3e3039p-19",  "-0x984719p-23", "-0xa57909p-26", "-0x1954e1p-19", "0x164051p-18",
                 "0x7ebfefp-21", "-0x32533dp-25", "-0x977b09p-25", "-0x1477e3p-17", "0xf2e38dp-27",  "-0xe1291fp-25",
                 "0xc62de9p-22", "0xda54bbp-24",  "0xab5ec7p-26",  "-0x307889p-24", "0xc1b105p-24",  "0xf836e9p-19",
                 "0x5661c1p-18", "-0x3825d1p-25", "-0x6e0f27p-22"},
          .exact_lo = "-0xd30f05p-7",
          .exact_hi = "0x48e21fp-8"},
         {.h = "0xb5f3c1p-27",
          .cf = {"0xda8ab7p-27",  "-0x8d7763p-23", "-0xb15d01p-20", "-0xceb979p-19", "-0x135293p-16", "-0x27ec07p-25",
                 "-0xe4b385p-19", "-0x6bfdd1p-26", "-0x87dcedp-27", "-0xbc2f83p-21", "0x764457p-23",  "-0x453f3dp-20",
                 "-0x684c63p-21", "-0x263775p-25", "-0xddc9bfp-25", "0x1acc7dp-19",  "-0x202601p-22", "-0x317c77p-22",
                 "0xdca76bp-19",  "-0xe23399p-21", "-0xfc7f57p-24"},
          .exact_lo = "-0x64ad8dp-26",
          .exact_hi = "0xda8ab7p-27"},
         {.h = "-0x88ed7bp-23",
          .cf = {"-0x8639ffp-26", "-0x91a0c9p-25", "-0x82472dp-21", "-0xb60357p-27", "-0xa16279p-19", "-0xd56f63p-20",
                 "0xf507f1p-23",  "-0xaeac87p-27", "-0xc15377p-24", "0xf59251p-23",  "0xab6c09p-22",  "0x5f6db3p-22",
                 "0xd45465p-21",  "0x4d6565p-26",  "0x64de87p-22",  "0x318b31p-19",  "-0xfc4d67p-23", "0x50a4f5p-18",
                 "0xfdcf07p-21",  "0xf521a7p-24",  "-0x2e26a7p-18"},
          .exact_lo = "-0xc8ed19p-17",
          .exact_hi = "0xdf5f9p-17"},
         {.h = "0xc391b5p-24",
          .cf = {"0x3ae275p-17",  "0x41abf3p-20",  "0x83893p-18",   "-0x8c6b99p-25", "0x4f0a57p-22",  "0xbee273p-26",
                 "-0xaf71e5p-25", "-0x38f1dfp-20", "-0x425db3p-20", "-0x1a601p-20",  "-0xdb1b5p-23",  "-0x5308c9p-18",
                 "0xcf8ef9p-20",  "-0xa795f1p-21", "0x646341p-18",  "0x23a15p-18",   "-0xb1c1c7p-24", "-0x25f38bp-23",
                 "-0x4a2471p-25", "-0x68ee89p-20", "0xad5f2fp-24"},
          .exact_lo = "0x3ae275p-17",
          .exact_hi = "0x887f31p-18"},
         {.h = "-0x265191p-22",
          .cf = {"0x4e51f3p-22",  "0x5aa313p-18",  "0x640039p-23",  "0xd04e35p-19",  "0x750f9bp-23",  "-0x6456c9p-20",
                 "0x92347bp-24",  "-0x270cbdp-22", "-0x998545p-25", "-0x337695p-17", "0x5a8d7bp-26",  "-0x60eb9bp-26",
                 "-0xbcea47p-19", "0xc4221p-15",   "0x4a24d3p-19",  "-0xbd9af5p-26", "-0xd2674dp-23", "-0x58f921p-25",
                 "-0x84a539p-21", "-0x4463abp-26", "-0x12643p-16"},
          .exact_lo = "-0x8d3641p-19",
          .exact_hi = "0x4e51f3p-22"},
     }},
    {53,
     {
         {.h = "0xa9f08fc41e393p-53",
          .cf = {"-0x1a713527b49987p-50", "-0x1d2ad795010d27p-50"},
          .exact_lo = "-0x483f1b2c29409p-47",
          .exact_hi = "-0x1a713527b49987p-50"},
         {.h = "-0x1f2c633e9987d7p-55",
          .cf = {"-0x6bda572c36a39p-52", "-0x11a79da15ba9fdp-52"},
          .exact_lo = "-0x6bda572c36a39p-52",
          .exact_hi = "-0x13876fb9cbba6bp-55"},
         {.h = "0x1169f883e482a3p-53",
          .cf = {"-0x9271f35d2a6ffp-51", "0x7c4537a3d7afbp-54"},
          .exact_lo = "-0x9271f35d2a6ffp-51",
          .exact_hi = "-0x113fbd348c5551p-52"},
         {.h = "-0x163e26f71e2747p-53",
          .cf = {"0x14c9d7e50ed70dp-48", "-0x132ce40d05f357p-52"},
          .exact_lo = "0x14c9d7e50ed70dp-48",
          .exact_hi = "0x159f1996d1e25p-44"},
         {.h = "0x1c4fa099edc505p-54",
          .cf = {"0x141fba2a1dbec7p-54", "-0x9cd7ec47a0e21p-52"},
          .exact_lo = "0x163a9a58dd2e89p-57",
          .exact_hi = "0x141fba2a1dbec7p-54"},
         {.h = "-0xc244b453594efp-51",
          .cf = {"0x109e7dd1f0a2a3p-52", "-0x12c451c5c6cc8dp-48"},
          .exact_lo = "0x109e7dd1f0a2a3p-52",
          .exact_hi = "0x1d8586889ccd07p-48"},
         {.h = "0x11daa22851518dp-54",
          .cf = {"0x1dcb0a4d8643dfp-56", "0x36343d4d22fcfp-47"},
          .exact_lo = "0x1dcb0a4d8643dfp-56",
          .exact_hi = "0x100d63e218ed3dp-51"},
         {.h = "-0x168180f3de5649p-54",
          .cf = {"0xefd868329e279p-48", "-0xe3f0d6a0535f9p-52"},
          .exact_lo = "0xefd868329e279p-48",
          .exact_hi = "0x1e9b5d134fabb1p-49"},
         {.h = "0x1c8f9d5f02b5abp-56",
          .cf = {"0x5d4891c4b26e1p-50", "-0x1cb0823888a769p-52", "-0x16a966a065abf1p-54"},
          .exact_lo = "0x140cb080883d13p-52",
          .exact_hi = "0x5d4891c4b26e1p-50"},
         {.h = "-0x18a64e348f841dp-53",
          .cf = {"0x1040bac6ae7279p-54", "-0x836704404cb41p-50", "-0x44a44e1a5decfp-52"},
          .exact_lo = "0x1040bac6ae7279p-54",
          .exact_hi = "0x1d5e395ff9277bp-52"},
         {.h = "0xc9005840e0bc3p-55",
          .cf = {"-0x1887027328359dp-52", "0x9bcdaa468503fp-54", "-0xd84d557934789p-47"},
          .exact_lo = "-0x1c749be828bea9p-52",
          .exact_hi = "-0x3093b035e8de1p-49"},
         {.h = "-0xe26f1aded207p-50",
          .cf = {"0x12dc30b8f58997p-48", "-0x1eeca44852b71p-51", "0x4346428bf596fp-48"},
          .exact_lo = "0x12dc30b8f58997p-48",
          .exact_hi = "0x131e7fcb3d30cbp-48"},
         {.h = "0x62901ecce1355p-53",
          .cf = {"-0x29d6428237e8bp-48", "0xc4768832b1b23p-49", "0x17a711417292ffp-49"},
          .exact_lo = "-0x29d6428237e8bp-48",
          .exact_hi = "-0x1fd443bd6774a7p-53"},
         {.h = "-0xe4096c9ab8ec7p-55",
          .cf = {"0x19e333fe28db07p-49", "0x2f566cf2a6635p-51", "-0x584d025e6ac41p-54"},
          .exact_lo = "0x19cd9281d52151p-49",
          .exact_hi = "0x19e333fe28db07p-49"},
         {.h = "0x940226e043f65p-55",
          .cf = {"0xf1a7ee3ced381p-50", "-0x14732357ea7b6bp-50", "-0xfe6ac3831fc9p-48"},
          .exact_lo = "0xd9ad6a54b5b99p-50",
          .exact_hi = "0xf1a7ee3ced381p-50"},
         {.h = "-0xf4216b4c290e5p-51",
          .cf = {"-0x1099daf4ae93c9p-52", "0x12da797eb21e49p-52", "-0x153b4365013d9fp-50"},
          .exact_lo = "-0x5a5f8cfea7107p-46",
          .exact_hi = "-0x1099daf4ae93c9p-52"},
         {.h = "0x1dcc3af4df804fp-55",
          .cf = {"-0x1f2bb8ece063fp-46", "-0x9b7292068987p-44", "-0x1a831dc4b1d2bbp-49", "0x326e3edd04477p-45",
                 "-0x19d701a0d27adfp-53", "0x17d039f30e735bp-49"},
          .exact_lo = "-0x158bad49b050edp-49",
          .exact_hi = "-0x1f2bb8ece063fp-46"},
         {.h = "-0x10b99a003748dp-51",
          .cf = {"0xbcb5e1d40ae23p-51", "0x1149c0ffe69ea3p-54", "-0x1258519b6650c7p-52", "-0x1178ace4fd83c5p-50",
                 "0x10d16795795755p-49", "0x10446817facfd3p-52"},
          .exact_lo = "0x5ad7ee49aae8bp-50",
          .exact_hi = "0xbcb5e1d40ae23p-51"},
         {.h = "0x26b610b95589dp-51",
          .cf = {"0x1a1b11d45bdd23p-52", "-0x30976589738e7p-45", "-0x445a20de9cb53p-46", "-0x1455a45eb5ffddp-48",
                 "-0x6a576fbcbbaa1p-48", "-0x155fbdfcfe1f43p-48"},
          .exact_lo = "-0x1fce1dbfccea4fp-50",
          .exact_hi = "0x1a1b11d45bdd23p-52"},
         {.h = "-0xbe1690d24b9d9p-55",
          .cf = {"-0x13e0badc913043p-51", "0x1e8d8010bb3de9p-51", "0x1e719876bde07fp-54", "-0xca4820a487923p-52",
                 "0x5e3d5bb414571p-54", "0xf26445bd1cb9fp-54"},
          .exact_lo = "-0x5adac6b153867p-49",
          .exact_hi = "-0x13e0badc913043p-51"},
         {.h = "0xb4d027f2db21p-48",
          .cf = {"-0x13739a6d23bf87p-54", "0x135c21fc36473fp-50", "0x13c2a9052ae75dp-54", "-0x88677916b4c29p-53",
                 "0x13a28106c92619p-51", "-0x1ace31500032fdp-50"},
          .exact_lo = "-0x13739a6d23bf87p-54",
          .exact_hi = "0x3c916890598fdp-48"},
         {.h = "-0x368f50b631d95p-50",
          .cf = {"-0x148f0ac0b5770fp-56", "-0x16d72f7430ecc1p-53", "-0x148ee8154a95b3p-49", "-0x12c28ab252aaadp-54",
                 "-0x1a55b63b75bf7fp-55", "0x138802a7cc0a6bp-52"},
          .exact_lo = "-0x76b3bd9e91ad7p-48",
          .exact_hi = "0x10e6d44dfb40b1p-53"},
         {.h = "0x1646eacaf69649p-53",
          .cf = {"-0x16d477b84ec599p-54", "0xaf3cb7430339fp-47", "-0x1a3bd1e2a085d7p-51", "-0x1aaf085e86fa3dp-53",
                 "-0x165d435a43a867p-50", "0x54abc5fcebd2dp-51"},
          .exact_lo = "-0x16d477b84ec599p-54",
          .exact_hi = "0x7723899b5caedp-47"},
         {.h = "-0x171e41fe7674a9p-52",
          .cf = {"0x44bf9c530e947p-53", "-0xedeea65726ce7p-51", "0x1604eccd530ab5p-56", "0x1fa4ef201a3afbp-51",
                 "-0x1824c99641451fp-50", "0x12408ca73d2a17p-50"},
          .exact_lo = "-0x7ff3725a1e157p-45",
          .exact_hi = "0x17ff504e9bdb2bp-51"},
         {.h = "0x14b2a2468f06a3p-54",
          .cf = {"-0x1612a8e8a35d61p-48", "0x1e1c77c887d0e5p-54",  "-0x5650653eb6309p-46", "0xf9bab03b0c857p-55",
                 "-0x1c18eb1fe36429p-52", "-0x1c6656a4617393p-55", "0x116ab8c08724f3p-49", "-0x101d665f5f1fbp-45",
                 "-0x6246c52f3a56fp-54",  "-0x16e38cb5c4615fp-50", "0xfc1e15343fe4dp-48",  "0x12e9bb379c7b7dp-55",
                 "0x1f0a183acc27b5p-56",  "0xc63bcc67790dp-46",    "0x15d6654618f433p-53", "0x7246caa7d7dfdp-53",
                 "-0x1ac836af4a13d3p-48", "-0xaa43a4e2f4a33p-47",  "-0x38f743a4991d5p-52", "0x2ea33abc3388fp-48",
                 "0x1b877df218bdd9p-51"},
          .exact_lo = "-0xc18c436203963p-47",
          .exact_hi = "-0x15ebb524cf162fp-48"},
         {.h = "-0x8c20e74cbbb1p-50",
          .cf = {"0x1f9f5b147e296fp-48",  "0x15253067b9f7d3p-50", "-0x1c71592cbca0b7p-49", "-0x1da5773955e7a1p-50",
                 "-0x1b0c0527efcec7p-53", "-0xf5223a1941963p-47", "0x1590f9e0fefae5p-53",  "-0xa44d83c447641p-49",
                 "-0x33da6e079033dp-52",  "0x5baeea7c53d4dp-46",  "0x40c0f0aa3fbffp-53",   "-0x1d3404a514281bp-51",
                 "0x452d87071149fp-46",   "0x15935ea5cb3f43p-51", "0x1aeff5ae415165p-50",  "-0x43562c2dea9efp-49",
                 "0x11b1b3b6ea29cfp-49",  "0x16cab3e30d7dbfp-51", "0x1a7c0d543b7b51p-54",  "0x1104617b7a0d9dp-53",
                 "-0x15299405dc1853p-51"},
          .exact_lo = "0x1ea1fcfe6f843bp-48",
          .exact_hi = "0x1f9f5b147e296fp-48"},
         {.h = "0xca551c8467637p-52",
          .cf = {"-0x17bd75b8b3a0a9p-56", "0x14dac62435555dp-53",  "-0x59bca189548c3p-49",  "0x69c6dd2ae8bf3p-50",
                 "-0xcdaed989337d7p-49",  "-0x8fb918e4efe1p-50",   "0x94001d7c93f95p-51",   "0xe92c9c5e9f9b1p-51",
                 "0xa7b41ed02fdbfp-51",   "0xc242b645aa48dp-54",   "-0x34e076e6e89e9p-48",  "0x1cc4bd5e142e47p-49",
                 "0x1e1fbbd7b0b9a3p-49",  "-0x1cc3eddd01f50dp-50", "-0xac3dc0ce0c0b5p-47",  "-0x1530e392703cbfp-48",
                 "0x172257b8abd05dp-52",  "-0x11383037d46051p-51", "-0x1b6efcf56a8291p-51", "-0x1ee20fb764c473p-50",
                 "-0xd4ee251caba67p-54"},
          .exact_lo = "-0x6213124adda91p-49",
          .exact_hi = "0x360f825ff234fp-51"},
         {.h = "-0x1491834e67c859p-55",
          .cf = {"-0x12da1dd4851aedp-56", "-0xdecb9372181c5p-48",  "0x1ef66e5e31ece1p-49",  "0x6406da68e739p-49",
                 "0x10089535e02905p-53",  "-0x441552aa161adp-51",  "-0x85b68ea48c431p-47",  "-0x11a8c8eccc5fedp-50",
                 "0x7da04ced5314dp-46",   "-0x40947146bb68bp-51",  "-0x11457cc38205d1p-51", "-0xbbea43e05271fp-47",
                 "0x1bd626f2185b5p-48",   "-0x8d9358ceaa4f7p-50",  "0xeff347bbf5ff9p-49",   "0x1f33f44dfdec97p-54",
                 "0xf6222fb0c69d7p-51",   "-0x143234d5dec577p-49", "0xae4e6585391b7p-55",   "0x1ffc83902a9d5p-50",
                 "-0x1a2a6eac0775cdp-49"},
          .exact_lo = "-0x12da1dd4851aedp-56",
          .exact_hi = "0x14826aa8d1ac79p-51"},
         {.h = "0xe1ea8ce74bd85p-52",
          .cf = {"-0x6e88c125cc8b1p-52",  "-0x128d7a0597c3bdp-53", "-0x88bf44f666985p-48",  "0x1fc489ef29a0d3p-53",
                 "-0x4562bc8d00f8bp-48",  "0x114776205d8273p-52",  "-0x1baa143e1c6199p-48", "-0x6a2c279503283p-47",
                 "-0x17fe420cd78ab7p-56", "0xb7e755325b9ddp-54",   "-0x1137a85173c8cfp-52", "-0x14f0c9709c4807p-51",
                 "0x8e0a0dba19ef7p-53",   "0x17b4a19df6211fp-48",  "0x7dfbeb3934783p-51",   "-0x6490cc26526f1p-53",
                 "0x1aedfdf703f4e7p-55",  "-0x190bc9233f42e3p-49", "-0x1429d289ec39c7p-49", "0x1a12f520696cebp-52",
                 "-0x1b7d616b7130c5p-55"},
          .exact_lo = "-0x720e5a468ad2bp-46",
          .exact_hi = "-0x6e88c125cc8b1p-52"},
         {.h = "-0x175bb638a54fa3p-55",
          .cf = {"-0x13cc4688680fb5p-56", "0xee6394c03976dp-47",   "-0x17d01715b14dc5p-56", "0x1eb9dd88d74f8dp-50",
                 "0x791b4acf2b25bp-52",   "0x12004acda6b429p-53",  "0x12507ce36e569p-45",   "-0x1f1a6236f10adbp-53",
                 "0x7783f0e9beccbp-50",   "0x1759d91a7f30dp-50",   "-0x1cba106d23dd85p-54", "0xc4aa16ce94e1p-46",
                 "0x300a6d032483p-49",    "0x19a6ecc7c3da21p-48",  "0xf526d104f4949p-54",   "-0x7e424e56379a5p-54",
                 "-0xd66d3deeeabdfp-51",  "-0x19126caad7e0d7p-55", "0xb00c28e1719fp-44",    "-0x19dbe1752b1213p-54",
                 "-0x1e641b9f9bd255p-55"},
          .exact_lo = "-0x164283a647fd0bp-50",
          .exact_hi = "-0x13cc4688680fb5p-56"},
         {.h = "0x158607da6b623dp-55",
          .cf = {"0x3bbdae7df631p-49",    "-0x185adf373032dfp-50", "-0x108998ebaced15p-48", "0x73309d7a55995p-48",
                 "-0x7cce2956477cfp-49",  "0x1219f6ef643c6dp-48",  "0x9a6d91154455dp-50",   "-0x137055f5f15c15p-56",
                 "-0x1a193aec697fafp-54", "0x13acc1811c8a49p-53",  "0x1841191f29aa3fp-50",  "-0x1b8da12bb76a97p-52",
                 "0x1f8898879553f7p-52",  "-0x12713dd147eac3p-55", "-0x1fab80b563df71p-50", "0x17d0be2b0e77c7p-52",
                 "-0x57aadecc0ed35p-48",  "-0x1885e9d8e62d3fp-53", "-0x66717bcc81529p-48",  "0x10ee6b4e882ebbp-55",
                 "-0xac20b232860fbp-47"},
          .exact_lo = "-0x57fc155a12a09p-50",
          .exact_hi = "0x3bbdae7df631p-49"},
         {.h = "-0xaa64d4cfeea03p-51",
          .cf = {"-0x196ad42f6f5383p-54", "0x2650a4218b707p-51",   "-0xc5312c0cc63dp-44",  "0xfdeb57588330bp-55",
                 "-0xaad7a9c637d43p-48",  "0x1295857331d539p-53",  "0x16a00f43d2c5afp-51", "-0x162136b3b38081p-53",
                 "-0x7ae976f4b862fp-46",  "-0x1e1dc52f596b75p-49", "-0xb0f50fe23ca11p-51", "-0x1955cdca02d6c3p-54",
                 "-0x3a980e0134309p-50",  "0x1a51c1027e2aebp-50",  "-0x2aa3771eb1845p-53", "0xb6ccc6da5d301p-53",
                 "-0x1b6c9e6f9507d5p-49", "0xe6fedf848bbf1p-49",   "-0x7c07cde36003fp-48", "0x881aa6326bed9p-48",
                 "-0x1a072cdbc79491p-53"},
          .exact_lo = "-0x1894ab3073d4a7p-40",
          .exact_hi = "-0x196ad42f6f5383p-54"},
     }},
    {64,
     {
         {.h = "0x86e03beb8ee93317p-64",
          .cf = {"-0xb90ae600b7d67e2bp-59", "0x903df90913ee47e5p-63"},
          .exact_lo = "-0xb90ae600b7d67e2bp-59",
          .exact_hi = "-0x2d12be5ecf4ce501p-57"},
         {.h = "-0x8a32286a7dac4297p-63",
          .cf = {"-0x161f9e3b05089979p-58", "-0x14e2f08499630b1dp-59"},
          .exact_lo = "-0x161f9e3b05089979p-58",
          .exact_hi = "-0xad930f46aa613ebp-58"},
         {.h = "0x453da5ed2e403ed3p-65",
          .cf = {"-0xca2b8fe27b84499fp-66", "-0xca21840b10ec0065p-67"},
          .exact_lo = "-0x35f59f5203feaeabp-64",
          .exact_hi = "-0xca2b8fe27b84499fp-66"},
         {.h = "-0x87271984b20c5edfp-64",
          .cf = {"-0x32e34ac9ca215a6bp-58", "0x872e05e54b69e9cbp-59"},
          .exact_lo = "-0x56924015faa4819bp-58",
          .exact_hi = "-0x32e34ac9ca215a6bp-58"},
         {.h = "0xf38c55740df865efp-65",
          .cf = {"-0x6563916b72a01be5p-64", "-0x5c6d459ee4a5fe17p-64"},
          .exact_lo = "-0x915ac3447cedadefp-64",
          .exact_hi = "-0x6563916b72a01be5p-64"},
         {.h = "-0x2dabebf59b30262fp-62",
          .cf = {"-0x4acf88ad3255c35dp-61", "-0xfe08f4f85dea9733p-61"},
          .exact_lo = "-0x4acf88ad3255c35dp-61",
          .exact_hi = "0x6a792c17bd14acf1p-61"},
         {.h = "0x884fa02cd8ac924fp-64",
          .cf = {"-0xfc5e4a40b9f28375p-65", "-0xfa902815b78da18dp-60"},
          .exact_lo = "-0x8d4d76dfebed76f9p-60",
          .exact_hi = "-0xfc5e4a40b9f28375p-65"},
         {.h = "-0xb6755d606dba9fefp-64",
          .cf = {"0x476fb372e5499189p-59", "0x192129a2c7555acbp-61"},
          .exact_lo = "0x85eada6fa85aa6cdp-60",
          .exact_hi = "0x476fb372e5499189p-59"},
         {.h = "0xfaf34a7bca915cf5p-67",
          .cf = {"-0xb469b7e1becf9f47p-60", "-0xa0882e50e4a65dp-52", "-0xd9db474e8557a5c1p-60"},
          .exact_lo = "-0xcb5acca8bfa77ddp-56",
          .exact_hi = "-0xb469b7e1becf9f47p-60"},
         {.h = "-0xbec20c923585729p-59",
          .cf = {"0x5a427b446f99d197p-66", "-0x87a6d638c63e48d3p-59", "0x5a12c9678ade0747p-58"},
          .exact_lo = "0x5a427b446f99d197p-66",
          .exact_hi = "0x4b5f08011f5a9cc7p-56"},
         {.h = "0xa9624744da236ba9p-65",
          .cf = {"0x2a6f11ba9c37523fp-57", "0x88a23a3c5fb72c99p-64", "-0x505e3f8998403251p-63"},
          .exact_lo = "0x2a6f11ba9c37523fp-57",
          .exact_hi = "0x5592f28e64369427p-58"},
         {.h = "-0x27c5ceefc0711873p-65",
          .cf = {"0xba0da87a8f88950dp-63", "-0x84479c7b933ec0cdp-65", "0xb97a35e5f7b8d2bbp-59"},
          .exact_lo = "0xba0da87a8f88950dp-63",
          .exact_hi = "0xce87ae6435acebc9p-63"},
         {.h = "0x65b99ea00e912a7bp-65",
          .cf = {"0x8e27c2fb7b7d3c1bp-59", "0xe773ade37d9fdbb5p-59", "0x52e8f490497ac7e5p-66"},
          .exact_lo = "0x8e27c2fb7b7d3c1bp-59",
          .exact_hi = "0xbc2a897a1f9aa2afp-59"},
         {.h = "-0xd2e70d1343340a3fp-63",
          .cf = {"0xec55385addf68a2dp-64", "0xb8ecf4846bc2e789p-63", "-0x244f9556a3cfd6dfp-64"},
          .exact_lo = "-0x1d7a3e468bcda395p-60",
          .exact_hi = "0xec55385addf68a2dp-64"},
         {.h = "0x3d0122d45ed2b853p-63",
          .cf = {"-0xdcd53329814fb505p-60", "0xce6c5844a321960dp-59", "-0xcf108358e8e1cc1fp-66"},
          .exact_lo = "-0xdcd53329814fb505p-60",
          .exact_hi = "-0xc0922801690c1e53p-63"},
         {.h = "-0xac5f6b6102e4fe43p-63",
          .cf = {"0xa16601ad14979effp-64", "-0x710261af233bcbbdp-64", "-0x3f518c5c4ab42993p-65"},
          .exact_lo = "0xa16601ad14979effp-64",
          .exact_hi = "0x4e655ce0a3fc30e5p-62"},
         {.h = "0x4cd8fddb099fd7e5p-64",
          .cf = {"0x637636ce5159ba7dp-64", "-0x7642a57c1cec41d1p-65", "-0x64435eb7737d5ff1p-63",
                 "-0xc8a4982e47a529b3p-64", "-0x6f81dea50eb61ebfp-63", "0x2e443ba3aa685d63p-63"},
          .exact_lo = "0x1c33a9be0d975e7fp-63",
          .exact_hi = "0x637636ce5159ba7dp-64"},
         {.h = "-0xa2df084c478c6745p-65",
          .cf = {"0x5240e1566ccb5bf7p-66", "0x77f79e52baf1821bp-62", "-0x23dc792c3c9e2d4fp-61",
                 "-0x2e9effa88e08fad7p-65", "-0x41d5ec8dbcefc7p-54", "-0xc88003851348114dp-59"},
          .exact_lo = "-0xa3106a06ff43c23fp-64",
          .exact_hi = "0x5240e1566ccb5bf7p-66"},
         {.h = "0xb4f7b832ca13d57fp-63",
          .cf = {"-0x1484a5bff7a203abp-57", "0x274ad1d1e0b03ec9p-61", "-0x1f5428480374f55p-59",
                 "-0x2834c410a52b63fbp-62", "-0x3f3f16465184169bp-59", "0x769f50b31e48784bp-60"},
          .exact_lo = "-0xa97fc893e9de79cbp-58",
          .exact_hi = "-0xfd97b9d1a8b7308bp-65"},
         {.h = "-0x1e3e6f3522c976a1p-62",
          .cf = {"0xe96ce0513c13d11bp-63", "-0xa6acfcb7f6dac301p-66", "-0x5337407e6fc33749p-63",
                 "0x16cf6c532d961fb3p-59", "-0xa11cb0b9bf6635abp-64", "0xfef63ca812b074b9p-59"},
          .exact_lo = "0xac0a91b5f11bb779p-64",
          .exact_hi = "0xf34556466ad6a465p-63"},
         {.h = "0x99cf43440556da11p-67",
          .cf = {"-0x2d529a510d49023p-60", "0xbe1235375b45ded9p-65", "0x898461985b8b12ebp-67", "0x8b71f5334b38cce5p-64",
                 "-0xa5c018920d70d1fp-57", "-0x67b08010404dd279p-64"},
          .exact_lo = "-0x2d529a510d49023p-60",
          .exact_hi = "-0x981df305a44047c9p-66"},
         {.h = "-0xcda099d233c20da9p-66",
          .cf = {"0x81166f6628480e21p-66", "-0x58722f9f191dc53p-57", "0x69dc5eb3b91b4eafp-66",
                 "-0x64e8cdb732c85b9bp-63", "-0x5ea2bea84a8f654fp-64", "0xc57c43b08bc894d3p-62"},
          .exact_lo = "0x81166f6628480e21p-66",
          .exact_hi = "0x2c43c09dd79f8c39p-62"},
         {.h = "0xc8730b11c34247b9p-65",
          .cf = {"0x8b5010a403805d9p-56", "0xfa89c8331391d4e9p-65", "0x2965995b9dff647fp-58", "0xc3680e03c60b947p-57",
                 "-0x8c8ba6e85ad17581p-65", "0x76ed9f3608667ef9p-59"},
          .exact_lo = "0x8b5010a403805d9p-56",
          .exact_hi = "0xafb4a905fe66eb5bp-60"},
         {.h = "-0x8e0345800890496fp-64",
          .cf = {"-0xa14b9acaa6b60223p-62", "-0xafaa909b059ece79p-62", "0x4f849a1fffdc9d33p-59",
                 "-0xaea9c5095cf9506fp-61", "0x4022931bfd188f2dp-60", "-0x2d5027cca04717fp-55"},
          .exact_lo = "-0xa14b9acaa6b60223p-62",
          .exact_hi = "0x3ab8d7a227d9aee7p-60"},
         {.h = "0x9485b2097a77bc0fp-64",
          .cf
          = {"-0x7dea5417b5b4db4bp-63", "0x46d2cb8bb17bf9d1p-65",  "-0x244b086800f00607p-59", "-0xe70c1d71f754c8a7p-60",
             "-0x530fb455257f9cf7p-64", "-0x62585c3915979e19p-64", "-0x2e36dcf9fcbfa8b3p-61", "0xc79a7b026c2412ddp-60",
             "-0x3f82c5fdfa65af4bp-64", "0x8dfe66955a27f7d5p-59",  "-0xe631cb56a1582f3dp-60", "0x660a58e9e623b9ffp-61",
             "0x2c330f91391d648fp-62",  "0xf740f5f542acc215p-62",  "0xe40dbf309e4d5c7p-58",   "0xb778c910d841ff9p-61",
             "0x9dae0d3a2a1fbf09p-66",  "-0xbf4976fbc9f4394fp-59", "0xda8922bb22c14ddp-58",   "0x48a9b991e7178e21p-60",
             "-0xc5c67631724ba29dp-61"},
          .exact_lo = "-0xabc224a0a950118fp-61",
          .exact_hi = "-0x1ce9275ed23d2adp-57"},
         {.h = "-0x4c6f97df0ca46ed5p-65",
          .cf
          = {"-0xa041e2cc3c2bdd89p-64", "0x2018d348be73409dp-60",  "-0xd55e4f5bb881077bp-66", "0xde9c3f794b45ff19p-65",
             "0x46883c578f4ee3a9p-62",  "0x5244af2f5677375dp-65",  "-0xe41533df45127d0dp-63", "0xb04c80dd61870b2fp-59",
             "0x1ac254d570d4755dp-56",  "0x72f72466fe143b8dp-61",  "0xd710e48e946c5961p-64",  "0xc18bd7b9ccf60ef5p-63",
             "-0x9296ffd67a11046dp-67", "-0x9713eb2a0c29c03bp-67", "0xf5035be71f863011p-66",  "0xd46087a6a8f185b9p-61",
             "0xd519d46e7d7a67d7p-63",  "0xa8cc8201401b457fp-64",  "0xc0d05e59e246867fp-61",  "0xd3ed466924f79fd1p-63",
             "-0xb278591948a944ebp-61"},
          .exact_lo = "-0x3b9efb66aba34acfp-62",
          .exact_hi = "-0xa041e2cc3c2bdd89p-64"},
         {.h = "0x1856ac03e2b532bp-56",
          .cf
          = {"0x3bc66e756023a343p-58",  "0x6d8ee05308ef3f8fp-66",  "0x5c09b782a06f30adp-62",  "0x79d18457b4ba94bp-55",
             "-0x6a9966fac7407531p-61", "0x8b0f348c3e774705p-66",  "0xa9ca0ed4f873eb33p-62",  "0x328edde35e7ee029p-63",
             "-0x4b1ccce6df77a2a9p-65", "-0xbd326f2e17806b07p-59", "-0x508b91752fa91753p-63", "0x9b345ae2b64ccc0dp-61",
             "-0x53df427f0227c3a3p-62", "-0xb878cada64022325p-64", "0x17856f04917963a1p-60",  "-0x6a3913a23c3cde75p-60",
             "0x6ca54aea0ea19111p-63",  "-0x1601604f5a682769p-57", "0x57a7c04b9e49d087p-60",  "0xb66f2b3b0da81d9fp-63",
             "-0x576b010dc8c87cb1p-58"},
          .exact_lo = "-0x604d272aee533493p-46",
          .exact_hi = "0xbf3a8c11a4c43cdp-53"},
         {.h = "-0xb4e724ba50b28b83p-63",
          .cf
          = {"-0x492c80e44a1c1715p-59", "-0x9fa6a06629b61e29p-59", "-0x1cb75ae772cb6909p-63", "0x386b397c2d0e6ed1p-65",
             "-0xc8a2b1bb87c390fdp-59", "-0x58f6f8b701b37f33p-63", "-0x75fbbd991bf0eebfp-63", "-0xa53bd7c551ba6629p-62",
             "0x767461b7f5223ae1p-60",  "-0xb357f4e53e3800cbp-66", "-0x9a59e5170c444399p-63", "-0x2f60e2c90274116fp-59",
             "0x90d6bc701a2b7c9p-61",   "-0x2c9ecb47d2f5b7efp-57", "0xa8f4be256c19a5d9p-64",  "-0x671e494c5cd82d3bp-59",
             "-0x1435d2ddd901c757p-62", "-0x57fa6f6800ed3191p-63", "-0x1fc20c0b8feb2cfp-54",  "0x34cf1e28e29aa69p-61",
             "-0x380a97f98fbd969dp-59"},
          .exact_lo = "-0xc4d7a07e63478f2bp-51",
          .exact_hi = "0x9854f63b0f5a4a7p-47"},
         {.h = "0x2c4a47ecdd8566b9p-62",
          .cf
          = {"0xafcbed5d461f87efp-61",  "-0x4a32e73f55d1e157p-61", "-0xe98a5e883f522657p-59", "0x8de6a72739e68ba1p-63",
             "-0xad48688c81326ba5p-59", "0x4f787cea09c8a3cbp-59",  "0xb369422a2c450265p-64",  "0x2339e40e84f2cbffp-60",
             "0xb25ab5ce78a43327p-67",  "0xfcbe3ff30213fc8fp-63",  "0xcac1c5097756425bp-65",  "0x3e6ab2b8a3b7102fp-57",
             "0x28313911e6bc6bafp-59",  "0xf1e7c89b7f38b68fp-62",  "0x6778b14242a0117fp-63",  "-0xb52e107166087cd3p-65",
             "-0x5642156701906887p-63", "-0x88ca632fe57c1175p-65", "0x8510062c61afeaa3p-64",  "-0x64b7cf789cf37b27p-58",
             "-0xa078b83ecba788fdp-67"},
          .exact_lo = "-0xeb12ee182f7fbeffp-60",
          .exact_hi = "0xafcbed5d461f87efp-61"},
         {.h = "-0xd44af611d53966ffp-65",
          .cf
          = {"0xfb276af9fbdcbf5dp-60",  "-0x86e7b82a5a2d4103p-65", "0xabfc3ded38b024b1p-65",  "-0xbbdb4992425ff827p-67",
             "0x8f6f0ca45b625265p-66",  "-0xec0f3317c293ff9fp-59", "-0xe7be6255c1693effp-59", "0xaf33f23783450845p-62",
             "-0x4838267dfdfec6cdp-62", "0x40f2f2e9f750268dp-58",  "-0x23b3a359faf2ab93p-61", "0x38e7bf73e37850c1p-61",
             "0xb8c9abb5e03cb9efp-61",  "-0xea7eb287cd33cd31p-60", "-0x34f042deca77f8c9p-59", "-0x53255018fbb18a1dp-66",
             "-0x61875771342208d1p-63", "0x2c83b10ab339cbf1p-63",  "0xda5eea62fb50cdf5p-67",  "0xe563c53ad53edab7p-60",
             "-0xc6f5cabdff540463p-63"},
          .exact_lo = "0xfb276af9fbdcbf5dp-60",
          .exact_hi = "0x81e434239cc3de1bp-59"},
         {.h = "0x98906412ca5665cbp-67",
          .cf
          = {"-0xc32188f5bac1c583p-66", "-0x4dd24a394e8a77d5p-65", "-0x97f9fdc2819a2b35p-60", "0x436f6b237486448fp-63",
             "-0x80a0cbae405db8ebp-63", "0x5c385c59e10f11f7p-61",  "0x9e007943ddcf88bdp-65",  "0xccdab5797400ce3bp-64",
             "-0x528be2c17a2c5b75p-62", "0x45f06a76bf0b53c3p-62",  "0xb4c173b80a8b1ee1p-63",  "-0x120818f07b0b732dp-58",
             "-0x163c8fedb24f356fp-63", "0x283d91cb398fd7d5p-62",  "0x843f5fc20822ab53p-59",  "-0x26bc659d7ada2c2dp-58",
             "0xe2fbb50dde790bdfp-62",  "-0x9cfe57917061a6f3p-60", "-0x77015a59f5adde09p-64", "0x63e9b9a790e6a54fp-64",
             "-0xfe806a03c430d96dp-60"},
          .exact_lo = "-0x8259cadfb937a99fp-65",
          .exact_hi = "-0xc32188f5bac1c583p-66"},
         {.h = "-0x2328b1c2497feb4fp-62",
          .cf
          = {"0x9db10582b38e249fp-67",  "-0xe91322ae379699bdp-59", "0x52040ff465cefb89p-65",  "-0x5a0a9e96c0f8f665p-62",
             "-0xbdf12273d52a8bf3p-64", "-0xa6f73dd93e9c6f57p-65", "0xfac7cf3433ce18d7p-60",  "0xbd64a0f6464ca8ffp-67",
             "-0x6b19897f723c26d5p-62", "0x622ccbd7760b6519p-62",  "-0x7f46a8fd3f05e967p-61", "-0xaa12a76a4841855fp-64",
             "-0xe57bbe1504636f6dp-62", "-0xdfe12d8559e6faa1p-62", "-0x484e8c0d5db554bdp-65", "-0xcd3c29c7cbecad61p-63",
             "0x815a1e2302f4e4f9p-61",  "0xbe408bd90d8320e5p-67",  "-0xdecfd9c020538715p-59", "-0xdff726df62674693p-61",
             "0xe9424de8dc0f88edp-67"},
          .exact_lo = "0x9db10582b38e249fp-67",
          .exact_hi = "0x85f26fcb95a1c23bp-59"},
     }},
    {113,
     {
         {.h = "0x18d9fcc2c493d9fe519da4981afa7p-112",
          .cf = {"-0x1d4aeb012249c51e8a356021ed11bp-112", "0x1c005b28322319ea2705d235e4ba7p-109"},
          .exact_lo = "-0x1d4aeb012249c51e8a356021ed11bp-112",
          .exact_hi = "0x9f52aa2bc093931dcbf80c1ab3f3p-107"},
         {.h = "-0xc71a1f83be8239326c36baf2ce03p-113",
          .cf = {"0x1123b4bc7d77b3e2dd5a7a8f612cdp-108", "0x140bd504692b56571c18ed7bd4185p-108"},
          .exact_lo = "0x12b02b372af784ad26e08fa2e539dp-109",
          .exact_hi = "0x1123b4bc7d77b3e2dd5a7a8f612cdp-108"},
         {.h = "0x16df39f73d5e15ac6761aefadd765p-112",
          .cf = {"0xbe878c51d633f62759b05e6c2b41p-115", "0xe66a91d25852342de06a29cfcfe9p-115"},
          .exact_lo = "0xbe878c51d633f62759b05e6c2b41p-115",
          .exact_hi = "0x81fa3139e6ef69e7d2b8f3bdf7abp-113"},
         {.h = "-0x55d38fc3c59995dd5495a71f4cebp-113",
          .cf = {"0x1137d424b9d53bce5b16fe0d1bebbp-109", "-0x3aee6821f3d012d3318a0f6f48d1p-105"},
          .exact_lo = "0x1137d424b9d53bce5b16fe0d1bebbp-109",
          .exact_hi = "0x1b18c23a656dce442f419182c9dc5p-109"},
         {.h = "0xbf08cdbbeea5acbc51a21e2d2171p-114",
          .cf = {"-0x696d8316e06231b0189aba5d78afp-114", "-0x1c9c227501599d4475f5b8cbe84e3p-115"},
          .exact_lo = "-0x12840ec7f60ae2602e5ced057d5ddp-115",
          .exact_hi = "-0x696d8316e06231b0189aba5d78afp-114"},
         {.h = "-0x84ef08a05f6600d3cc37d069af75p-112",
          .cf = {"-0x17261f15b5ae67d88bd7f7f2a151fp-109", "-0x2eeaccf3f3aa4efb13d0ea532abbp-110"},
          .exact_lo = "-0x17261f15b5ae67d88bd7f7f2a151fp-109",
          .exact_hi = "-0x16633811f6ce2913fd4f181dc46abp-109"},
         {.h = "0x2d4fd3317c41d5c3567d189b7117p-109",
          .cf = {"-0x97c3c33f7f74eb7edbdc5867eabfp-113", "-0x16390e5d6f69f3c03a5691a44dbadp-111"},
          .exact_lo = "-0x10eb5d6336db62defb32401b4119fp-110",
          .exact_hi = "-0x97c3c33f7f74eb7edbdc5867eabfp-113"},
         {.h = "-0x1b8a2024845365d51ed188c55d51bp-115",
          .cf = {"-0x14d28db1a89dc0236438047958389p-116", "-0x1ba9a39b115ebd165d3543c7913e7p-111"},
          .exact_lo = "-0x14d28db1a89dc0236438047958389p-116",
          .exact_hi = "0xa9a1bd7b1930d67ea22b87d3d955p-112"},
         {.h = "0xa686a957b01ee4ea33dbbb9b0e23p-114",
          .cf = {"0x24961dceba9697c7c9e68907ded1p-112", "0xe8b30b10b44779e38bd6371ef621p-115",
                 "-0x89b82300d9f36b72a668af7d2e1fp-107"},
          .exact_lo = "-0x4b3b81eaa9cf258d60b4dae5206fp-112",
          .exact_hi = "0x14a888eaa766d69720ab9e1d9da17p-115"},
         {.h = "-0x4a4f1dc634fed3839db3f9ab5469p-112",
          .cf = {"-0x19fde5930b94c88e6e2dc15dc4523p-115", "-0xf1604b95df010c4d2b0359ffcbcbp-110",
                 "0x4e197d2abaf7ffa4f4843e7ba29fp-110"},
          .exact_lo = "-0x19fde5930b94c88e6e2dc15dc4523p-115",
          .exact_hi = "0x3fa610bd61d7094e976b1583f295p-110"},
         {.h = "0x73947a62fb10565a33d7fd7c59adp-114",
          .cf = {"0xf33e1d3e9820a69b0b931f9f5d57p-114", "0x18be0fb36eddf04205a463255233fp-114",
                 "-0x6bbc3d909043d79be5dfc6f6bcb3p-113"},
          .exact_lo = "0xf33e1d3e9820a69b0b931f9f5d57p-114",
          .exact_hi = "0x11fed0849e44da020044d95d1f9c3p-114"},
         {.h = "-0xbc21e7cb089857a1df194b53a3afp-115",
          .cf = {"-0x14f2722d17134952b46595e98eb71p-112", "0x9a84457796c36b2713f7e655d10bp-113",
                 "-0x28061f3709bd4a5a1ad1f285f205p-113"},
          .exact_lo = "-0x1566b379f14303d6bac33f3430d93p-112",
          .exact_hi = "-0x14f2722d17134952b46595e98eb71p-112"},
         {.h = "0x138926383c87c3fd0c0513c691579p-115",
          .cf = {"-0x56ab925b5b9a7c4eb0bba00ddd1fp-106", "-0x1dd08943539087d17035a80e45d73p-111",
                 "0x1e04e4ba9f3338fd2e4095fee9a63p-109"},
          .exact_lo = "-0x163c8166143828f54883acf0cc933p-108",
          .exact_hi = "-0x56ab925b5b9a7c4eb0bba00ddd1fp-106"},
         {.h = "-0x4c2aa02fa022e43883adfa65b163p-114",
          .cf = {"-0x37a44b8e640fbdd886bf021a5fe9p-106", "0xd611b11f91df811d67651b7d3a9dp-108",
                 "0xac1ed448374697c2ad149ac6d3dp-105"},
          .exact_lo = "-0x1dcfacf2ac8a9d13209a453ccc1f7p-109",
          .exact_hi = "-0x37a44b8e640fbdd886bf021a5fe9p-106"},
         {.h = "0x145de8f8062e29e5fabce24654aa9p-116",
          .cf = {"-0x14171810f94283b9cff5171f2d063p-110", "0x15c2438b85331fc8e063c94e94afp-106",
                 "0xff7283c38cc2435a51406025f8fdp-113"},
          .exact_lo = "-0x14171810f94283b9cff5171f2d063p-110",
          .exact_hi = "-0x1258b3973602ae1343975b67a9edbp-110"},
         {.h = "-0x1d162d89eb57b580cff55cba716f1p-113",
          .cf = {"-0x190f6cacd79a8b26b2c00f4ad8267p-108", "0x174245fc49cfd0ac940164d51184bp-111",
                 "-0x6038c3435cc59b09c2a12b513d1dp-106"},
          .exact_lo = "-0x17c9eea8026025a75cca6ba41fa2bp-107",
          .exact_hi = "-0x190f6cacd79a8b26b2c00f4ad8267p-108"},
         {.h = "0x171fc7418d3ba11557e21fa504b85p-116",
          .cf = {"-0x11e6839a5d82e5c85beaa995a0fd7p-109", "-0x15809e2dac984c2cce0b5c1acbc31p-110",
                 "0x18a13105aadd9fe169dbf43a69077p-115", "0x16132f17be4a5c985c191c177c1d5p-109",
                 "-0x1fd68125fb3c7949e44c56f86d40bp-115", "0x1ee485e6b17165f183ea3a0822da7p-112"},
          .exact_lo = "-0x12df205d3cc4f2743c53c3d5de535p-109",
          .exact_hi = "-0x11e6839a5d82e5c85beaa995a0fd7p-109"},
         {.h = "-0x5b535a685232011a121f75a108dbp-114",
          .cf = {"0x1905069e46d2a86682a636b2578bbp-116", "0xabb7597e07cab576cd6ea9b71b8bp-112",
                 "0x1eac37e8f4bb17abc5d93a0bccbb7p-113", "-0x19d8bca2507bee1c034f6e8ff02bfp-109",
                 "0x1840db75b7d1428ac848e5faa0731p-111", "0x7f0ff66ba7bda064d0ecb6f32fc5p-108"},
          .exact_lo = "0x136902496dca0e9c3f55b3132691fp-117",
          .exact_hi = "0x1905069e46d2a86682a636b2578bbp-116"},
         {.h = "0xd8b41365c89855c2d7b0afb5594dp-112",
          .cf = {"0x1a0abf2de4e73a1f463ba3ac2094bp-114", "0x9ca70d770be964eed35b832ef013p-108",
                 "0x177b4da7c6c3ac9016c8ecae1508dp-109", "0x1902d75a98f19ab45922b5fcf55ffp-112",
                 "-0xea8d4f1d7a91d7b276b9a9262c79p-112", "0xac569a1510dedaf5c18cf5bd98ffp-107"},
          .exact_lo = "0x1a0abf2de4e73a1f463ba3ac2094bp-114",
          .exact_hi = "0x1af2d9d4302d3d557a83eec7aaffdp-108"},
         {.h = "-0x164afba33dd5cc2fad0af2722727bp-115",
          .cf = {"0x1038532d72c962bfa1a5cec45c77p-107", "-0x1a1caf783bac03ffea99dc5b1f59dp-114",
                 "0x1cb9ab68ef9d3c09822c0d3f40a13p-113", "-0x707f32ba084be45cb8df9c6a3bc9p-112",
                 "-0xab690e42153459d5a970bb295d3bp-108", "0xdd8657d6a5eefc21087d636c5339p-113"},
          .exact_lo = "0x1038532d72c962bfa1a5cec45c77p-107",
          .exact_hi = "0x8832fa332ac797cb58c7bf6de081p-110"},
         {.h = "0x143c57b069125b09beb7d3700babdp-112",
          .cf = {"-0x1f76fbee5963861bb2371f3b6bfe9p-108", "0xf7a3e0bb04efecf0f04c605f98c3p-109",
                 "-0x151ef10de8c01cfee97ae2aa406b9p-116", "0x1a2c5c6f28cc7eb204f0744c99d2dp-112",
                 "0x120cce41c4d133f1445c04a5566bfp-111", "-0x1d812b150366c713b6863b23248cbp-112"},
          .exact_lo = "-0x1f76fbee5963861bb2371f3b6bfe9p-108",
          .exact_hi = "-0x32e8bb8075181a3ed91b5f43be6bp-106"},
         {.h = "-0x1864686107a725ef5e78fbec6f85p-112",
          .cf = {"0x94c354c4f53ccbf619ad756c7fb1p-113", "0x1ba6f67955e3b14dfeb27a6f443b1p-108",
                 "-0xba8a89e04dd330d4eaf2eb186c01p-115", "0x1cdfeec249166aca686a8b05ad1f5p-110",
                 "-0xbecdc692f1e47a9a3bd0e4669433p-115", "-0xc470420cea021d54c5849b6fe9cdp-111"},
          .exact_lo = "-0x967b777a3ffb0a8c72a18befb7c7p-110",
          .exact_hi = "0x94c354c4f53ccbf619ad756c7fb1p-113"},
         {.h = "0x1cca2eed9dcbc386362149972d2dp-112",
          .cf = {"0x141d0886e5103d9f19e671413b7fbp-111", "-0x15bec28173f9ce9fab1088d8550bfp-114",
                 "0x56df3e9f1e02fd9e8af831abc123p-106", "0x1a7fd1efcc6fca0f97b5df539a261p-115",
                 "-0x646586923d6eacd720bf7afea04dp-114", "-0x14e10ca8ab5ee4f60bfcbe762dd3fp-113"},
          .exact_lo = "0x9e763a6f0fb87669b7b91eb08173p-110",
          .exact_hi = "0x1601e9b6dae17c446e044f6719541p-111"},
         {.h = "-0x1520e830347a8fdb5d8365a34cc7dp-113",
          .cf = {"0x3ee0c126733ca84a6130707f8febp-108", "-0x17ef8f560e77f33a0c3eac0e0cf19p-112",
                 "0x1f05e6ef0e441d3680a72c76d06e5p-109", "-0xe3e296af5633d5927e0b08ba771fp-111",
                 "-0x1feb019cfda9a004ea259925eadabp-115", "0x1b04a89fa8a7731687254e2f5beefp-109"},
          .exact_lo = "0x3ee0c126733ca84a6130707f8febp-108",
          .exact_hi = "0x18627f25015c14a649aea13bfd49p-105"},
         {.h = "0x1cd44cc3f4fd1703882a637c6ffb7p-113",
          .cf = {"0x158254a2261cda3c265cfbe96a821p-108",  "0xceb002bf9c3fecf6ff12122a083bp-112",
                 "0x1707ce30debddcde552739bfe154dp-108",  "-0x17c4595a31563952ff3abd23b8bc5p-109",
                 "-0x10a038abec35ca988ba29ddaa559fp-111", "0xe3db75a322271e2ebb9a118eb25p-104",
                 "0x4ed0f5b51d30d90147783298ca7p-102",    "0x7e6451f9668529e307b2892446f9p-112",
                 "0x173692ed3cf807cb6e7b836f22a5dp-114",  "0x32d99565d48c66ca1e72c15d9107p-112",
                 "-0x3f2806361e247edb06a2379a3021p-108",  "-0xd7d562cfe387d3f1f5e891e29ad3p-109",
                 "0xfab1ad2c487195e58a3b63bfcce9p-110",   "0x52b32b7661305c68b3fa2132939dp-107",
                 "-0x195b0b9ac7242be5cecf1991838c7p-115", "0x17a6dd3610ed181e70151ed4e2b23p-109",
                 "0x941a906e922967a8c397050c52a3p-113",   "-0x168ca2c97fb2625d98a0294d96179p-111",
                 "0x1bd475156cc8fccf361b15a1980e5p-114",  "0xfbaaf1ad2dc22f45cb5ee0b5a553p-108",
                 "-0x1b09450abaf175915e9668e9c45e3p-116"},
          .exact_lo = "0x158254a2261cda3c265cfbe96a821p-108",
          .exact_hi = "0xdb45ae625f71cabbeb9cef689435p-106"},
         {.h = "-0x11b51cb88c57f702454119035d1fbp-116",
          .cf = {"0x32625949bf03f7b89ac753c02399p-112",   "0x191c398b8c20434ff9135dbd96b4fp-113",
                 "0x168deaefbff24183bca36841daf57p-108",  "-0x412d8a442141b6e9765d0be7bc59p-107",
                 "0xe057f70bd56ec9ae4314e449453p-108",    "0x848385926996e100a1fae1fe3a69p-110",
                 "-0xcd1d8af9d9002964f9e4d6c24315p-107",  "0x1cf3027ab2d81e78350b62ac73eedp-108",
                 "0x13e0c93fbce17b692b1fe1dc76ac1p-114",  "-0x24c6b895dc242f6a8eaf60f4e3a9p-113",
                 "-0x2d41903e55229d80d631de16396fp-109",  "-0x1adc2ddbfbc38693e67d839876f35p-112",
                 "-0x3f597573cb1d8b310f8cc465ebabp-106",  "0x15e353d0eaa96ec51fd2c9703701bp-114",
                 "-0x1066f8a024e2dcfa54841266ad3cfp-109", "-0x14d402ed35295d62ec0840e128a27p-114",
                 "0x74e0733f2d716d457778b69cfb05p-109",   "-0x5aeb479be23c22d5a8d05a53786dp-112",
                 "-0x162c17dbf7667913d813060c1567p-108",  "-0x17424cdcb33ef80450bfef811fcbfp-112",
                 "-0x76b2a0797713257762f8d0248c85p-111"},
          .exact_lo = "0x123e9e444e3e5f12687c328775665p-115",
          .exact_hi = "0x1033d19867b76513e2f2f4dae0fc1p-114"},
         {.h = "0x1681851f8236560ce26376241d05p-111",
          .cf = {"-0xee81919e26e0ca83067aad409395p-108",  "0x93ff9e3f986d3a0759b8daa6b72dp-112",
                 "-0x13f0ebc5141694b2f77a3559c6bfp-110",  "0xd8476b75ff2a0acccdb1760267edp-113",
                 "-0x16522fc32d96dbe907448e8d35927p-113", "0xb6d4cc2338b38d1cd382eda0cb33p-109",
                 "0x1a1da51d4aba3c7782c2d4ae1a61dp-112",  "-0xbe31235a0b0398b63fbc10d08e81p-112",
                 "0x60753c13dc2b5cd4b8d4ea0947d5p-108",   "-0x1587a2ce70822dc96a387a1246461p-109",
                 "-0x174a7c6ce46a439880d95117405f7p-112", "-0x197c7f5618d3ab60715db3556b7e5p-116",
                 "0x1413b2690d1fe80e51412e59ce92bp-110",  "-0xffb72e4625cfafa0222dd8fdcb87p-115",
                 "-0x13726f09fe4e4a516ecd165597653p-114", "0xde931fe96bc2f106c8c57527f63fp-108",
                 "0xd0c4e1b7e4f30e144b6e91d980bfp-108",   "0x1db491fed99fc2177b63d68ee5e6bp-108",
                 "0x569ffaf51b81f6181c050f1b82a5p-108",   "0x1c585eed74232eed8f8e483d7f711p-112",
                 "-0x11d32b59c098e7ef9639b1ae49aa9p-113"},
          .exact_lo = "-0xee81919e26e0ca83067aad409395p-108",
          .exact_hi = "-0x1d9c26d24bf5bad89884cfa019657p-109"},
         {.h = "-0x695c630c74a07d5804fdcc5496d9p-110",
          .cf = {"-0x178f065e0232d951f0fe12f23716dp-115", "-0x1f081b06e1fb92bcead4e96d06ff3p-113",
                 "0xa74e5714864bdb0e53ba0cd9ce0bp-112",   "0x190dd084547d3338755f3b1b5d7fdp-113",
                 "0x1a54ef8530e5d12eda221b69226f3p-114",  "-0x13f43959aed38c427766ba2fd7121p-108",
                 "-0x113e3a249c90c6d279e534d055c39p-113", "-0x33feab76ec577b1510695ae064dp-108",
                 "-0xf8819d77acb754680004de65ea63p-108",  "-0x5943a207179cb96dca7a44295465p-113",
                 "-0x14feb09653ff445db422fbd4f2c59p-109", "0x6f8b624b01326b7857cb933e119p-104",
                 "0x3ed57ec3b81996ad93add6c1f0e9p-113",   "0x17c4d6de9c86624ce5166a1ebdd05p-108",
                 "-0x173f33d02c6341b21618066119c93p-110", "0x1f7050a436cdf5b072c2ca6ec8245p-111",
                 "-0x18bcbbfa352256a0d574ba8f60a4bp-116", "0x52c718ae9d41ce89f80abdae38c3p-109",
                 "0x819796d12db1fef78361e5d9c4bp-108",    "0x369dff745a1c6e91bd788be59567p-113",
                 "0x1f4cca885d7a2d0d6fb0832a9d95dp-110"},
          .exact_lo = "-0x581f9667cfe5933d76c728a8b277p-95",
          .exact_hi = "0x1e76bd353da223e8564f6904d07cfp-96"},
         {.h = "0x17142f5dd863af49c7650bf88486dp-116",
          .cf = {"-0x91cf934281533bc83ccf348b9f45p-107", "-0xbd1bbac8b21775944369e3154c59p-110",
                 "0xfb6c92bbf180cc6c558e900dd3d9p-113",  "-0x90339d904adf08c9de7f88fbd68bp-111",
                 "0x8b785a0c2cb3c35a3f7decef84dbp-111",  "0xce72232a911d9d93ae8133757b29p-107",
                 "0x995010f79b94f806c84326a93975p-114",  "-0x1784239a9f780807b8cf645a9ef25p-112",
                 "0x13a2ba615db4118f0156eec7926afp-115", "0x359bff6c5da457293f6b510b1ddfp-105",
                 "-0x12282d3073cdaaa1f7a4ca8ddf93p-106", "0x1f2fdab56babbaaa197a2b0719173p-108",
                 "0x100330edbc4c900887a302fc511d1p-113", "0x15cc139ece0a291aae2c58bb23c45p-114",
                 "-0x21e2bb60d22581744fe68cf4afbbp-106", "-0x1741b3830b3602c94d2f3d23f28b9p-115",
                 "0x937f910d09d3c19d46d00592f3e1p-115",  "-0x103aec2ccbac6eed220539369e7f5p-115",
                 "-0xd7acd449eb0633ad2ab06cea0ccfp-109", "-0x177c81629c79485db97224b10b1d3p-109",
                 "-0xb7fbce5dca0ff77538f2c9dc83cbp-107"},
          .exact_lo = "-0x93f12020828fd4022dff95b02d91p-107",
          .exact_hi = "-0x91cf934281533bc83ccf348b9f45p-107"},
         {.h = "-0xb074934b3e3ec743f2b7ea3ce287p-114",
          .cf = {"0x1d60cf6b34b5783eeaeba2cea67bfp-110",  "0x1cfe3eea8c6ee546bce18d2cfa653p-115",
                 "-0x172c5b6b6add524476c4cc4f3ae81p-114", "-0x1a820c0f54ba5093fa6981a064a35p-111",
                 "-0x17a398beafc377d0be2cf828dfa3dp-113", "-0x219ec521da179b579084ba583761p-113",
                 "-0x168525df72720602592a6e89b8731p-109", "0x9ca05e833a320eaecb3013961981p-108",
                 "0x13e7e4488970688e8bc61d98cf8efp-113",  "-0xa9d42fefc5453830a993ca3f14afp-115",
                 "0x17ea67f2981958c02d844783fc5dfp-116",  "-0xe9c08b84136242ec968b7eccb7f7p-108",
                 "0xbf75850b75cc997e31f2ec009bc7p-115",   "0xcb70d73fef7e26c8b5f6737ecfadp-109",
                 "0x8e7c2b158185afef0efe3316bd91p-110",   "-0x334864a6e9e26ede8bb31792879dp-107",
                 "-0x5d4521798ec7cff2f025cdb5f3abp-111",  "-0x1781eb4a80a3a3b7c015c3feb6c9fp-112",
                 "-0x8f54771739c49b0cd372fcd73cfp-106",   "0x16481b29d42dbddd5e43eac3e3723p-108",
                 "-0x1f54df4e759d08dc8e137c5798ce1p-112"},
          .exact_lo = "0xe96ea773800f36e8a741d5220881p-109",
          .exact_hi = "0x1d60cf6b34b5783eeaeba2cea67bfp-110"},
         {.h = "0x1a6c8d017c28139402a2d8e793affp-113",
          .cf = {"-0xeaeeae5798dd6b3e020237c00fcfp-113", "0x16c74b7636966d101c9b24ea193f1p-112",
                 "0xe39a80b97c42d9424fe305d356bdp-107",  "-0x791fe93f4e6e53809557b35d506dp-114",
                 "0x1af662e0bb1ac1716375299b4be6dp-108", "0x11c14bcd00f127885a2020c93a51bp-110",
                 "-0x69d62e5ee672aa3493639cde10c1p-112", "0x1788c409746e3c9a881e7d555bdc9p-116",
                 "0xfd3f48b6924157a238f43e526c25p-114",  "0x19235336abc335f81dd1813cfd89dp-114",
                 "0x13ed7965c8eaf138def975533389dp-110", "-0x1ceb94bebc2c4488128bfa6d609cdp-114",
                 "-0xffd8121a4cfff6ca36956d0b1bp-100",   "0x1c8323f079e3647d1516e8e1d60cfp-111",
                 "0x9a945e98c04dfd92d60d60655969p-114",  "0xf194b6656b6ce3f8fd25feb7642dp-114",
                 "0x1193903e057d816724d77ab9d2c8bp-110", "0x75723bac37a2aab23d177144cfb9p-107",
                 "0x12d988724afbe2dba6d27c27f580dp-115", "-0x163f9b3a2fce48fe3640963b3bb45p-116",
                 "-0x1704100ec9072af34e315d5210125p-110"},
          .exact_lo = "-0xeaeeae5798dd6b3e020237c00fcfp-113",
          .exact_hi = "0x11852c42b3c004c37752aa7784ca5p-107"},
         {.h = "-0x1990317f2efd6e20e6f22cee28a27p-115",
          .cf = {"0x1714aae1cc2af5a028a5d8a761fffp-110",  "0xdd9905706a2ad585b452117ea5fp-103",
                 "-0x1a3d8c8319eedd71b81e40887c5abp-108", "-0x92bbba27eb8f73100a502cbf6f97p-111",
                 "0xacb3c924431dec1d1381409c777dp-109",   "0x78309ab3898996c25419904ef437p-112",
                 "-0x1492aea23db9b9c7c8a30f9102661p-112", "0x1f3369754f3189ca404c6807b95f1p-111",
                 "0x8da500ca01e941b83005a58bd4fbp-114",   "0x9fb42b98cf19edff82cff416ef19p-107",
                 "0x1cc8d350b6df3a80f58cb2c3581bbp-110",  "-0x11bb30aaa5503438fcd71505d3dafp-109",
                 "0x11b50cb5b85feee41bd1dd989f6bfp-115",  "-0x1983b260da6001dfc2e7eacebd5c7p-109",
                 "-0x1173ee8a132d23645d97f5f61868bp-113", "-0xe8a313d0513744eb4189aaccf8ebp-108",
                 "-0xa66df9fa5284607e2c8981659e79p-113",  "0x1995fd726257eecfdc7ec06e38095p-111",
                 "-0x15eb2537323a31e3463c42af51063p-108", "-0x79aca9e8459c2a2f5c2433a8eb29p-109",
                 "-0x1393ab3cf803467e980a05a2138d5p-110"},
          .exact_lo = "-0xcef444499344347ccd616486c2f5p-112",
          .exact_hi = "0x1714aae1cc2af5a028a5d8a761fffp-110"},
     }},
};

// The polynomial degrees in the test cases.
const std::uint32_t degrees[] = {1, 2, 5, 20};

// Conversion of a hexadecimal floating-point string to T. The conversion is exact if the value is representable in T.
template <typename T>
T from_hex(const char *str)
{
    if constexpr (std::is_same_v<T, float>) {
        return std::strtof(str, nullptr);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::strtod(str, nullptr);
#if defined(HEYOKA_HAVE_REAL128)
    } else if constexpr (std::is_same_v<T, mppp::real128>) {
        return mppp::real128(str);
#endif
    } else {
        static_assert(std::is_same_v<T, long double>);
        return std::strtold(str, nullptr);
    }
}

// The test cases for the type T with polynomial degree n.
template <typename T>
std::vector<test_case> get_test_cases(std::uint32_t n)
{
    std::vector<test_case> retval;

    for (const auto &tc : test_cases.at(std::numeric_limits<T>::digits)) {
        if (tc.cf.size() == n + 1u) {
            retval.push_back(tc);
        }
    }

    REQUIRE(!retval.empty());

    return retval;
}

// The polynomial coefficients of a test case, converted to T.
template <typename T>
std::vector<T> get_cf(const test_case &tc)
{
    std::vector<T> retval;

    for (const auto *str : tc.cf) {
        retval.push_back(from_hex<T>(str));
    }

    return retval;
}

// Pointer to a fast exclusion check function.
template <typename T>
using fex_check_t = void (*)(const T *, const T *, std::uint32_t *, T *);

// Add a fast exclusion check function for polynomials of degree n to s, compile s and return a pointer to the function.
template <typename T>
fex_check_t<T> add_fex_check(llvm_state &s, std::uint32_t n, std::uint32_t batch_size)
{
    detail::llvm_ed_add_fex_check(s, "fex_check", detail::to_external_llvm_type<T>(s.context()), n, batch_size);

    s.compile();

    return reinterpret_cast<fex_check_t<T>>(s.jit_lookup("fex_check"));
}

// Check the output of a fast exclusion check function (the enclosure [lo, hi] and the flag out) for a test case.
template <typename T>
void check_output(const test_case &tc, T lo, T hi, std::uint32_t out)
{
    // The computed enclosure must contain the exact enclosure.
    REQUIRE(lo <= from_hex<T>(tc.exact_lo));
    REQUIRE(from_hex<T>(tc.exact_hi) <= hi);

    // out must be 1 if and only if the computed enclosure does not contain zero.
    REQUIRE(out == static_cast<std::uint32_t>(lo > 0 || hi < 0));
}

TEST_CASE("scalar")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 1u, 2u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};
                const auto fex_check = add_fex_check<fp_t>(s, n, 1);

                for (const auto &tc : get_test_cases<fp_t>(n)) {
                    const auto cf = get_cf<fp_t>(tc);
                    const auto h = from_hex<fp_t>(tc.h);

                    fp_t enc[2] = {};
                    std::uint32_t out = 0;
                    fex_check(cf.data(), &h, &out, enc);

                    check_output(tc, enc[0], enc[1], out);
                }
            }
        }
    });
}

TEST_CASE("batch")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        if (!ival_arith_supported<fp_t>()) {
            WARN("Skipping the test: the arithmetic does not honour the rounding mode on this platform");
            return;
        }

        for (const auto n : degrees) {
            const auto cur_test_cases = get_test_cases<fp_t>(n);
            const auto n_test_cases = cur_test_cases.size();

            for (const auto batch_size : {2u, 4u, 8u, 16u}) {
                for (const auto opt_level : {0u, 1u, 2u, 3u}) {
                    llvm_state s{kw::opt_level = opt_level};
                    const auto fex_check = add_fex_check<fp_t>(s, n, batch_size);

                    std::vector<fp_t> cf((n + 1u) * batch_size), h(batch_size), enc(2u * batch_size);
                    std::vector<std::uint32_t> out(batch_size);

                    // Batch lane j gets the test case with index (start + j) % n_test_cases. Forward and backward test
                    // cases alternate, hence each batch mixes the two integration directions.
                    for (std::size_t start = 0; start < n_test_cases; ++start) {
                        for (std::uint32_t j = 0; j < batch_size; ++j) {
                            const auto &tc = cur_test_cases[(start + j) % n_test_cases];
                            const auto tc_cf = get_cf<fp_t>(tc);

                            h[j] = from_hex<fp_t>(tc.h);
                            for (std::uint32_t k = 0; k <= n; ++k) {
                                cf[k * batch_size + j] = tc_cf[k];
                            }
                        }

                        fex_check(cf.data(), h.data(), out.data(), enc.data());

                        for (std::uint32_t j = 0; j < batch_size; ++j) {
                            const auto &tc = cur_test_cases[(start + j) % n_test_cases];

                            check_output(tc, enc[j], enc[batch_size + j], out[j]);
                        }
                    }
                }
            }
        }
    });
}

#if BOOST_ARCH_X86_64

// The types for which the fast exclusion check operates only on the MXCSR register (i.e., the types which are mapped to
// the LLVM float/double types, see llvm_set_rounding_up()).
template <typename T>
constexpr bool uses_mxcsr = detail::is_ieee754_binary32<T> || detail::is_ieee754_binary64<T>;

// Mask of the rounding control field (bits 13 and 14) in the MXCSR register.
constexpr unsigned mxcsr_rc_mask = 0x6000;

// Mask of all the bits of the MXCSR register except the rounding control field.
constexpr unsigned mxcsr_non_rc_mask = 0xFFFF9FFF;

// Values of the rounding control field for rounding up and down.
constexpr unsigned mxcsr_rc_up = 0x4000;
constexpr unsigned mxcsr_rc_down = 0x2000;

#endif

// Fetch the rounding mode for the floating-point type T in the same way as the fast exclusion check does (see
// llvm_set_rounding_up()): on x86-64 with float/double, the rounding control field of the MXCSR register, otherwise the
// value returned by std::fegetround().
template <typename T>
int get_rounding()
{
#if BOOST_ARCH_X86_64
    if constexpr (uses_mxcsr<T>) {
        return static_cast<int>(_mm_getcsr() & mxcsr_rc_mask);
    } else {
        return std::fegetround();
    }
#else
    return std::fegetround();
#endif
}

// Restore the rounding mode r returned by get_rounding() for the floating-point type T.
template <typename T>
void restore_rounding(int r)
{
#if BOOST_ARCH_X86_64
    if constexpr (uses_mxcsr<T>) {
        _mm_setcsr((_mm_getcsr() & mxcsr_non_rc_mask) | static_cast<unsigned>(r));
    } else {
        REQUIRE(std::fesetround(r) == 0);
    }
#else
    REQUIRE(std::fesetround(r) == 0);
#endif
}

// Set the rounding mode for the floating-point type T to rounding up, in the same way as get_rounding().
template <typename T>
void set_rounding_up()
{
#if BOOST_ARCH_X86_64
    if constexpr (uses_mxcsr<T>) {
        _mm_setcsr((_mm_getcsr() & mxcsr_non_rc_mask) | mxcsr_rc_up);
    } else {
        REQUIRE(std::fesetround(FE_UPWARD) == 0);
    }
#else
    REQUIRE(std::fesetround(FE_UPWARD) == 0);
#endif
}

// Set the rounding mode for the floating-point type T to rounding down, in the same way as get_rounding().
template <typename T>
void set_rounding_down()
{
#if BOOST_ARCH_X86_64
    if constexpr (uses_mxcsr<T>) {
        _mm_setcsr((_mm_getcsr() & mxcsr_non_rc_mask) | mxcsr_rc_down);
    } else {
        REQUIRE(std::fesetround(FE_DOWNWARD) == 0);
    }
#else
    REQUIRE(std::fesetround(FE_DOWNWARD) == 0);
#endif
}

TEST_CASE("caller rounding mode")
{
    tuple_for_each(fp_types, [](auto x) {
        using fp_t = decltype(x);

        const std::uint32_t n = 20;

        // NOTE: if AVX512 is available, the AVX512 implementation is used for float and double with batch size 1. The
        // portable implementation is always used with batch size 2.
        for (const auto batch_size : {1u, 2u}) {
            llvm_state s;
            const auto fex_check = add_fex_check<fp_t>(s, n, batch_size);

            for (const auto &tc : get_test_cases<fp_t>(n)) {
                // Put the test case in all batch lanes.
                const auto tc_cf = get_cf<fp_t>(tc);
                std::vector<fp_t> cf((n + 1u) * batch_size), h(batch_size, from_hex<fp_t>(tc.h));
                for (std::uint32_t k = 0; k <= n; ++k) {
                    for (std::uint32_t j = 0; j < batch_size; ++j) {
                        cf[k * batch_size + j] = tc_cf[k];
                    }
                }

                std::vector<fp_t> enc_orig(2u * batch_size), enc_up(2u * batch_size), enc_down(2u * batch_size);
                std::vector<std::uint32_t> out_orig(batch_size), out_up(batch_size), out_down(batch_size);

                // Run the fast exclusion check with the caller's original rounding mode.
                fex_check(cf.data(), h.data(), out_orig.data(), enc_orig.data());

                // The caller's original rounding mode.
                const auto orig_rounding = get_rounding<fp_t>();

                // NOTE: the test code does not perform any floating-point computation while the caller's rounding mode
                // is changed.

                // Run the fast exclusion check with the caller's rounding mode set to rounding up.
                set_rounding_up<fp_t>();
                const auto rounding_up = get_rounding<fp_t>();
                fex_check(cf.data(), h.data(), out_up.data(), enc_up.data());
                const auto rounding_up_after = get_rounding<fp_t>();

                // Run the fast exclusion check with the caller's rounding mode set to rounding down.
                set_rounding_down<fp_t>();
                const auto rounding_down = get_rounding<fp_t>();
                fex_check(cf.data(), h.data(), out_down.data(), enc_down.data());
                const auto rounding_down_after = get_rounding<fp_t>();

                restore_rounding<fp_t>(orig_rounding);

                // The caller's rounding mode must be restored.
                REQUIRE(rounding_up_after == rounding_up);
                REQUIRE(rounding_down_after == rounding_down);

                // The output must not depend on the caller's rounding mode.
                REQUIRE(enc_up == enc_orig);
                REQUIRE(out_up == out_orig);
                REQUIRE(enc_down == enc_orig);
                REQUIRE(out_down == out_orig);
            }
        }
    });
}

#if defined(HEYOKA_HAVE_REAL)

// NOTE: the multiprecision implementation uses MPFR with rounding toward +inf. All the values in the test cases are
// well inside the normal range, where MPFR at precision p rounds exactly like an IEEE format with p significand bits.
// Hence, the test cases for all precisions can be reused.
TEST_CASE("real")
{
    using fp_t = mppp::real;

    for (const auto &prec_cases : test_cases) {
        const auto prec = prec_cases.first;
        const auto &cases = prec_cases.second;

        // Conversion of a hexadecimal floating-point string to fp_t with precision prec. The conversion is exact if the
        // value is representable with prec bits.
        const auto from_hex_real = [prec](const char *str) { return fp_t{str, 16, prec}; };

        for (const auto n : degrees) {
            for (const auto opt_level : {0u, 3u}) {
                llvm_state s{kw::opt_level = opt_level};

                // NOTE: the function operates on mppp::real values with precision prec. All the mppp::real objects
                // passed to it (inputs and outputs) must have this precision.
                detail::llvm_ed_add_fex_check(s, "fex_check", detail::internal_llvm_type_like(s, fp_t{0, prec}), n, 1);

                s.compile();

                auto *fex_check = reinterpret_cast<void (*)(const fp_t *, const fp_t *, std::uint32_t *, fp_t *)>(
                    s.jit_lookup("fex_check"));

                for (const auto &tc : cases) {
                    if (tc.cf.size() != n + 1u) {
                        continue;
                    }

                    std::vector<fp_t> cf;
                    for (const auto *str : tc.cf) {
                        cf.push_back(from_hex_real(str));
                    }
                    const auto h = from_hex_real(tc.h);

                    std::vector<fp_t> enc(2u, fp_t{0, prec});
                    std::uint32_t out = 0;
                    fex_check(cf.data(), &h, &out, enc.data());

                    // The function must not alter the precision of the output values.
                    REQUIRE(enc[0].get_prec() == prec);
                    REQUIRE(enc[1].get_prec() == prec);

                    // The computed enclosure must contain the exact enclosure.
                    REQUIRE(enc[0] <= from_hex_real(tc.exact_lo));
                    REQUIRE(from_hex_real(tc.exact_hi) <= enc[1]);

                    // out must be 1 if and only if the computed enclosure does not contain zero.
                    REQUIRE(out == static_cast<std::uint32_t>(enc[0] > 0 || enc[1] < 0));
                }
            }
        }
    }
}

#endif
