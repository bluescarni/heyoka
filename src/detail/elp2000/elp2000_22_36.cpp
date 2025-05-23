// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdint>

#include <heyoka/config.hpp>
#include <heyoka/detail/elp2000/elp2000_22_36.hpp>

HEYOKA_BEGIN_NAMESPACE

namespace model::detail
{

const std::int8_t elp2000_idx_22[3][5] = {{0, 1, 1, -1, -1}, {0, 1, 1, 0, -1}, {0, 1, 1, 1, -1}};
const double elp2000_phi_A_22[3][2] = {{3.3673797902679174, 1.9392547244381442e-10},
                                       {3.3673797902679174, 3.9754721850981949e-09},
                                       {3.3673797902679174, 1.9392547244381442e-10}};
const std::int8_t elp2000_idx_23[2][5] = {{0, 1, 1, 0, -2}, {0, 1, 1, 0, 0}};
const double elp2000_phi_A_23[2][2]
    = {{3.3673794412020674, 1.9392547244381442e-10}, {3.3673796157349924, 1.9392547244381442e-10}};
const std::int8_t elp2000_idx_24[2][5] = {{0, 1, 1, -1, -1}, {0, 1, 1, 1, -1}};
const double elp2000_phi_A_24[2][2]
    = {{4.938176117062814, 4.0000000000000003e-05}, {1.7965834634730211, 4.0000000000000003e-05}};
const std::int8_t elp2000_idx_25[6][5]
    = {{0, 0, 0, 1, 0}, {0, 0, 0, 2, 0}, {0, 2, 0, -2, 0}, {0, 2, 0, -1, 0}, {0, 2, 0, 0, 0}, {0, 2, 0, 1, 0}};
const double elp2000_phi_A_25[6][2]
    = {{0, 2.8119193504353089e-09}, {0, 1.9392547244381442e-10}, {0, 9.696273622190721e-11},
       {0, 1.0181087303300257e-09}, {0, 4.3633231299858244e-10}, {0, 4.8481368110953605e-11}};
const std::int8_t elp2000_idx_26[4][5] = {{0, 0, 0, 0, 1}, {0, 0, 0, 1, -1}, {0, 0, 0, 1, 1}, {0, 2, 0, 0, -1}};
const double elp2000_phi_A_26[4][2] = {{3.1415926535897931, 2.4240684055476799e-10},
                                       {0, 1.4544410433286079e-10},
                                       {0, 1.4544410433286079e-10},
                                       {0, 4.8481368110953605e-11}};
const std::int8_t elp2000_idx_27[5][5]
    = {{0, 0, 0, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 2, 0}, {0, 2, 0, -1, 0}, {0, 2, 0, 0, 0}};
const double elp2000_phi_A_27[5][2] = {{1.5707963267948966, 0.0035599999999999998},
                                       {4.7123889803846897, 0.00072000000000000005},
                                       {4.7123889803846897, 3.0000000000000001e-05},
                                       {4.7123889803846897, 0.00019000000000000001},
                                       {4.7123889803846897, 0.00012999999999999999}};
const std::int8_t elp2000_idx_28[20][5]
    = {{0, 0, 0, 0, 1},   {0, 0, 0, 1, -1}, {0, 0, 0, 2, -2}, {0, 0, 0, 3, -2}, {0, 0, 1, -1, 0},
       {0, 0, 1, 0, 0},   {0, 0, 1, 1, 0},  {0, 1, 0, -1, 0}, {0, 1, 0, 0, 0},  {0, 1, 1, -1, 0},
       {0, 2, -1, -1, 0}, {0, 2, -1, 0, 0}, {0, 2, 0, -3, 0}, {0, 2, 0, -2, 0}, {0, 2, 0, -1, 0},
       {0, 2, 0, 0, -2},  {0, 2, 0, 0, 0},  {0, 2, 1, -2, 0}, {0, 2, 1, -1, 0}, {0, 2, 1, 0, 0}};
const double elp2000_phi_A_28[20][2]
    = {{5.3051350829531261, 1.9392547244381442e-10},    {4.5358302515224675, 7.7570188977525768e-10},
       {0.007508406442079606, 1.9392547244381439e-09},  {0.0075710637622262026, 9.696273622190721e-11},
       {6.2831605235042085, 6.7873915355335027e-10},    {6.2831821655869327, 1.0811345088742653e-08},
       {6.2831785003955041, 6.7873915355335027e-10},    {6.283068544652628, 4.3633231299858244e-10},
       {6.2831044984352191, 4.8481368110953605e-11},    {0.0011201523139299607, 1.4544410433286079e-10},
       {3.1416092342176869, 1.9392547244381442e-10},    {3.1415950970507458, 1.4544410433286079e-10},
       {3.1412655788879693, 4.8481368110953605e-11},    {3.1413074667900172, 1.2120342027738401e-09},
       {3.1415294726708707, 6.7873915355335027e-10},    {3.1408707854111686, 1.4544410433286079e-10},
       {3.1415758984289743, 9.696273622190721e-11},     {3.1414502347228304, 9.696273622190721e-11},
       {5.4628805587422514e-05, 9.696273622190721e-11}, {6.2831791985272041, 9.696273622190721e-11}};
const std::int8_t elp2000_idx_29[12][5]
    = {{0, 0, 0, 1, -1}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 0, 2, -3},  {0, 0, 0, 2, -1}, {0, 0, 1, -1, -1},
       {0, 0, 1, 0, -1}, {0, 0, 1, 0, 1}, {0, 0, 1, 1, 1}, {0, 2, 0, -2, -1}, {0, 2, 0, -2, 1}, {0, 2, 0, 0, -1}};
const double elp2000_phi_A_29[12][2]
    = {{0.00038379790251355305, 1.4544410433286079e-10}, {4.2933471206868399, 4.8481368110953605e-11},
       {9.2502450355699473e-05, 4.8481368110953605e-11}, {0.0073797756762076236, 9.696273622190721e-11},
       {0.013003575591983752, 4.8481368110953605e-11},   {6.2831821655869327, 4.8481368110953605e-11},
       {6.2831821655869327, 4.8481368110953598e-10},     {6.2831821655869327, 4.8481368110953598e-10},
       {6.2831821655869327, 4.8481368110953605e-11},     {3.1413057214607654, 4.8481368110953605e-11},
       {3.1413051978619899, 4.8481368110953605e-11},     {3.1415015474028389, 2.4240684055476799e-10}};
const std::int8_t elp2000_idx_30[14][5]
    = {{0, 0, 0, 0, 0},  {0, 0, 0, 0, 1},  {0, 0, 0, 0, 2},  {0, 0, 0, 1, 0},   {0, 0, 0, 3, -2},
       {0, 0, 1, -1, 0}, {0, 0, 1, 0, 0},  {0, 0, 1, 1, 0},  {0, 2, -1, -1, 0}, {0, 2, -1, 0, 0},
       {0, 2, 0, -2, 0}, {0, 2, 0, -1, 0}, {0, 2, 1, -1, 0}, {0, 2, 1, 0, 0}};
const double elp2000_phi_A_30[14][2]
    = {{1.5707963267948966, 0.0012999999999999999},  {3.7342575983480115, 3.0000000000000001e-05},
       {4.7130426061895614, 2.0000000000000002e-05}, {1.5721222534276367, 4.0000000000000003e-05},
       {4.7199687707931757, 2.0000000000000002e-05}, {1.5707821896279555, 0.00012999999999999999},
       {4.7123902021151665, 0.00022000000000000001}, {4.7123720506909459, 0.00011},
       {1.5707640382037347, 2.0000000000000002e-05}, {1.5708054025070068, 3.0000000000000001e-05},
       {4.712142016295533, 5.0000000000000002e-05},  {1.5707724157841443, 0.00012999999999999999},
       {4.7123858387920361, 2.0000000000000002e-05}, {4.7123858387920361, 3.0000000000000001e-05}};
const std::int8_t elp2000_idx_31[11][5]
    = {{0, 0, 1, -1, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 1, 0}, {0, 1, 0, 0, 0},  {0, 1, 1, 0, 0}, {0, 2, -1, -1, 0},
       {0, 2, 0, -1, 0}, {0, 2, 0, 0, 0}, {0, 2, 0, 1, 0}, {0, 2, 1, -1, 0}, {0, 4, 0, -1, 0}};
const double elp2000_phi_A_31[11][2]
    = {{3.1404534771870165, 2.9088820866572159e-10}, {3.1413364392556002, 3.9269908169872414e-09},
       {3.1409508960238353, 2.4240684055476799e-10}, {1.7453292519943297e-07, 6.3025778544239667e-10},
       {3.1419909377250983, 4.8481368110953605e-11}, {0.00039514254265151624, 9.696273622190721e-11},
       {6.2829804055254028, 9.696273622190721e-11},  {3.1415959697153721, 2.666475246102448e-09},
       {3.1415956206495217, 2.9088820866572159e-10}, {3.1546745944651913, 4.8481368110953605e-11},
       {3.1415987622421753, 4.8481368110953605e-11}};
const std::int8_t elp2000_idx_32[4][5] = {{0, 0, 1, 0, -1}, {0, 0, 1, 0, 1}, {0, 2, 0, 0, -1}, {0, 2, 0, 0, 1}};
const double elp2000_phi_A_32[4][2] = {{3.1415582706035288, 1.9392547244381442e-10},
                                       {3.1415573979389033, 1.9392547244381442e-10},
                                       {6.2831521459237987, 9.696273622190721e-11},
                                       {3.1415971914458485, 9.696273622190721e-11}};
const std::int8_t elp2000_idx_33[10][5]
    = {{0, 0, 0, 0, 0}, {0, 0, 0, 1, 0},  {0, 0, 1, -1, 0}, {0, 0, 1, 0, 0}, {0, 0, 1, 1, 0},
       {0, 1, 0, 0, 0}, {0, 2, -1, 0, 0}, {0, 2, 0, -1, 0}, {0, 2, 0, 0, 0}, {0, 2, 0, 1, 0}};
const double elp2000_phi_A_33[10][2] = {{4.7123889803846897, 0.0082799999999999992},
                                        {1.5707952795973452, 0.00042999999999999999},
                                        {4.7112182135224527, 5.0000000000000002e-05},
                                        {4.7125474562807703, 9.0000000000000006e-05},
                                        {1.5700571798566769, 5.0000000000000002e-05},
                                        {4.7123893294505406, 6.0000000000000002e-05},
                                        {1.5702851198569874, 2.0000000000000002e-05},
                                        {4.7122785010430386, 3.0000000000000001e-05},
                                        {1.5707987702558495, 0.00106},
                                        {1.5707980721241486, 8.0000000000000007e-05}};
const std::int8_t elp2000_idx_34[28][5]
    = {{0, 0, 1, -2, 0},  {0, 0, 1, -1, 0},  {0, 0, 1, 0, 0},   {0, 0, 1, 1, 0},   {0, 0, 1, 2, 0},  {0, 0, 2, -1, 0},
       {0, 0, 2, 0, 0},   {0, 0, 2, 1, 0},   {0, 1, 1, 0, 0},   {0, 2, -2, -1, 0}, {0, 2, -2, 0, 0}, {0, 2, -2, 1, 0},
       {0, 2, -1, -2, 0}, {0, 2, -1, -1, 0}, {0, 2, -1, 0, -2}, {0, 2, -1, 0, 0},  {0, 2, -1, 1, 0}, {0, 2, 0, -1, 0},
       {0, 2, 0, 0, 0},   {0, 2, 1, -2, 0},  {0, 2, 1, -1, 0},  {0, 2, 1, 0, -2},  {0, 2, 1, 0, 0},  {0, 2, 1, 1, 0},
       {0, 2, 2, -1, 0},  {0, 4, -1, -2, 0}, {0, 4, -1, -1, 0}, {0, 4, -1, 0, 0}};
const double elp2000_phi_A_34[28][2] = {{0, 3.3936957677667514e-10},
                                        {0, 5.235987755982989e-09},
                                        {0, 2.3610426270034402e-08},
                                        {0, 3.8785094488762879e-09},
                                        {0, 2.9088820866572159e-10},
                                        {0, 1.9392547244381442e-10},
                                        {0, 5.3329504922048958e-10},
                                        {0, 9.696273622190721e-11},
                                        {3.1415926535897931, 6.3025778544239667e-10},
                                        {3.1415926535897931, 5.3329504922048958e-10},
                                        {3.1415926535897931, 5.8177641733144318e-10},
                                        {3.1415926535897931, 4.8481368110953605e-11},
                                        {3.1415926535897931, 2.9088820866572159e-10},
                                        {3.1415926535897931, 7.27220521664304e-09},
                                        {3.1415926535897931, 9.696273622190721e-11},
                                        {3.1415926535897931, 5.817764173314431e-09},
                                        {3.1415926535897931, 5.3329504922048958e-10},
                                        {0, 9.696273622190721e-11},
                                        {0, 1.4544410433286079e-10},
                                        {3.1415926535897931, 9.696273622190721e-11},
                                        {0, 1.0181087303300257e-09},
                                        {0, 4.8481368110953605e-11},
                                        {0, 8.7266462599716487e-10},
                                        {0, 9.696273622190721e-11},
                                        {0, 1.9392547244381442e-10},
                                        {3.1415926535897931, 9.696273622190721e-11},
                                        {3.1415926535897931, 1.4544410433286079e-10},
                                        {3.1415926535897931, 4.8481368110953605e-11}};
const std::int8_t elp2000_idx_35[13][5]
    = {{0, 0, 1, -1, -1}, {0, 0, 1, -1, 1},  {0, 0, 1, 0, -1},   {0, 0, 1, 0, 1},   {0, 0, 1, 1, -1},
       {0, 0, 1, 1, 1},   {0, 2, -2, 0, -1}, {0, 2, -1, -1, -1}, {0, 2, -1, -1, 1}, {0, 2, -1, 0, -1},
       {0, 2, -1, 0, 1},  {0, 2, -1, 1, -1}, {0, 2, 1, 0, -1}};
const double elp2000_phi_A_35[13][2] = {{0, 2.4240684055476799e-10},
                                        {0, 1.9392547244381442e-10},
                                        {0, 1.9392547244381442e-10},
                                        {0, 2.4240684055476799e-10},
                                        {0, 1.9392547244381442e-10},
                                        {0, 1.9392547244381442e-10},
                                        {3.1415926535897931, 9.696273622190721e-11},
                                        {3.1415926535897931, 2.4240684055476799e-10},
                                        {3.1415926535897931, 2.9088820866572159e-10},
                                        {3.1415926535897931, 1.0665900984409792e-09},
                                        {3.1415926535897931, 2.9088820866572159e-10},
                                        {3.1415926535897931, 4.8481368110953605e-11},
                                        {0, 4.3633231299858244e-10}};
const std::int8_t elp2000_idx_36[19][5]
    = {{0, 0, 1, -2, 0},  {0, 0, 1, -1, 0}, {0, 0, 1, 0, 0},   {0, 0, 1, 1, 0},  {0, 0, 1, 2, 0},
       {0, 0, 2, -1, 0},  {0, 1, 1, 0, 0},  {0, 2, -2, -1, 0}, {0, 2, -2, 0, 0}, {0, 2, -1, -2, 0},
       {0, 2, -1, -1, 0}, {0, 2, -1, 0, 0}, {0, 2, -1, 1, 0},  {0, 2, 0, 0, 0},  {0, 2, 1, -1, 0},
       {0, 2, 1, 0, 0},   {0, 2, 1, 1, 0},  {0, 2, 2, -1, 0},  {0, 4, -1, -1, 0}};
const double elp2000_phi_A_36[19][2] = {{1.5707963267948966, 5.0000000000000002e-05},
                                        {1.5707963267948966, 0.00095},
                                        {4.7123889803846897, 0.00036000000000000002},
                                        {4.7123889803846897, 0.00076999999999999996},
                                        {4.7123889803846897, 4.0000000000000003e-05},
                                        {1.5707963267948966, 3.0000000000000001e-05},
                                        {1.5707963267948966, 0.00012},
                                        {1.5707963267948966, 6.9999999999999994e-05},
                                        {1.5707963267948966, 0.00013999999999999999},
                                        {4.7123889803846897, 6.9999999999999994e-05},
                                        {1.5707963267948966, 0.0011100000000000001},
                                        {1.5707963267948966, 0.00149},
                                        {1.5707963267948966, 9.0000000000000006e-05},
                                        {4.7123889803846897, 4.0000000000000003e-05},
                                        {4.7123889803846897, 0.00018000000000000001},
                                        {4.7123889803846897, 0.00023000000000000001},
                                        {4.7123889803846897, 2.0000000000000002e-05},
                                        {4.7123889803846897, 3.0000000000000001e-05},
                                        {1.5707963267948966, 3.0000000000000001e-05}};

} // namespace model::detail

HEYOKA_END_NAMESPACE
