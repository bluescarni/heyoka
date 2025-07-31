// Copyright 2020-2025 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef HEYOKA_DETAIL_ERFA_DECLS_HPP
#define HEYOKA_DETAIL_ERFA_DECLS_HPP

// NOTE: this header contains declarations of erfa functions used within heyoka.

extern "C" {
int eraUtctai(double, double, double *, double *);
int eraTaitt(double, double, double *, double *);
int eraUtcut1(double, double, double, double *, double *);
int eraTaiutc(double, double, double *, double *);
double eraEra00(double, double);
int eraCal2jd(int, int, int, double *, double *);
int eraJd2cal(double, double, int *, int *, int *, double *);
int eraTttai(double, double, double *, double *);
}

#endif
