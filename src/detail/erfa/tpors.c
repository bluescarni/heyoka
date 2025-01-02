#include "erfa.h"

int eraTpors(double xi, double eta, double a, double b,
             double *a01, double *b01, double *a02, double *b02)
/*
**  - - - - - - - - -
**   e r a T p o r s
**  - - - - - - - - -
**
**  In the tangent plane projection, given the rectangular coordinates
**  of a star and its spherical coordinates, determine the spherical
**  coordinates of the tangent point.
**
**  Given:
**     xi,eta     double  rectangular coordinates of star image (Note 2)
**     a,b        double  star's spherical coordinates (Note 3)
**
**  Returned:
**     *a01,*b01  double  tangent point's spherical coordinates, Soln. 1
**     *a02,*b02  double  tangent point's spherical coordinates, Soln. 2
**
**  Returned (function value):
**                int     number of solutions:
**                        0 = no solutions returned (Note 5)
**                        1 = only the first solution is useful (Note 6)
**                        2 = both solutions are useful (Note 6)
**
**  Notes:
**
**  1) The tangent plane projection is also called the "gnomonic
**     projection" and the "central projection".
**
**  2) The eta axis points due north in the adopted coordinate system.
**     If the spherical coordinates are observed (RA,Dec), the tangent
**     plane coordinates (xi,eta) are conventionally called the
**     "standard coordinates".  If the spherical coordinates are with
**     respect to a right-handed triad, (xi,eta) are also right-handed.
**     The units of (xi,eta) are, effectively, radians at the tangent
**     point.
**
**  3) All angular arguments are in radians.
**
**  4) The angles a01 and a02 are returned in the range 0-2pi.  The
**     angles b01 and b02 are returned in the range +/-pi, but in the
**     usual, non-pole-crossing, case, the range is +/-pi/2.
**
**  5) Cases where there is no solution can arise only near the poles.
**     For example, it is clearly impossible for a star at the pole
**     itself to have a non-zero xi value, and hence it is meaningless
**     to ask where the tangent point would have to be to bring about
**     this combination of xi and dec.
**
**  6) Also near the poles, cases can arise where there are two useful
**     solutions.  The return value indicates whether the second of the
**     two solutions returned is useful;  1 indicates only one useful
**     solution, the usual case.
**
**  7) The basis of the algorithm is to solve the spherical triangle PSC,
**     where P is the north celestial pole, S is the star and C is the
**     tangent point.  The spherical coordinates of the tangent point are
**     [a0,b0];  writing rho^2 = (xi^2+eta^2) and r^2 = (1+rho^2), side c
**     is then (pi/2-b), side p is sqrt(xi^2+eta^2) and side s (to be
**     found) is (pi/2-b0).  Angle C is given by sin(C) = xi/rho and
**     cos(C) = eta/rho.  Angle P (to be found) is the longitude
**     difference between star and tangent point (a-a0).
**
**  8) This function is a member of the following set:
**
**         spherical      vector         solve for
**
**         eraTpxes      eraTpxev         xi,eta
**         eraTpsts      eraTpstv          star
**       > eraTpors <    eraTporv         origin
**
**  Called:
**     eraAnp       normalize angle into range 0 to 2pi
**
**  References:
**
**     Calabretta M.R. & Greisen, E.W., 2002, "Representations of
**     celestial coordinates in FITS", Astron.Astrophys. 395, 1077
**
**     Green, R.M., "Spherical Astronomy", Cambridge University Press,
**     1987, Chapter 13.
**
**  This revision:   2018 January 2
**
**  Copyright (C) 2013-2023, NumFOCUS Foundation.
**  Derived, with permission, from the SOFA library.  See notes at end of file.
*/
{
   double xi2, r, sb, cb, rsb, rcb, w2, w, s, c;


   xi2 = xi*xi;
   r = sqrt(1.0 + xi2 + eta*eta);
   sb = sin(b);
   cb = cos(b);
   rsb = r*sb;
   rcb = r*cb;
   w2 = rcb*rcb - xi2;
   if ( w2 >= 0.0 ) {
      w = sqrt(w2);
      s = rsb - eta*w;
      c = rsb*eta + w;
      if ( xi == 0.0 && w == 0.0 ) w = 1.0;
      *a01 = eraAnp(a - atan2(xi,w));
      *b01 = atan2(s,c);
      w = -w;
      s = rsb - eta*w;
      c = rsb*eta + w;
      *a02 = eraAnp(a - atan2(xi,w));
      *b02 = atan2(s,c);
      return (fabs(rsb) < 1.0) ? 1 : 2;
   } else {
      return 0;
   }

/* Finished. */

}
/*----------------------------------------------------------------------
**  
**  
**  Copyright (C) 2013-2023, NumFOCUS Foundation.
**  All rights reserved.
**  
**  This library is derived, with permission, from the International
**  Astronomical Union's "Standards of Fundamental Astronomy" library,
**  available from http://www.iausofa.org.
**  
**  The ERFA version is intended to retain identical functionality to
**  the SOFA library, but made distinct through different function and
**  file names, as set out in the SOFA license conditions.  The SOFA
**  original has a role as a reference standard for the IAU and IERS,
**  and consequently redistribution is permitted only in its unaltered
**  state.  The ERFA version is not subject to this restriction and
**  therefore can be included in distributions which do not support the
**  concept of "read only" software.
**  
**  Although the intent is to replicate the SOFA API (other than
**  replacement of prefix names) and results (with the exception of
**  bugs;  any that are discovered will be fixed), SOFA is not
**  responsible for any errors found in this version of the library.
**  
**  If you wish to acknowledge the SOFA heritage, please acknowledge
**  that you are using a library derived from SOFA, rather than SOFA
**  itself.
**  
**  
**  TERMS AND CONDITIONS
**  
**  Redistribution and use in source and binary forms, with or without
**  modification, are permitted provided that the following conditions
**  are met:
**  
**  1 Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
**  
**  2 Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in
**    the documentation and/or other materials provided with the
**    distribution.
**  
**  3 Neither the name of the Standards Of Fundamental Astronomy Board,
**    the International Astronomical Union nor the names of its
**    contributors may be used to endorse or promote products derived
**    from this software without specific prior written permission.
**  
**  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
**  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
**  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
**  FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
**  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
**  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
**  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
**  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
**  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
**  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
**  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
**  POSSIBILITY OF SUCH DAMAGE.
**  
*/
