#include "erfa.h"

int eraTpxes(double a, double b, double a0, double b0,
             double *xi, double *eta)
/*
**  - - - - - - - - -
**   e r a T p x e s
**  - - - - - - - - -
**
**  In the tangent plane projection, given celestial spherical
**  coordinates for a star and the tangent point, solve for the star's
**  rectangular coordinates in the tangent plane.
**
**  Given:
**     a,b       double  star's spherical coordinates
**     a0,b0     double  tangent point's spherical coordinates
**
**  Returned:
**     *xi,*eta  double  rectangular coordinates of star image (Note 2)
**
**  Returned (function value):
**               int     status:  0 = OK
**                                1 = star too far from axis
**                                2 = antistar on tangent plane
**                                3 = antistar too far from axis
**
**  Notes:
**
**  1) The tangent plane projection is also called the "gnomonic
**     projection" and the "central projection".
**
**  2) The eta axis points due north in the adopted coordinate system.
**     If the spherical coordinates are observed (RA,Dec), the tangent
**     plane coordinates (xi,eta) are conventionally called the
**     "standard coordinates".  For right-handed spherical coordinates,
**     (xi,eta) are also right-handed.  The units of (xi,eta) are,
**     effectively, radians at the tangent point.
**
**  3) All angular arguments are in radians.
**
**  4) This function is a member of the following set:
**
**         spherical      vector         solve for
**
**       > eraTpxes <    eraTpxev         xi,eta
**         eraTpsts      eraTpstv          star
**         eraTpors      eraTporv         origin
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
   const double TINY = 1e-6;
   int j;
   double sb0, sb, cb0, cb, da, sda, cda, d;


/* Functions of the spherical coordinates. */
   sb0 = sin(b0);
   sb = sin(b);
   cb0 = cos(b0);
   cb = cos(b);
   da = a - a0;
   sda = sin(da);
   cda = cos(da);

/* Reciprocal of star vector length to tangent plane. */
   d = sb*sb0 + cb*cb0*cda;

/* Check for error cases. */
   if ( d > TINY ) {
      j = 0;
   } else if ( d >= 0.0 ) {
      j = 1;
      d = TINY;
   } else if ( d > -TINY ) {
      j = 2;
      d = -TINY;
   } else {
      j = 3;
   }

/* Return the tangent plane coordinates (even in dubious cases). */
   *xi = cb*sda / d;
   *eta = (sb*cb0 - cb*sb0*cda) / d;

/* Return the status. */
   return j;

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
