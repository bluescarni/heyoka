#include "erfa.h"
#include "erfam.h"

double eraEpb(double dj1, double dj2)
/*
**  - - - - - - -
**   e r a E p b
**  - - - - - - -
**
**  Julian Date to Besselian Epoch.
**
**  Given:
**     dj1,dj2    double     Julian Date (Notes 3,4)
**
**  Returned (function value):
**                double     Besselian Epoch.
**
**  Notes:
**
**  1) Besselian Epoch is a method of expressing a moment in time as a
**     year plus fraction.  It was superseded by Julian Year (see the
**     function eraEpj).
**
**  2) The start of a Besselian year is when the right ascension of
**     the fictitious mean Sun is 18h 40m, and the unit is the tropical
**     year.  The conventional definition (see Lieske 1979) is that
**     Besselian Epoch B1900.0 is JD 2415020.31352 and the length of the
**     year is 365.242198781 days.
**
**  3) The time scale for the JD, originally Ephemeris Time, is TDB,
**     which for all practical purposes in the present context is
**     indistinguishable from TT.
**
**  4) The Julian Date is supplied in two pieces, in the usual ERFA
**     manner, which is designed to preserve time resolution.  The
**     Julian Date is available as a single number by adding dj1 and
**     dj2.  The maximum resolution is achieved if dj1 is 2451545.0
**     (J2000.0).
**
**  Reference:
**
**     Lieske, J.H., 1979. Astron.Astrophys., 73, 282.
**
**  This revision:  2023 May 5
**
**  Copyright (C) 2013-2023, NumFOCUS Foundation.
**  Derived, with permission, from the SOFA library.  See notes at end of file.
*/
{
/* J2000.0-B1900.0 (2415019.81352) in days */
   const double D1900 = 36524.68648;

   return 1900.0 + ((dj1 - ERFA_DJ00) + (dj2 + D1900)) / ERFA_DTY;

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
