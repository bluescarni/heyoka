/*
** Copyright (C) 2016-2017, NumFOCUS Foundation.
**
** Licensed under a 3-clause BSD style license - see LICENSE
**
** This file is NOT derived from SOFA sources
*/

/*
 * config.h defines the version information macros;
 * it is auto-generated in the autotools build process.
 * without it, the macros have to be defined explicitly
 * in the call to the compiler.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */


const char* eraVersion(void) {
  return PACKAGE_VERSION;
}


int eraVersionMajor(void) {
  return PACKAGE_VERSION_MAJOR;
}


int eraVersionMinor(void) {
  return PACKAGE_VERSION_MINOR;
}


int eraVersionMicro(void) {
  return PACKAGE_VERSION_MICRO;
}


const char* eraSofaVersion(void) {
  return SOFA_VERSION;
}
