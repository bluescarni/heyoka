#include <stdexcept>
#include <string>

#include "SGP4.h"

#define pi 3.14159265358979323846

const auto deg2rad = [](auto x) { return x * 2. * pi / 360.; };

const auto revday2radmin = [](auto x) { return x * 2. * pi / 1440.; };

// NOTE: this main function uses the official code from
//
// https://celestrak.org/software/vallado-sw.php
//
// to run a couple of propagations for validation purposes.
int main() {
  std::cout.precision(16);

  elsetrec satrec;

  {
    const auto tle_y = 2024;
    const auto tle_dayfrac = 177.64786330;

    int mon, day, hr, minute;
    double sec;

    SGP4Funcs::days2mdhms_SGP4(tle_y, tle_dayfrac, mon, day, hr, minute, sec);

    double jd, jdFrac;

    SGP4Funcs::jday_SGP4(tle_y, mon, day, hr, minute, sec, jd, jdFrac);

    std::cout << "TLE jd=" << jd << ", jdFrac=" << jdFrac << '\n';

    const auto status = SGP4Funcs::sgp4init(
        wgs72, 'a', "00900", jd + jdFrac, .75863e-3, .00000735, 0., .0024963,
        deg2rad(320.5956), deg2rad(90.2039), deg2rad(91.4738),
        revday2radmin(13.75091047972192), deg2rad(55.5633), satrec);

    if (satrec.error != 0) {
      throw std::invalid_argument("sgp4init() returned error code " +
                                  std::to_string(satrec.error));
    }

    double r[3], v[3];

    SGP4Funcs::sgp4(satrec, 1440., r, v);

    std::cout << r[0] << '\n';
    std::cout << r[1] << '\n';
    std::cout << r[2] << '\n';

    std::cout << v[0] << '\n';
    std::cout << v[1] << '\n';
    std::cout << v[2] << '\n';
  }

  {
    const auto tle_y = 2019;
    const auto tle_dayfrac = 343.69339541;

    int mon, day, hr, minute;
    double sec;

    SGP4Funcs::days2mdhms_SGP4(tle_y, tle_dayfrac, mon, day, hr, minute, sec);

    double jd, jdFrac;

    SGP4Funcs::jday_SGP4(tle_y, mon, day, hr, minute, sec, jd, jdFrac);

    std::cout << "TLE jd=" << jd << ", jdFrac=" << jdFrac << '\n';

    const auto status = SGP4Funcs::sgp4init(
        wgs72, 'a', "25544", jd + jdFrac, .38792e-4, .00001764, 0., .0007417,
        deg2rad(17.6667), deg2rad(51.6439), deg2rad(85.6398),
        revday2radmin(15.50103472202482), deg2rad(211.2001), satrec);

    if (satrec.error != 0) {
      throw std::invalid_argument("sgp4init() returned error code " +
                                  std::to_string(satrec.error));
    }

    double r[3], v[3];

    SGP4Funcs::sgp4(satrec, 1440., r, v);

    std::cout << r[0] << '\n';
    std::cout << r[1] << '\n';
    std::cout << r[2] << '\n';

    std::cout << v[0] << '\n';
    std::cout << v[1] << '\n';
    std::cout << v[2] << '\n';
  }
}
