// Copyright 2020-2026 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the heyoka library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <vector>

#include <heyoka/detail/debug.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/kw.hpp>
#include <heyoka/model/egm2008.hpp>
#include <heyoka/model/eo_dynamics.hpp>
#include <heyoka/taylor.hpp>

#include "catch.hpp"
#include "test_utils.hpp"

using namespace heyoka;
using namespace heyoka_test;

TEST_CASE("default")
{
    // Default-constructed eo_dynamics: no kwargs should give Keplerian dynamics using Earth's gravitational parameter
    // from EGM2008.
    auto dyn = model::eo_dynamics();

    // Six equations.
    REQUIRE(dyn.size() == 6u);

    // LHS of each equation: positions in the first three, velocities in the last three.
    REQUIRE(dyn[0].first == "x"_var);
    REQUIRE(dyn[1].first == "y"_var);
    REQUIRE(dyn[2].first == "z"_var);
    REQUIRE(dyn[3].first == "vx"_var);
    REQUIRE(dyn[4].first == "vy"_var);
    REQUIRE(dyn[5].first == "vz"_var);

    // RHS of the first three equations: velocity passthrough.
    REQUIRE(dyn[0].second == "vx"_var);
    REQUIRE(dyn[1].second == "vy"_var);
    REQUIRE(dyn[2].second == "vz"_var);

    // RHS of the last three equations: Keplerian acceleration -mu_E r / |r|^3. Build a cfunc and evaluate at a test
    // point, comparing against the analytical formula.
    auto [x, y, z] = make_vars("x", "y", "z");
    cfunc<double> acc_cf{{dyn[3].second, dyn[4].second, dyn[5].second}, std::vector{x, y, z}};

    // Earth's gravitational parameter in km^3/s^2 (matches eo_dynamics' internal rescaling).
    const double earth_mu = model::get_egm2008_mu() / 1e9;

    // A test point off-axis so all three components are non-trivial.
    const std::array<double, 3> pos{7000.0, 1000.0, -2000.0};
    const auto r2 = (pos[0] * pos[0]) + (pos[1] * pos[1]) + (pos[2] * pos[2]);
    const auto r_inv3 = 1.0 / (r2 * std::sqrt(r2));

    std::array<double, 3> out{};
    acc_cf(out, pos);

    REQUIRE(out[0] == approximately(-earth_mu * pos[0] * r_inv3, 100.));
    REQUIRE(out[1] == approximately(-earth_mu * pos[1] * r_inv3, 100.));
    REQUIRE(out[2] == approximately(-earth_mu * pos[2] * r_inv3, 100.));
}

// Under J2-only geopotential, the orbital plane precesses about the Earth's rotation axis at the secular rate
//
//     dΩ/dt = -(3/2) n J2 (R_⊕/p)^2 cos(i)
//
// where n is the mean motion, p the semi-latus rectum, i the inclination. We integrate circular LEOs at several
// inclinations for one day and check the numerical RAAN drift against this formula, covering:
//   - two prograde cases (i < 90°) to probe the cos(i) scaling
//   - i = 90° where the secular rate vanishes
//   - a retrograde case (i > 90°) where cos(i) flips sign
TEST_CASE("J2 nodal precession")
{
    detail::edb_disabler ed;

    // J2-only geopotential.
    auto dyn = model::eo_dynamics(kw::max_geo_degree = 2u, kw::max_geo_order = 0u);
    REQUIRE(dyn.size() == 6u);

    // Constants in km, s.
    const double earth_mu = model::get_egm2008_mu() / 1e9;
    const double earth_a = model::get_egm2008_a() / 1e3;
    // Unnormalised J2 from EGM2008 (J2 = -sqrt(5) C_{2,0}).
    const double J2 = 1.0826267e-3;

    // Circular LEO at a = 7000 km.
    const double a = 7000.0;
    const double v_circ = std::sqrt(earth_mu / a);
    const double n = std::sqrt(earth_mu / (a * a * a));
    constexpr double duration = 86400.0;

    // Helper: propagate a circular orbit at the given inclination for `duration` seconds and return the final RAAN. ICs
    // place the spacecraft at the ascending node on the +x axis, so Ω_0 = 0.
    auto propagate_and_measure_Omega = [&](double inc) {
        const std::vector<double> ic{
            a, 0.0, 0.0, 0.0, v_circ * std::cos(inc), v_circ * std::sin(inc),
        };

        taylor_adaptive<double> ta{dyn, ic, kw::compact_mode = true};
        ta.propagate_until(duration);

        const auto &s = ta.get_state();
        const double hx = (s[1] * s[5]) - (s[2] * s[4]);
        const double hy = (s[2] * s[3]) - (s[0] * s[5]);
        return std::atan2(hx, -hy);
    };

    for (const double inc_deg : {30.0, 60.0, 90.0, 150.0}) {
        const double inc = inc_deg * std::numbers::pi_v<double> / 180.0;
        const double Omega_final = propagate_and_measure_Omega(inc);

        const double Omega_dot = -1.5 * n * J2 * (earth_a / a) * (earth_a / a) * std::cos(inc);
        const double Omega_expected = Omega_dot * duration;

        if (std::abs(std::cos(inc)) < 1e-10) {
            // Polar orbit: the secular rate vanishes exactly. Check absolutely - short-period residual should
            // be well below 1 mrad over one day.
            REQUIRE(std::abs(Omega_final) < 1e-3);
        } else {
            REQUIRE(std::abs((Omega_final - Omega_expected) / Omega_expected) < 0.02);
        }
    }
}

// Under J2-only geopotential, the argument of perigee precesses at the secular rate
//
//     dω/dt = (3/4) n J2 (R_⊕/p)^2 (5 cos²i − 1)
//
// We integrate elliptical LEOs at several inclinations for a couple of days and check the numerical drift against this
// formula, covering:
//   - two prograde cases (i < i_c) where ω̇ > 0
//   - the critical inclination i_c = arccos(1/√5) ≈ 63.4°, where the first-order secular rate vanishes exactly
//   - i = 90° (polar) where ω̇ < 0
TEST_CASE("J2 perigee precession")
{
    detail::edb_disabler ed;

    // J2-only geopotential.
    auto dyn = model::eo_dynamics(kw::max_geo_degree = 2u, kw::max_geo_order = 0u);
    REQUIRE(dyn.size() == 6u);

    // Constants in km, s.
    const double earth_mu = model::get_egm2008_mu() / 1e9;
    const double earth_a = model::get_egm2008_a() / 1e3;
    const double J2 = 1.0826267e-3;

    const double a = 8000.0;
    const double e_init = 0.1;
    const double r_p = a * (1.0 - e_init);
    const double v_p = std::sqrt(earth_mu * (1.0 + e_init) / (a * (1.0 - e_init)));
    const double n = std::sqrt(earth_mu / (a * a * a));
    const double p = a * (1.0 - (e_init * e_init));
    constexpr double duration = 86400.0 * 4;

    // Helper: propagate an elliptical orbit at the given inclination for `duration` seconds and extract ω from the
    // final state via ω = atan2(e·(ĥ × N̂), e·N̂). ICs place periapsis at the ascending node on the +x axis, so Ω_0 = ω_0
    // = 0.
    auto propagate_and_measure_omega = [&](double inc) {
        const std::vector<double> ic{
            r_p, 0.0, 0.0, 0.0, v_p * std::cos(inc), v_p * std::sin(inc),
        };

        taylor_adaptive<double> ta{dyn, ic, kw::compact_mode = true};
        ta.propagate_until(duration);

        const auto &s = ta.get_state();

        // h = r × v.
        const double hx = (s[1] * s[5]) - (s[2] * s[4]);
        const double hy = (s[2] * s[3]) - (s[0] * s[5]);
        const double hz = (s[0] * s[4]) - (s[1] * s[3]);
        const double h_mag = std::sqrt((hx * hx) + (hy * hy) + (hz * hz));

        // e_vec = (v × h)/μ − r/|r|.
        const double r_mag = std::sqrt((s[0] * s[0]) + (s[1] * s[1]) + (s[2] * s[2]));
        const double vxh_x = (s[4] * hz) - (s[5] * hy);
        const double vxh_y = (s[5] * hx) - (s[3] * hz);
        const double vxh_z = (s[3] * hy) - (s[4] * hx);
        const double ex = (vxh_x / earth_mu) - (s[0] / r_mag);
        const double ey = (vxh_y / earth_mu) - (s[1] / r_mag);
        const double ez = (vxh_z / earth_mu) - (s[2] / r_mag);

        // Node vector N = ẑ × h = (-hy, hx, 0). ω components.
        const double Nx = -hy;
        const double Ny = hx;
        const double x_comp = (ex * Nx) + (ey * Ny);
        const double hxN_x = -hz * Ny;
        const double hxN_y = hz * Nx;
        const double hxN_z = (hx * Ny) - (hy * Nx);
        const double y_comp = ((hxN_x * ex) + (hxN_y * ey) + (hxN_z * ez)) / h_mag;
        return std::atan2(y_comp, x_comp);
    };

    // Critical inclination: 5 cos²i = 1. i_c = arccos(1/√5).
    const double inc_critical = std::acos(1.0 / std::sqrt(5.0));

    const std::array inclinations{
        std::numbers::pi_v<double> / 6.0, // 30°, factor 2.75 → positive ω̇
        std::numbers::pi_v<double> / 4.0, // 45°, factor 1.5  → positive ω̇
        inc_critical,                     // ~63.4°, factor 0  → ω̇ = 0 at first order
        std::numbers::pi_v<double> / 2.0, // 90°, factor -1    → negative ω̇
    };

    for (const double inc : inclinations) {
        const double omega_final = propagate_and_measure_omega(inc);
        const double factor = (5.0 * std::cos(inc) * std::cos(inc)) - 1.0;
        const double omega_dot = 0.75 * n * J2 * (earth_a / p) * (earth_a / p) * factor;
        const double omega_expected = omega_dot * duration;

        if (std::abs(factor) < 1e-10) {
            // Critical inclination: first-order ω̇ vanishes.
            REQUIRE(std::abs(omega_final) < 3e-2);
        } else {
            // Non-critical: relative tolerance.
            REQUIRE(std::abs((omega_final - omega_expected) / omega_expected) < 0.10);
        }
    }
}

TEST_CASE("drag pointwise check")
{
    detail::edb_disabler ed;

    // Kepler baseline and drag-enabled dynamics with Cb as a runtime parameter.
    auto dyn_kep = model::eo_dynamics();
    auto dyn_drag = model::eo_dynamics(kw::Cb = par[0]);

    REQUIRE(dyn_kep.size() == 6u);
    REQUIRE(dyn_drag.size() == 6u);

    // Expressions must differ.
    REQUIRE(dyn_drag[3].second != dyn_kep[3].second);

    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    cfunc<double> kep_acc_cf{{dyn_kep[3].second, dyn_kep[4].second, dyn_kep[5].second},
                             std::vector{x, y, z, vx, vy, vz}};
    cfunc<double> drag_acc_cf{{dyn_drag[3].second, dyn_drag[4].second, dyn_drag[5].second},
                              std::vector{x, y, z, vx, vy, vz}};

    {
        // LEO circular prograde equatorial orbit at ~400 km altitude - drag regime where default sw/eop data yields
        // meaningful density at t=0 (J2000). Cb = 1e-2 m²/kg (representative high-area-to-mass object).
        const std::array<double, 6> state{6778.0, 0.0, 0.0, 0.0, 7.67, 0.0};
        const std::vector<double> pars{1e-2};

        std::array<double, 3> kep_acc{};
        std::array<double, 3> full_acc{};
        kep_acc_cf(kep_acc, state, kw::time = 0.0);
        drag_acc_cf(full_acc, state, kw::time = 0.0, kw::pars = pars);

        const std::array<double, 3> drag_acc{
            full_acc[0] - kep_acc[0],
            full_acc[1] - kep_acc[1],
            full_acc[2] - kep_acc[2],
        };

        const double drag_mag2
            = (drag_acc[0] * drag_acc[0]) + (drag_acc[1] * drag_acc[1]) + (drag_acc[2] * drag_acc[2]);
        REQUIRE(drag_mag2 > 0.0);

        const double drag_dot_v = (drag_acc[0] * state[3]) + (drag_acc[1] * state[4]) + (drag_acc[2] * state[5]);
        REQUIRE(drag_dot_v < 0.0);

        // Quantitative magnitude check - baselined against a first-principles evaluation with default sw/eop data at
        // t=0 (J2000).
        const double drag_mag_expected = 1.66292e-9;
        REQUIRE(std::abs(std::sqrt(drag_mag2) - drag_mag_expected) < 1e-4 * drag_mag_expected);
    }

    // Spot-check against the drag notebook at heyoka.py: https://bluescarni.github.io/heyoka.py/notebooks/eo_atmo.html
    {
        const double earth_mu_local = model::get_egm2008_mu() / 1e9;
        const double r_nb = 6910.0;
        const double v_nb = std::sqrt(earth_mu_local / r_nb);
        const std::array<double, 6> state_nb{r_nb, 0.0, 0.0, 0.0, v_nb, 0.0};
        const std::vector<double> pars_nb{0.00019366446 * 2 / 0.15696615};

        std::array<double, 3> kep_nb{};
        std::array<double, 3> full_nb{};
        kep_acc_cf(kep_nb, state_nb, kw::time = 0.0);
        drag_acc_cf(full_nb, state_nb, kw::time = 0.0, kw::pars = pars_nb);

        const std::array<double, 3> drag_nb{
            full_nb[0] - kep_nb[0],
            full_nb[1] - kep_nb[1],
            full_nb[2] - kep_nb[2],
        };
        const double drag_nb_mag_km_s2
            = std::sqrt((drag_nb[0] * drag_nb[0]) + (drag_nb[1] * drag_nb[1]) + (drag_nb[2] * drag_nb[2]));

        // km/s² → m/s² (state is in km, notebook reports in m/s²).
        const double drag_nb_m_s2 = drag_nb_mag_km_s2 * 1e3;

        const double drag_nb_expected = 6.181088320949107e-8;
        REQUIRE(std::abs(drag_nb_m_s2 - drag_nb_expected) < 1e-5 * drag_nb_expected);
    }
}

// Drag end-to-end dissipation: propagate a LEO circular orbit with drag and verify that the specific orbital energy E =
// ½|v|² − μ/r strictly decreases over the propagation.
TEST_CASE("drag dissipates orbital energy")
{
    detail::edb_disabler ed;

    auto dyn = model::eo_dynamics(kw::Cb = 1e-2_dbl);
    REQUIRE(dyn.size() == 6u);

    const double earth_mu = model::get_egm2008_mu() / 1e9;

    // LEO circular at ~400 km.
    const double r_init = 6778.0;
    const double v_init = std::sqrt(earth_mu / r_init);
    const std::vector<double> ic{r_init, 0.0, 0.0, 0.0, v_init, 0.0};

    const double E_init = (0.5 * v_init * v_init) - (earth_mu / r_init);

    taylor_adaptive<double> ta{dyn, ic, kw::compact_mode = true};
    ta.propagate_until(86400.0);

    const auto &s = ta.get_state();
    const double r_final = std::sqrt((s[0] * s[0]) + (s[1] * s[1]) + (s[2] * s[2]));
    const double v2_final = (s[3] * s[3]) + (s[4] * s[4]) + (s[5] * s[5]);
    const double E_final = (0.5 * v2_final) - (earth_mu / r_final);

    REQUIRE(E_final < E_init);
}

// Third-body (Sun + Moon) perturbation cross-check against the heyoka.py notebook at
//
//   https://bluescarni.github.io/heyoka.py/notebooks/eo_third_body.html
//
// Evaluates the 3rd-body tidal perturbation at a fixed LEO state and compares against the notebook's result
// component-by-component. The 3rd-body acceleration depends only on position (and time through the ephemerides),
// not on velocity, so velocity values are arbitrary here.
TEST_CASE("third body notebook cross-check")
{
    detail::edb_disabler ed;

    auto dyn_kep = model::eo_dynamics();
    auto dyn_tb = model::eo_dynamics(kw::elp2000_thresh = 1e-5, kw::vsop2013_thresh = 1e-3);

    REQUIRE(dyn_kep.size() == 6u);
    REQUIRE(dyn_tb.size() == 6u);

    // Expressions must differ - catches "3rd-body kwargs silently ignored".
    REQUIRE(dyn_tb[3].second != dyn_kep[3].second);

    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");

    cfunc<double> kep_acc_cf{{dyn_kep[3].second, dyn_kep[4].second, dyn_kep[5].second},
                             std::vector{x, y, z, vx, vy, vz}};
    cfunc<double> tb_acc_cf{{dyn_tb[3].second, dyn_tb[4].second, dyn_tb[5].second}, std::vector{x, y, z, vx, vy, vz}};

    // Notebook state: pos_leo = (6370 + 410, 0, 0) km = (6780, 0, 0) km. At J2000.
    const std::array<double, 6> state{6780.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    std::array<double, 3> kep_acc{};
    std::array<double, 3> full_acc{};
    kep_acc_cf(kep_acc, state, kw::time = 0.0);
    tb_acc_cf(full_acc, state, kw::time = 0.0);

    // Perturbation in km/s², converted to m/s² to match the notebook's units (state is in km here).
    const std::array<double, 3> tb_acc_m_s2{
        (full_acc[0] - kep_acc[0]) * 1e3,
        (full_acc[1] - kep_acc[1]) * 1e3,
        (full_acc[2] - kep_acc[2]) * 1e3,
    };

    // Values from the notebook at J2000 with vsop2013_thresh = 1e-3, elp2000_thresh = 1e-5.
    const std::array<double, 3> expected{4.1438356577485833e-08, 5.8312040927821496e-07, 1.4592088870817727e-07};

    REQUIRE(std::abs(tb_acc_m_s2[0] - expected[0]) < 1e-5 * std::abs(expected[0]));
    REQUIRE(std::abs(tb_acc_m_s2[1] - expected[1]) < 1e-5 * std::abs(expected[1]));
    REQUIRE(std::abs(tb_acc_m_s2[2] - expected[2]) < 1e-5 * std::abs(expected[2]));
}

// Error handling: elp2000_thresh and vsop2013_thresh must travel together.
TEST_CASE("error handling: elp2000/vsop2013 pairing")
{
    REQUIRE_THROWS_AS(model::eo_dynamics(kw::elp2000_thresh = 1e-3), std::invalid_argument);
    REQUIRE_THROWS_AS(model::eo_dynamics(kw::vsop2013_thresh = 1e-3), std::invalid_argument);

    // Both thresholds present: OK (third-body enabled).
    REQUIRE_NOTHROW(model::eo_dynamics(kw::elp2000_thresh = 1e-3, kw::vsop2013_thresh = 1e-3));

    // Neither present: OK (third-body disabled - default).
    REQUIRE_NOTHROW(model::eo_dynamics());
}

// Verify the optional_from kwarg semantics: callers may pass a std::optional<> as kwarg, with an empty optional being
// equivalent to not providing the kwarg at all.
TEST_CASE("optional kwarg semantics")
{
    detail::edb_disabler ed;

    const auto [x, y, z, vx, vy, vz] = make_vars("x", "y", "z", "vx", "vy", "vz");
    const std::vector<expression> vars{x, y, z, vx, vy, vz};
    const std::array<double, 6> state{6778.0, 100.0, -100.0, 1.0, 7.67, 0.5};

    auto eval_rhs = [&](const std::vector<std::pair<expression, expression>> &dyn, const std::vector<double> &pars) {
        std::vector<expression> rhs;
        rhs.reserve(dyn.size());
        for (const auto &p : dyn) {
            rhs.push_back(p.second);
        }
        cfunc<double> cf{rhs, vars};
        std::array<double, 6> out{};
        cf(out, state, kw::time = 0.0, kw::pars = pars);
        return out;
    };

    // Empty optional for Cb ≡ no kwarg (drag disabled).
    {
        const auto out_no_kwarg = eval_rhs(model::eo_dynamics(), {});
        const auto out_empty = eval_rhs(model::eo_dynamics(kw::Cb = std::optional<expression>{}), {});
        REQUIRE(out_no_kwarg == out_empty);
    }

    // Filled optional for Cb ≡ passing the expression directly.
    {
        const std::vector<double> pars{1e-2};
        const auto out_direct = eval_rhs(model::eo_dynamics(kw::Cb = par[0]), pars);
        const auto out_opt = eval_rhs(model::eo_dynamics(kw::Cb = std::optional<expression>(par[0])), pars);
        REQUIRE(out_direct == out_opt);
    }

    // Cb as a double-precision value.
    {
        const auto out_direct = eval_rhs(model::eo_dynamics(kw::Cb = 1e-2), {});
        const auto out_opt = eval_rhs(model::eo_dynamics(kw::Cb = std::optional<expression>(1e-2_dbl)), {});
        REQUIRE(out_direct == out_opt);
    }

    // Both thresholds empty ≡ no kwargs (third-body disabled; pairing invariant not triggered).
    {
        const auto out_no_kwargs = eval_rhs(model::eo_dynamics(), {});
        const auto out_empty_pair = eval_rhs(model::eo_dynamics(kw::elp2000_thresh = std::optional<double>{},
                                                                kw::vsop2013_thresh = std::optional<double>{}),
                                             {});
        REQUIRE(out_no_kwargs == out_empty_pair);
    }

    // Both thresholds filled ≡ passing scalars directly.
    {
        const auto out_direct = eval_rhs(model::eo_dynamics(kw::elp2000_thresh = 1e-5, kw::vsop2013_thresh = 1e-3), {});
        const auto out_opt = eval_rhs(model::eo_dynamics(kw::elp2000_thresh = std::optional<double>(1e-5),
                                                         kw::vsop2013_thresh = std::optional<double>(1e-3)),
                                      {});
        REQUIRE(out_direct == out_opt);
    }

    // Pairing invariant still fires when one is empty and the other is provided.
    REQUIRE_THROWS_AS(model::eo_dynamics(kw::elp2000_thresh = std::optional<double>{}, kw::vsop2013_thresh = 1e-3),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(model::eo_dynamics(kw::elp2000_thresh = 1e-5, kw::vsop2013_thresh = std::optional<double>{}),
                      std::invalid_argument);
}
