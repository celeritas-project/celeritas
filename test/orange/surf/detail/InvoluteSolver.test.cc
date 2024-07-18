//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvoluteSolver.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/InvoluteSolver.hh"

#include "corecel/Constants.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
using constants::pi;
using Sign = InvoluteSolver::Sign;
using SurfaceSate = celeritas::SurfaceState;

//! Python reference can be found in \file test/orange/surf/doc/involute.py
TEST(SolveSurface, no_roots)
{
    // Solve for rb = 3.0, a = pi, sign = CCW
    // Point (0,-2) Direction (1,0)
    // tmin = 0.5 and tmax = 4
    {
        double r_b = 1.0;
        double a = 0;
        auto sign = InvoluteSolver::counterclockwise;

        double x = 0;
        double y = -2;
        double u = 1;
        double v = 0;

        double tmin = 0.5;
        double tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.75, a = -2*pi, sign = CCW
    // Point (-7,-1) Direction (0.894427191,-0.4472135955)
    // tmin = 2 and tmax = 4
    {
        double r_b = 0.75;
        double a = -2 * pi;
        auto sign = InvoluteSolver::counterclockwise;

        double x = -7;
        double y = -1;
        double u = 0.894427191;
        double v = -0.4472135955;

        double tmin = 2;
        double tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.25, a = -2*pi, sign = CCW
    // Point (-2,1) Direction (0.4472135955,0.894427191)
    // tmin = 2 and tmax = 4
    {
        double r_b = 0.75;
        double a = -2 * pi;
        auto sign = InvoluteSolver::counterclockwise;

        double x = -7;
        double y = -1;
        double u = 0.4472135955;
        double v = -0.894427191;

        double tmin = 2;
        double tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
}

TEST(SolveSurface, one_root)
{
    // Solve for rb = 1.0, a = 0, sign = CCW
    // Point (0,0) Direction (0,1)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.0;
        double a = 0;
        auto sign = InvoluteSolver::counterclockwise;

        double x = 0;
        double y = 0;
        double u = 0;
        double v = 1;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(2.9716938706909275, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.5, a = 0, sign = CCW
    // Point (-1.5,1) Direction (0.2,0.9797958971)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.5;
        double a = 0;
        auto sign = InvoluteSolver::counterclockwise;

        double x = -1.5;
        double y = 1.0;
        double u = 0.2;
        double v = 0.9797958971;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(3.7273045229725681, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 3.0, a = pi, sign = CCW
    // Point (-4.101853006408607,-5.443541628262038) Direction (0.0,1.0)
    // tmin = 2 and tmax = 4
    {
        double r_b = 3.0;
        double a = pi;
        auto sign = InvoluteSolver::counterclockwise;

        double x = -4.101853006408607;
        double y = -5.443541628262038;
        double u = 0.0;
        double v = 1.0;

        double tmin = 2;
        double tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(0, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.5, a = 0.4*pi, sign = CW
    // Point (-4,2) Direction (0.894427191,-0.4472135955)
    // tmin = 2 and tmax = 4
    {
        double r_b = 0.5;
        double a = 0.6 * pi;
        auto sign = InvoluteSolver::clockwise;

        double x = -4.0;
        double y = 2.0;
        double u = 0.894427191;
        double v = -0.4472135955;

        double tmin = 2;
        double tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(6.0371012183803652, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.1, a = 0.5*pi, sign = CW
    // Point (-0.2,1.1) Direction (0,0)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.1;
        double a = 0.5 * pi;
        auto sign = InvoluteSolver::clockwise;

        double x = -0.2;
        double y = 1.1;
        double u = 0;
        double v = 0;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
}

TEST(SolveSurface, two_roots)
{
    // Solve for rb = 1.1, a = 0.5*pi, sign = CW
    // Point (-0.2,1.1) Direction (1,0)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.1;
        double a = 0.5 * pi;
        auto sign = InvoluteSolver::clockwise;

        double x = -0.2;
        double y = 1.1;
        double u = 1;
        double v = 0;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(0.2, dist[0]);
        EXPECT_SOFT_EQ(2.764234602725404, dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.1, a = -0.5*pi, sign = CW
    // Point (-0.0001,-1.11) Direction (-0.1,0.9949874371)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.1;
        double a = 1.5 * pi;
        auto sign = InvoluteSolver::clockwise;

        double x = -0.0001;
        double y = -1.11;
        double u = -0.1;
        double v = 0.9949874371;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(0.0036178081060022097, dist[0]);
        EXPECT_SOFT_EQ(6.0284475629193013, dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.1, a = -0.5*pi, sign = CCW
    // Point (0.0058102462574510716,-1.1342955336941216)
    // Direction (0.7071067812,0.7071067812)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.1;
        double a = -0.5 * pi;
        auto sign = InvoluteSolver::counterclockwise;

        double x = 0.0058102462574510716;
        double y = -1.1342955336941216;
        double u = 0.7071067812;
        double v = 0.7071067812;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(0.0, dist[0]);
        EXPECT_SOFT_EQ(4.6528327548038506, dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
}
TEST(SolveSurface, three_roots)
{
    // Solve for rb = 1.1, a = -0.5*pi, sign = CCW
    // Point (-6.865305298657132,-0.30468305643505367)
    // Direction (0.9933558377574788,-0.11508335932330707)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.1;
        double a = -0.5 * pi;
        auto sign = InvoluteSolver::counterclockwise;

        double x = -6.8653052986571326;
        double y = -0.30468305643505367;
        double u = 0.9933558377574788;
        double v = -0.11508335932330707;

        double tmin = 0;
        double tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(0.0, dist[0]);
        EXPECT_SOFT_EQ(6.9112249151160547, dist[1]);
        EXPECT_SOFT_EQ(9.167603472624553, dist[2]);
    }
}
TEST(Components, line_angle_param)
{
    // Direction (0,1)
    // beta = -pi*0.5
    {
        double u = 0;
        double v = 1;

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(-pi * 0.5, beta);
    }

    // Direction (0,-1)
    // beta = pi*0.5
    {
        double u = 0;
        double v = -1;

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(pi * 0.5, beta);
    }

    // Direction (0.5,sin(pi/3))
    // beta = -pi/3
    {
        double u = 0.5;
        double v = std::sin(pi / 3);

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(-pi / 3, beta);
    }
}
TEST(Components, calc_dist)
{
    double r_b = 1.1;
    double a = 0.5 * pi;
    auto sign = InvoluteSolver::clockwise;

    double x = -0.2;
    double y = 1.1;
    double u = 1;
    double v = 0;

    double tmin = 0;
    double tmax = 1.99 * pi;

    double t_gamma = 0;

    InvoluteSolver calc_dist(r_b, a, sign, tmin, tmax);

    auto dist = calc_dist.calc_dist(x, y, u, v, t_gamma);
    EXPECT_SOFT_EQ(0.2, dist);
}
}  // namespace test
}  // namespace detail
}  // namespace celeritas