//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
constexpr real_type pi{constants::pi};
using Sign = Chirality;
Sign ccw = Chirality::left;
Sign cw = Chirality::right;
using SurfaceSate = celeritas::SurfaceState;

TEST(SolveSurface, no_roots)
{
    // Solve for rb = 3.0, a = 0, sign = CCW
    // Point (0,-2) Direction (1,0)
    // tmin = 0.5 and tmax = 4
    {
        real_type r_b = 3.0;
        real_type a = 0;
        auto sign = ccw;

        real_type x = 0;
        real_type y = -2;
        real_type u = 1;
        real_type v = 0;

        real_type tmin = 0.5;
        real_type tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.75, a = 0, sign = CCW
    // Point (-7,-1) Direction (0.894427191,-0.4472135955)
    // tmin = 2 and tmax = 4
    {
        real_type r_b = 0.75;
        real_type a = 0;
        auto sign = ccw;

        real_type x = -7;
        real_type y = -1;
        real_type u = 0.894427191;
        real_type v = -0.4472135955;

        real_type tmin = 2;
        real_type tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.25, a = 0, sign = CCW
    // Point (-2,1) Direction (0.4472135955,0.894427191)
    // tmin = 2 and tmax = 4
    {
        real_type r_b = 0.75;
        real_type a = 0;
        auto sign = ccw;

        real_type x = -7;
        real_type y = -1;
        real_type u = 0.4472135955;
        real_type v = -0.894427191;

        real_type tmin = 2;
        real_type tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.1, a = 0.5*pi, sign = CW
    // Point (-0.2,1.1) Direction (0,0)
    // tmin = 0 and tmax = 1.99*pi
    {
        real_type r_b = 1.1;
        real_type a = 0.5 * pi;
        auto sign = cw;

        real_type x = -0.2;
        real_type y = 1.1;
        real_type u = 0;
        real_type v = 0;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 3.0, a = pi, sign = CCW
    // Point (-4.101853006408607,-5.443541628262038) Direction (0.0,1.0)
    // tmin = 2 and tmax = 4
    {
        real_type r_b = 3.0;
        real_type a = pi;
        auto sign = ccw;

        real_type x = -4.101853006408607;
        real_type y = -5.443541628262038;
        real_type u = 0.0;
        real_type v = 1.0;

        real_type tmin = 2;
        real_type tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

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
        real_type r_b = 1.0;
        real_type a = 0;
        auto sign = ccw;

        real_type x = 0;
        real_type y = 0;
        real_type u = 0;
        real_type v = 1;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(2.9716938706703964, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.5, a = 0, sign = CCW
    // Point (-1.5,1) Direction (0.2,0.9797958971)
    // tmin = 0 and tmax = 1.99*pi
    {
        real_type r_b = 1.5;
        real_type a = 0;
        auto sign = ccw;

        real_type x = -1.5;
        real_type y = 1.0;
        real_type u = 0.2;
        real_type v = 0.9797958971;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(3.7273045229446105, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 0.5, a = 0.4*pi, sign = CW
    // Point (-4,2) Direction (0.894427191,-0.4472135955)
    // tmin = 2 and tmax = 4
    {
        real_type r_b = 0.5;
        real_type a = 0.6 * pi;
        auto sign = cw;

        real_type x = -4.0;
        real_type y = 2.0;
        real_type u = 0.894427191;
        real_type v = -0.4472135955;

        real_type tmin = 2;
        real_type tmax = 4;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(6.0371012194546871, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Solve for rb = 1.1, a = 1.5*pi, sign = CCW
    // Point (0.0058102462574510716,-1.1342955336941216)
    // Direction (0.7071067812,0.7071067812)
    // tmin = 0 and tmax = 1.99*pi
    {
        real_type r_b = 1.1;
        real_type a = 1.5 * pi;
        auto sign = ccw;

        real_type x = 0.0058102462574510716;
        real_type y = -1.1342955336941216;
        real_type u = 0.7071067812;
        real_type v = 0.7071067812;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(4.6528327550189861, dist[0]);
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
        real_type r_b = 1.1;
        real_type a = 0.5 * pi;
        auto sign = cw;

        real_type x = -0.2;
        real_type y = 1.1;
        real_type u = 1;
        real_type v = 0;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_EQ(0.2, dist[0]);
            EXPECT_SOFT_EQ(2.7642346005578582, dist[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(0.2, dist[0]);
            EXPECT_SOFT_EQ(2.7642346005578582, dist[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist[2]);
        }
    }

    // Solve for rb = 1.1, a = 1.5*pi, sign = CW
    // Point (-0.0001,-1.11) Direction (-0.1,0.9949874371)
    // tmin = 0 and tmax = 1.99*pi
    {
        real_type r_b = 1.1;
        real_type a = 1.5 * pi;
        auto sign = cw;

        real_type x = -0.0001;
        real_type y = -1.11;
        real_type u = -0.1;
        real_type v = 0.9949874371;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_EQ(0.0036177579725445259, dist[0]);
        }
        else
        {
            // Float loses precision
            EXPECT_LT(dist[0], 0.005);
        }
        EXPECT_SOFT_EQ(6.0284475639586041, dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
}
TEST(SolveSurface, three_roots)
{
    // Solve for rb = 1.1, a = 1.5*pi, sign = CCW
    // Point (-6.8653259986571326,-0.30468105643505367)
    // Direction (0.9933558377574788,-0.11508335932330707)
    // tmin = 0 and tmax = 1.99*pi
    {
        real_type r_b = 1.1;
        real_type a = 1.5 * pi;
        auto sign = ccw;

        real_type x = -6.8653259986571326;
        real_type y = -0.30468105643505367;
        real_type u = 0.9933558377574788;
        real_type v = -0.11508335932330707;

        real_type tmin = 0;
        real_type tmax = 1.99 * pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(6.9112457587355429, dist[0]);
        EXPECT_SOFT_EQ(9.1676238065759748, dist[1]);
        EXPECT_SOFT_EQ(2.0792209373995243e-05, dist[2]);
    }
}
TEST(SolveSurface, tangents)
{
    // TODO: can we make the solver more numerically stable and replace
    // EXPECT_SOFT_NEAR with EXPECT_SOFT_EQ?

    // Solve for 0.5*pi tangents for rb = 1.0 a = 0 sign = CCW
    // Direction (0,1)
    real_type r_b = 1.0;
    real_type a = 0;
    auto sign = ccw;
    real_type tmin = 0.33 * pi;
    real_type tmax = 0.67 * pi;
    InvoluteSolver solve(r_b, a, sign, tmin, tmax);
    real_type u = 0;
    real_type v = 1;

    // Tangent Point (1.5707963267948966, 0.9999999999999999)
    {
        real_type x = 1.5707963267948966;
        real_type y = 0.9999999999999999;

        auto dist = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(no_intersection(), dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }

    // Secant Point (1.5707953267948966, 0.9999999999999999)
    {
        real_type x = 1.5707953267948966;
        real_type y = 0.9999999999999999;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.0017713715293786088, dist_on[0], 1e-10);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.0017713715293786088, dist_off[0], 1e-10);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(1.4354699260366033e-06, dist_off[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
    }

    // Secant Point (1.57079622679489656, 0.9999999999999999)
    {
        real_type x = 1.5707962267948965;
        real_type y = 0.9999999999999999;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.00053216327674743909, dist_on[0], 1e-8);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.00053216327674743909, dist_off[0], 1e-8);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(2.6656007889869215e-07, dist_off[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
    }

    // Secant Point (1.5707963167948966, 0.9999999999999999)
    {
        real_type x = 1.5707963167948966;
        real_type y = 0.9999999999999999;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_EQ(1.2715542669661114e-08, dist_off[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(1.2715542669661114e-08, dist_off[0]);
            EXPECT_SOFT_EQ(1.1920928955078125e-07, dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
    }

    // Parallel Point (1.5707963367948965, 0.9999999999999999)
    {
        real_type x = 1.5707963367948965;
        real_type y = 0.9999999999999999;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_EQ(no_intersection(), dist_off[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
        else
        {
            // This test fails
            EXPECT_SOFT_EQ(1.1920928955078125e-07, dist_off[0]);
            EXPECT_SOFT_EQ(1.1920928955078125e-07, dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
    }

    // Secant Point (1.5707960166494566, 1.0019871577672377)
    {
        real_type x = 1.5707960166494566;
        real_type y = 1.0019871577672377;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist_off[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
    }

    // Secant Point (1.5707960166494566, 1.0009871677672377)
    {
        real_type x = 1.5707960166494566;
        real_type y = 1.0009871677672377;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        EXPECT_SOFT_EQ(no_intersection(), dist_off[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
    }

    // Secant Point (1.5707960168148234, 0.999029247016899)
    {
        real_type x = 1.5707960168148234;
        real_type y = 0.999029247016899;

        auto dist_on
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::on);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.0019504376639951655, dist_on[0], 1e-10);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(no_intersection(), dist_on[0]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_on[2]);
        }

        auto dist_off
            = solve(Real3{x, y, 0.0}, Real3{u, v, 0.0}, SurfaceState::off);

        // Float and double produce different results
        if constexpr (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
        {
            EXPECT_SOFT_NEAR(0.0019504376639951655, dist_off[0], 1e-10);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
        else
        {
            EXPECT_SOFT_EQ(0.00096935039618983865, dist_off[0]);
            EXPECT_SOFT_EQ(0.00097101932624354959, dist_off[1]);
            EXPECT_SOFT_EQ(no_intersection(), dist_off[2]);
        }
    }
}
TEST(Components, line_angle_param)
{
    // Direction (0,1)
    // beta = -pi*0.5
    {
        real_type u = 0;
        real_type v = 1;

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(-pi * 0.5, beta);
    }

    // Direction (0,-1)
    // beta = pi*0.5
    {
        real_type u = 0;
        real_type v = -1;

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(pi * 0.5, beta);
    }

    // Direction (0.5,sin(pi/3))
    // beta = -pi/3
    {
        real_type u = 0.5;
        real_type v = std::sin(pi / 3);

        auto beta = InvoluteSolver::line_angle_param(u, v);
        EXPECT_SOFT_EQ(-pi / 3, beta);
    }
}
TEST(Components, calc_dist)
{
    real_type r_b = 1.1;
    real_type a = 0.5 * pi;
    auto sign = cw;

    real_type x = -0.2;
    real_type y = 1.1;
    real_type u = 1;
    real_type v = 0;

    real_type tmin = 0;
    real_type tmax = 1.99 * pi;

    real_type t_gamma = 0;

    InvoluteSolver calc_dist(r_b, a, sign, tmin, tmax);

    auto dist = calc_dist.calc_dist(x, y, u, v, t_gamma);
    EXPECT_SOFT_EQ(0.2, dist);
}
}  // namespace test
}  // namespace detail
}  // namespace celeritas
