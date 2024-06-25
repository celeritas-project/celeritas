//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/InvoluteSolver.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/InvoluteSolver.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
const double pi = 3.14159265358979323846;

TEST(SolveSurface,one_root) 
{
    // Solve for rb = 1.0, a = 0, sign = 1
    // Point (0,0) Direction (0,1)
    // tmin = 0 and tmax = 1.99*pi
    {
        double r_b = 1.0;
        double a = 0;
        double sign = 1.0;

        double x = 0;
        double y = 0;
        double u = 0;
        double v = 0;

        double tmin = 0;
        double tmax = 1.99*pi;

        InvoluteSolver solve(r_b, a, sign, tmin, tmax);
        auto dist = solve(x,y,0,u,v,0);

        EXPECT_SOFT_EQ(2.97169387, dist[0]);
        EXPECT_SOFT_EQ(no_intersection(), dist[1]);
        EXPECT_SOFT_EQ(no_intersection(), dist[2]);
    }
    
}

}
}
}