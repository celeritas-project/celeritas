//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/TridiagonalSolver.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/TridiagonalSolver.hh"

#include <utility>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(TridiagonalSolverTest, basic)
{
    // Solve the following linear system Ax = d:
    //
    // [  2  -1   0   0  ][ x_0 ]   [ 1 ]
    // [ -1   2  -1   0  ][ x_1 ] = [ 0 ]
    // [  0  -1   2  -1  ][ x_2 ]   [ 0 ]
    // [  0   0  -1   2  ][ x_3 ]   [ 0 ]

    TridiagonalSolver::Coefficients coeffs;
    coeffs.a = {0, -1, -1, -1};
    coeffs.b = {2, 2, 2, 2};
    coeffs.c = {-1, -1, -1, 0};
    coeffs.d = {1, 0, 0, 0};

    TridiagonalSolver solve(std::move(coeffs));
    EXPECT_VEC_SOFT_EQ(
        std::vector<real_type>({4 / 5.0, 3 / 5.0, 2 / 5.0, 1 / 5.0}), solve());
}

TEST(TridiagonalSolverTest, small)
{
    // Solve the following linear system Ax = d:
    //
    // [  2  -1][ x_0 ] = [ 1 ]
    // [ -1   2][ x_1 ]   [ 0 ]

    TridiagonalSolver::Coefficients coeffs;
    coeffs.a = {0, -1};
    coeffs.b = {2, 2};
    coeffs.c = {-1, 0};
    coeffs.d = {1, 0};

    TridiagonalSolver solve(std::move(coeffs));
    EXPECT_VEC_SOFT_EQ(std::vector<real_type>({2 / 3.0, 1 / 3.0}), solve());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
