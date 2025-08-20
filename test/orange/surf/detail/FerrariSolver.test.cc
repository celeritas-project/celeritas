
//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/detail/QuadraticSolver.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/detail/QuadraticSolver.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(SolveNonsurface, no_roots)
{

}

TEST(SolveNonsurface, one_root) 
{
    // Does this imply we need to check for double roots? Would that impact the analysis?
}

TEST(SolveNonsurface, two_roots)
{

}

TEST(SolveNonsurface, two_double_roots)
{

}

TEST(SolveNonsurface, three_roots)
{

}

TEST(SolveNonsurface, four_roots)
{

}

TEST(SolveSurface, zero_roots)
{

}

TEST(SolveSurface, one_root)
{

}

TEST(SolveSurface, one_double_root)
{}

TEST(SolveSurface, two_roots)
{}

TEST(SolveSurface, two_roots_one_double)
{}

TEST(SolveSurface, three_roots)
{}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
