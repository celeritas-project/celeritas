//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/RegulaRootFinder.test.cc
//---------------------------------------------------------------------------//
#include <cmath>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/RegulaFalsiRootFinder.hh"
#include "orange/OrangeTypes.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

// Solve (x-2)^2 = 0
Test(RegulaFalsi, sqrt_two)
{
    auto root = [](t) { return std::sqrt(ipow<2>(t) - 2); }

    real_type root_two
        = RegulaFalsi::RegulaFalsi(root, 1e-13);

    EXPECT_SOFT_EQ(std::sqrt(2), root_two(1.0, 2.0));
}

// Solve (x-3)^2 = 0
Test(RegulaFalsi, sqrt_two)
{
    auto root = [](t) { return std::sqrt(ipow<2>(t) - 3); }

    real_type root_three
        = RegulaFalsi::RegulaFalsi(root, 1e-13);

    EXPECT_SOFT_EQ(std::sqrt(3), root_three(1.0, 2.0));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas