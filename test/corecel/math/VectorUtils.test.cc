//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/VectorUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/math/VectorUtils.hh"

#include "celeritas_test.hh"

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(VectorUtils, linspace)
{
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(linspace(1.23, 4.56, 0), DebugError);
        EXPECT_THROW(linspace(1.23, 4.56, 1), DebugError);
    }

    {
        auto result = linspace(10, 20, 2);

        static const real_type expected[] = {10, 20};
        EXPECT_VEC_SOFT_EQ(expected, result);
    }
    {
        auto result = linspace(10, 20, 5);

        static const real_type expected[] = {10, 12.5, 15, 17.5, 20};
        EXPECT_VEC_SOFT_EQ(expected, result);
    }
    {
        // Guard against accumulation error
        const real_type exact_third = 1.0 / 3.0;
        auto            result = linspace(exact_third, 2 * exact_third, 32768);
        ASSERT_EQ(32768, result.size());
        if (sizeof(real_type) == sizeof(double))
        {
            EXPECT_DOUBLE_EQ(exact_third, result.front());
            EXPECT_DOUBLE_EQ(2 * exact_third, result.back());
        }
    }
}
