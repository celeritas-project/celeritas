//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/detail/PolygonUtils.test.cc
//---------------------------------------------------------------------------//
#include "geocel/detail/PolygonUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

using Real2 = Array<real_type, 2>;
using VecReal2 = std::vector<Real2>;
using namespace celeritas::geoutils;

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(PolygonUtilsTest, IsPolygonConvex)
{
    VecReal2 cw{{1, 1}, {1, 2}, {2, 2}, {2, 1}};
    EXPECT_TRUE(IsPolygonConvex(cw));

    VecReal2 ccw{{1, 1}, {2, 1}, {2, 2}, {1, 2}};
    EXPECT_TRUE(IsPolygonConvex(ccw));

    VecReal2 bad{{1, 1}, {2, 2}, {2, 1}, {1, 2}};
    EXPECT_FALSE(IsPolygonConvex(bad));

    VecReal2 bad2{{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    EXPECT_FALSE(IsPolygonConvex(bad2));

    VecReal2 oct{8};
    for (size_type i = 0; i < 8; ++i)
    {
        oct[i] = {std::cos(2 * m_pi * i / 8), std::sin(2 * m_pi * i / 8)};
    }
    EXPECT_TRUE(IsPolygonConvex(oct));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
