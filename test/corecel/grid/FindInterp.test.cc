//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/FindInterp.test.cc
//---------------------------------------------------------------------------//
#include "corecel/grid/FindInterp.hh"

#include "celeritas_test.hh"
#include "corecel/grid/UniformGrid.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(FindInterpTest, uniform_real)
{
    auto data = UniformGridData::from_bounds(1.0, 5.0, 3);
    UniformGrid const grid(data);

    {
        auto interp = find_interp(grid, 1.0);
        EXPECT_EQ(0, interp.index);
        EXPECT_SOFT_EQ(0.0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 3.0);
        EXPECT_EQ(1, interp.index);
        EXPECT_SOFT_EQ(0.0, interp.fraction);
    }
    {
        auto interp = find_interp(grid, 4.0);
        EXPECT_EQ(1, interp.index);
        EXPECT_SOFT_EQ(0.5, interp.fraction);
    }
#if CELERITAS_DEBUG
    EXPECT_THROW(find_interp(grid, 0.999), DebugError);
    EXPECT_THROW(find_interp(grid, 5.0), DebugError);
    EXPECT_THROW(find_interp(grid, 5.001), DebugError);
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
