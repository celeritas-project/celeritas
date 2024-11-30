//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/InverseCdfFinder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/InverseCdfFinder.hh"

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/grid/NonuniformGrid.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class InverseCdfFinderTest : public Test
{
  protected:
    using GridT = NonuniformGrid<real_type>;

    void SetUp() override
    {
        CollectionBuilder build(&data);
        values = build.insert_back({0.0, 0.2, 0.5, 1.5, 6.0, 10.0});
        ref = data;
    }

    std::vector<real_type> build_cdf()
    {
        return std::vector<real_type>{0.0, 0.1, 0.2, 0.8, 0.9, 1.0};
    }

    ItemRange<real_type> values;
    Collection<real_type, Ownership::value, MemSpace::host> data;
    Collection<real_type, Ownership::const_reference, MemSpace::host> ref;
};

TEST_F(InverseCdfFinderTest, all)
{
    InverseCdfFinder find(GridT(values, ref), this->build_cdf());

    EXPECT_SOFT_EQ(0, find(0));
    EXPECT_SOFT_EQ(0.1, find(0.05));
    EXPECT_SOFT_EQ(0.2, find(0.1));
    EXPECT_SOFT_EQ(1, find(0.5));
    EXPECT_SOFT_EQ(6.0, find(0.9));
    EXPECT_SOFT_EQ(8.0, find(0.95));
    EXPECT_SOFT_EQ(9.99996, find(0.999999));
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(find(1.0), DebugError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
