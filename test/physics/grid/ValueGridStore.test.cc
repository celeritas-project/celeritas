//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ValueGridStore.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/ValueGridStore.hh"

#include <algorithm>
#include "celeritas_test.hh"
#include "base/Range.hh"

using celeritas::real_type;
using celeritas::size_type;
using celeritas::UniformGridData;
using celeritas::ValueGridStore;
using celeritas::XsGridPointers;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class ValueGridStoreTest : public celeritas::Test
{
  protected:
    XsGridPointers make_xsgrid(size_type count)
    {
        CELER_EXPECT(count > 0);
        XsGridPointers result;
        result.log_energy  = UniformGridData::from_bounds(0.0, 1.0, count);
        result.prime_index = count / 2;
        temp_real.resize(count);
        for (auto i : celeritas::range(temp_real.size()))
        {
            temp_real[i] = i + 1;
        }
        result.value = celeritas::make_span(temp_real);

        CELER_ENSURE(result);
        return result;
    }

    std::vector<real_type> temp_real;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(ValueGridStoreTest, all)
{
    ValueGridStore vgs;
    EXPECT_EQ(0, vgs.size());
    EXPECT_EQ(0, vgs.capacity());

    // Allocate for two grids, one with three values and one with four
    vgs = ValueGridStore(2, 3 + 4);
    EXPECT_EQ(0, vgs.size());
    EXPECT_EQ(2, vgs.capacity());

    vgs.push_back(make_xsgrid(3));
    vgs.push_back(make_xsgrid(4));
    EXPECT_EQ(2, vgs.size());

    auto ptrs = vgs.host_pointers();
    ASSERT_EQ(2, ptrs.size());

    {
        const XsGridPointers& xs = ptrs[0];
        EXPECT_TRUE(bool(xs));
        const real_type expected[] = {1, 2, 3};
        EXPECT_VEC_EQ(expected, xs.value);
    }

    {
        const XsGridPointers& xs = ptrs[1];
        EXPECT_TRUE(bool(xs));
        const real_type expected[] = {1, 2, 3, 4};
        EXPECT_VEC_EQ(expected, xs.value);
    }
}
