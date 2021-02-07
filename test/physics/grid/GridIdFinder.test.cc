//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GridIdFinder.test.cc
//---------------------------------------------------------------------------//
#include "physics/grid/GridIdFinder.hh"

#include "celeritas_test.hh"
#include "physics/base/Units.hh"

using celeritas::GridIdFinder;
using celeritas::make_span;
// using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GridIdFinderTest : public celeritas::Test
{
  protected:
    using Energy  = celeritas::units::MevEnergy;
    using IdT     = celeritas::OpaqueId<struct Foo>;
    using FinderT = GridIdFinder<Energy, IdT>;

    std::vector<Energy::value_type> grid;
    std::vector<IdT>                ids;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GridIdFinderTest, all)
{
    grid = {1e-3, 1, 10, 11};
    ids  = {IdT{5}, IdT{3}, IdT{7}};

    FinderT find_id(make_span(grid), make_span(ids));
    EXPECT_EQ(IdT{}, find_id(Energy{1e-6}));
    EXPECT_EQ(IdT{5}, find_id(Energy{1e-3}));
    EXPECT_EQ(IdT{5}, find_id(Energy{0.1}));
    EXPECT_EQ(IdT{3}, find_id(Energy{1}));
    EXPECT_EQ(IdT{3}, find_id(Energy{3}));
    EXPECT_EQ(IdT{7}, find_id(Energy{10}));
    EXPECT_EQ(IdT{7}, find_id(Energy{11}));
    EXPECT_EQ(IdT{}, find_id(Energy{100}));
}
