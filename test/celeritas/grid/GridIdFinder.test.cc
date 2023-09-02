//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GridIdFinder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/GridIdFinder.hh"

#include "corecel/OpaqueId.hh"
#include "celeritas/Quantities.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GridIdFinderTest : public Test
{
  protected:
    using Energy = units::MevEnergy;
    using IdT = OpaqueId<struct Foo_>;
    using FinderT = GridIdFinder<Energy, IdT>;

    std::vector<Energy::value_type> grid;
    std::vector<IdT> ids;
};

TEST_F(GridIdFinderTest, all)
{
    constexpr auto invalid = IdT{}.unchecked_get();

    grid = {1e-3, 1, 10, 11};
    ids = {IdT{5}, IdT{3}, IdT{7}};

    FinderT find_id(make_span(grid), make_span(ids));
    EXPECT_EQ(invalid, find_id(Energy{1e-6}).unchecked_get());
    EXPECT_EQ(5, find_id(Energy{1e-3}).unchecked_get());
    EXPECT_EQ(5, find_id(Energy{0.1}).unchecked_get());
    EXPECT_EQ(3, find_id(Energy{1}).unchecked_get());
    EXPECT_EQ(3, find_id(Energy{3}).unchecked_get());
    EXPECT_EQ(7, find_id(Energy{10}).unchecked_get());
    EXPECT_EQ(7, find_id(Energy{11}).unchecked_get());
    EXPECT_EQ(invalid, find_id(Energy{100}).unchecked_get());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
