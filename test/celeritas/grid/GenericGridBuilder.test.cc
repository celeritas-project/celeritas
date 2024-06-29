//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/GenericGridBuilder.hh"

#include <iostream>
#include <vector>

#include "celeritas/grid/GenericGridInserter.hh"
#include "celeritas/io/ImportPhysicsVector.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GenericGridBuilderTest : public ::celeritas::test::Test
{
  protected:
    using GridIndex = OpaqueId<struct GenericGridTag_>;

    static Span<real_type const> span_grid() { return make_span(grid_); }
    static Span<real_type const> span_values() { return make_span(values_); }

    Collection<real_type, Ownership::value, MemSpace::host> scalars_;
    Collection<GenericGridData, Ownership::value, MemSpace::host, GridIndex> grids_;

    constexpr static real_type grid_[] = {0.0, 0.4, 0.9, 1.3};
    constexpr static real_type values_[] = {-31.0, 12.1, 15.5, 92.0};

    void check(GridIndex grid_index) const
    {
        ASSERT_TRUE(static_cast<bool>(grid_index));
        ASSERT_EQ(1, grids_.size());
        ASSERT_LT(grid_index, grids_.size());

        GenericGridData const& grid_data = grids_[grid_index];

        ASSERT_TRUE(grid_data);
        ASSERT_EQ(8, scalars_.size());
        ASSERT_EQ(4, grid_data.grid.size());
        ASSERT_EQ(4, grid_data.value.size());

        EXPECT_VEC_EQ(grid_, scalars_[grid_data.grid]);
        EXPECT_VEC_EQ(values_, scalars_[grid_data.value]);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GenericGridBuilderTest, build_span)
{
    GenericGridBuilder builder(span_grid(), span_values());
    GridIndex grid_index
        = builder.build(GenericGridInserter{&scalars_, &grids_});

    check(grid_index);
}

TEST_F(GenericGridBuilderTest, TEST_IF_CELERITAS_DOUBLE(from_geant))
{
    auto builder = GenericGridBuilder::from_geant(span_grid(), span_values());
    GridIndex grid_index
        = builder->build(GenericGridInserter{&scalars_, &grids_});

    check(grid_index);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
