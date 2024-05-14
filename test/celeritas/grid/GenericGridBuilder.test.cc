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
    GenericGridBuilder make_builder() { return GenericGridBuilder(&scalars_); }

    static Span<double const> span_grid() { return make_span(grid_); }

    static Span<double const> span_values() { return make_span(values_); }

    Collection<double, Ownership::value, MemSpace::host> scalars_;

    constexpr static double grid_[] = {0.0, 0.4, 0.9, 1.3};
    constexpr static double values_[] = {-31.0, 12.1, 15.5, 92.0};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GenericGridBuilderTest, build_span)
{
    auto builder = make_builder();

    GenericGridData grid_data = builder(span_grid(), span_values());

    ASSERT_TRUE(grid_data);
    ASSERT_EQ(8, scalars_.size());
    ASSERT_EQ(4, grid_data.grid.size());
    ASSERT_EQ(4, grid_data.value.size());

    EXPECT_VEC_SOFT_EQ(grid_, scalars_[grid_data.grid]);
    EXPECT_VEC_SOFT_EQ(values_, scalars_[grid_data.value]);
}

TEST_F(GenericGridBuilderTest, build_vec)
{
    ImportPhysicsVector vect;
    vect.vector_type = ImportPhysicsVectorType::free;
    vect.x = std::vector<double>(span_grid().begin(), span_grid().end());
    vect.y = std::vector<double>(span_values().begin(), span_values().end());

    auto builder = make_builder();

    GenericGridData grid_data = builder(vect);

    ASSERT_TRUE(grid_data);
    ASSERT_EQ(8, scalars_.size());
    ASSERT_EQ(4, grid_data.grid.size());
    ASSERT_EQ(4, grid_data.value.size());

    EXPECT_VEC_SOFT_EQ(grid_, scalars_[grid_data.grid]);
    EXPECT_VEC_SOFT_EQ(values_, scalars_[grid_data.value]);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
