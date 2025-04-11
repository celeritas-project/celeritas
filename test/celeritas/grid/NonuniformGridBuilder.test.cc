//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/NonuniformGridBuilder.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/grid/NonuniformGridBuilder.hh"

#include <iostream>
#include <vector>

#include "celeritas/inp/Physics.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class NonuniformGridBuilderTest : public ::celeritas::test::Test
{
  protected:
    NonuniformGridBuilder make_builder()
    {
        return NonuniformGridBuilder(&scalars_);
    }

    static Span<real_type const> span_grid() { return make_span(grid_); }

    static Span<real_type const> span_values() { return make_span(values_); }

    Collection<real_type, Ownership::value, MemSpace::host> scalars_;

    constexpr static real_type grid_[] = {0.0, 0.4, 0.9, 1.3};
    constexpr static real_type values_[] = {-31.0, 12.1, 15.5, 92.0};
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(NonuniformGridBuilderTest, build_span)
{
    auto builder = make_builder();

    NonuniformGridRecord grid_data = builder(span_grid(), span_values());

    ASSERT_TRUE(grid_data);
    ASSERT_EQ(8, scalars_.size());
    ASSERT_EQ(4, grid_data.grid.size());
    ASSERT_EQ(4, grid_data.value.size());

    EXPECT_VEC_SOFT_EQ(grid_, scalars_[grid_data.grid]);
    EXPECT_VEC_SOFT_EQ(values_, scalars_[grid_data.value]);
}

TEST_F(NonuniformGridBuilderTest, TEST_IF_CELERITAS_DOUBLE(build_vec))
{
    inp::Grid vect;
    vect.x = std::vector<double>(span_grid().begin(), span_grid().end());
    vect.y = std::vector<double>(span_values().begin(), span_values().end());

    auto builder = make_builder();

    NonuniformGridRecord grid_data = builder(vect);

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
