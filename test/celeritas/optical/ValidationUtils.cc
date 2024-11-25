//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ValidationUtils.cc
//---------------------------------------------------------------------------//
#include "ValidationUtils.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//
/*!
 * Compare expected and actual \c ImportPhysicsVector expecting them to be
 * exactly equal.
 */
void check_physics_vector(ImportPhysicsVector const& expected,
                          ImportPhysicsVector const& actual)
{
    EXPECT_EQ(expected.vector_type, actual.vector_type);
    EXPECT_VEC_EQ(expected.x, actual.x);
    EXPECT_VEC_EQ(expected.y, actual.y);
}

//---------------------------------------------------------------------------//
/*!
 * Construct validator for with the underlying storage.
 */
GridValidator::GridValidator(Items<real_type>* reals, Items<Grid>* grids)
    : reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals_);
    CELER_EXPECT(grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Check the imported data is built under the given grid ID range.
 */
void GridValidator::check_built_table(ImportPhysicsTable const& table,
                                      ItemRange<Grid> grid_ids,
                                      Softness soft)
{
    ASSERT_EQ(table.size(), grid_ids.size());

    for (auto i : range(grid_ids.size()))
    {
        this->check_built_grid(table[i], grid_ids[i], soft);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Check the imported data is built under the given ID range.
 */
void GridValidator::check_built_grid(ImportPhysicsVector const& expected,
                                     GridId grid_id,
                                     Softness soft)
{
    ASSERT_LT(grid_id, grids_->size());
    Grid const& grid = (*grids_)[grid_id];
    ASSERT_TRUE(grid);

    this->check_built_vector(expected.x, grid.grid, soft);
    this->check_built_vector(expected.y, grid.value, soft);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an MFP builder with the underlying collections.
 */
MfpBuilder GridValidator::create_mfp_builder()
{
    return MfpBuilder(reals_, grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Check the imported data is built in the given data range.
 */
void GridValidator::check_span(Span<real_type const> const& t,
                               ItemRange<real_type> const& real_ids,
                               Softness soft)
{
    ASSERT_LT(real_ids.front(), real_ids.back());
    ASSERT_LT(real_ids.back(), reals_->size());

    switch (soft)
    {
        case Soft:
            EXPECT_VEC_SOFT_EQ(t, (*reals_)[real_ids]);
            break;
        case Exact:
            EXPECT_VEC_EQ(t, (*reals_)[real_ids]);
            break;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with internal collections.
 */
GridStorage::GridStorage() : GridValidator(&reals_, &grids_) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
