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
namespace testdetail
{
//---------------------------------------------------------------------------//
/*!
 * Template specialization to compare \c ImportPhysicsVector for equivalence.
 *
 * Uses exact comparison to compare imported vectors that have been copied
 * or referenced, and therefore should have the same bit representation as
 * the source vectors.
 */
template<>
::testing::AssertionResult IsVecEq<ImportPhysicsVector, ImportPhysicsVector>(
    char const* expected_expr,
    char const* actual_expr,
    ImportPhysicsVector const& expected,
    ImportPhysicsVector const& actual)
{
    auto x_result = IsVecEq(expected_expr, actual_expr, expected.x, actual.x);
    auto y_result = IsVecEq(expected_expr, actual_expr, expected.y, actual.y);

    ::testing::AssertionResult result(x_result && y_result);
    if (!x_result)
    {
        result << "x values:\n" << x_result.message();
    }
    if (!y_result)
    {
        result << "y values:\n" << y_result.message();
    }

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace testdetail

namespace optical
{
namespace test
{
using namespace ::celeritas::test;
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
 * Construct with internal collections.
 */
OwningGridValidator::OwningGridValidator() : GridValidator(&reals_, &grids_) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
