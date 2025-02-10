//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Construct validator for with the underlying storage.
 */
GridAccessor::GridAccessor(Items<real_type>* reals, Items<Grid>* grids)
    : reals_(reals), grids_(grids)
{
    CELER_EXPECT(reals_);
    CELER_EXPECT(grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve a span of reals built on the storage.
 */
Span<real_type const>
GridAccessor::operator()(ItemRange<real_type> const& real_ids) const
{
    CELER_EXPECT(real_ids.front() < real_ids.back());
    CELER_EXPECT(real_ids.back() < reals_->size());
    return (*reals_)[real_ids];
}

//---------------------------------------------------------------------------//
/*!
 * Retrieve a table of grid views built on the storage.
 *
 * Each grid view is a pair of spans representing the grid and value of a
 * \c NonuniformGridRecord.
 */
auto GridAccessor::operator()(ItemRange<Grid> grid_ids) const
    -> std::vector<GridView>
{
    std::vector<GridView> grids;
    grids.reserve(grid_ids.size());

    for (GridId grid_id : grid_ids)
    {
        CELER_EXPECT(grid_id < grids_->size());
        auto const& grid = (*grids_)[grid_id];
        CELER_EXPECT(grid);
        grids.emplace_back((*this)(grid.grid), (*this)(grid.value));
    }

    return grids;
}

//---------------------------------------------------------------------------//
/*!
 * Construct an MFP builder with the underlying collections.
 */
MfpBuilder GridAccessor::create_mfp_builder()
{
    return MfpBuilder(reals_, grids_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with internal collections.
 */
OwningGridAccessor::OwningGridAccessor() : GridAccessor(&reals_, &grids_) {}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
