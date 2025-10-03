//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/MfpBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/Logger.hh"
#include "celeritas/grid/NonuniformGridInserter.hh"
#include "celeritas/inp/Grid.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for optical models to build MFP tables.
 *
 * Tracks individual grid IDs that have been built, and returns them
 * as an ItemRange which may be used by model MFP tables.
 */
class MfpBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using GridId = OpaqueId<NonuniformGridRecord>;
    using GridInserter = NonuniformGridInserter<GridId>;
    using GridIdRange = Range<GridId>;

    using Values = typename GridInserter::Values;
    using GridValues = typename GridInserter::GridValues;
    //!@}

  public:
    // Construct with given inserter
    inline MfpBuilder(Values* real_data, GridValues* grid_data);

    // Build the grid
    inline void operator()(inp::Grid const& grid);

    // Build an empty grid for models that do not apply in the material
    inline void operator()();

    // Get the range of grid IDs that have been built
    inline GridIdRange grid_ids() const;

  private:
    GridInserter insert_grid_;
    GridValues* grid_data_;
    GridId const grid_id_first_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with given collections.
 */
MfpBuilder::MfpBuilder(Values* real_data, GridValues* grid_data)
    : insert_grid_(real_data, grid_data)
    , grid_data_(grid_data)
    , grid_id_first_(grid_data->size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Build a grid.
 */
void MfpBuilder::operator()(inp::Grid const& grid)
{
    CELER_EXPECT(!grid || grid.x.front() >= 0);
    if (!grid)
    {
        // Build empty cross sections
        return (*this)();
    }

    insert_grid_(grid);
}

//---------------------------------------------------------------------------//
/*!
 * Build an empty grid for zero interaction probability.
 */
void MfpBuilder::operator()()
{
    insert_grid_();
}

//---------------------------------------------------------------------------//
/*!
 * Get the range of grid IDs that have been built.
 */
auto MfpBuilder::grid_ids() const -> GridIdRange
{
    return GridIdRange(grid_id_first_, GridId{grid_data_->size()});
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
