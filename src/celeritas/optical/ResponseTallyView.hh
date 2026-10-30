//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyView.hh
//! \sa ResponseTally.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/grid/UniformGrid.hh"
#include "geocel/Types.hh"

#include "ResponseTallyData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Map a detector, position, and time to a compressed tally bin.
 */
class ResponseTallyView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<ResponseTallyParamsData>;
    //!@}

  public:
    //! Bin ID for a point outside the grids or detector range
    static CELER_CONSTEXPR_FUNCTION size_type invalid_bin()
    {
        return static_cast<size_type>(-1);
    }

    // Construct with params
    explicit inline CELER_FUNCTION ResponseTallyView(ParamsRef const& params);

    //! Total number of bins
    CELER_FUNCTION size_type num_bins() const { return params_.num_bins(); }

    // Calculate the flattened bin, or \c invalid_bin if out of range
    inline CELER_FUNCTION size_type bin_id(
        size_type detector, Real3 const& pos, real_type time) const;

  private:
    ParamsRef const& params_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with params.
 */
CELER_FUNCTION ResponseTallyView::ResponseTallyView(ParamsRef const& params)
    : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the flattened bin ID of a hit.
 *
 * Points outside any grid (including those at or beyond the upper edge) and
 * detector IDs beyond the number of detectors are not scored.
 */
CELER_FUNCTION size_type ResponseTallyView::bin_id(
    size_type detector, Real3 const& pos, real_type time) const
{
    if (detector >= params_.dims[0])
    {
        return invalid_bin();
    }

    Array<size_type, 5> coords;
    coords[0] = detector;

    UniformGridData const* const grids[]
        = {&params_.x_grid, &params_.y_grid, &params_.z_grid};
    for (size_type i = 0; i < 3; ++i)
    {
        UniformGrid grid(*grids[i]);
        if (!(pos[i] >= grid.front() && pos[i] < grid.back()))
        {
            return invalid_bin();
        }
        coords[i + 1] = grid.find(pos[i]);
    }

    UniformGrid time_grid(params_.time_grid);
    if (!(time >= time_grid.front() && time < time_grid.back()))
    {
        return invalid_bin();
    }
    coords[4] = time_grid.find(time);

    HyperslabIndexer<5> index(params_.dims);
    return index(coords);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
