//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyView.hh
//! \sa ResponseTally.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
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
 * Map photon start and end points to a compressed tally bin.
 *
 * The bin of a detected photon combines where it started (x, y, z) with where
 * and when it ended (detector, time). Because the flattened bin index is
 * linear in its coordinates, the two halves can be computed independently
 * and added:
 * - \c start_bin gives the bin with detector and time coordinates of zero,
 *   and is saved when the track is created;
 * - \c end_bin adds the detector and time contributions when the photon hits
 *   a detector.
 *
 * Points outside the grids, and detectors beyond the tally's range, result in
 * a null bin ID which is not scored.
 */
class ResponseTallyView
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<ResponseTallyParamsData>;
    //!@}

  public:
    // Construct with params
    explicit inline CELER_FUNCTION ResponseTallyView(ParamsRef const& params);

    //! Total number of bins
    CELER_FUNCTION size_type num_bins() const { return params_.num_bins(); }

    // Calculate the partial bin from the spatial starting point
    inline CELER_FUNCTION ResponseBinId start_bin(Real3 const& pos) const;

    // Add the detector and time contribution to a starting bin
    inline CELER_FUNCTION ResponseBinId end_bin(
        ResponseBinId start, DetectorId detector, real_type time) const;

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
 * Calculate the partial bin ID from the spatial starting point.
 *
 * The detector and time coordinates of the result are zero. Positions outside
 * any spatial grid (including at or beyond the upper edge) are not scored and
 * result in a null ID.
 */
CELER_FUNCTION ResponseBinId ResponseTallyView::start_bin(
    Real3 const& pos) const
{
    Array<UniformGridData const*, 3> const grids{
        &params_.x_grid, &params_.y_grid, &params_.z_grid};

    // Detector and time coordinates are filled in at the end
    Array<size_type, 5> coords{};
    for (size_type i : range(size_type{3}))
    {
        UniformGrid grid(*grids[i]);
        if (!(pos[i] >= grid.front() && pos[i] < grid.back()))
        {
            return nullid;
        }
        coords[i + 1] = grid.find(pos[i]);
    }

    HyperslabIndexer<5> index(params_.dims);
    return ResponseBinId{index(coords)};
}

//---------------------------------------------------------------------------//
/*!
 * Add the detector and time contribution to a partial bin ID.
 *
 * A null starting bin or a time outside the time grid gives a null result, as
 * does a detector ID outside the range of the tally.
 */
CELER_FUNCTION ResponseBinId ResponseTallyView::end_bin(
    ResponseBinId start, DetectorId detector, real_type time) const
{
    CELER_EXPECT(detector);
    if (!start || detector.unchecked_get() >= params_.dims[0])
    {
        return nullid;
    }

    UniformGrid time_grid(params_.time_grid);
    if (!(time >= time_grid.front() && time < time_grid.back()))
    {
        return nullid;
    }

    Array<size_type, 5> coords{};
    coords[0] = detector.unchecked_get();
    coords[4] = time_grid.find(time);

    HyperslabIndexer<5> index(params_.dims);
    return ResponseBinId{start.unchecked_get() + index(coords)};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
