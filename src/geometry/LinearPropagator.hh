//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/NumericLimits.hh"
#include "GeoTrackView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate (move) a particle in a straight line.
 */
class LinearPropagator
{
  public:
    //! Output results
    struct result_type
    {
        real_type distance; //!< Distance traveled
        VolumeId  volume;   //!< Post-propagation volume
    };

  public:
    //! Construct from persistent and state data
    inline CELER_FUNCTION LinearPropagator(GeoTrackView* track);

    //! Move track to next volume boundary.
    inline CELER_FUNCTION result_type operator()();

    //! Move track by a user-provided distance, or to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

  private:
    GeoTrackView& track_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "LinearPropagator.i.hh"
