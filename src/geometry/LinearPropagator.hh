//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoTrackView.hh"
#include "base/NumericLimits.hh"

namespace celeritas
{
/*!
 * \brief Handler applying linear propagation to neutral tracks.
 */
class LinearPropagator
{
    using Initializer_t = GeoStateInitializer;

  public:
    //! Construct from persistent and state data
    CELER_FUNCTION
    LinearPropagator(GeoTrackView& track) : track_(track) {}

    /*!
     * \brief Move track by a user-provided distance.
     *
     * Step must be within current volume. User can ask next_step() for maximum
     * distance allowed before reaching volume boundary.
     */
    CELER_FORCEINLINE_FUNCTION
    void operator()(real_type dist);

    //! Move track by next_step(), which takes it to next volume boundary.
    CELER_FORCEINLINE_FUNCTION
    void operator()();

  private:
    //! Fast move, update only the position.
    CELER_FORCEINLINE_FUNCTION
    void apply_linear_step(real_type step);

  private:
    //@{
    //! Referenced thread-local data
    GeoTrackView& track_;
    //@}
};

} // namespace celeritas

#include "geometry/LinearPropagator.i.hh"
