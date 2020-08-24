//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file LinearPropagationHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geometry/GeoTrackView.hh"

namespace celeritas {

/**
 * @brief Handler grouping neutral tracks and performing linear propagation.
 */

class LinearPropagationHandler {

protected:
private:
  LinearPropagationHandler(const LinearPropagationHandler &) = delete;
  LinearPropagationHandler &operator=(const LinearPropagationHandler &) = delete;

  CELER_FORCEINLINE_FUNCTION
  void quickLinearStep(GeoTrackView &track, real_type step);

  CELER_FORCEINLINE_FUNCTION
  void commitStepUpdates(GeoTrackView &track);
  
public:
  /** @brief Default constructor */
  LinearPropagationHandler() = default;

  /** @briefdestructor */
  ~LinearPropagationHandler() = default;

  /** @brief Scalar DoIt interface */
  CELER_FORCEINLINE_FUNCTION
  bool Propagate(GeoTrackView &track);

  CELER_FORCEINLINE_FUNCTION
  bool not_at_boundary(GeoTrackView const& track) const;
};

} // namespace celeritas

#include "geometry/LinearPropagationHandler.i.hh"
