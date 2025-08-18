//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "SurfacePhysicsMapData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access surface physics mappings for a particular surface.
 *
 * This simply encapsulates the \c SurfaceParamsData class.
 */
class SurfacePhysicsMapView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsMapData>;
    using ModelSurfaceId = SurfaceModel::ModelSurfaceId;
    //!@}

  public:
    // Construct from data and current surface
    CELER_FUNCTION
    SurfacePhysicsMapView(SurfaceParamsRef const& params, SurfaceId surface);

    // Get the action ID for the current surface, if any
    CELER_FUNCTION ActionId action_id() const;

    // Get the subindex inside that model
    CELER_FUNCTION ModelSurfaceId model_surface_id() const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceId surface_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data and current surface.
 */
CELER_FUNCTION
SurfacePhysicsMapView::SurfacePhysicsMapView(SurfaceParamsRef const& params,
                                             SurfaceId surface)
    : params_{params}, surface_{surface}
{
    CELER_EXPECT(params_);
    CELER_EXPECT(surface_ < params.action_ids.size());
}

//---------------------------------------------------------------------------//
/*!
 * Get the action ID for the current surface, if any.
 */
CELER_FUNCTION ActionId SurfacePhysicsMapView::action_id() const
{
    return params_.action_ids[surface_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the subindex inside that model.
 */
CELER_FUNCTION auto SurfacePhysicsMapView::model_surface_id() const
    -> ModelSurfaceId
{
    return params_.model_surface_ids[surface_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
