//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/SurfacePhysicsMapView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "SurfaceModel.hh"
#include "SurfacePhysicsMapData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Access surface physics mappings for a particular physics surface.
 *
 * This simply encapsulates the \c SurfaceParamsData class.
 */
class SurfacePhysicsMapView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsMapData>;
    //!@}

  public:
    // Construct from data and current surface
    inline CELER_FUNCTION SurfacePhysicsMapView(SurfaceParamsRef const& params,
                                                PhysSurfaceId surface);

    // Construct from data and "default" surface
    explicit inline CELER_FUNCTION
    SurfacePhysicsMapView(SurfaceParamsRef const& params);

    // Get the model ID for the current surface, if any
    inline CELER_FUNCTION SurfaceModelId surface_model_id() const;

    //! Current physics surface ID
    CELER_FUNCTION PhysSurfaceId surface_id() const { return surface_; }

    // Get the subindex inside that model
    inline CELER_FUNCTION SubModelId internal_surface_id() const;

  private:
    SurfaceParamsRef const& params_;
    PhysSurfaceId surface_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data and current surface.
 */
CELER_FUNCTION
SurfacePhysicsMapView::SurfacePhysicsMapView(SurfaceParamsRef const& params,
                                             PhysSurfaceId surface)
    : params_{params}, surface_{surface}
{
    CELER_EXPECT(params_);
    CELER_EXPECT(surface_ < params.surface_models.size());
}

//---------------------------------------------------------------------------//
/*!
 * Construct from data and "no surface".
 *
 * This provides default surface models for boundaries without user-specified
 * surfaces.
 */
CELER_FUNCTION
SurfacePhysicsMapView::SurfacePhysicsMapView(SurfaceParamsRef const& params)
    : SurfacePhysicsMapView{
          params, id_cast<PhysSurfaceId>(params.surface_models.size() - 1)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for the current surface, if any.
 */
CELER_FUNCTION auto SurfacePhysicsMapView::surface_model_id() const
    -> SurfaceModelId
{
    return params_.surface_models[surface_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the subindex for data inside that model.
 */
CELER_FUNCTION auto SurfacePhysicsMapView::internal_surface_id() const
    -> SubModelId
{
    return params_.internal_surface_ids[surface_];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
