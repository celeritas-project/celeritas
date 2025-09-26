//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceModelView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/Types.hh"
#include "celeritas/phys/SurfacePhysicsMapView.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Optical surface data for a model.
 *
 * Wraps common behavior for querying the surface model data for a given
 * physics surface interface.
 */
class SurfaceModelView
{
  public:
    // Construct from map view and materials
    inline CELER_FUNCTION SurfaceModelView(SurfacePhysicsMapView,
                                           OptMatId pre_mat,
                                           OptMatId post_mat);

    // Get surface model ID
    inline CELER_FUNCTION SurfaceModelId model_id() const;

    // Get internal surface ID for the model
    inline CELER_FUNCTION SubModelId internal_surface_id() const;

    // Get pre-volume optical material
    inline CELER_FUNCTION OptMatId pre_material() const;

    // Get post-volume optical material
    inline CELER_FUNCTION OptMatId post_material() const;

  private:
    SurfacePhysicsMapView physics_map_;
    OptMatId pre_material_;
    OptMatId post_material_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from physics map view and materials.
 */
CELER_FUNCTION
SurfaceModelView::SurfaceModelView(SurfacePhysicsMapView physics_map,
                                   OptMatId pre_material,
                                   OptMatId post_material)
    : physics_map_(physics_map)
    , pre_material_(pre_material)
    , post_material_(post_material)
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface model for this physics surface.
 */
CELER_FUNCTION SurfaceModelId SurfaceModelView::model_id() const
{
    return physics_map_.surface_model_id();
}

//---------------------------------------------------------------------------//
/*!
 * Get the internal surface ID for the physics surface in this model.
 */
CELER_FUNCTION auto SurfaceModelView::internal_surface_id() const -> SubModelId
{
    return physics_map_.internal_surface_id();
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical material before the interface.
 */
CELER_FUNCTION OptMatId SurfaceModelView::pre_material() const
{
    return pre_material_;
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical material after the interface.
 */
CELER_FUNCTION OptMatId SurfaceModelView::post_material() const
{
    return post_material_;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
