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
    //!@{
    //! \name Type aliases
    using InternalSurfaceId = SurfacePhysicsMapView::InternalSurfaceId;
    //!@}

  public:
    // Construct from a direction and map view
    inline CELER_FUNCTION
    SurfaceModelView(SubsurfaceDirection, SurfacePhysicsMapView const&);

    // Get subsurface track direction
    inline CELER_FUNCTION SubsurfaceDirection direction() const;

    // Get surface model ID
    inline CELER_FUNCTION SurfaceModelId surface_model() const;

    // Get internal surface ID for the model
    inline CELER_FUNCTION InternalSurfaceId internal_surface_id() const;

    // Get pre-volume optical material
    inline CELER_FUNCTION OptMatId pre_material() const;

    // Get post-volume optical material
    inline CELER_FUNCTION OptMatId post_material() const;

  private:
    SubsurfaceDirection dir_;
    SurfacePhysicsMapView physics_map_;
    // OptMatId pre_material_;
    // OptMatId post_material_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from track direction and physics map view.
 */
CELER_FUNCTION
SurfaceModelView::SurfaceModelView(SubsurfaceDirection dir,
                                   SurfacePhysicsMapView const& physics_map)
    : dir_(dir), physics_map_(physics_map)
{
}

//---------------------------------------------------------------------------//
/*!
 * Get the subsurface track direction pointing to this surface.
 */
CELER_FUNCTION SubsurfaceDirection SurfaceModelView::direction() const
{
    return SubsurfaceDirection::forward;
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface model for this physics surface.
 */
CELER_FUNCTION SurfaceModelId SurfaceModelView::surface_model() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the internal surface ID for the physics surface in this model.
 */
CELER_FUNCTION auto SurfaceModelView::internal_surface_id() const
    -> InternalSurfaceId
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical material before the interface.
 */
CELER_FUNCTION OptMatId SurfaceModelView::pre_material() const
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Get the optical material after the interface.
 */
CELER_FUNCTION OptMatId SurfaceModelView::post_material() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
