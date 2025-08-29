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
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    SurfaceModelView ...;
   \endcode
 */
class SurfaceModelView
{
  public:
    //!@{
    //! \name Type aliases
    using InternalSurfaceId = SurfacePhysicsMapView::InternalSurfaceId;
    //!@}

  public:
    inline CELER_FUNCTION
    SurfaceModelView(SubsurfaceDirection, SurfacePhysicsMapView const&);

    inline CELER_FUNCTION SubsurfaceDirection direction() const;
    // inline CELER_FUNCTION PhysSurfaceId phys_surface_id() const;
    inline CELER_FUNCTION SurfaceModelId surface_model() const;
    inline CELER_FUNCTION InternalSurfaceId internal_surface_id() const;
    inline CELER_FUNCTION OptMatId pre_material() const;
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
 */
CELER_FUNCTION
SurfaceModelView::SurfaceModelView(SubsurfaceDirection dir,
                                   SurfacePhysicsMapView const& physics_map)
    : dir_(dir), physics_map_(physics_map)
{
}

CELER_FUNCTION SubsurfaceDirection SurfaceModelView::direction() const
{
    return SubsurfaceDirection::forward;
}
CELER_FUNCTION SurfaceModelId SurfaceModelView::surface_model() const
{
    return {};
}
CELER_FUNCTION auto SurfaceModelView::internal_surface_id() const
    -> InternalSurfaceId
{
    return {};
}
CELER_FUNCTION OptMatId SurfaceModelView::pre_material() const
{
    return {};
}
CELER_FUNCTION OptMatId SurfaceModelView::post_material() const
{
    return {};
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
