//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/Types.hh"

#include "SurfacePhysicsData.hh"

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
    SurfacePhysicsView ...;
   \endcode
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    using SurfaceStateRef = NativeRef<SurfacePhysicsStateData>;
    //!@}

    struct Initializer
    {
        GeometricSurfaceId surface;
        SubsurfaceDirection orientation;
    };

  public:
    inline CELER_FUNCTION SurfacePhysicsView(SurfaceParamsRef const&,
                                             SurfaceStateRef const&,
                                             TrackSlotId);

    inline CELER_FUNCTION SurfacePhysicsView& operator=(Initializer const&);

    inline CELER_FUNCTION GeometricSurfaceId surface() const;
    inline CELER_FUNCTION SubsurfaceDirection orientation() const;
    inline CELER_FUNCTION bool is_crossing_boundary() const;
    inline CELER_FUNCTION void reset() const;
    inline CELER_FUNCTION bool in_pre_volume() const;
    inline CELER_FUNCTION bool in_post_volume() const;
    inline CELER_FUNCTION SurfaceTrackPosition subsurface_position() const;
    inline CELER_FUNCTION SurfaceTrackPosition& subsurface_position();
    inline CELER_FUNCTION SurfaceTrackPosition::size_type num_positions() const;
    inline CELER_FUNCTION OptMatId subsurface_material() const;
    inline CELER_FUNCTION
        PhysicsSurfaceId subsurface_interface(SubsurfaceDirection) const;
    inline CELER_FUNCTION void
        cross_subsurface_interface(SubsurfaceDirection) const;

  private:
    SurfaceTrackPosition temp_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */
CELER_FUNCTION SurfacePhysicsView::SurfacePhysicsView(SurfaceParamsRef const&,
                                                      SurfaceStateRef const&,
                                                      TrackSlotId)
{
}

CELER_FUNCTION SurfacePhysicsView&
SurfacePhysicsView::operator=(Initializer const&)
{
    return *this;
}

CELER_FUNCTION GeometricSurfaceId SurfacePhysicsView::surface() const
{
    return {};
}

CELER_FUNCTION SubsurfaceDirection SurfacePhysicsView::orientation() const
{
    return SubsurfaceDirection::forward;
}

CELER_FUNCTION bool SurfacePhysicsView::in_pre_volume() const
{
    return false;
}

CELER_FUNCTION bool SurfacePhysicsView::in_post_volume() const
{
    return false;
}

CELER_FUNCTION SurfaceTrackPosition SurfacePhysicsView::subsurface_position() const
{
    return {};
}

CELER_FUNCTION SurfaceTrackPosition& SurfacePhysicsView::subsurface_position()
{
    return temp_;
}

CELER_FUNCTION SurfaceTrackPosition::size_type
SurfacePhysicsView::num_positions() const
{
    return 0;
}

CELER_FUNCTION bool SurfacePhysicsView::is_crossing_boundary() const
{
    return false;
}

CELER_FUNCTION void SurfacePhysicsView::reset() const {}

CELER_FUNCTION OptMatId SurfacePhysicsView::subsurface_material() const
{
    return {};
}

CELER_FUNCTION PhysicsSurfaceId
SurfacePhysicsView::subsurface_interface(SubsurfaceDirection) const
{
    return {};
}

CELER_FUNCTION void
SurfacePhysicsView::cross_subsurface_interface(SubsurfaceDirection) const
{
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
