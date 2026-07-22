//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfacePhysicsView.hh
//! \sa celeritas/optical/SurfacePhysics.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/Types.hh"
#include "celeritas/optical/Types.hh"

#include "SurfacePhysicsData.hh"
#include "SurfacePhysicsUtils.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Access physics data for a local surface based on traversal orientation.
 *
 * "Local" surfaces, which are boundaries between layers on a geometric
 * surface, are ordered in storage between two volumes based on user input.
 * These layers can be traversed based on the \c orientation, whether the track
 * entered the surface crossing in \c forward order (e.g., \em out of the
 * volume defining a boundary surface) or \c reverse (e.g., \em into that
 * volume).
 *
 * The surface physics record only stores the behavior on the surfaces and
 * interstitial materials. Because one geometric surface can be surrounded by
 * different pairs of pre/post materials (due to the physical volume
 * adjacency), it \em cannot store the pre- and post-crossing materials.
 * Those are managed by the \c SurfacePhysicsTrackView which has extra state.
 *
 * A single geometric surface with \em N interstitial layers will have
 * \f$ N + 2 \f$ "local materials" and \f$ N + 1 \f$ "local surfaces".
 * Most geometric surfaces have no interstial layers, just a pre- and post-
 * volume material with a single surface, and therefore have two local material
 * regions surrounding a single local surface.
 *
 * During crossing \c LocalSurfaceId{0} separates the pre-volume material (\c
 * LocalPositionId{0} ) and the next.
 * Physics surface IDs are contiguous <em>by construction</em>
 * in the surface physics loader and optical physics input.
 *
 * \par Example traversal
 *
 * This surface record has two interstitial materials, implying three surfaces
 * and a total of four materials during the crossing.
 * The IDs and corresponding physics data, when traversing in the forward
 * direction, are:
 * \verbatim
  local surface ID        :       LS0   LS1    LS2
  local material ID       :  LM0   | LM1 | LM2  |  LM3
  interstitial materials  : (pre)  | IM0 | IM1  | (post)
  optical materials       :  ----  | A   | B    | ----
  physics surfaces        :        | X   | X+1  |
  \endverbatim
 * Only the two interstitial materials' optical material IDs can be accessed by
 * the view.
 * The orientation of the local surfaces and interstitial materials align
 * with the \c SurfacePhysicsRecord storage \em only in the \c forward
 * orientation.
 * When traversing in \c reverse  orientation, the data layout stays the same
 * but the IDs are still relative to the pre/post volumes:
 * \verbatim
  local surface ID        :       LS2   LS1    LS0
  local material ID       :  LM3   | LM2 | LM1  |  LM0
  interstitial materials  : (post) | IM1 | IM0  | (pre)
  optical materials       :  ----  | A   | B    | ----
  physics surfaces        :        | X   | X+1  |
  \endverbatim
 *
 */
class SurfacePhysicsView
{
  public:
    //!@{
    //! \name Type aliases
    using SurfaceParamsRef = NativeCRef<SurfacePhysicsParamsData>;
    //!@}

  public:
    // Construct view from data and state
    inline CELER_FUNCTION
    SurfacePhysicsView(SurfaceParamsRef const&, SurfaceId, LocalDirection);

    //! Get geometric surface ID the track is currently on
    CELER_FIF SurfaceId surface() const { return surface_; }

    //! Get traversal orientation on the current surface
    CELER_FIF LocalDirection orientation() const { return orientation_; }

    // Get optical pre/interstitial/post material at the given track position
    inline CELER_FUNCTION OptMatId interstitial_material(LocalPositionId) const;

    // Get the physics surface at the given position and direction
    inline CELER_FUNCTION
        PhysSurfaceId interface(LocalPositionId, LocalDirection) const;

  private:
    SurfaceParamsRef const& params_;
    SurfaceId surface_;
    LocalDirection orientation_;

    // Get record data for the current surface
    inline CELER_FUNCTION SurfacePhysicsRecord const& surface_record() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from data, states, and a given track ID.
 */
CELER_FUNCTION
SurfacePhysicsView::SurfacePhysicsView(SurfaceParamsRef const& params,
                                       SurfaceId surface,
                                       LocalDirection orientation)
    : params_(params), surface_(surface), orientation_(orientation)
{
    CELER_EXPECT(surface_ < params_.surfaces.size());
}

//---------------------------------------------------------------------------//
/*!
 * Return the interstitial material ID at the given track position.
 *
 * Position should be in the range [1,N] where N is the number of subsurface
 * materials.
 */
CELER_FUNCTION OptMatId SurfacePhysicsView::interstitial_material(
    LocalPositionId pos) const
{
    auto const& ids = this->surface_record().interstitial_mat_ids;
    CELER_EXPECT(pos > LocalPositionId{0} && pos <= ids.size());

    // Position 0 is pre-volume material, and N+1 is post-volume material.
    // Offset position so it maps to the interstitial material range.
    LocalPositionId::size_type interstit_idx = pos.unchecked_get();
    if (orientation_ == LocalDirection::reverse)
    {
        interstit_idx = ids.size() - interstit_idx;
    }
    else
    {
        --interstit_idx;
    }

    ItemId<OptMatId> opt_mat_id_id = ids[interstit_idx];
    CELER_ASSERT(opt_mat_id_id < params_.opt_mat_ids.size());
    return params_.opt_mat_ids[opt_mat_id_id];
}

//---------------------------------------------------------------------------//
/*!
 * Return the next physics surface at the given position along the direction.
 */
CELER_FUNCTION PhysSurfaceId SurfacePhysicsView::interface(
    LocalPositionId pos, LocalDirection dir) const
{
    auto const& ids = this->surface_record().local_surface_ids;
    CELER_EXPECT(pos >= LocalPositionId{0} && pos <= ids.size());

    auto local_surf = local_surf_id(pos, dir);
    if (orientation_ == LocalDirection::reverse)
    {
        // Reverse index in ids: 0 -> N-1; 1 -> N-2; ...
        local_surf = id_cast<LocalSurfaceId>(
            ids.size() - 1 - local_surf.unchecked_get());
    }

    CELER_ENSURE(local_surf < ids.size());
    return ids[local_surf];
}

//---------------------------------------------------------------------------//
/*!
 * Get surface record of current geometric surface.
 */
CELER_FUNCTION SurfacePhysicsRecord const&
SurfacePhysicsView::surface_record() const
{
    return params_.surfaces[this->surface()];
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
