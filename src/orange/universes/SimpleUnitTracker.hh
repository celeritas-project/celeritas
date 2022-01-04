//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SimpleUnitTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include "orange/Data.hh"
#include "orange/surfaces/Surfaces.hh"
#include "detail/LogicEvaluator.hh"
#include "detail/SenseCalculator.hh"
#include "detail/SurfaceFunctors.hh"
#include "detail/Types.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Track a particle in a universe of well-connected volumes.
 *
 * The simple unit tracker is based on a set of non-overlapping volumes
 * comprised of surfaces. It is a faster but less "user-friendly" version of
 * the masked unit tracker because it requires all volumes to be exactly
 * defined by their connected surfaces. It does *not* check for overlaps.
 */
class SimpleUnitTracker
{
  public:
    //!@{
    //! Type aliases
    using ParamsRef
        = OrangeParamsData<Ownership::const_reference, MemSpace::native>;
    using Initialization = detail::Initialization;
    using Intersection   = detail::Intersection;
    using LocalState     = detail::LocalState;
    //!@}

  public:
    // Construct with parameters (surfaces, cells)
    inline CELER_FUNCTION SimpleUnitTracker(const ParamsRef& params);

    // Find the local cell and possibly surface ID.
    inline CELER_FUNCTION Initialization initialize(LocalState state) const;

    // Calculate the distance to an exiting face for the current volume.
    inline CELER_FUNCTION Intersection intersect(LocalState state) const;

  private:
    //// DATA ////
    const ParamsRef& params_;

    //// METHODS ////
    inline CELER_FUNCTION
        Intersection simple_intersect(LocalState, VolumeView) const;
    inline CELER_FUNCTION
        Intersection complex_intersect(LocalState, VolumeView) const;
};

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent parameter data.
 *
 * \todo When adding multiple universes, this will calculate range of VolumeIds
 * that belong to this unit. For now we assume all volumes and surfaces belong
 * to us.
 */
CELER_FUNCTION SimpleUnitTracker::SimpleUnitTracker(const ParamsRef& params)
    : params_(params)
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the local cell and possibly surface ID.
 *
 * This function is valid for initialization from a point, *and* for
 * initialization across a boundary.
 */
CELER_FUNCTION auto SimpleUnitTracker::initialize(LocalState state) const
    -> Initialization
{
    CELER_EXPECT(params_);

    detail::SenseCalculator calc_senses(
        Surfaces{params_.surfaces}, state.pos, state.temp_sense);

    for (VolumeId volid : range(VolumeId{params_.volumes.size()}))
    {
        if (state.surface && volid == state.volume)
        {
            // Cannot cross surface into the same cell
            continue;
        }

        VolumeView vol{params_.volumes, volid};

        // Calculate the local senses and face, and see if we're inside.
        auto logic_state
            = calc_senses(vol, detail::find_face(vol, state.surface));
        bool found = detail::LogicEvaluator(vol.logic())(logic_state.senses);
        if (!found)
        {
            // Try the next cell
            continue;
        }
        if (!state.surface && logic_state.face)
        {
            // Initialized on a boundary in this cell but wasn't known
            // to be crossing a surface. Fail safe by letting the multi-level
            // tracking geometry bump and try again.
            break;
        }

        // Found and not unexpectedly on a boundary!
        return {volid, get_surface(vol, logic_state.face)};
    }

    // Failed to find a valid volume containing the point
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 */
CELER_FUNCTION auto SimpleUnitTracker::intersect(LocalState state) const
    -> Intersection
{
    CELER_EXPECT(state.volume && !state.temp_sense.empty());

    // Resize temporaries based on volume properties
    VolumeView vol{params_.volumes, state.volume};
    CELER_ASSERT(state.temp_next.size >= vol.num_intersections());
    state.temp_next.size = vol.num_intersections();
    const bool is_simple = !(vol.flags() & VolumeView::internal_surfaces);

    // Find all surface intersection distances inside this volume
    {
        auto calc_intersections = make_surface_action(
            Surfaces{params_.surfaces},
            detail::CalcIntersections{
                state.pos,
                state.dir,
                state.surface ? vol.find_face(state.surface.id()) : FaceId{},
                is_simple,
                state.temp_next});
        for (SurfaceId surface : vol.faces())
        {
            calc_intersections(surface);
        }
        CELER_ASSERT(calc_intersections.action().face_idx() == vol.num_faces());
        CELER_ASSERT(calc_intersections.action().isect_idx()
                     == vol.num_intersections());
    }

    if (is_simple)
    {
        // No interior surfaces: closest distance is next boundary
        return this->simple_intersect(state, vol);
    }
    else
    {
        return this->complex_intersect(state, vol);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance to the next boundary for nonreentrant cells.
 */
CELER_FUNCTION auto
SimpleUnitTracker::simple_intersect(LocalState state, VolumeView vol) const
    -> Intersection
{
    CELER_EXPECT(state.temp_next && vol.num_intersections() > 0);

    // Crossing any surface will leave the cell; perform a linear search for
    // the smallest (but positive) distance
    size_type distance_idx;
    {
        const real_type* distance_ptr = celeritas::min_element(
            state.temp_next.distance,
            state.temp_next.distance + state.temp_next.size,
            detail::CloserPositiveDistance{});
        CELER_ASSERT(*distance_ptr > 0);
        distance_idx = distance_ptr - state.temp_next.distance;
    }

    if (state.temp_next.distance[distance_idx] == no_intersection())
        return {};

    // Determine the crossing surface
    SurfaceId surface;
    {
        FaceId face = state.temp_next.face[distance_idx];
        CELER_ASSERT(face);
        surface = vol.get_surface(face);
        CELER_ASSERT(surface);
    }

    Sense cur_sense;
    if (surface == state.surface.id())
    {
        // Crossing the same surface that we're currently on and inside (e.g.
        // on the inside surface of a sphere, and the next intersection is the
        // other side)
        cur_sense = state.surface.sense();
    }
    else
    {
        auto calc_sense = make_surface_action(Surfaces{params_.surfaces},
                                              detail::CalcSense{state.pos});

        SignedSense ss = calc_sense(surface);
        CELER_ASSERT(ss != SignedSense::on);
        cur_sense = to_sense(ss);
    }

    // Post-surface sense will be on the other side of the surface
    Intersection result;
    result.surface  = {surface, flip_sense(cur_sense)};
    result.distance = state.temp_next.distance[distance_idx];
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate boundary distance if internal surfaces are present.
 *
 * In "complex" cells, crossing a surface can still leave the particle in an
 * "inside" state.
 *
 * We have to iteratively track through all surfaces, in order of minimum
 * distance, to determine whether crossing them in sequence will cause us to
 * exit the cell.
 */
CELER_FUNCTION auto
SimpleUnitTracker::complex_intersect(LocalState state, VolumeView vol) const
    -> Intersection
{
    // Partition intersections (enumerated from 0 as the `idx` array) into
    // valid (finite positive) and invalid (infinite-or-negative) groups.
    size_type num_isect = celeritas::partition(
                              state.temp_next.isect,
                              state.temp_next.isect + state.temp_next.size,
                              detail::IntersectionPartitioner{state.temp_next})
                          - state.temp_next.isect;
    CELER_ASSERT(num_isect <= state.temp_next.size);

    // Sort these finite distances in ascending order
    celeritas::sort(state.temp_next.isect,
                    state.temp_next.isect + num_isect,
                    [&state](size_type a, size_type b) {
                        return state.temp_next.distance[a]
                               < state.temp_next.distance[b];
                    });

    // Calculate local senses, taking current face into account
    auto logic_state = detail::SenseCalculator(
        Surfaces{params_.surfaces}, state.pos, state.temp_sense)(
        vol, detail::find_face(vol, state.surface));

    // Current senses should put us inside the cell
    detail::LogicEvaluator is_inside(vol.logic());
    CELER_ASSERT(is_inside(logic_state.senses));

    // Loop over distances and surface indices to cross by iterating over
    // temp_next.isect[:num_isect].
    // Evaluate the logic expression at each crossing to determine whether
    // we're actually leaving the cell.
    for (size_type isect_idx = 0; isect_idx != num_isect; ++isect_idx)
    {
        // Index into the distance/face arrays
        const size_type isect = state.temp_next.isect[isect_idx];
        // Face being crossed in this ordered intersection
        FaceId face = state.temp_next.face[isect];
        // Flip the sense of the face being crossed
        Sense new_sense = flip_sense(logic_state.senses[face.get()]);
        logic_state.senses[face.unchecked_get()] = new_sense;
        if (!is_inside(logic_state.senses))
        {
            // Flipping this sense puts us outside the current volume: in
            // other words, only after crossing all the internal surfaces along
            // this direction do we hit a surface that actually puts us
            // outside.
            Intersection result;
            result.surface  = {vol.get_surface(face), new_sense};
            result.distance = state.temp_next.distance[isect];
            CELER_ENSURE(result.distance > 0 && !std::isinf(result.distance));
            return result;
        }
    }

    // No intersection: perhaps leaving an exterior cell? Perhaps geometry
    // error.
    return {};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
