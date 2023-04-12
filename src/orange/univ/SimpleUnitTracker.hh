//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/SimpleUnitTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>

#include "corecel/Assert.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"
#include "orange/surf/Surfaces.hh"

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
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    using Initialization = detail::Initialization;
    using Intersection = detail::Intersection;
    using LocalState = detail::LocalState;
    //!@}

  public:
    // Construct with parameters (unit definitions and this one's ID)
    inline CELER_FUNCTION
    SimpleUnitTracker(ParamsRef const& params, SimpleUnitId id);

    //// ACCESSORS ////

    //! Number of local volumes
    CELER_FUNCTION LocalVolumeId::size_type num_volumes() const
    {
        return unit_record_.volumes.size();
    }

    //! Number of local surfaces
    CELER_FUNCTION LocalSurfaceId::size_type num_surfaces() const
    {
        return unit_record_.surfaces.size();
    }

    //! SimpleUnitRecord for this tracker
    CELER_FUNCTION SimpleUnitRecord const& unit_record() const
    {
        return unit_record_;
    }

    // DaughterId of universe embedded in a given volume
    inline CELER_FUNCTION DaughterId daughter(LocalVolumeId vol) const;

    //// OPERATIONS ////

    // Find the local volume from a position
    inline CELER_FUNCTION Initialization
    initialize(LocalState const& state) const;

    // Find the new volume by crossing a surface
    inline CELER_FUNCTION Initialization
    cross_boundary(LocalState const& state) const;

    // Calculate the distance to an exiting face for the current volume
    inline CELER_FUNCTION Intersection intersect(LocalState const& state) const;

    // Calculate nearby distance to an exiting face for the current volume
    inline CELER_FUNCTION Intersection intersect(LocalState const& state,
                                                 real_type max_dist) const;

    // Calculate closest distance to a surface in any direction
    inline CELER_FUNCTION real_type safety(Real3 const& pos,
                                           LocalVolumeId vol) const;

    // Calculate the local surface normal
    inline CELER_FUNCTION Real3 normal(Real3 const& pos,
                                       LocalSurfaceId surf) const;

  private:
    //// DATA ////
    ParamsRef const& params_;
    SimpleUnitRecord const& unit_record_;

    //// METHODS ////

    // Get volumes that have the given surface as a "face" (connectivity)
    inline CELER_FUNCTION Span<LocalVolumeId const>
        get_neighbors(LocalSurfaceId) const;

    template<class F>
    inline CELER_FUNCTION Intersection intersect_impl(LocalState const&,
                                                      F) const;

    inline CELER_FUNCTION Intersection simple_intersect(LocalState const&,
                                                        VolumeView const&,
                                                        size_type) const;
    inline CELER_FUNCTION Intersection complex_intersect(LocalState const&,
                                                         VolumeView const&,
                                                         size_type) const;
    inline CELER_FUNCTION Intersection background_intersect(LocalState const&,
                                                            size_type) const;

    // Create a Surfaces object from the params
    inline CELER_FUNCTION Surfaces make_local_surfaces() const;

    // Create a Volumes object from the params
    inline CELER_FUNCTION VolumeView make_local_volume(LocalVolumeId vid) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent parameter data.
 *
 * \todo When adding multiple universes, this will calculate range of
 * LocalVolumeIds that belong to this unit. For now we assume all volumes and
 * surfaces belong to us.
 */
CELER_FUNCTION
SimpleUnitTracker::SimpleUnitTracker(ParamsRef const& params, SimpleUnitId suid)
    : params_(params), unit_record_(params.simple_units[suid])
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the local volume from a position.
 *
 * To avoid edge cases and inconsistent logical/physical states, it is
 * prohibited to initialize from an arbitrary point directly onto a surface.
 */
CELER_FUNCTION auto
SimpleUnitTracker::initialize(LocalState const& state) const -> Initialization
{
    CELER_EXPECT(params_);
    CELER_EXPECT(!state.surface && !state.volume);

    detail::SenseCalculator calc_senses(
        this->make_local_surfaces(), state.pos, state.temp_sense);

    // Loop over all volumes (TODO: use BVH)
    for (LocalVolumeId volid : range(LocalVolumeId{this->num_volumes()}))
    {
        VolumeView vol = this->make_local_volume(volid);

        // Calculate the local senses, and see if we're inside.
        auto logic_state = calc_senses(vol);

        // Evalulate whether the senses are "inside" the volume
        if (!detail::LogicEvaluator(vol.logic())(logic_state.senses))
        {
            // State is *not* inside this volume: try the next one
            continue;
        }
        if (logic_state.face)
        {
            // Initialized on a boundary in this volume but wasn't known
            // to be crossing a surface. Fail safe by letting the multi-level
            // tracking geometry (NOT YET IMPLEMENTED in GPU ORANGE) bump and
            // try again.
            break;
        }

        // Found and not unexpectedly on a surface!
        return {volid, {}};
    }

    // Not found, or default to background volume
    return {unit_record_.background, {}};
}

//---------------------------------------------------------------------------//
/*!
 * Find the local volume on the opposite side of a surface.
 */
CELER_FUNCTION auto
SimpleUnitTracker::cross_boundary(LocalState const& state) const
    -> Initialization
{
    CELER_EXPECT(state.surface && state.volume);
    detail::SenseCalculator calc_senses(
        this->make_local_surfaces(), state.pos, state.temp_sense);

    // Loop over all connected surfaces (TODO: intersect with BVH)
    for (LocalVolumeId volid : this->get_neighbors(state.surface.id()))
    {
        if (volid == state.volume)
        {
            // Cannot cross surface into the same volume
            continue;
        }
        VolumeView vol = this->make_local_volume(volid);

        // Calculate the local senses and face
        auto logic_state
            = calc_senses(vol, detail::find_face(vol, state.surface));

        // Evaluate whether the senses are "inside" the volume
        if (!detail::LogicEvaluator(vol.logic())(logic_state.senses))
        {
            // Not inside the volume
            continue;
        }

        // Found the volume! Convert the face to a surface ID and return
        return {volid, get_surface(vol, logic_state.face)};
    }

    return {unit_record_.background, state.surface};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 */
CELER_FUNCTION auto SimpleUnitTracker::intersect(LocalState const& state) const
    -> Intersection
{
    Intersection result = this->intersect_impl(state, detail::IsFinite{});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 */
CELER_FUNCTION auto
SimpleUnitTracker::intersect(LocalState const& state, real_type max_dist) const
    -> Intersection
{
    CELER_EXPECT(max_dist > 0);
    Intersection result
        = this->intersect_impl(state, detail::IsNotFurtherThan{max_dist});
    if (!result)
    {
        result.distance = max_dist;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate nearest distance to a surface in any direction.
 *
 * The safety calculation uses a very limited method for calculating the safety
 * distance: it's the nearest distance to any surface, for a certain subset of
 * surfaces.  Other surface types will return a safety distance of zero.
 * Complex surfaces might return the distance to internal surfaces that do not
 * represent the edge of a volume. Such distances are conservative but will
 * necessarily slow down the simulation.
 */
CELER_FUNCTION real_type SimpleUnitTracker::safety(Real3 const& pos,
                                                   LocalVolumeId volid) const
{
    CELER_EXPECT(volid);

    VolumeView vol = this->make_local_volume(volid);
    if (!vol.simple_safety())
    {
        // Has a tricky surface: we can't use the simple algorithm to calculate
        // the safety, so return a conservative estimate.
        return 0;
    }

    // Calculate minimim distance to all local faces
    real_type result = numeric_limits<real_type>::infinity();
    auto calc_safety = make_surface_action(this->make_local_surfaces(),
                                           detail::CalcSafetyDistance{pos});
    for (LocalSurfaceId surface : vol.faces())
    {
        result = celeritas::min(result, calc_safety(surface));
    }

    CELER_ENSURE(result >= 0);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the local surface normal.
 */
CELER_FUNCTION auto
SimpleUnitTracker::normal(Real3 const& pos, LocalSurfaceId surf) const -> Real3
{
    CELER_EXPECT(surf);

    auto calc_normal = make_surface_action(this->make_local_surfaces(),
                                           detail::CalcNormal{pos});

    return calc_normal(surf);
}

//---------------------------------------------------------------------------//
// PRIVATE INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get volumes that have the given surface as a "face" (connectivity).
 */
CELER_FUNCTION auto SimpleUnitTracker::get_neighbors(LocalSurfaceId surf) const
    -> Span<LocalVolumeId const>
{
    CELER_EXPECT(surf < this->num_surfaces());

    OpaqueId<Connectivity> conn_id
        = unit_record_.connectivity[surf.unchecked_get()];
    Connectivity const& conn = params_.connectivities[conn_id];

    CELER_ENSURE(!conn.neighbors.empty());
    return params_.local_volume_ids[conn.neighbors];
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 *
 * The algorithm is:
 * - Use the current volume to find potential intersecting surfaces and maximum
 *   number of intersections.
 * - Loop over all surfaces and calculate the distance to intercept based on
 *   the given physical and logical state. Save to the thread-local buffer
 *   *only* intersections that are valid (either finite *or* less than the
 *   user-supplied maximum). The buffer contains the distances, the face
 *   indices, and an index used for sorting (if the volume has internal
 *   surfaes).
 * - If no intersecting surfaces are found, return immediately. (Rely on the
 *   caller to set the "maximum distance" if we're not searching to infinity.)
 * - If the volume has no special cases, find the closest surface by calling \c
 *   simple_intersect.
 * - If the volume has internal surfaces call \c complex_intersect.
 * - If the volume is the "background" then search externally for the next
 *   volume with \c background_intersect (equivalent of DistanceToIn for
 *   Geant4)
 */
template<class F>
CELER_FUNCTION auto
SimpleUnitTracker::intersect_impl(LocalState const& state, F is_valid) const
    -> Intersection
{
    CELER_EXPECT(state.volume && !state.temp_sense.empty());

    // Resize temporaries based on volume properties
    VolumeView vol = this->make_local_volume(state.volume);
    CELER_ASSERT(state.temp_next.size >= vol.max_intersections());

    // Find all valid (nearby or finite, depending on F) surface intersection
    // distances inside this volume. Fill the `isect` array if the tracking
    // algorithm requires sorting.
    auto calc_intersections = make_surface_action(
        this->make_local_surfaces(),
        detail::CalcIntersections<F const&>{
            state.pos,
            state.dir,
            is_valid,
            state.surface ? vol.find_face(state.surface.id()) : FaceId{},
            vol.simple_intersection(),
            state.temp_next});
    for (LocalSurfaceId surface : vol.faces())
    {
        calc_intersections(surface);
    }
    CELER_ASSERT(calc_intersections.action().face_idx() == vol.num_faces());
    size_type num_isect = calc_intersections.action().isect_idx();
    CELER_ASSERT(num_isect <= vol.max_intersections());

    if (num_isect == 0)
    {
        // No intersection (no surfaces in this volume, no finite distances, or
        // no "nearby" distances depending on F)
        return {};
    }
    else if (vol.simple_intersection())
    {
        // No internal surfaces nor implicit volume: the closest distance is
        // the next boundary
        return this->simple_intersect(state, vol, num_isect);
    }
    else
    {
        // Sort valid intersection distances in ascending order
        celeritas::sort(state.temp_next.isect,
                        state.temp_next.isect + num_isect,
                        [&state](size_type a, size_type b) {
                            return state.temp_next.distance[a]
                                   < state.temp_next.distance[b];
                        });

        if (vol.internal_surfaces())
        {
            // Internal surfaces: find closest surface that puts us outside
            return this->complex_intersect(state, vol, num_isect);
        }
        else if (vol.implicit_vol())
        {
            // Search all the volumes "externally"
            return this->background_intersect(state, num_isect);
        }
    }

    CELER_ASSERT_UNREACHABLE();  // Unexpected set of flags
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance to the next boundary for nonreentrant volumes.
 */
CELER_FUNCTION auto
SimpleUnitTracker::simple_intersect(LocalState const& state,
                                    VolumeView const& vol,
                                    size_type num_isect) const -> Intersection
{
    CELER_EXPECT(num_isect > 0);

    // Crossing any surface will leave the volume; perform a linear search for
    // the smallest (but positive) distance
    size_type distance_idx
        = celeritas::min_element(state.temp_next.distance,
                                 state.temp_next.distance + num_isect,
                                 Less<real_type>{})
          - state.temp_next.distance;
    CELER_ASSERT(distance_idx < num_isect);

    // Determine the crossing surface
    LocalSurfaceId surface;
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
        auto calc_sense = make_surface_action(this->make_local_surfaces(),
                                              detail::CalcSense{state.pos});

        SignedSense ss = calc_sense(surface);
        CELER_ASSERT(ss != SignedSense::on);
        cur_sense = to_sense(ss);
    }

    // Post-surface sense will be on the other side of the surface
    Intersection result;
    result.surface = {surface, cur_sense};
    result.distance = state.temp_next.distance[distance_idx];
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate boundary distance if internal surfaces are present.
 *
 * In "complex" volumes, crossing a surface can still leave the particle in an
 * "inside" state.
 *
 * We have to iteratively track through all surfaces, in order of minimum
 * distance, to determine whether crossing them in sequence will cause us to
 * exit the volume.
 *
 * \pre The `state.temp_next.isect` array must be sorted by the caller by
 * ascending distance.
 */
CELER_FUNCTION auto
SimpleUnitTracker::complex_intersect(LocalState const& state,
                                     VolumeView const& vol,
                                     size_type num_isect) const -> Intersection
{
    CELER_ASSERT(num_isect > 0);

    // Calculate local senses, taking current face into account
    auto logic_state = detail::SenseCalculator(
        this->make_local_surfaces(), state.pos, state.temp_sense)(
        vol, detail::find_face(vol, state.surface));

    // Current senses should put us inside the volume
    detail::LogicEvaluator is_inside(vol.logic());
    CELER_ASSERT(is_inside(logic_state.senses));

    // Loop over distances and surface indices to cross by iterating over
    // temp_next.isect[:num_isect].
    // Evaluate the logic expression at each crossing to determine whether
    // we're actually leaving the volume.
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
            result.surface = {vol.get_surface(face), flip_sense(new_sense)};
            result.distance = state.temp_next.distance[isect];
            CELER_ENSURE(result.distance > 0 && !std::isinf(result.distance));
            return result;
        }
    }

    // No intersection: perhaps leaving an exterior volume? Perhaps geometry
    // error.
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance from the background volume to enter any other volume.
 *
 * This is a slimmed-down version of the masked unit tracker's intersection
 * method. We loop over all surface intersections in ascending order, and test
 * all volumes that are connected to each surface. At the intersection point
 * being tested, we see whether each potential connected volume is "inside".
 * The first such volume gives our next surface.
 *
 * It's not cheap, as there are many embedded loops:
 * - Intersection points
 * - Volumes connected to the surface being intersected
 * - Surfaces connected to the target volume (sense evaluation) plus number of
 *   elements in the logic array ("is_inside" evaluation)
 *
 * \pre The `state.temp_next.isect` array must be sorted by the caller by
 * ascending distance.
 * \pre The "faces" for the background volume are *all* the surfaces in the
 * volume (alternatively we could introduce a mapping between Face and
 * LocalSurfaceId).
 */
CELER_FUNCTION auto
SimpleUnitTracker::background_intersect(LocalState const& state,
                                        size_type num_isect) const
    -> Intersection
{
    // Calculate bump distance
    const real_type bump_dist
        = detail::BumpCalculator{params_.scalars}(state.pos);

    // Loop over distances and surface indices to cross by iterating over
    // temp_next.isect[:num_isect].
    for (size_type isect_idx = 0; isect_idx != num_isect; ++isect_idx)
    {
        // Index into the distance/face arrays
        const size_type isect = state.temp_next.isect[isect_idx];
        // Inside the "background" volume, Face and Surface are the same
        const LocalSurfaceId surface{
            state.temp_next.face[isect].unchecked_get()};

        // Calculate position just past the surface in order to evaluate
        // senses, since we can't know the change in sense of the
        // target surface without marching through all interior surfaces.
        // Assume that bumping past the surface means not on any surface.
        Real3 pos{state.pos};
        axpy(state.temp_next.distance[isect] + bump_dist, state.dir, &pos);

        // Loop over volumes connected to this surface.
        // TODO: intersection of this with BVH/acceleration grid
        for (LocalVolumeId vid : this->get_neighbors(surface))
        {
            CELER_ASSERT(vid != state.volume);
            VolumeView vol = this->make_local_volume(vid);
            auto logic_state = detail::SenseCalculator{
                this->make_local_surfaces(), pos, state.temp_sense}(vol);

            if (detail::LogicEvaluator{vol.logic()}(logic_state.senses))
            {
                // We are in this new volume by crossing the tested surface.
                // Get the sense corresponding to this "crossed" surface.
                auto face = vol.find_face(surface);
                CELER_ASSERT(face);

                Intersection result;
                result.distance = state.temp_next.distance[isect];
                result.surface = detail::OnLocalSurface{
                    surface,
                    flip_sense(logic_state.senses[face.unchecked_get()])};
                return result;
            }
        }
    }

    // No intersection in this unit
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Create a Surfaces object from the params for this unit.
 */
CELER_FORCEINLINE_FUNCTION Surfaces SimpleUnitTracker::make_local_surfaces() const
{
    return Surfaces{params_, unit_record_.surfaces};
}

//---------------------------------------------------------------------------//
/*!
 * Create a Volume view object from the params for this unit.
 */
CELER_FORCEINLINE_FUNCTION VolumeView
SimpleUnitTracker::make_local_volume(LocalVolumeId vid) const
{
    return VolumeView{params_, unit_record_, vid};
}

//---------------------------------------------------------------------------//
/*!
 * DaughterId of universe embedded in a given volume.
 */
CELER_FORCEINLINE_FUNCTION DaughterId
SimpleUnitTracker::daughter(LocalVolumeId vol) const
{
    return params_.volume_records[unit_record_.volumes[vol]].daughter_id;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
