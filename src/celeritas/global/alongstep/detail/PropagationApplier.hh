//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/PropagationApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/math/Algorithms.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/global/detail/ApplierTraits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply propagation over the step (implementation).
 */
template<class MP>
struct PropagationApplierBaseImpl
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MP make_propagator;
};

//---------------------------------------------------------------------------//
/*!
 * Apply propagation over the step.
 *
 * \tparam MP Propagator factory
 *
 * MP should be a function-like object:
 * \code Propagator(*)(CoreTrackView const&) \endcode
 *
 * This class is partially specialized with a second template argument to
 * extract any launch bounds from the MP class. TODO: we could probably inherit
 * from a helper class to pull in those constants (if available).
 */
template<class MP, typename = void>
struct PropagationApplier : public PropagationApplierBaseImpl<MP>
{
    CELER_FUNCTION PropagationApplier(MP&& mp)
        : PropagationApplierBaseImpl<MP>{celeritas::forward<MP>(mp)}
    {
    }
};

template<class MP>
struct PropagationApplier<MP, std::enable_if_t<kernel_max_blocks_min_warps<MP>>>
    : public PropagationApplierBaseImpl<MP>
{
    static constexpr int max_block_size = MP::max_block_size;
    static constexpr int min_warps_per_eu = MP::min_warps_per_eu;

    CELER_FUNCTION PropagationApplier(MP&& mp)
        : PropagationApplierBaseImpl<MP>{celeritas::forward<MP>(mp)}
    {
    }
};

template<class MP>
struct PropagationApplier<MP, std::enable_if_t<kernel_max_blocks<MP>>>
    : public PropagationApplierBaseImpl<MP>
{
    static constexpr int max_block_size = MP::max_block_size;

    CELER_FUNCTION PropagationApplier(MP&& mp)
        : PropagationApplierBaseImpl<MP>{celeritas::forward<MP>(mp)}
    {
    }
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MP>
CELER_FUNCTION PropagationApplier(MP&&)->PropagationApplier<MP>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class MP>
CELER_FUNCTION void
PropagationApplierBaseImpl<MP>::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.step_length() == 0)
    {
        // Track is stopped: no movement or energy loss will happen
        // (could be a stopped positron waiting for annihilation, or a
        // particle waiting to decay?)
        CELER_ASSERT(track.make_particle_view().is_stopped());
        CELER_ASSERT(sim.post_step_action()
                     == track.make_physics_view().scalars().discrete_action());
        CELER_ASSERT(track.make_physics_view().has_at_rest());
        return;
    }

    bool tracks_can_loop;
    Propagation p;
    {
        auto propagate = make_propagator(track);
        p = propagate(sim.step_length());
        tracks_can_loop = propagate.tracks_can_loop();
    }

    if (tracks_can_loop)
    {
        sim.update_looping(p.looping);
    }
    if (tracks_can_loop && p.looping)
    {
        // The track is looping, i.e. progressing little over many
        // integration steps in the field propagator (likely a low energy
        // particle in a low density material/strong magnetic field).
        sim.step_length(p.distance);

        // Kill the track if it's stable and below the threshold energy or
        // above the threshold number of steps allowed while looping.
        auto particle = track.make_particle_view();
        sim.post_step_action([&track, &particle, &sim] {
            if (particle.is_stable()
                && sim.is_looping(particle.particle_id(), particle.energy()))
            {
                return track.abandon_looping_action();
            }
            return track.propagation_limit_action();
        }());

        if (sim.post_step_action() == track.abandon_looping_action())
        {
            // TODO: move this branch into a separate post-step kernel.
            // If the track is looping (or if it's a stuck track that was
            // flagged as looping), deposit the energy locally.
            auto deposited = particle.energy().value();
            if (particle.is_antiparticle())
            {
                // Energy conservation for killed positrons
                deposited += 2 * particle.mass().value();
            }
            track.make_physics_step_view().deposit_energy(
                ParticleTrackView::Energy{deposited});
            particle.subtract_energy(particle.energy());

            // Mark that this track was abandoned while looping
            sim.status(TrackStatus::killed);
        }
    }
    else if (p.boundary)
    {
        // Stopped at a geometry boundary: this is the new step action.
        CELER_ASSERT(p.distance <= sim.step_length());
        sim.step_length(p.distance);
        sim.post_step_action(track.boundary_action());
    }
    else if (p.distance < sim.step_length())
    {
        // Some tracks may get stuck on a boundary and fail to move at
        // all in the field propagator, and will get bumped a small
        // distance. This primarily occurs with reentrant tracks on a
        // boundary with VecGeom.
        sim.step_length(p.distance);
        sim.post_step_action(track.propagation_limit_action());
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
