//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/PropagationApplier.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/math/Algorithms.hh"
#include "corecel/sys/KernelTraits.hh"
#include "celeritas/global/CoreTrackView.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#endif

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
    inline CELER_FUNCTION void operator()(CoreTrackView& track);

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
CELER_FUNCTION PropagationApplier(MP&&) -> PropagationApplier<MP>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
template<class MP>
CELER_FUNCTION void
PropagationApplierBaseImpl<MP>::operator()(CoreTrackView& track)
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
        CELER_ASSERT(track.make_physics_view().at_rest_process());
        return;
    }

    bool tracks_can_loop;
    Propagation p;
    {
#if CELERITAS_DEBUG
        Real3 const orig_pos = track.make_geo_view().pos();
#endif
        auto propagate = make_propagator(track);
        p = propagate(sim.step_length());
        tracks_can_loop = propagate.tracks_can_loop();
        CELER_ASSERT(p.distance > 0);
#if CELERITAS_DEBUG
        if (CELER_UNLIKELY(track.make_geo_view().pos() == orig_pos))
        {
            // This unusual case happens when the step length is less than
            // machine epsilon compared to the actual position. This case seems
            // to occur in vecgeom when "stuck" on a boundary, and when in a
            // field taking a small step when the track's position has a large
            // magnitude.
#    if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(error)
                << "Propagation of step length " << repr(sim.step_length())
                << " due to post-step action "
                << sim.post_step_action().unchecked_get()
                << " leading to distance " << repr(p.distance)
                << (p.boundary  ? " (boundary hit)"
                    : p.looping ? " (**LOOPING**)"
                                : "")
                << " failed to change position";
#    endif
            track.apply_errored();
            return;
        }
#endif
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
        sim.post_step_action([&track, &sim] {
            auto particle = track.make_particle_view();
            if (particle.is_stable()
                && sim.is_looping(particle.particle_id(), particle.energy()))
            {
#if !CELER_DEVICE_COMPILE
                CELER_LOG_LOCAL(debug)
                    << "Track (pid=" << particle.particle_id().get()
                    << ", E=" << particle.energy().value() << ' '
                    << ParticleTrackView::Energy::unit_type::label()
                    << ") is looping after " << sim.num_looping_steps()
                    << " steps";
#endif
                return track.tracking_cut_action();
            }
            return track.propagation_limit_action();
        }());
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
