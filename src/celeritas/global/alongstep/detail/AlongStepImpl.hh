//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/AlongStepImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "orange/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class MH>
struct MscStepLimitApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION MscStepLimitApplier(MH&&)->MscStepLimitApplier<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply MSC step limiters.
 *
 * TODO: think about integrating this into the pre-step sequence. Maybe the
 * geo/phys path transformation would be best suited to the \c
 * apply_propagation step?
 */
template<class MH>
CELER_FUNCTION void
MscStepLimitApplier<MH>::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (msc.is_applicable(track, sim.step_limit().step))
    {
        // Apply MSC step limiters and transform "physical" step (with MSC) to
        // "geometrical" step (smooth curve)
        msc.limit_step(track, &sim.step_limit());

        auto step_view = track.make_physics_step_view();
        CELER_ASSERT(step_view.msc_step().geom_path > 0);
    }
    else
    {
        // TODO: hack flag for saving "use_msc"
        auto step_view = track.make_physics_step_view();
        step_view.msc_step().geom_path = 0;
    }
}

//---------------------------------------------------------------------------//
template<class MP>
struct PropagationApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MP make_propagator;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MP>
CELER_FUNCTION PropagationApplier(MP&&)->PropagationApplier<MP>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply propagaion.
 */
template<class MP>
CELER_FUNCTION void
PropagationApplier<MP>::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    StepLimit& step_limit = sim.step_limit();
    if (step_limit.step == 0)
    {
        // Track is stopped: no movement or energy loss will happen
        // (could be a stopped positron waiting for annihilation, or a
        // particle waiting to decay?)
        CELER_ASSERT(track.make_particle_view().is_stopped());
        CELER_ASSERT(step_limit.action
                     == track.make_physics_view().scalars().discrete_action());
        CELER_ASSERT(track.make_physics_view().has_at_rest());
        return;
    }

    auto geo = track.make_geo_view();
    auto particle = track.make_particle_view();
    auto propagate = make_propagator(particle, &geo);
    Propagation p = propagate(step_limit.step);
    if (propagate.tracks_can_loop())
    {
        sim.update_looping(p.looping);
    }
    if (propagate.tracks_can_loop() && p.looping)
    {
        // The track is looping, i.e. progressing little over many
        // integration steps in the field propagator (likely a low energy
        // particle in a low density material/strong magnetic field).
        step_limit.step = p.distance;

        // Kill the track if it's stable and below the threshold energy or
        // above the threshold number of steps allowed while looping.
        step_limit.action = [&track, &particle, &sim] {
            if (particle.is_stable()
                && sim.is_looping(particle.particle_id(), particle.energy()))
            {
                return track.abandon_looping_action();
            }
            return track.propagation_limit_action();
        }();

        if (step_limit.action == track.abandon_looping_action())
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
        CELER_ASSERT(p.distance <= step_limit.step);
        step_limit.step = p.distance;
        step_limit.action = track.boundary_action();
    }
    else if (p.distance < step_limit.step)
    {
        // Some tracks may get stuck on a boundary and fail to move at
        // all in the field propagator, and will get bumped a small
        // distance. This primarily occurs with reentrant tracks on a
        // boundary with VecGeom.
        step_limit.step = p.distance;
        step_limit.action = track.propagation_limit_action();
    }
}

//---------------------------------------------------------------------------//
template<class MH>
struct MscApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    MH msc;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class MH>
CELER_FUNCTION MscApplier(MH&&)->MscApplier<MH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

/*!
 * Apply multiple scattering.
 *
 * This does three key things:
 * - Replaces the "geometrical" step (continuous) with the "physical" step
 *   (including multiple scattering)
 * - Likely changes the direction of the track
 * - Possibly displaces the particle
 */
template<class MH>
CELER_FUNCTION void MscApplier<MH>::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.status() == TrackStatus::killed)
    {
        // Active track killed during propagation: don't apply MSC
        return;
    }

    if (track.make_physics_step_view().msc_step().geom_path > 0)
    {
        // Scatter the track and transform the "geometrical" step back to
        // "physical" step
        msc.apply_step(track, &sim.step_limit());
    }
}

//---------------------------------------------------------------------------//
struct TimeUpdater
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update the lab frame time.
 */
CELER_FUNCTION void TimeUpdater::operator()(CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    real_type speed = native_value_from(particle.speed());
    CELER_ASSERT(speed >= 0);
    if (speed > 0)
    {
        // For very small energies (< numeric_limits<real_type>::epsilon)
        // the calculated speed can be zero.
        auto sim = track.make_sim_view();
        real_type delta_time = sim.step_limit().step / speed;
        sim.add_time(delta_time);
    }
}

//---------------------------------------------------------------------------//
template<class EH>
struct ElossApplier
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);

    EH eloss;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class EH>
CELER_FUNCTION ElossApplier(EH&&)->ElossApplier<EH>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Apply energy loss.
 *
 * TODO: move apply-cut out of mean/fluct eloss to this function to reduce
 * duplicate code?
 */
template<class EH>
CELER_FUNCTION void ElossApplier<EH>::operator()(CoreTrackView const& track)
{
    auto particle = track.make_particle_view();
    if (!eloss.is_applicable(track) || particle.is_stopped())
    {
        return;
    }

    auto sim = track.make_sim_view();
    StepLimit const& step_limit = sim.step_limit();

    // Calculate energy loss, possibly applying tracking cuts
    bool apply_cut = (step_limit.action != track.boundary_action());
    auto deposited = eloss.calc_eloss(track, step_limit.step, apply_cut);
    CELER_ASSERT(deposited <= particle.energy());
    CELER_ASSERT(apply_cut || deposited != particle.energy());

    if (deposited > zero_quantity())
    {
        // Deposit energy loss
        auto step = track.make_physics_step_view();
        step.deposit_energy(deposited);
        particle.subtract_energy(deposited);
    }

    // Energy loss helper *must* apply the tracking cutoff
    CELER_ASSERT(
        particle.energy()
            >= track.make_physics_view().scalars().lowest_electron_energy
        || !apply_cut || particle.is_stopped());

    if (particle.is_stopped())
    {
        // Particle lost all energy over the step
        CELER_ASSERT(step_limit.action != track.boundary_action());
        auto const phys = track.make_physics_view();
        if (!phys.has_at_rest())
        {
            // Immediately kill stopped particles with no at rest processes
            sim.status(TrackStatus::killed);
            sim.step_limit().action = phys.scalars().range_action();
        }
        else
        {
            // Particle slowed down to zero: force a discrete interaction
            sim.step_limit().action = phys.scalars().discrete_action();
        }
    }
}

//---------------------------------------------------------------------------//
struct TrackUpdater
{
    inline CELER_FUNCTION void operator()(CoreTrackView const& track);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Finish the step.
 *
 * TODO: we may need to save the pre-step speed and apply the time update using
 * an average here.
 */
CELER_FUNCTION void TrackUpdater::operator()(CoreTrackView const& track)
{
    auto sim = track.make_sim_view();
    if (sim.status() != TrackStatus::killed)
    {
        StepLimit const& step_limit = sim.step_limit();
        CELER_ASSERT(step_limit.step > 0
                     || track.make_particle_view().is_stopped());
        CELER_ASSERT(step_limit.action);
        auto phys = track.make_physics_view();
        if (step_limit.action != phys.scalars().discrete_action())
        {
            // Reduce remaining mean free paths to travel. The 'discrete
            // action' case is launched separately and resets the
            // interaction MFP itself.
            auto step = track.make_physics_step_view();
            real_type mfp = phys.interaction_mfp()
                            - step_limit.step * step.macro_xs();
            CELER_ASSERT(mfp > 0);
            phys.interaction_mfp(mfp);
        }
    }

    // Increment the step counter
    sim.increment_num_steps();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
