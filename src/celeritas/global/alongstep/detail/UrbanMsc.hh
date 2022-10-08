//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/detail/UrbanMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/em/distribution/UrbanMscScatter.hh"
#include "celeritas/em/distribution/UrbanMscStepLimit.hh"
#include "celeritas/global/alongstep/AlongStep.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Apply Urban multiple scattering to a track.
 */
class UrbanMsc
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<UrbanMscData>;
    //!@}

  public:
    // Construct from MSC params
    explicit inline CELER_FUNCTION UrbanMsc(const ParamsRef& params);

    // Whether MSC applies to the current track
    inline CELER_FUNCTION bool
    is_applicable(CoreTrackView const&, real_type step) const;

    // Update the physical and geometric step lengths
    inline CELER_FUNCTION void
    calc_step(CoreTrackView const&, AlongStepLocalState*) const;

    // Apply MSC
    inline CELER_FUNCTION void
    apply_step(CoreTrackView const&, AlongStepLocalState*) const;

  private:
    const ParamsRef& msc_params_;

    // Whether a
    static inline CELER_FUNCTION bool
    is_geo_limited(CoreTrackView const&, const StepLimit&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION UrbanMsc::UrbanMsc(const ParamsRef& params)
    : msc_params_(params)
{
}

//---------------------------------------------------------------------------//
/*!
 * Whether MSC applies to the current track.
 */
CELER_FUNCTION bool
UrbanMsc::is_applicable(CoreTrackView const& track, real_type step) const
{
    if (!msc_params_)
        return false;

    if (step <= msc_params_.params.geom_limit)
        return false;

    auto phys = track.make_physics_view();
    if (!phys.msc_ppid())
        return false;

    auto particle = track.make_particle_view();
    return particle.energy() > msc_params_.params.energy_limit;
}

//---------------------------------------------------------------------------//
/*!
 * Update the physical and geometric step lengths.
 */
CELER_FUNCTION void UrbanMsc::calc_step(CoreTrackView const& track,
                                        AlongStepLocalState* local) const
{
    CELER_EXPECT(msc_params_);

    auto particle = track.make_particle_view();
    auto geo      = track.make_geo_view();
    auto phys     = track.make_physics_view();
    auto sim      = track.make_sim_view();

    // Sample multiple scattering step length
    UrbanMscStepLimit msc_step_limit(msc_params_,
                                     particle,
                                     phys,
                                     track.make_material_view().material_id(),
                                     sim.num_steps() == 0,
                                     geo.find_safety(),
                                     local->step_limit.step);

    auto rng             = track.make_rng_engine();
    auto msc_step_result = msc_step_limit(rng);
    track.make_physics_step_view().msc_step(msc_step_result);

    // Use "straight line" path calculated for geometry step
    local->geo_step = msc_step_result.geom_path;

    if (msc_step_result.true_path < local->step_limit.step)
    {
        // True/physical step might be further limited by MSC
        // TODO: this is already kinda sorta determined inside the
        // UrbanMscStepLimit calculation
        local->step_limit.step   = msc_step_result.true_path;
        local->step_limit.action = msc_params_.ids.action;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Apply MSC.
 */
CELER_FUNCTION void UrbanMsc::apply_step(CoreTrackView const& track,
                                         AlongStepLocalState* local) const
{
    CELER_EXPECT(msc_params_);

    auto par  = track.make_particle_view();
    auto geo  = track.make_geo_view();
    auto phys = track.make_physics_view();
    auto mat  = track.make_material_view();

    // Replace step with actual geometry distance traveled
    auto msc_step_result      = track.make_physics_step_view().msc_step();
    msc_step_result.geom_path = local->geo_step;

    UrbanMscScatter msc_scatter(msc_params_,
                                par,
                                &geo,
                                phys,
                                mat.make_material_view(),
                                msc_step_result,
                                is_geo_limited(track, local->step_limit));

    auto rng        = track.make_rng_engine();
    auto msc_result = msc_scatter(rng);

    // Update full path length traveled along the step based on MSC to
    // correctly calculate energy loss, step time, etc.
    CELER_ASSERT(local->geo_step <= msc_result.step_length
                 && msc_result.step_length <= local->step_limit.step);
    local->step_limit.step = msc_result.step_length;

    // Update direction and position
    if (msc_result.action != MscInteraction::Action::unchanged)
    {
        // Changing direction during a boundary crossing is OK
        geo.set_dir(msc_result.direction);
    }
    if (msc_result.action == MscInteraction::Action::displaced)
    {
        // Displacment during a boundary crossing is *not* OK
        CELER_ASSERT(!geo.is_on_boundary());
        Real3 new_pos;
        for (int i = 0; i < 3; ++i)
        {
            new_pos[i] = geo.pos()[i] + msc_result.displacement[i];
        }
        geo.move_internal(new_pos);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Whether the step was limited by geometry.
 *
 * Usually the track is limited only if it's on the boundary (in which case
 * it should be "boundary action") but in rare circumstances the propagation
 * has to pause before the end of the step is reached.
 */
CELER_FUNCTION bool
UrbanMsc::is_geo_limited(CoreTrackView const& track, const StepLimit& limit)
{
    return (limit.action == track.boundary_action()
            || limit.action == track.propagation_limit_action());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
