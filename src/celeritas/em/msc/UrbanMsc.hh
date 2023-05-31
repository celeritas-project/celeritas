//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/msc/UrbanMsc.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/data/UrbanMscData.hh"
#include "celeritas/global/CoreTrackView.hh"

#include "detail/MscStepFromGeo.hh"  // IWYU pragma: associated
#include "detail/MscStepToGeo.hh"  // IWYU pragma: associated
#include "detail/UrbanMscHelper.hh"  // IWYU pragma: associated
#include "detail/UrbanMscScatter.hh"  // IWYU pragma: associated
#include "detail/UrbanMscStepLimit.hh"  // IWYU pragma: associated

namespace celeritas
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
    explicit inline CELER_FUNCTION UrbanMsc(ParamsRef const& params);

    // Whether MSC applies to the current track
    inline CELER_FUNCTION bool
    is_applicable(CoreTrackView const&, real_type step) const;

    // Update the physical and geometric step lengths
    inline CELER_FUNCTION void limit_step(CoreTrackView const&, StepLimit*);

    // Apply MSC
    inline CELER_FUNCTION void apply_step(CoreTrackView const&, StepLimit*);

  private:
    ParamsRef const& msc_params_;

    // Whether the step was limited by geometry
    static inline CELER_FUNCTION bool
    is_geo_limited(CoreTrackView const&, StepLimit const&);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CELER_FUNCTION UrbanMsc::UrbanMsc(ParamsRef const& params)
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

    if (track.make_sim_view().status() != TrackStatus::alive)
        return false;

    auto par = track.make_particle_view();
    if (par.particle_id() != msc_params_.ids.electron
        && par.particle_id() != msc_params_.ids.positron)
        return false;

    return par.energy() > msc_params_.params.low_energy_limit
           && par.energy() < msc_params_.params.high_energy_limit;
}

//---------------------------------------------------------------------------//
/*!
 * Update the physical and geometric step lengths.
 */
CELER_FUNCTION void
UrbanMsc::limit_step(CoreTrackView const& track, StepLimit* step_limit)
{
    CELER_EXPECT(msc_params_);

    auto phys = track.make_physics_view();
    auto par = track.make_particle_view();
    detail::UrbanMscHelper msc_helper(msc_params_, par, phys);

    bool displaced = false;

    // Sample multiple scattering step length
    real_type const true_path = [&] {
        if (step_limit->step <= msc_params_.params.limit_min_fix())
        {
            // Very short step: don't displace or limit
            return step_limit->step;
        }

        auto geo = track.make_geo_view();

        real_type safety = 0;
        if (!geo.is_on_boundary())
        {
            // Because the MSC behavior changes based on whether the *total*
            // track range is close to a boundary, rather than whether the next
            // steps are closer, we need to find the safety distance up to the
            // potential travel radius of the particle at its current energy.
            real_type const max_step = msc_helper.max_step();
            safety = geo.find_safety(max_step);
            if (safety >= max_step)
            {
                // The nearest boundary is further than the maximum expected
                // travel distance of the particle: don't displace or limit
                return step_limit->step;
            }
        }

        displaced = true;
        detail::UrbanMscStepLimit calc_limit(msc_params_,
                                             msc_helper,
                                             par.energy(),
                                             &phys,
                                             phys.material_id(),
                                             geo.is_on_boundary(),
                                             safety,
                                             step_limit->step);
        auto rng = track.make_rng_engine();
        return calc_limit(rng);
    }();
    CELER_ASSERT(true_path <= step_limit->step);

    bool limited = (true_path < step_limit->step);

    // Always apply the step transformation, even if the physical step wasn't
    // necessarily limited. This transformation will be reversed in
    // `apply_step` below.
    auto gp = [&] {
        detail::MscStepToGeo calc_geom_path(msc_params_,
                                            msc_helper,
                                            par.energy(),
                                            msc_helper.msc_mfp(),
                                            phys.dedx_range());
        auto gp = calc_geom_path(true_path);

        // Limit geometrical step to 1 MSC MFP
        if (gp.step > msc_helper.msc_mfp())
        {
            gp.step = msc_helper.msc_mfp();
            limited = true;
        }

        return gp;
    }();
    CELER_ASSERT(0 < gp.step && gp.step <= true_path);

    // Save MSC step for later
    track.make_physics_step_view().msc_step([&] {
        MscStep result;
        result.is_displaced = displaced;
        result.true_path = true_path;
        result.geom_path = gp.step;
        result.alpha = gp.alpha;
        return result;
    }());

    step_limit->step = gp.step;
    if (limited)
    {
        // Physical step was further limited by MSC
        step_limit->action = phys.scalars().msc_action();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Apply MSC.
 */
CELER_FUNCTION void
UrbanMsc::apply_step(CoreTrackView const& track, StepLimit* step_limit)
{
    CELER_EXPECT(msc_params_);

    auto par = track.make_particle_view();
    auto geo = track.make_geo_view();
    auto phys = track.make_physics_view();

    // Replace step with actual geometry distance traveled
    detail::UrbanMscHelper msc_helper(msc_params_, par, phys);
    auto msc_step = track.make_physics_step_view().msc_step();
    if (this->is_geo_limited(track, *step_limit))
    {
        // Convert geometrical distance to equivalent physical distance, which
        // will be greater than (or in edge cases equal to) that distance and
        // less than the original physical step limit.
        msc_step.geom_path = step_limit->step;
        detail::MscStepFromGeo geo_to_true(msc_params_.params,
                                           msc_step,
                                           phys.dedx_range(),
                                           msc_helper.msc_mfp());
        msc_step.true_path = geo_to_true(msc_step.geom_path);
        CELER_ASSERT(msc_step.true_path >= msc_step.geom_path);

        // Disable displacement on boundary
        msc_step.is_displaced = false;
    }

    // Update full path length traveled along the step based on MSC to
    // correctly calculate energy loss, step time, etc.
    step_limit->step = msc_step.true_path;

    auto msc_result = [&] {
        auto mat = track.make_material_view().make_material_view();
        detail::UrbanMscScatter sample_scatter(
            msc_params_, msc_helper, par, phys, mat, &geo, msc_step);

        auto rng = track.make_rng_engine();
        return sample_scatter(rng);
    }();

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
UrbanMsc::is_geo_limited(CoreTrackView const& track, StepLimit const& limit)
{
    return (limit.action == track.boundary_action()
            || limit.action == track.propagation_limit_action());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
