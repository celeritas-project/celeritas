//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/PropagatorLauncher.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{

struct UrbanMsc
{
    const UrbanMscRef& urban_data;

    CELER_FUNCTION bool
    is_applicable(CoreTrackView const& track, real_type step) const
    {
        if (step <= this->urban_data.params.geom_limit)
            return false;

        auto phys = track.make_physics_view();
        if (!phys.msc_ppid())
            return false;

        auto particle = track.make_particle_view();
        return particle.energy() > this->urban_data.params.energy_limit;
    }

    CELER_FUNCTION void
    calc_step(CoreTrackView const& track, AlongStepLocalState* local) const
    {
        auto particle = track.make_particle_view();
        auto geo      = track.make_geo_view();
        auto phys     = track.make_physics_view();

        // Sample multiple scattering step length
        UrbanMscStepLimit msc_step_limit(
            this->urban_data,
            particle,
            phys,
            track.make_material_view().material_id(),
            sim.num_steps() == 0,
            local->step_limit.step);

        auto rng             = track.make_rng_engine();
        auto msc_step_result = msc_step_limit(rng);
        track.make_physics_step_view().msc_step(msc_step_result);

        // Use "straight line" path calculated for geometry step
        ? local->geo_step = msc_step_result.geom_path;

        if (msc_step_result.true_path < local->step_limit.step)
        {
            // True/physical step might be further limited by MSC
            // TODO: this is already kinda sorta determined inside the
            // UrbanMscStepLimit calculation
            local->step_limit.step   = msc_step_result.true_path;
            local->step_limit.action = this->urban_data.ids.action;
        }
    }

    CELER_FUNCTION void
    apply_step(CoreTrackView const& track, AlongStepLocalState* local) const
    {
        // Replace step with actual geometry distance traveled
        auto msc_step_result      = track.make_physics_step_view().msc_step();
        msc_step_result.geom_path = geo_step;

        UrbanMscScatter msc_scatter(this->urban_data,
                                    particle,
                                    &geo,
                                    phys,
                                    mat.make_material_view(),
                                    msc_step_result);

        auto msc_result = msc_scatter(rng);

        // Update full path length traveled along the step based on MSC to
        // correctly calculate energy loss, step time, etc.
        CELER_ASSERT(geo_step <= msc_result.step_length
                     && msc_result.step_length <= local->step_limit.step);
        local->step_limit.step = msc_result.step_length;

        // Update direction and position
        geo.set_dir(msc_result.direction);
        Real3 new_pos;
        for (int i : range(3))
        {
            new_pos[i] = geo.pos()[i] + msc_result.displacement[i];
        }
        geo.move_internal(new_pos);
    }
};

//---------------------------------------------------------------------------//
// PROPAGATE
//---------------------------------------------------------------------------//

struct UniformMagPropagatorFactory
{
    UniformMagPropagatorParams const& field_params;

    CELER_FUNCTION decltype(auto)
    operator()(const ParticleTrackView& particle, GeoTrackView* geo) const
    {
        return make_mag_field_propagator<DormandPrinceStepper>(
            field_params.field, field_params.driver_options, particle, geo);
    };
};

//---------------------------------------------------------------------------//
} // namespace celeritas
