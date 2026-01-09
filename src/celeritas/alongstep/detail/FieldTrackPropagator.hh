//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/global/CoreTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a mapped magnetic field.
 *
 * This class template unifies field propagation for mapped field types
 * (RZ map, cylindrical map, cartesian map). It creates the field
 * propagator internally and executes the propagation step.
 *
 * \tparam Field The field type (e.g., RZMapField, CylMapField, CartMapField)
 */
template<class Field>
struct FieldTrackPropagator
{
    //!@{
    //! \name Type aliases
    using ParamsRef = typename Field::FieldParamsRef;
    //!@}

    //! Create propagator, execute propagation, and return result
    [[nodiscard]] CELER_FUNCTION Propagation operator()(CoreTrackView& track) const
    {
        auto sim = track.sim();
        auto propagator = make_mag_field_propagator<DormandPrinceIntegrator>(
            Field{field}, field.options, track.particle(), track.geometry());

        Propagation p = propagator(sim.step_length());

        sim.update_looping(p.looping);
        if (p.looping)
        {
            sim.step_length(p.distance);
            sim.post_step_action([&track, &sim] {
                auto particle = track.particle();
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
        return p;
    }

    //// DATA ////

    ParamsRef field;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
