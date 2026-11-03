//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/detail/FieldTrackPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "celeritas/field/DormandPrinceIntegrator.hh"
#include "celeritas/field/FieldPropagationCounters.hh"
#include "celeritas/field/MakeMagFieldPropagator.hh"
#include "celeritas/field/detail/CountingGeoTrackView.hh"
#include "celeritas/field/detail/CountingIntegrator.hh"
#include "celeritas/global/CoreTrackView.hh"
#include "celeritas/user/FieldDiagnosticData.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a mapped magnetic field.
 *
 * This class moves the track a single step based on the current sim step
 * length. It updates the particle with looping behavior if necessary.
 *
 * \tparam Field The field type (e.g., RZMapField, CylMapField, CartMapField)
 */
template<class Field>
struct FieldTrackPropagator
{
    //!@{
    //! \name Type aliases
    using ParamsRef = typename Field::ParamsRef;
    //!@}

    //! Create propagator, execute propagation, and return result
    [[nodiscard]] CELER_FUNCTION Propagation operator()(
        CoreTrackView& track) const;

    //// DATA ////

    ParamsRef field;
};

//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a field and record its propagation counters.
 *
 * This is the instrumented counterpart of \c FieldTrackPropagator used by
 * \c FieldDiagnostic . It counts substepper and propagator loop events
 * through \c CountingFieldObserver and wrappers around the integrator and
 * geometry, and it writes them to the track slot's \c FieldStepRecord .
 * If requested, it calculates the safety distance at the start of the step
 * (outside of the counters) to measure how many boundary searches a safety
 * sphere could avoid.
 */
template<class Field>
struct DiagnosticFieldTrackPropagator
{
    //!@{
    //! \name Type aliases
    using ParamsRef = typename Field::ParamsRef;
    //!@}

    //! Create propagator, execute propagation, and return result
    [[nodiscard]] CELER_FUNCTION Propagation operator()(
        CoreTrackView& track) const;

    //// DATA ////

    ParamsRef field;
    NativeCRef<FieldDiagnosticParamsData> diag_params;
    NativeRef<FieldDiagnosticStateData> diag_state;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Update the track after propagation if it is looping.
 */
inline CELER_FUNCTION void update_field_looping(CoreTrackView& track,
                                                Propagation const& p)
{
    auto sim = track.sim();
    sim.update_looping(p.looping);
    if (p.looping)
    {
        sim.step_length(p.distance);
        auto particle = track.particle();
        if (particle.is_stable()
            && sim.is_looping(particle.particle_id(), particle.energy()))
        {
#if !CELER_DEVICE_COMPILE
            CELER_LOG_LOCAL(debug)
                << "Track " << track.track_slot_id().get()
                << " (pid=" << particle.particle_id().get()
                << ", E=" << particle.energy() << ") is looping after "
                << sim.num_looping_steps() << " steps";
#endif
            track.apply_cut();
        }
        else
        {
            sim.post_step_action(track.propagation_limit_action());
        }
    }
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Create propagator, execute propagation, and return result.
 */
template<class Field>
inline CELER_FUNCTION Propagation FieldTrackPropagator<Field>::operator()(
    CoreTrackView& track) const
{
    auto propagator = make_mag_field_propagator<DormandPrinceIntegrator>(
        Field{field}, field.options, track.particle(), track.geometry());

    Propagation p = propagator(track.sim().step_length());
    update_field_looping(track, p);
    return p;
}

//---------------------------------------------------------------------------//
/*!
 * Create an instrumented propagator, execute propagation, and record it.
 */
template<class Field>
inline CELER_FUNCTION Propagation
DiagnosticFieldTrackPropagator<Field>::operator()(CoreTrackView& track) const
{
    auto geo = track.geometry();
    auto particle = track.particle();

    FieldStepRecord rec;
    rec.step = track.sim().step_length();
    rec.energy = value_as<units::MevEnergy>(particle.energy());
    rec.volume = geo.volume_id();
    if (diag_params.oracle_safety && !geo.is_on_boundary())
    {
        rec.start_safety = geo.find_safety();
    }

    FieldPropagationCounters counters;
    CountingGeoTrackView counting_geo{geo, &counters};
    if (rec.start_safety > 0)
    {
        counting_geo.set_safety_sphere(geo.pos(), rec.start_safety);
    }
    auto propagator = make_field_propagator(
        CountingIntegrator{make_mag_field_integrator<DormandPrinceIntegrator>(
                               Field{field}, particle.charge()),
                           &counters},
        field.options,
        particle,
        counting_geo,
        CountingFieldObserver{&counters});

    Propagation p = propagator(rec.step);

    rec.counters = counters;
    rec.distance = p.distance;
    rec.boundary = p.boundary;
    rec.looping = p.looping;
    rec.valid = true;
    diag_state.step[track.track_slot_id()] = rec;

    update_field_looping(track, p);
    return p;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
