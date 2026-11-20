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

#include "FieldFunctors.hh"
#include "PropagationApplier.hh"

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
 * length, searching for boundaries along each substep. It updates the
 * particle with looping behavior if necessary.
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
 * Propagate a track whose whole step fits inside its safety sphere.
 *
 * This is like \c FieldTrackPropagator but only integrates the field (see
 * \c SafetyFieldPropagator ), so that a kernel using it doesn't include the
 * boundary search code. The track's step must fit inside its sphere (see \c
 * is_step_in_safety_sphere ).
 *
 * \tparam Field The field type (e.g., RZMapField, CylMapField, CartMapField)
 */
template<class Field>
struct SafetyFieldTrackPropagator
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
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Update the track after field propagation if it is looping.
 */
inline CELER_FUNCTION void apply_field_looping(CoreTrackView& track,
                                               Propagation const& p)
{
    auto sim = track.sim();
    sim.update_looping(p.looping);
    if (!p.looping)
    {
        return;
    }

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

//---------------------------------------------------------------------------//
/*!
 * Propagate a track in a field, using the safety sphere if the step fits.
 *
 * This is for host loops that apply every along-step function to one track.
 * Device code instead launches a separate kernel for each propagator.
 */
template<class Field>
inline CELER_FUNCTION void apply_field_propagation(
    CoreTrackView& track, typename Field::ParamsRef const& field)
{
    auto const& options = field.options;
    if (options.use_safety_sphere
        && is_step_in_safety_sphere(track, options.delta_intersection))
    {
        PropagationApplier{SafetyFieldTrackPropagator<Field>{field}}(track);
    }
    else
    {
        PropagationApplier{FieldTrackPropagator<Field>{field}}(track);
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
    apply_field_looping(track, p);
    return p;
}

//---------------------------------------------------------------------------//
/*!
 * Create propagator, execute propagation, and return result.
 */
template<class Field>
inline CELER_FUNCTION Propagation SafetyFieldTrackPropagator<Field>::operator()(
    CoreTrackView& track) const
{
    auto propagator = make_mag_field_safety_propagator<DormandPrinceIntegrator>(
        Field{field},
        field.options,
        track.particle(),
        track.geometry(),
        track.physics_step().safety_sphere());

    Propagation p = propagator(track.sim().step_length());
    apply_field_looping(track, p);
    return p;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
