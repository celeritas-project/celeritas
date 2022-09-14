//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MakeMagFieldPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "FieldDriver.hh"
#include "FieldPropagator.hh"
#include "MagFieldEquation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create a stepper for a charge in a magnetic field.
 *
 * \example
 * \code
 * auto step = make_stepper<DormandPrinceStepper>(
 *    UniformField{{1, 2, 3}},
 *    particle.charge());
 * \endcode
 */
template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_stepper(FieldT&& field, units::ElementaryCharge charge)
{
    using Equation_t = MagFieldEquation<FieldT>;
    return StepperT<Equation_t>{
        Equation_t{::celeritas::forward<FieldT>(field), charge}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a field propagator from an existing stepper.
 *
 * \example
 * \code
 * FieldDriverOptions driver_options,
 * auto propagate = make_field_propagator(
 *    stepper,
 *    driver_options,
 *    particle,
 *    &geo);
 * propagate(0.123);
 * \endcode
 */
template<class StepperT>
CELER_FUNCTION decltype(auto)
make_field_propagator(StepperT&&                stepper,
                      const FieldDriverOptions& options,
                      const ParticleTrackView&  particle,
                      GeoTrackView*             geometry)
{
    CELER_ASSERT(geometry);
    using Driver_t = FieldDriver<StepperT>;
    return FieldPropagator<Driver_t>{
        Driver_t{options, ::celeritas::forward<StepperT>(stepper)},
        particle,
        geometry};
}

//---------------------------------------------------------------------------//
/*!
 * Create a magnetic field propagator.
 *
 * \example
 * \code
 * FieldDriverOptions driver_options,
 * auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
 *    UniformField{{1, 2, 3}},
 *    driver_options,
 *    particle,
 *    &geo);
 * propagate(0.123);
 * \endcode
 */
template<template<class EquationT> class StepperT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_propagator(FieldT&&                  field,
                          const FieldDriverOptions& options,
                          const ParticleTrackView&  particle,
                          GeoTrackView*             geometry)
{
    return make_field_propagator(
        make_mag_field_stepper<StepperT>(::celeritas::forward<FieldT>(field),
                                         particle.charge()),
        options,
        particle,
        geometry);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
