//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MakeMagFieldPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "FieldDriver.hh"
#include "FieldDriverOptions.hh"
#include "FieldPropagator.hh"
#include "MagFieldEquation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create a stepper for a charge in a magnetic field.
 *
 * Example:
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
 * Example:
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
template<class StepperT, class GTV>
CELER_FUNCTION decltype(auto)
make_field_propagator(StepperT&& stepper,
                      FieldDriverOptions const& options,
                      ParticleTrackView const& particle,
                      GTV&& geometry)
{
    return FieldPropagator{
        FieldDriver{options, ::celeritas::forward<StepperT>(stepper)},
        particle,
        ::celeritas::forward<GTV>(geometry)};
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
template<template<class EquationT> class StepperT, class FieldT, class GTV>
CELER_FUNCTION decltype(auto)
make_mag_field_propagator(FieldT&& field,
                          FieldDriverOptions const& options,
                          ParticleTrackView const& particle,
                          GTV&& geometry)
{
    return make_field_propagator(
        make_mag_field_stepper<StepperT>(::celeritas::forward<FieldT>(field),
                                         particle.charge()),
        options,
        particle,
        ::celeritas::forward<GTV>(geometry));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
