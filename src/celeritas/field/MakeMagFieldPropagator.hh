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
 * Create a magnetic field propagator.
 *
 * \example
 * \code
 * auto propagate = make_mag_field_propagator<DormandPrinceStepper>(
 *    UniformField{{1, 2, 3}},
 *    FieldDriverOptions{},
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
    CELER_ASSERT(geometry);
    using Equation_t = MagFieldEquation<FieldT>;
    using Stepper_t  = StepperT<Equation_t>;
    using Driver_t   = FieldDriver<Stepper_t>;
    return FieldPropagator<Driver_t>{
        Driver_t{options,
                 Stepper_t{Equation_t{::celeritas::forward<FieldT>(field),
                                      particle.charge()}}},
        particle,
        geometry};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
