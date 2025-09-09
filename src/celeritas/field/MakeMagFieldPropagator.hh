//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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

#include "FieldDriverOptions.hh"
#include "FieldPropagator.hh"
#include "FieldSubstepper.hh"
#include "MagFieldEquation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Create an integrator for moving a charge in a magnetic field.
 *
 * Example:
 * \code
 * auto step = make_stepper<DormandPrinceIntegrator>(
 *    UniformField{{1, 2, 3}},
 *    particle.charge());
 * \endcode
 */
template<template<class EquationT> class IntegratorT, class FieldT>
CELER_FUNCTION decltype(auto)
make_mag_field_integrator(FieldT&& field, units::ElementaryCharge charge)
{
    using Equation_t = MagFieldEquation<FieldT>;
    return IntegratorT<Equation_t>{
        Equation_t{::celeritas::forward<FieldT>(field), charge}};
}

//---------------------------------------------------------------------------//
/*!
 * Create a field propagator from an existing integrator.
 *
 * Example:
 * \code
 * FieldDriverOptions driver_options,
 * auto propagate = make_field_propagator(
 *    integrate,
 *    driver_options,
 *    particle,
 *    &geo);
 * propagate(0.123);
 * \endcode
 */
template<class IntegratorT, class GTV>
CELER_FUNCTION decltype(auto)
make_field_propagator(IntegratorT&& integrate,
                      FieldDriverOptions const& options,
                      ParticleTrackView const& particle,
                      GTV&& geometry)
{
    return FieldPropagator{
        FieldSubstepper{options, ::celeritas::forward<IntegratorT>(integrate)},
        particle,
        ::celeritas::forward<GTV>(geometry)};
}

//---------------------------------------------------------------------------//
/*!
 * Create a magnetic field propagator.
 *
 * Example:
 * \code
 * FieldDriverOptions driver_options,
 * auto propagate = make_mag_field_propagator<DormandPrinceIntegrator>(
 *    UniformField{{1, 2, 3}},
 *    driver_options,
 *    particle,
 *    &geo);
 * propagate(0.123);
 * \endcode
 */
template<template<class EquationT> class IntegratorT, class FieldT, class GTV>
CELER_FUNCTION decltype(auto)
make_mag_field_propagator(FieldT&& field,
                          FieldDriverOptions const& options,
                          ParticleTrackView const& particle,
                          GTV&& geometry)
{
    return make_field_propagator(
        make_mag_field_integrator<IntegratorT>(
            ::celeritas::forward<FieldT>(field), particle.charge()),
        options,
        particle,
        ::celeritas::forward<GTV>(geometry));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
