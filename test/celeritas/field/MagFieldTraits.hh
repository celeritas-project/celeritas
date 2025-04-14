//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldTraits.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class FieldT>
class MagFieldEquation;
template<class IntegratorT>
class FieldDriver;
template<class DriverT>
class FieldPropagator;

//---------------------------------------------------------------------------//
/*!
 * Manage class types for different magnetic fields and stepping classes.
 *
 * The Integrator must take an Equation function-like operator as a template
 * parameter.
 */
template<class FieldT, template<class EquationT> class IntegratorT>
struct MagFieldTraits
{
    using Field_t = FieldT;
    using Equation_t = MagFieldEquation<Field_t const&>;
    using Integrator_t = IntegratorT<Equation_t const&>;
    using Driver_t = FieldDriver<Integrator_t const&>;
    using Propagator_t = FieldPropagator<Driver_t const&>;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
