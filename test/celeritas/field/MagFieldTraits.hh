//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
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
template<class StepperT>
class FieldDriver;
template<class DriverT>
class FieldPropagator;

//---------------------------------------------------------------------------//
/*!
 * Manage class types for different magnetic fields and stepping classes.
 *
 * The Stepper must take an Equation function-like operator as a template
 * parameter.
 */
template<class FieldT, template<class EquationT> class StepperT>
struct MagFieldTraits
{
    using Field_t      = FieldT;
    using Equation_t   = MagFieldEquation<const Field_t&>;
    using Stepper_t    = StepperT<const Equation_t&>;
    using Driver_t     = FieldDriver<const Stepper_t&>;
    using Propagator_t = FieldPropagator<const Driver_t&>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
