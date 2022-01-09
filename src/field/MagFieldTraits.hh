//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagFieldTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "field/MagFieldEquation.hh"
#include "field/FieldDriver.hh"
#include "field/FieldPropagator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A trait class that encapsulates a set of template classes for the
 * propagation of the charged particle in a magnetic field.
 */
template<class Field, template<class> class Stepper>
struct MagFieldTraits
{
    using Field_t      = Field;
    using Equation_t   = MagFieldEquation<Field_t>;
    using Stepper_t    = Stepper<Equation_t>;
    using Driver_t     = FieldDriver<Stepper_t>;
    using Propagator_t = FieldPropagator<Driver_t>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
