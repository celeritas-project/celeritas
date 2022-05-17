//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "FieldDriver.hh"
#include "FieldPropagator.hh"
#include "MagFieldEquation.hh"

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
    using Equation_t   = MagFieldEquation<const Field_t&>;
    using Stepper_t    = Stepper<const Equation_t&>;
    using Driver_t     = FieldDriver<const Stepper_t&>;
    using Propagator_t = FieldPropagator<const Driver_t&>;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
