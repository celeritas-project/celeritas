//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagTestTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "field/MagFieldEquation.hh"
#include "field/FieldDriver.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * A trait class that encapsulates a set of template classes for testing
 * the magnetic field driver and stepper.
 */
template<class Field, template<class> class Stepper>
struct MagTestTraits
{
    using Field_t    = Field;
    using Equation_t = MagFieldEquation<Field_t>;
    using Stepper_t  = Stepper<Equation_t>;
    using Driver_t   = FieldDriver<Stepper_t>;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
