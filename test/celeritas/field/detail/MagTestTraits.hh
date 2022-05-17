//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/MagTestTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/MagFieldEquation.hh"

namespace celeritas_test
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
    using Equation_t = celeritas::MagFieldEquation<const Field_t&>;
    using Stepper_t  = Stepper<const Equation_t&>;
    using Driver_t   = celeritas::FieldDriver<const Stepper_t&>;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas_test
