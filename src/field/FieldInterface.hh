//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/ArrayUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A utility array of the equation of motion based on \ref celeritas::Array
 */
struct OdeState
{
    Real3 pos{0, 0, 0}; //!< Particle position
    Real3 mom{0, 0, 0}; //!< Particle momentum
};

//---------------------------------------------------------------------------//
/*!
 * The result of the integration stepper
 */
struct StepperResult
{
    OdeState end_state; //!< OdeState at the end
    OdeState mid_state; //!< OdeState at the middle
    OdeState err_state; //!< Delta between one full step and two half steps
};

//---------------------------------------------------------------------------//
} // namespace celeritas
