//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Types.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/cont/Array.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// STRUCTS
//---------------------------------------------------------------------------//
/*!
 * A utility array of the equation of motion based on \ref celeritas::Array .
 */
struct OdeState
{
    using MomentumUnits = units::MevMomentum;
    using Real3 = Array<real_type, 3>;

    Real3 pos;  //!< Particle position
    Real3 mom;  //!< Particle momentum
};

//---------------------------------------------------------------------------//
/*!
 * The result of the integration stepper.
 */
struct FieldStepperResult
{
    OdeState mid_state;  //!< OdeState at the middle
    OdeState end_state;  //!< OdeState at the end
    OdeState err_state;  //!< Delta between one full step and two half steps
};

//---------------------------------------------------------------------------//
/*!
 * The result of moving up to a certain distance along a step.
 */
struct DriverResult
{
    OdeState state;  //!< Post-step state
    real_type step;  //!< Actual curved step
};

//---------------------------------------------------------------------------//
// FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Perform y <- ax + y for OdeState.
 */
inline CELER_FUNCTION void axpy(real_type a, OdeState const& x, OdeState* y)
{
    axpy(a, x.pos, &y->pos);
    axpy(a, x.mom, &y->mom);
}

}  // namespace celeritas
