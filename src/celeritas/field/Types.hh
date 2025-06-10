//------------------------------ -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Store a track's position and momentum for field integration.
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
 * The result of a single integration.
 */
struct FieldIntegration
{
    OdeState mid_state;  //!< OdeState at the middle
    OdeState end_state;  //!< OdeState at the end
    OdeState err_state;  //!< Delta between one full step and two half steps
};

//---------------------------------------------------------------------------//
/*!
 * The result of moving up to a certain distance along a step.
 */
struct Substep
{
    OdeState state;  //!< Post-step state
    real_type length;  //!< Actual curved step
};

//! \cond (CELERITAS_DOC_DEV)
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
//! \endcond

}  // namespace celeritas
