//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldEquation.i.cuh
//---------------------------------------------------------------------------//
#include "MagField.hh"
#include <cmath>
#include "base/Constants.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a magnetic field.
 */
CELER_FUNCTION
FieldEquation::FieldEquation(MagField& field) : field_(field)
{
    // set the default charge and the lorentz_cof with -eplus
    set_charge(-1.0);
}

//---------------------------------------------------------------------------//
/*!
 * Set charge for this track and recalcuate lorentz_cof
 */

CELER_FUNCTION void FieldEquation::set_charge(real_type charge)
{
    charge_ = charge;
    coeffi_ = charge_ * constants::c_light * 1.0e-8;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the field equation
 */

CELER_FUNCTION void FieldEquation::
                    operator()(const ode_type y, ode_type& dydx) const
{
    this->evaluate_rhs(field_(), y, dydx);
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the Lorentz equation for a given B
 */

CELER_FUNCTION
void FieldEquation::evaluate_rhs(const Real3    B,
                                 const ode_type y,
                                 ode_type&      dy) const
{
    /*** m d^2x/dt^2 = (q/c)(vxB), s = |v|t, y = dx/ds
         rhs:  dx/ds = v/|v|
               dy/ds = (q/pc)(yxB)
    */

    dy[0] = y[3];
    dy[1] = y[4];
    dy[2] = y[5];
    dy[3] = coeffi_ * (y[4] * B[2] - y[5] * B[1]);
    dy[4] = coeffi_ * (y[5] * B[0] - y[3] * B[2]);
    dy[5] = coeffi_ * (y[3] * B[1] - y[4] * B[0]);

    dy *= y.momentum_inv();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
