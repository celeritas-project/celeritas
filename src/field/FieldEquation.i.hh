//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldEquation.i.hh
//---------------------------------------------------------------------------//
#include "MagField.hh"

#include "base/Constants.hh"
#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a const magnetic field.
 */
CELER_FUNCTION
FieldEquation::FieldEquation(const MagField& field) : field_(field)
{
    // set the default charge and the lorentz_cof with -eplus
    set_charge(units::ElementaryCharge{-1.0});
}

//---------------------------------------------------------------------------//
/*!
 * Set charge for this track and recalcuate lorentz_cof
 */

CELER_FUNCTION void FieldEquation::set_charge(units::ElementaryCharge charge)
{
    charge_ = charge;
    coeffi_ = charge_.value() * constants::c_light * this->scale();
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the Lorentz equation
 *
 * \f[
    m \frac{d^2 \vec{x}}{d t^2} = (q/c)(\vec{v} \times  \vec{B})
    s = |v|t
    \vec{y} = d\vec{x}/ds
    \frac{d\vec{x}}{ds} = \vec{v}/|v|
    \frac{d\vec{y}}{ds} = (q/pc)(\vec{y} \times \vec{B})
   \f]
 */
CELER_FUNCTION
auto FieldEquation::operator()(const OdeArray& y) const -> OdeArray
{
    // Get the magnetic field value at the given position
    Real3 B = field_({y[0], y[1], y[2]});

    // Evalue the right-hand-side of the equation
    OdeArray rhs;

    real_type momentum_mag2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    CELER_ASSERT(momentum_mag2 > 0.0);
    real_type momenum_inv = 1.0 / std::sqrt(momentum_mag2);

    rhs[0] = y[3] * momenum_inv;
    rhs[1] = y[4] * momenum_inv;
    rhs[2] = y[5] * momenum_inv;
    rhs[3] = coeffi_ * (y[4] * B[2] - y[5] * B[1]) * momenum_inv;
    rhs[4] = coeffi_ * (y[5] * B[0] - y[3] * B[2]) * momenum_inv;
    rhs[5] = coeffi_ * (y[3] * B[1] - y[4] * B[0]) * momenum_inv;

    return rhs;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
