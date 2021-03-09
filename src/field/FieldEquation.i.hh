//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldEquation.i.hh
//---------------------------------------------------------------------------//
#include "MagField.hh"

#include "base/Constants.hh"
#include "base/Array.hh"
#include "base/ArrayUtils.hh"
#include "base/Algorithms.hh"

#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a magnetic field.
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
OdeArray<real_type, 6> FieldEquation::operator()(const ode_type& y) const
{
    ode_type dy;

    dy.position(y.momentum());
    dy.momentum(
        cross_product(y.momentum_scaled(coeffi_), field_(y.position())));
    dy *= y.momentum_inv();

    return dy;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
