//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagFieldEquation.i.hh
//---------------------------------------------------------------------------//
#include "MagField.hh"

#include "base/Constants.hh"
#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a constant magnetic field.
 */
CELER_FUNCTION
MagFieldEquation::MagFieldEquation(const MagField&         field,
                                   units::ElementaryCharge charge)
    : field_(field), charge_(charge)
{
    // The (Lorentz) coefficent in ElementaryCharge and MevMomentum
    coeffi_ = unit_cast(charge_) / unit_cast(units::MevMomentum{1});
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the Lorentz equation.
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
auto MagFieldEquation::operator()(const OdeState& y) const -> OdeState
{
    // Get a magnetic field value at a given position
    Real3 mag_vec = field_();

    real_type momentum_mag2 = dot_product(y.mom, y.mom);
    CELER_ASSERT(momentum_mag2 > 0.0);
    real_type momentum_inv = 1.0 / std::sqrt(momentum_mag2);

    // Evaluate the right-hand-side of the equation
    OdeState result;

    axpy(momentum_inv, y.mom, &result.pos);
    axpy(coeffi_ * momentum_inv, cross_product(y.mom, mag_vec), &result.mom);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
