//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldEquation.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the force applied by a magnetic field.
 *
 * The templated \c FieldT must provide the operator(Real3 position) which
 * returns a magnetic field value of Real3 at a given position.
 */
template<class FieldT>
class MagFieldEquation
{
  public:
    //!@{
    //! Type aliases
    using Field_t = FieldT;
    //!@}

  public:
    // Construct with a magnetic field
    inline CELER_FUNCTION
    MagFieldEquation(FieldT&& field, units::ElementaryCharge q);

    // Evaluate the right hand side of the field equation
    inline CELER_FUNCTION auto operator()(const OdeState& y) const -> OdeState;

  private:
    Field_t&& calc_field_;
    real_type coeffi_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a magnetic field equation.
 */
template<class FieldT>
CELER_FUNCTION
MagFieldEquation<FieldT>::MagFieldEquation(FieldT&&                field,
                                           units::ElementaryCharge charge)
    : calc_field_(::celeritas::forward<FieldT>(field))
{
    CELER_EXPECT(charge != zero_quantity());

    // The (Lorentz) coefficent in ElementaryCharge and MevMomentum
    coeffi_ = native_value_from(charge)
              / native_value_from(units::MevMomentum{1});
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the Lorentz equation.
 *
 * This calculates the force based on the current magnetic field state
 * (position and momentum).
 *
 * \f[
    m \frac{d^2 \vec{x}}{d t^2} = (q/c)(\vec{v} \times  \vec{B})
    s = |v|t
    \vec{y} = d\vec{x}/ds
    \frac{d\vec{x}}{ds} = \vec{v}/|v|
    \frac{d\vec{y}}{ds} = (q/pc)(\vec{y} \times \vec{B})
   \f]
 */
template<class FieldT>
CELER_FUNCTION auto
MagFieldEquation<FieldT>::operator()(const OdeState& y) const -> OdeState
{
    // Get a magnetic field value at a given position
    Real3 mag_vec = calc_field_(y.pos);

    real_type momentum_mag2 = dot_product(y.mom, y.mom);
    CELER_ASSERT(momentum_mag2 > 0);
    real_type momentum_inv = 1 / std::sqrt(momentum_mag2);

    // Evaluate the right-hand-side of the equation
    OdeState result;

    axpy(momentum_inv, y.mom, &result.pos);
    axpy(coeffi_ * momentum_inv, cross_product(y.mom, mag_vec), &result.mom);

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
