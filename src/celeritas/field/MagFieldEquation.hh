//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
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
#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Evaluate the force applied by a magnetic field.
 *
 * The templated \c FieldT must be a function-like object with the signature
 * \code
 * Real3 (*)(const Real3&)
 * \endcode
 * which returns a magnetic field vector at a given position. The field
 * strength is in Celeritas native units, so multiply by \c units::tesla if
 * necessary.
 */
template<class FieldT>
class MagFieldEquation
{
  public:
    //!@{
    //! \name Type aliases
    using Field_t = FieldT;
    //!@}

  public:
    // Construct with a magnetic field
    inline CELER_FUNCTION
    MagFieldEquation(FieldT&& field, units::ElementaryCharge q);

    // Evaluate the right hand side of the field equation
    inline CELER_FUNCTION OdeState operator()(OdeState const& y) const;

  private:
    // Field evaluator
    Field_t calc_field_;

    // The (Lorentz) coefficent in ElementaryCharge and MevMomentum
    real_type coeffi_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class FieldT>
CELER_FUNCTION MagFieldEquation(FieldT&&, units::ElementaryCharge)
    ->MagFieldEquation<FieldT>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a magnetic field equation.
 */
template<class FieldT>
CELER_FUNCTION
MagFieldEquation<FieldT>::MagFieldEquation(FieldT&& field,
                                           units::ElementaryCharge charge)
    : calc_field_(::celeritas::forward<FieldT>(field))
    , coeffi_{native_value_from(charge)
              / native_value_from(units::MevMomentum{1})}
{
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the right hand side of the Lorentz equation.
 *
 * This calculates the force based on the given magnetic field state
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
MagFieldEquation<FieldT>::operator()(OdeState const& y) const -> OdeState
{
    CELER_EXPECT(y.mom[0] != 0 || y.mom[1] != 0 || y.mom[2] != 0);
    real_type momentum_inv = celeritas::rsqrt(dot_product(y.mom, y.mom));

    // Evaluate the rate of change in particle's position per unit length: this
    // is just the direction
    OdeState result;
    result.pos = detail::ax(momentum_inv, y.mom);

    // Calculate the magnetic field value at the current position
    // to calculate the force on the particle
    result.mom = detail::ax(coeffi_ * momentum_inv,
                            cross_product(y.mom, calc_field_(y.pos)));

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
