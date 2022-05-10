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
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The MagFieldEquation evaluates the right hand side of the Lorentz equation
 * for a given magnetic field value.
 * The templated \c FieldT must provide the operator(Real3 position) which
 * returns a magnetic field value of Real3 at a given position
 */
template<class FieldT>
class MagFieldEquation
{
  public:
    //!@{
    //! Type aliases
    using field_type = FieldT;
    //!@}

  public:
    // Construct with a magnetic field
    inline CELER_FUNCTION
    MagFieldEquation(const FieldT& field, units::ElementaryCharge q);

    // Evaluate the right hand side of the field equation
    inline CELER_FUNCTION auto operator()(const OdeState& y) const -> OdeState;

  private:
    const field_type&       field_;
    units::ElementaryCharge charge_;
    real_type               coeffi_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a constant magnetic field.
 */
template<class FieldT>
CELER_FUNCTION
MagFieldEquation<FieldT>::MagFieldEquation(const FieldT&           field,
                                           units::ElementaryCharge charge)
    : field_(field), charge_(charge)
{
    // The (Lorentz) coefficent in ElementaryCharge and MevMomentum
    coeffi_ = native_value_from(charge_)
              / native_value_from(units::MevMomentum{1});
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
template<class FieldT>
CELER_FUNCTION auto
MagFieldEquation<FieldT>::operator()(const OdeState& y) const -> OdeState
{
    // Get a magnetic field value at a given position
    Real3 mag_vec = field_(y.pos);

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
