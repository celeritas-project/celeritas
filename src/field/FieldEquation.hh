//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldEquation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "MagField.hh"
#include "base/Types.hh"
#include "field/base/OdeArray.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The FieldEquation evaluates the righ hand side of the Lorentz equation
 * for a given magnetic field value.
 */
class FieldEquation
{
  public:
    //@{
    //! Type aliases
    using ode_type = OdeArray;
    //@}

  public:
    // Construct with a magnetic field
    CELER_FUNCTION FieldEquation(MagField& field);

    // set charge if q != -eplus
    CELER_FUNCTION void set_charge(real_type q);

    // evaluate the right hand side
    // XXX: add the position arguement for a non-uniform field
    CELER_FUNCTION void operator()(const ode_type y, ode_type& dydx) const;

    // evaluate the right hand side for a given B
    CELER_FUNCTION void
    evaluate_rhs(const Real3 B, const ode_type y, ode_type& dydx) const;

  private:
    real_type charge_;
    real_type coeffi_;
    MagField& field_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldEquation.i.hh"
