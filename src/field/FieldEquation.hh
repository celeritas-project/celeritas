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

    // Set charge if q != -eplus
    CELER_FUNCTION void set_charge(real_type q);

    // Evaluate the right hand side
    CELER_FUNCTION ode_type operator()(const ode_type& y);
    // XXX: overload with the position arguement for a non-uniform field

    // Evaluate the right hand side for a given B
    CELER_FUNCTION ode_type evaluate_rhs(const Real3& B, const ode_type& y);

  private:
    // XXX: remove once scaled_array is added to ArrayUtils
    CELER_FUNCTION Real3 scaled_real3(const real_type a, const Real3& x)
    {
        Real3 result;
        for (std::size_t i = 0; i != 3; ++i)
        {
            result[i] = a * x[i];
        }
        return result;
    }

  private:
    real_type charge_;
    real_type coeffi_;
    MagField& field_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldEquation.i.hh"
