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
#include "physics/base/Units.hh"

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
    using OdeArray = Array<real_type, 6>;
    //@}

  public:
    //! Construct with a magnetic field
    CELER_FUNCTION FieldEquation(const MagField& field);

    //! Set charge if q != -eplus
    CELER_FUNCTION void set_charge(units::ElementaryCharge q);

    //! Evaluate the right hand side of the field equation
    CELER_FUNCTION auto operator()(const OdeArray& y) const -> OdeArray;

  private:
    //! Scale factor for the coefficient of the equation (temporary)
    static CELER_CONSTEXPR_FUNCTION real_type scale() { return 1e-14; }

  private:
    units::ElementaryCharge charge_;
    real_type               coeffi_;
    const MagField&         field_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "FieldEquation.i.hh"
