//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagFieldEquation.hh
//---------------------------------------------------------------------------//
#pragma once

#include "MagField.hh"
#include "FieldInterface.hh"

#include "base/Types.hh"
#include "physics/base/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * The MagFieldEquation evaluates the righ hand side of the Lorentz equation
 * for a given magnetic field value.
 */
class MagFieldEquation
{
  public:
    // Construct with a magnetic field
    inline CELER_FUNCTION
    MagFieldEquation(const MagField& field, units::ElementaryCharge q);

    // Evaluate the right hand side of the field equation
    inline CELER_FUNCTION auto operator()(const OdeState& y) const -> OdeState;

  private:
    const MagField&         field_;
    units::ElementaryCharge charge_;
    real_type               coeffi_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MagFieldEquation.i.hh"
