//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsConstants
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
#include "base/Macros.hh"
#include "base/Constants.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Physical constants which are derived from fundamental constants.
 *
 * Derived constants    | Unit                  | Notes
 * -------------------- | --------------------- | ------------
 * electron_mass_c2()   | g * (m/s)^2           |
 * migdal_constant()    | cm^3                  | Bremsstrahlung
 * lpm_constant()       | MeV/cm                | Relativistic Bremsstrahlung
 */

//!@{
//! Constant functions
CELER_CONSTEXPR_FUNCTION real_type electron_mass_c2()
{
    using namespace constants;

    return electron_mass * c_light * c_light;
}

CELER_CONSTEXPR_FUNCTION real_type migdal_constant()
{
    using namespace constants;
    using namespace units;

    return 4 * pi * r_electron * lambdabar_electron * lambdabar_electron;
}

CELER_CONSTEXPR_FUNCTION real_type lpm_constant()
{
    using namespace constants;
    using namespace units;

    return alpha_fine_structure * electron_mass_c2() * electron_mass_c2()
           / (2 * h_planck * c_light);
}
//!@}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
