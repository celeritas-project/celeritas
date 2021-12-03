//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PhysicsConstants
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"
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
 * electron_mass_c2     | g * (m/s)^2           |
 * electron_mass_mev    | g * (m/s)^2/MeV       | Geant4 electron mass
 * migdal_constant      | cm^3                  | Bremsstrahlung
 * lpm_constant         | MeV/cm                | Relativistic Bremsstrahlung
 */

using namespace constants;
using namespace units;

//!@{
//! Derivative constants with units
constexpr real_type electron_mass_c2  = electron_mass * c_light * c_light;
constexpr real_type electron_mass_mev = electron_mass_c2
                                        / (1e6 * e_electron * volt);
constexpr real_type migdal_constant = 4 * pi * r_electron * lambdabar_electron
                                      * lambdabar_electron;
constexpr real_type lpm_constant = alpha_fine_structure * electron_mass_c2
                                   * electron_mass_c2
                                   / (2 * h_planck * c_light);
//!@}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
