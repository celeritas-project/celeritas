//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Types.hh"
#include "base/SystemOfUnits.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//
/*!
 * \namespace constants
 *
 * Mathematical and numerical constants.
 */

  constexpr real_type pi     = 3.14159265358979323846;
  constexpr real_type two_pi = 2. * pi;

  static constexpr real_type cLight = 2.99792458e+8 * units::meter / units::second;
  static constexpr real_type electron_mass_c2 = 0.510998910 * units::MeV;

} // namespace constants

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
