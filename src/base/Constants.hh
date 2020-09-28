//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Types.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//
/*!
 * \namespace constants
 *
 * Mathematical and numerical constants. Physical constants (some of whose
 * values depend on the unit system) are defined in
 * `physics/base/Constants.hh`.
 */

constexpr real_type pi     = 3.14159265358979323846;
constexpr real_type two_pi = 2. * pi;

} // namespace constants

//---------------------------------------------------------------------------//
} // namespace celeritas
