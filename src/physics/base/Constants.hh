//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Constants.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Units.hh"

namespace celeritas
{
namespace constants
{
//---------------------------------------------------------------------------//
/*
 * Physical constants. Some of these values depend on the unit system.
 * Purely mathematical constants belong in base/Constants.hh
 */

//@{
//! Speed of light
constexpr real_type speed_of_light    = 1.; // Natural unit
constexpr real_type speed_of_light_sq = 1.;
//@}

//@{
//! Derived quantities
constexpr real_type mev_csq = units::mega_electron_volt * speed_of_light_sq;
//@}

//---------------------------------------------------------------------------//
} // namespace constants
} // namespace celeritas
