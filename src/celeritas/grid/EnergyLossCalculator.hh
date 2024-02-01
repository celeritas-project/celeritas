//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/EnergyLossCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "XsCalculator.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * For now, energy loss calculation has the same behavior as cross sections.
 *
 * The return value is [MeV / len] but isn't wrapped with a Quantity.
 */
using EnergyLossCalculator = XsCalculator;

//---------------------------------------------------------------------------//
}  // namespace celeritas
