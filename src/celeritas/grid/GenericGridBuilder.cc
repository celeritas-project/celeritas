//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/GenericGridBuilder.cc
//---------------------------------------------------------------------------//
#include "GenericGridBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the builder from imported Geant grids.
 */
std::unique_ptr<GenericGridBuilder>
GenericGridBuilder::from_geant(SpanConstDbl grid, SpanConstDbl values)
{
    return std::make_unique<GenericGridBuilder>(grid, values);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the builder directly from grids.
 */
GenericGridBuilder::GenericGridBuilder(SpanConstDbl grid, SpanConstDbl values)
    : grid_(grid), values_(values)
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
