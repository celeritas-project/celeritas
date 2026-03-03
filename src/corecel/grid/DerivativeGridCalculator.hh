//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/grid/DerivativeGridCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/inp/Grid.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Calculate the derivatives of a given grid.
inp::Grid construct_derivative_grid(inp::Grid const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
