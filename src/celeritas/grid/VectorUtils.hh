//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/VectorUtils.hh
//! \brief Grid creation helpers
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Return evenly spaced numbers over a specific interval
std::vector<double> linspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
// Return logarithmically spaced numbers over a specific interval
std::vector<double> logspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
// True if the grid values are monotonically increasing
bool is_monotonic_increasing(Span<double const> grid);

//---------------------------------------------------------------------------//
}  // namespace celeritas
