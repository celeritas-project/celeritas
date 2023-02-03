//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/grid/VectorUtils.hh
//! \brief Grid creation helpers
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Return evenly spaced numbers over a specific interval
std::vector<double> linspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
// Return logarithmically spaced numbers over a specific interval
std::vector<double> logspace(double start, double stop, size_type n);

//---------------------------------------------------------------------------//
}  // namespace celeritas
