//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DetectorUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "../DetectorPointers.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Bin energy deposition from the hit buffer into the tally grid
void bin_buffer(const DetectorPointers& device_ptrs);

//---------------------------------------------------------------------------//
// Multiply tally deposition by the given value
void normalize(const DetectorPointers& device_ptrs, real_type norm);

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
