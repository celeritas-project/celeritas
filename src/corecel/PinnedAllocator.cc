//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/PinnedAllocator.cc
//---------------------------------------------------------------------------//

#include "Types.hh"
#include "data/PinnedAllocator.t.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// Explicit instantiations
template struct PinnedAllocator<real_type>;
template struct PinnedAllocator<size_type>;
//---------------------------------------------------------------------------//
}  // namespace celeritas