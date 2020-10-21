//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocatorStore.cc
//---------------------------------------------------------------------------//
#include "Secondary.hh"
#include "base/StackAllocatorStore.t.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Explicitly instantiate stack allocator
template class StackAllocatorStore<Secondary>;

//---------------------------------------------------------------------------//
} // namespace celeritas
