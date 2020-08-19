//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file HostDebugSecondaryStorage.cc
//---------------------------------------------------------------------------//
#include "HostDebugSecondaryStorage.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
/*!
 * Resize and give unassigned valuess to the secondaries.
 *
 * The values here are to help ensure that the secondary allocator is properly
 * initializing the data when it operates.
 */
void HostDebugSecondaryStorage::resize(size_type capacity)
{
    Secondary unallocated_secondary;
    unallocated_secondary.def_id    = celeritas::ParticleDefId{0xdeadbeef};
    unallocated_secondary.direction = {-10, -20, -30};
    unallocated_secondary.energy    = -1;
    storage_.assign(capacity, unallocated_secondary);
    size_             = 0;
    pointers_.storage = celeritas::make_span(storage_);
    pointers_.size    = &size_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas_test
