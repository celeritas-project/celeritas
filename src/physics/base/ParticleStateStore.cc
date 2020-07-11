//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleStateStore.cc
//---------------------------------------------------------------------------//
#include "ParticleStateStore.hh"

#include "base/Array.hh"
#include "ParticleStatePointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with number of parallel tracks.
 */
ParticleStateStore::ParticleStateStore(size_type size) : vars_(size)
{
    REQUIRE(size > 0);
    ENSURE(!vars_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Number of threads stored in the state.
 */
size_type ParticleStateStore::size() const
{
    return vars_.size();
}

//---------------------------------------------------------------------------//
/*!
 * View to on-device state data.
 */
ParticleStatePointers ParticleStateStore::device_pointers()
{
    ParticleStatePointers result;
    result.vars = vars_.device_pointers();

    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
