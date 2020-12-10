//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StateStore.cc
//---------------------------------------------------------------------------//
#include "StateStore.hh"

#include "base/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the track state storage objects.
 */
StateStore::StateStore(const Input& inp)
    : particle_states_(ParticleStateStore(inp.num_tracks))
    , geo_states_(GeoStateStore(*inp.geo, inp.num_tracks))
    , sim_states_(SimStateStore(inp.num_tracks))
    , rng_states_(RngStateStore(inp.num_tracks, inp.host_seed))
    , interactions_(this->size())
{
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
StatePointers StateStore::device_pointers()
{
    StatePointers result;
    result.particle     = particle_states_.device_pointers();
    result.geo          = geo_states_.device_pointers();
    result.sim          = sim_states_.device_pointers();
    result.rng          = rng_states_.device_pointers();
    result.interactions = interactions_.device_pointers();
    ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
