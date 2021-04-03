//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StateStore.cc
//---------------------------------------------------------------------------//
#include "StateStore.hh"

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the track state storage objects.
 */
StateStore::StateStore(const Input& inp)
    : sim_states_(SimStateStore(inp.num_tracks)), interactions_(inp.num_tracks)
{
    CELER_EXPECT(inp.geo);
    make_builder(&particle_states_.state).resize(inp.num_tracks);

    // TODO: Input struct should take RngParams
    RngParamsData<Ownership::value, MemSpace::host> rng_params;
    rng_params.seed = inp.host_seed;
    resize(&rng_states_, make_const_ref(rng_params), inp.num_tracks);

    resize(&geo_states_, inp.geo->host_pointers(), inp.num_tracks);

    CELER_ENSURE(inp.num_tracks == this->size());
}

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
StatePointers StateStore::device_pointers()
{
    StatePointers result;
    result.particle     = particle_states_;
    result.rng          = rng_states_;
    result.geo          = geo_states_;
    result.sim          = sim_states_.device_pointers();
    result.interactions = interactions_.device_pointers();
    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
