//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ParticleTallyData.cc
//---------------------------------------------------------------------------//
#include "ParticleTallyData.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize based on number of bins and particle types.
 */
template<MemSpace M>
inline void resize(ParticleTallyStateData<Ownership::value, M>* state,
                   HostCRef<ParticleTallyParamsData> const& params,
                   StreamId,
                   size_type)
{
    CELER_EXPECT(params);
    resize(&state->counts, params.num_bins * params.num_particles);
    fill(size_type(0), &state->counts);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void
resize(ParticleTallyStateData<Ownership::value, MemSpace::host>* state,
       HostCRef<ParticleTallyParamsData> const& params,
       StreamId,
       size_type);

template void
resize(ParticleTallyStateData<Ownership::value, MemSpace::device>* state,
       HostCRef<ParticleTallyParamsData> const& params,
       StreamId,
       size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
