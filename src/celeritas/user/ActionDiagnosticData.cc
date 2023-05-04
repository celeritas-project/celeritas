//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ActionDiagnosticData.cc
//---------------------------------------------------------------------------//
#include "ActionDiagnosticData.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize based on number of actions and particle types.
 */
template<MemSpace M>
inline void resize(ActionDiagnosticStateData<Ownership::value, M>* state,
                   HostCRef<ActionDiagnosticParamsData> const& params,
                   StreamId,
                   size_type)
{
    CELER_EXPECT(params);
    resize(&state->counts, params.num_actions * params.num_particles);
    fill(size_type(0), &state->counts);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void
resize(ActionDiagnosticStateData<Ownership::value, MemSpace::host>* state,
       HostCRef<ActionDiagnosticParamsData> const& params,
       StreamId,
       size_type);

template void
resize(ActionDiagnosticStateData<Ownership::value, MemSpace::device>* state,
       HostCRef<ActionDiagnosticParamsData> const& params,
       StreamId,
       size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
