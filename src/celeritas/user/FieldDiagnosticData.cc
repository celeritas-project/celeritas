//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/FieldDiagnosticData.cc
//---------------------------------------------------------------------------//
#include "FieldDiagnosticData.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Resize based on number of bins.
 */
template<MemSpace M>
inline void resize(FieldDiagnosticStateData<Ownership::value, M>* state,
                   HostCRef<FieldDiagnosticParamsData> const& params,
                   StreamId,
                   size_type)
{
    CELER_EXPECT(params);
    size_type num_energy_bins = params.energy.size - 1;
    resize(&state->counts, num_energy_bins * params.num_substep_bins);
    fill(size_type(0), &state->counts);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void
resize(FieldDiagnosticStateData<Ownership::value, MemSpace::host>* state,
       HostCRef<FieldDiagnosticParamsData> const& params,
       StreamId,
       size_type);

template void
resize(FieldDiagnosticStateData<Ownership::value, MemSpace::device>* state,
       HostCRef<FieldDiagnosticParamsData> const& params,
       StreamId,
       size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
