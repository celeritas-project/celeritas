//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ResponseTallyData.cc
//---------------------------------------------------------------------------//
#include "ResponseTallyData.hh"

#include "corecel/Assert.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Resize based on the number of bins and track slots.
 */
template<MemSpace M>
void resize(ResponseTallyStateData<Ownership::value, M>* state,
            HostCRef<ResponseTallyParamsData> const& params,
            StreamId,
            size_type size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);

    resize(&state->bin_ids, size);
    fill(ResponseTallyStateData<Ownership::value, M>::invalid_bin(),
         &state->bin_ids);

    resize(&state->counts, params.num_bins());
    fill(0_sz, &state->counts);

    CELER_ENSURE(*state);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void resize(ResponseTallyStateData<Ownership::value, MemSpace::host>*,
                     HostCRef<ResponseTallyParamsData> const&,
                     StreamId,
                     size_type);

template void resize(
    ResponseTallyStateData<Ownership::value, MemSpace::device>*,
    HostCRef<ResponseTallyParamsData> const&,
    StreamId,
    size_type);

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
