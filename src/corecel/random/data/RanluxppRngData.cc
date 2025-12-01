//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/RanluxppRngData.cc
//---------------------------------------------------------------------------//
#include "RanluxppRngData.hh"

#include <random>
#include <vector>

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/random/data/RanluxppTypes.hh"
#include "corecel/random/data/detail/RanluxppRngStateInit.hh"

namespace celeritas
{

//---------------------------------------------------------------------------//
/*!
 * Resize and seed the RNG states.
 */
template<MemSpace M>
void resize(RanluxppRngStateData<Ownership::value, M>* state,
            HostCRef<RanluxppRngParamsData> const& params,
            StreamId stream,
            size_type size)
{
    CELER_EXPECT(params);
    CELER_EXPECT(stream);
    CELER_EXPECT(size > 0);
    CELER_EXPECT(M == MemSpace::host || celeritas::device());

    // Create a temporary "native" copy of the params so that we can initialize
    // the ranlux state
    RanluxppRngParamsData<Ownership::value, M> p;
    p = params;

    // Resize the state collection and initialize the state for each stream on
    // device
    resize(&state->state, size);
    celeritas::detail::ranlux_state_init(
        make_const_ref(p), make_ref(*state), stream);

    CELER_ENSURE(*state);
    CELER_ENSURE(state->size() == size);
}

//---------------------------------------------------------------------------//
// Explicit instantiations
template void resize(HostVal<RanluxppRngStateData>*,
                     HostCRef<RanluxppRngParamsData> const&,
                     StreamId,
                     size_type);

template void resize(RanluxppRngStateData<Ownership::value, MemSpace::device>*,
                     HostCRef<RanluxppRngParamsData> const&,
                     StreamId,
                     size_type);

//---------------------------------------------------------------------------//
}  // namespace celeritas
