//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/data/RanluxppRngData.cc
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
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

    // Move params to device
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
