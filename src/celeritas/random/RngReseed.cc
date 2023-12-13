//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngReseed.cc
//---------------------------------------------------------------------------//
#include "RngReseed.hh"

#include "corecel/cont/Range.hh"
#include "corecel/sys/ThreadId.hh"

#include "RngEngine.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reinitialize the RNG states on host at the start of an event.
 *
 * Each thread's state is initialized using same seed and skipped ahead a
 * different number of subsequences so the sequences on different threads will
 * not have statistically correlated values.
 */
void reseed_rng(HostCRef<RngParamsData> const& params,
                HostRef<RngStateData> const& state,
                size_type event_id)
{
    for (auto tid : range(TrackSlotId{state.size()}))
    {
        RngEngine::Initializer_t init;
#if CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW
        init.seed = params.seed[0];
#else
        init.seed = params.seed;
#endif
        init.subsequence = event_id * state.size() + tid.get();
        RngEngine engine(params, state, tid);
        engine = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
