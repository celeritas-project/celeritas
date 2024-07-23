//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
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
    auto size = state.size();
#if CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (TrackSlotId::size_type i = 0; i < size; ++i)
    {
        RngEngine::Initializer_t init;
        init.seed = params.seed;
        init.subsequence = event_id * size + i;
        RngEngine engine(params, state, TrackSlotId{i});
        engine = init;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
