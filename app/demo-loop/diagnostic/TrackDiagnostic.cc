//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.cc
//---------------------------------------------------------------------------//
#include "TrackDiagnostic.hh"

#include "base/Macros.hh"
#include "base/Range.hh"

using namespace celeritas;

namespace demo_loop
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Sums the number of 'alive' tracks.
 *
 * This function is nearly identical to its device-side counterpart.
 */
size_type reduce_alive(const CoreStateHostRef& states)
{
    auto sim_states = states.sim.state[AllItems<SimTrackState>{}].data();

    size_type result = 0;
    for (auto i : range(states.size()))
    {
        if (sim_states[i].alive)
            result += 1;
    }
    return result;
}
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Add the current step's number of alive tracks to this diagnostic.
 */
template<>
void TrackDiagnostic<MemSpace::host>::end_step(const StateRef& states)
{
    num_alive_per_step_.push_back(demo_loop::reduce_alive(states));
}

#if !CELER_USE_DEVICE
template<>
void TrackDiagnostic<MemSpace::device>::end_step(const StateRef&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace demo_loop
