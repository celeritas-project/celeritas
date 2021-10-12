//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TrackDiagnostic.cc
//---------------------------------------------------------------------------//
#include "TrackDiagnostic.hh"

#include "base/Macros.hh"
#include <thrust/transform_reduce.h>

using namespace celeritas;

namespace demo_loop
{
template<>
void TrackDiagnostic<MemSpace::host>::end_step(const StateDataRef& states)
{
    num_alive_per_step_.push_back(demo_loop::reduce_alive(states));
}
/*!
 * Sums the number of 'alive' tracks.
 *
 * This function is nearly identical to its device-side counterpart.
 */
size_type
reduce_alive(const StateData<Ownership::reference, MemSpace::host>& states)
{
    auto sim_states = states.sim.state[AllItems<SimTrackState>{}].data();

    return thrust::transform_reduce(
        thrust::raw_pointer_cast(sim_states),
        thrust::raw_pointer_cast(sim_states) + states.size(),
        OneIfAlive(),
        0,
        thrust::plus<unsigned int>());
}
} // namespace demo_loop