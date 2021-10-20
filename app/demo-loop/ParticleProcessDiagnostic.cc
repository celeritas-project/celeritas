//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.cc
//---------------------------------------------------------------------------//
#include "ParticleProcessDiagnostic.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Tally the particle/process combinations that occur at each step.
 */
void count_particle_process(
    const ParamsHostRef&                                        params,
    const StateHostRef&                                         states,
    Collection<size_type, Ownership::reference, MemSpace::host> counts)
{
    for (auto tid : range(ThreadId{states.size()}))
    {
        auto model_id = states.physics.state[tid].model_id;
        if (model_id)
        {
            size_type index = model_id.get()
                                  * params.physics.process_groups.size()
                              + states.particles.state[tid].particle_id.get();
            CELER_ASSERT(index < counts.size());
            ++counts[ItemId<size_type>(index)];
        }
    }
}
} // namespace demo_loop
