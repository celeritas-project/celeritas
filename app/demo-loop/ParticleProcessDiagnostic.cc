//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.cc
//---------------------------------------------------------------------------//
#include "ParticleProcessDiagnostic.hh"

#include "physics/base/PhysicsTrackView.hh"

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
        ParticleTrackView particle(params.particles, states.particles, tid);
        PhysicsTrackView  physics(
            params.physics, states.physics, particle.particle_id(), {}, tid);

        if (physics.model_id())
        {
            size_type index = physics.model_id().get() * physics.num_particles()
                              + particle.particle_id().get();
            CELER_ASSERT(index < counts.size());
            atomic_add(&counts[ItemId<size_type>(index)], 1u);
        }
    }
}
} // namespace demo_loop
