//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.i.hh
//---------------------------------------------------------------------------//

#include "base/CollectionBuilder.hh"
#include "physics/base/PhysicsTrackView.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
template<MemSpace M>
ParticleProcessDiagnostic<M>::ParticleProcessDiagnostic(
    const ParamsDataRef& params,
    SPConstParticles     particles,
    SPConstPhysics       physics)
    : params_(params), particles_(particles), physics_(physics)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(particles_);
    CELER_EXPECT(physics_);

    // Counts of particle/model combinations that underwent discrete
    // interactions (indexed as model_id * num_particles + particle_id)
    size_type size = particles_->size() * physics_->num_models();
    resize(&counts_, size);
}

//---------------------------------------------------------------------------//
/*!
 * Tally the particle/process combinations that occur at each step.

 * This must be called after the post-step kernel and before the
 * post-processing.
 */
template<MemSpace M>
void ParticleProcessDiagnostic<M>::mid_step(const StateDataRef& states)
{
    using ItemsRef = celeritas::Collection<size_type, Ownership::reference, M>;

    ItemsRef counts(counts_);
    count_particle_process(params_, states, counts);
}

//---------------------------------------------------------------------------//
/*!
 * Counts of particle/model combinations that underwent discrete interaction.
 */
template<MemSpace M>
std::unordered_map<std::string, celeritas::size_type>
ParticleProcessDiagnostic<M>::particle_processes() const
{
    using BinId = celeritas::ItemId<size_type>;
    using HostItems
        = celeritas::Collection<size_type, Ownership::value, MemSpace::host>;

    // Copy result to host if necessary
    HostItems counts(counts_);

    // Map particle ID/model ID to particle and process name and store counts
    std::unordered_map<std::string, size_type> result;
    for (auto model_id : range(celeritas::ModelId{physics_->num_models()}))
    {
        const auto& process = physics_->process(physics_->process_id(model_id));
        for (auto particle_id :
             range(celeritas::ParticleId{particles_->size()}))
        {
            size_type index = model_id.get() * particles_->size()
                              + particle_id.get();
            CELER_ASSERT(index < counts.size());

            size_type count = counts[BinId{index}];
            if (count > 0)
            {
                // Accumulate the result for this process
                std::string label = process.label() + " "
                                    + particles_->id_to_label(particle_id);
                result[label] += count;
            }
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
template<MemSpace M>
CELER_FUNCTION
ParticleProcessLauncher<M>::ParticleProcessLauncher(const ParamsDataRef& params,
                                                    const StateDataRef& states,
                                                    ItemsRef&           counts)
    : params_(params), states_(states), counts_(counts)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(states_);
}

//---------------------------------------------------------------------------//
/*!
 * Create track views and tally particle/processes.
 */
template<MemSpace M>
CELER_FUNCTION void ParticleProcessLauncher<M>::operator()(ThreadId tid) const
{
    using BinId = celeritas::ItemId<size_type>;

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::PhysicsTrackView physics(
        params_.physics, states_.physics, particle.particle_id(), {}, tid);

    if (physics.model_id())
    {
        size_type index = physics.model_id().get() * physics.num_particles()
                          + particle.particle_id().get();
        CELER_ASSERT(index < counts_.size());
        celeritas::atomic_add(&counts_[BinId(index)], size_type{1});
    }
}
} // namespace demo_loop
