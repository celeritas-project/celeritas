//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ParticleProcessDiagnostic.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <unordered_map>

#include "base/CollectionBuilder.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/PhysicsParams.hh"
#include "physics/base/PhysicsTrackView.hh"

#include "Diagnostic.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Diagnostic class for collecting information on particle and process types.
 *
 * Tallies the particle/process combinations that underwent a discrete
 * interaction at each step.
 */
template<MemSpace M>
class ParticleProcessDiagnostic : public Diagnostic<M>
{
  public:
    //!@{
    //! Type aliases
    using size_type = celeritas::size_type;
    using Items     = celeritas::Collection<size_type, Ownership::value, M>;
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    using SPConstParticles  = std::shared_ptr<const celeritas::ParticleParams>;
    using SPConstPhysics    = std::shared_ptr<const celeritas::PhysicsParams>;
    using TransporterResult = celeritas::TransporterResult;
    //!@}

  public:
    // Construct with shared problem data
    ParticleProcessDiagnostic(const ParamsDataRef& params,
                              SPConstParticles     particles,
                              SPConstPhysics       physics);

    // Particle/model tallied after sampling discrete interaction
    void mid_step(const StateDataRef& states) final;

    // Collect diagnostic results
    void get_result(TransporterResult* result) final;

    // Get particle-process and number of times the interaction occured
    std::unordered_map<std::string, size_type> particle_processes() const;

  private:
    // Shared problem data
    const ParamsDataRef& params_;
    // Shared particle data for getting particle name from particle ID
    SPConstParticles particles_;
    // Shared physics data for getting process name from model ID
    SPConstPhysics physics_;
    // Count of particle/model combinations that underwent discrete interaction
    Items counts_;
};

//---------------------------------------------------------------------------//
// KERNEL LAUNCHER(S)
//---------------------------------------------------------------------------//
/*!
 * Diagnostic kernel launcher
 */
template<MemSpace M>
class ParticleProcessLauncher
{
  public:
    //!@{
    //! Type aliases
    using size_type = celeritas::size_type;
    using ThreadId  = celeritas::ThreadId;
    using ItemsRef = celeritas::Collection<size_type, Ownership::reference, M>;
    using ParamsDataRef = celeritas::ParamsData<Ownership::const_reference, M>;
    using StateDataRef  = celeritas::StateData<Ownership::reference, M>;
    //!@}

  public:
    // Construct with shared and state data
    CELER_FUNCTION ParticleProcessLauncher(const ParamsDataRef& params,
                                           const StateDataRef&  states,
                                           ItemsRef&            counts);

    //! Create track views and tally particle/processes
    inline CELER_FUNCTION void operator()(ThreadId tid) const;

  private:
    const ParamsDataRef& params_;
    const StateDataRef&  states_;
    ItemsRef&            counts_;
};

void count_particle_process(
    const celeritas::ParamsHostRef&                   params,
    const celeritas::StateHostRef&                    states,
    ParticleProcessLauncher<MemSpace::host>::ItemsRef counts);

void count_particle_process(
    const celeritas::ParamsDeviceRef&                   params,
    const celeritas::StateDeviceRef&                    states,
    ParticleProcessLauncher<MemSpace::device>::ItemsRef counts);

#if !CELER_USE_DEVICE
inline void
count_particle_process(const celeritas::ParamsDeviceRef&,
                       const celeritas::StateDeviceRef&,
                       ParticleProcessLauncher<MemSpace::device>::ItemsRef)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
 * Collect the diagnostic results.
 */
template<MemSpace M>
void ParticleProcessDiagnostic<M>::get_result(TransporterResult* result)
{
    result->process = this->particle_processes();
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

//---------------------------------------------------------------------------//
} // namespace demo_loop
