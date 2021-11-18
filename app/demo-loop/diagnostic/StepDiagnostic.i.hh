//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StepDiagnostic.i.hh
//---------------------------------------------------------------------------//

#include <algorithm>
#include "base/CollectionBuilder.hh"
#include "physics/base/PhysicsTrackView.hh"
#include "sim/SimTrackView.hh"

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Construct from shared data.
 */
template<MemSpace M>
StepDiagnostic<M>::StepDiagnostic(const ParamsDataRef& params,
                                  SPConstParticles     particles,
                                  size_type            num_tracks,
                                  size_type            max_steps)
    : params_(params), particles_(particles)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(particles_);
    CELER_EXPECT(num_tracks > 0);
    CELER_EXPECT(max_steps > 0);

    StepDiagnosticData<Ownership::value, MemSpace::host> host_data;
    host_data.max_steps     = max_steps;
    host_data.num_particles = particles_->size();

    // Initialize current number of steps in active tracks to zero
    std::vector<size_type> zeros(num_tracks);
    celeritas::make_builder(&host_data.steps)
        .insert_back(zeros.begin(), zeros.end());

    // Tracks binned by number of steps and particle type (indexed as
    // particle_id * max_steps + num_steps). The final bin is for overflow.
    zeros.resize((host_data.max_steps + 2) * host_data.num_particles);
    celeritas::make_builder(&host_data.counts)
        .insert_back(zeros.begin(), zeros.end());

    data_ = host_data;
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the distribution of steps per track.
 *
 * This must be called after the interactors have been launched but before the
 * post-processing.
 */
template<MemSpace M>
void StepDiagnostic<M>::mid_step(const StateDataRef& states)
{
    StepDiagnosticDataRef<M> data_ref;
    data_ref = data_;
    count_steps(params_, states, data_ref);
}

//---------------------------------------------------------------------------//
/*!
 * Get distribution of steps per track for each particle type.
 *
 * For i in [0, \c max_steps + 1], steps[particle][i] is the number of tracks
 * of the given particle type that took i steps. The final bin stores the
 * number of tracks that took greater than \c max_steps steps.
 */
template<MemSpace M>
std::unordered_map<std::string, std::vector<celeritas::size_type>>
StepDiagnostic<M>::steps()
{
    using BinId = celeritas::ItemId<size_type>;

    // Copy result to host if necessary
    StepDiagnosticData<Ownership::value, MemSpace::host> data;
    data = data_;

    // Map particle ID to particle name and store steps per track distribution
    std::unordered_map<std::string, std::vector<size_type>> result;
    for (auto particle_id : range(celeritas::ParticleId{particles_->size()}))
    {
        auto start = BinId{particle_id.get() * data.max_steps};
        auto end   = BinId{start.get() + data.max_steps + 2};
        CELER_ASSERT(end.get() <= data.counts.size());
        auto counts = data.counts[celeritas::ItemRange<size_type>{start, end}];

        // Export non-trivial particle's counts
        if (std::any_of(counts.begin(), counts.end(), [](size_type x) {
                return x > 0;
            }))
        {
            result[particles_->id_to_label(particle_id)]
                = {counts.begin(), counts.end()};
        }
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
template<MemSpace M>
CELER_FUNCTION StepLauncher<M>::StepLauncher(const ParamsDataRef&     params,
                                             const StateDataRef&      states,
                                             StepDiagnosticDataRef<M> data)
    : params_(params), states_(states), data_(data)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(states_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Increment step count and tally number of steps for tracks that were killed.
 *
 * At this point, active tracks will either be marked "alive" in the sim state
 * or will have a killing action in their interaction result. Tracks that were
 * killed in a discrete interaction will still be marked as alive (these are
 * killed in post-processing), but their action will be killing.
 */
template<MemSpace M>
CELER_FUNCTION void StepLauncher<M>::operator()(ThreadId tid) const
{
    using BinId = celeritas::ItemId<size_type>;

    celeritas::ParticleTrackView particle(
        params_.particles, states_.particles, tid);
    celeritas::SimTrackView sim(states_.sim, tid);

    const auto& interaction = states_.interactions[tid];

    // Increment the step count if this is an active track
    if (sim.alive() || celeritas::action_killed(interaction.action))
    {
        ++data_.steps[tid];
    }

    // Tally the number of steps if the track was killed
    if (celeritas::action_killed(interaction.action))
    {
        // TODO: Add an ndarray-type class?
        auto get = [this](size_type i, size_type j) -> size_type& {
            size_type index = i * data_.max_steps + j;
            CELER_ENSURE(index < data_.counts.size());
            return data_.counts[BinId(index)];
        };

        size_type num_steps = data_.steps[tid] <= data_.max_steps
                                  ? data_.steps[tid]
                                  : data_.max_steps + 1;

        // Increment the bin corresponding to the given particle and step count
        auto& bin = get(particle.particle_id().get(), num_steps);
        celeritas::atomic_add(&bin, 1u);

        // Reset the track's step counter
        data_.steps[tid] = 0;
    }
}
} // namespace demo_loop
