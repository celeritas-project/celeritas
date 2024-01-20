//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/HostKNDemoRunner.cc
//---------------------------------------------------------------------------//
#include "HostKNDemoRunner.hh"

#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/data/StackAllocatorData.hh"
#include "corecel/grid/UniformGridData.hh"
#include "corecel/sys/Stopwatch.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/ParticleView.hh"
#include "celeritas/phys/Secondary.hh"

#include "Detector.hh"
#include "DetectorData.hh"
#include "KNDemoKernel.hh"
#include "KernelUtils.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters.
 */
HostKNDemoRunner::HostKNDemoRunner(constSPParticleParams particles,
                                   constSPXsGridParams xs)
    : pparams_(std::move(particles)), xsparams_(std::move(xs))
{
    CELER_EXPECT(pparams_);
    CELER_EXPECT(xsparams_);

    // Set up KN interactor data;
    kn_data_.ids.action = ActionId{0};  // Unused but needed for error check
    kn_data_.ids.electron = pparams_->find(pdg::electron());
    kn_data_.ids.gamma = pparams_->find(pdg::gamma());
    kn_data_.inv_electron_mass
        = 1 / pparams_->get(kn_data_.ids.electron).mass().value();
    CELER_ENSURE(kn_data_);
}

//---------------------------------------------------------------------------//
/*!
 * Run given number of particles each for max steps.
 */
auto HostKNDemoRunner::operator()(celeritas::app::KNDemoRunArgs args)
    -> result_type
{
    CELER_EXPECT(args.energy > 0);
    CELER_EXPECT(args.num_tracks > 0);

    // Initialize results
    result_type result;
    result.time.reserve(args.max_steps);
    result.alive.resize(args.max_steps + 1);
    result.edep.reserve(args.max_steps);

    // Start timer for overall execution and transport-only time
    Stopwatch total_time;
    double transport_time = 0.0;

    // Random number generations
    std::mt19937 rng(args.seed);

    // Particle data
    HostVal<ParticleStateData> track_states;
    resize(&track_states, pparams_->host_ref(), 1);

    // Make secondary store
    StackAllocatorData<Secondary, Ownership::value, MemSpace::host> secondaries;
    resize(&secondaries, args.max_steps);

    // Detector data
    DetectorParamsData detector_params;
    detector_params.tally_grid = args.tally_grid;
    HostVal<DetectorStateData> detector_states;
    resize(&detector_states, detector_params, args.max_steps);

    // Construct references
    HostCRef<ParamsData> params;
    params.particle = pparams_->host_ref();
    params.tables = xsparams_->host_ref();
    params.kn_interactor = kn_data_;
    params.detector = detector_params;

    // Construct initialization
    InitialData initial;
    initial.particle = ParticleTrackInitializer{kn_data_.ids.gamma,
                                                units::MevEnergy(args.energy)};

    HostRef<StateData> state;
    state.particle = track_states;
    state.secondaries = secondaries;
    state.detector = detector_states;

    // Loop over particle tracks and events per track
    for ([[maybe_unused]] auto n : range(args.num_tracks))
    {
        // Storage for track state
        Real3 position = {0, 0, 0};
        Real3 direction = {0, 0, 1};
        real_type time = 0;
        bool alive = true;

        // Create and initialize particle view
        ParticleTrackView particle(
            params.particle, state.particle, TrackSlotId{0});

        // Create helper classes
        StackAllocator<Secondary> allocate_secondaries(state.secondaries);
        Detector detector(params.detector, state.detector);
        XsCalculator calc_xs(params.tables.xs, params.tables.reals);

        CELER_ASSERT(state.secondaries.capacity() == args.max_steps);
        CELER_ASSERT(state.detector.hit_buffer.capacity() == args.max_steps);
        CELER_ASSERT(allocate_secondaries.get().size() == 0);
        CELER_ASSERT(detector.num_hits() == 0);

        // Counters
        size_type num_steps = 0;
        auto remaining_steps = args.max_steps;
        Stopwatch elapsed_time;

        particle = initial.particle;

        while (alive && --remaining_steps > 0)
        {
            // Increment alive counter
            CELER_ASSERT(num_steps < result.alive.size());
            result.alive[num_steps]++;
            ++num_steps;

            // Move to collision
            celeritas::app::move_to_collision(
                particle, calc_xs, direction, &position, &time, rng);

            // Hit analysis
            Hit h;
            h.pos = position;
            h.dir = direction;
            h.track_slot = TrackSlotId{0};
            h.time = time;

            // Check for below energy cutoff
            if (particle.energy() < units::MevEnergy{0.01})
            {
                // Particle is below interaction energy
                h.energy_deposited = particle.energy();

                // Deposit energy and kill
                detector.buffer_hit(h);
                alive = false;
                continue;
            }

            // Construct the KN interactor
            KleinNishinaInteractor interact(
                kn_data_, particle, direction, allocate_secondaries);

            // Perform interactions - emits a single particle
            Interaction interaction = interact(rng);
            CELER_ASSERT(interaction.action != Interaction::Action::failed);
            CELER_ASSERT(interaction.secondaries.size() == 1);

            // Deposit energy from the secondary (all local)
            {
                auto const& secondary = interaction.secondaries.front();
                h.dir = secondary.direction;
                h.energy_deposited = units::MevEnergy{
                    secondary.energy.value()
                    + interaction.energy_deposition.value()};
                detector.buffer_hit(h);
            }

            // Update the energy and direction in the state from the
            // interaction
            direction = interaction.direction;
            particle.energy(interaction.energy);
        }
        CELER_ASSERT(num_steps < args.max_steps
                         ? allocate_secondaries.get().size() == num_steps - 1
                         : allocate_secondaries.get().size() == num_steps);
        CELER_ASSERT(detector.num_hits() == num_steps);

        // Store transport time
        transport_time += elapsed_time();

        // Clear secondaries
        allocate_secondaries.clear();
        CELER_ASSERT(allocate_secondaries.get().size() == 0);

        // Bin the tally results from the buffer onto the grid
        for (auto hit_id : range(Detector::HitId{detector.num_hits()}))
        {
            detector.process_hit(hit_id);
        }
        detector.clear_buffer();
    }

    // Copy integrated energy deposition
    std::vector<real_type> edep(detector_params.tally_grid.size);
    celeritas::app::finalize(params, state, make_span(edep));
    result.edep.assign(edep.begin(), edep.end());

    // Store timings
    result.time.push_back(transport_time);
    result.total_time = total_time();

    // Remove trailing zeros from preallocated "alive" size
    while (!result.alive.empty() && result.alive.back() == 0)
    {
        result.alive.pop_back();
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
