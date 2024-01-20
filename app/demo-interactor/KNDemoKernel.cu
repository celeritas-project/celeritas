//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-interactor/KNDemoKernel.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include "corecel/Assert.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "celeritas/em/interactor/KleinNishinaInteractor.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/random/RngEngine.hh"

#include "Detector.hh"
#include "KernelUtils.hh"

namespace celeritas
{
namespace app
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Kernel to initialize particle data.
 */
__global__ void initialize_kernel(DeviceCRef<ParamsData> const params,
                                  DeviceRef<StateData> const states,
                                  InitialData const init)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size())
    {
        return;
    }

    ParticleTrackView particle(
        params.particle, states.particle, TrackSlotId(tid));
    particle = init.particle;

    // Particles begin alive and in the +z direction
    states.direction[tid] = {0, 0, 1};
    states.position[tid] = {0, 0, 0};
    states.time[tid] = 0;
    states.alive[tid] = true;
}

//---------------------------------------------------------------------------//
/*!
 * Sample cross sections and move to the collision point.
 */
__global__ void move_kernel(DeviceCRef<ParamsData> const params,
                            DeviceRef<StateData> const states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(
        params.particle, states.particle, TrackSlotId(tid));
    RngEngine rng(params.rng, states.rng, TrackSlotId(tid));

    // Move to collision
    XsCalculator calc_xs(params.tables.xs, params.tables.reals);
    celeritas::app::move_to_collision(particle,
                                      calc_xs,
                                      states.direction[tid],
                                      &states.position[tid],
                                      &states.time[tid],
                                      rng);
}

//---------------------------------------------------------------------------//
/*!
 * Perform the iteraction plus cleanup.
 *
 * The interaction:
 * - Allocates and emits a secondary
 * - Kills the secondary, depositing its local energy
 * - Applies the interaction (updating track direction and energy)
 */
__global__ void interact_kernel(DeviceCRef<ParamsData> const params,
                                DeviceRef<StateData> const states)
{
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(
        params.particle, states.particle, TrackSlotId(tid));
    RngEngine rng(params.rng, states.rng, TrackSlotId(tid));

    Detector detector(params.detector, states.detector);

    Hit h;
    h.pos = states.position[tid];
    h.dir = states.direction[tid];
    h.track_slot = TrackSlotId(tid);
    h.time = states.time[tid];

    if (particle.energy() < units::MevEnergy{0.01})
    {
        // Particle is below interaction energy
        h.energy_deposited = particle.energy();

        // Deposit energy and kill
        detector.buffer_hit(h);
        states.alive[tid] = false;
        return;
    }

    // Construct RNG and interaction interfaces
    KleinNishinaInteractor interact(
        params.kn_interactor, particle, h.dir, allocate_secondaries);

    // Perform interaction: should emit a single particle (an electron)
    Interaction interaction = interact(rng);
    CELER_ASSERT(interaction.action != Interaction::Action::failed);

    // Deposit energy from the secondary (effectively, an infinite energy
    // cutoff)
    {
        auto const& secondary = interaction.secondaries.front();
        h.dir = secondary.direction;
        h.energy_deposited = units::MevEnergy{
            secondary.energy.value() + interaction.energy_deposition.value()};
        detector.buffer_hit(h);
    }

    // Update post-interaction state (apply interaction)
    states.direction[tid] = interaction.direction;
    particle.energy(interaction.energy);
}

//---------------------------------------------------------------------------//
/*!
 * Bin detector hits.
 */
__global__ void process_hits_kernel(DeviceCRef<ParamsData> const params,
                                    DeviceRef<StateData> const states)
{
    Detector detector(params.detector, states.detector);
    Detector::HitId hid{blockIdx.x * blockDim.x + threadIdx.x};

    if (hid < detector.num_hits())
    {
        detector.process_hit(hid);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clear secondaries and detector hits.
 */
__global__ void cleanup_kernel(DeviceCRef<ParamsData> const params,
                               DeviceRef<StateData> const states)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    Detector detector(params.detector, states.detector);
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);

    if (thread_idx == 0)
    {
        allocate_secondaries.clear();
        detector.clear_buffer();
    }
}

}  // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Initialize particle states.
 */
void initialize(DeviceGridParams const& opts,
                DeviceCRef<ParamsData> const& params,
                DeviceRef<StateData> const& states,
                InitialData const& initial)
{
    CELER_EXPECT(states.alive.size() == states.size());
    CELER_EXPECT(states.rng.size() == states.size());
    CELER_LAUNCH_KERNEL(initialize, states.size(), 0, params, states, initial);
}

//---------------------------------------------------------------------------//
/*!
 * Run an iteration.
 */
void iterate(DeviceGridParams const& opts,
             DeviceCRef<ParamsData> const& params,
             DeviceRef<StateData> const& states)
{
    // Move to the collision site
    CELER_LAUNCH_KERNEL(move, states.size(), 0, params, states);

    // Perform the interaction
    CELER_LAUNCH_KERNEL(interact, states.size(), 0, params, states);

    if (opts.sync)
    {
        // Synchronize for granular kernel timing diagnostics
        CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clean up after an iteration.
 */
void cleanup(DeviceGridParams const& opts,
             DeviceCRef<ParamsData> const& params,
             DeviceRef<StateData> const& states)
{
    // Process hits from buffer to grid
    CELER_LAUNCH_KERNEL(
        process_hits, states.detector.capacity(), 0, params, states);

    // Clear buffers
    CELER_LAUNCH_KERNEL(cleanup, 1, 0, params, states);

    if (opts.sync)
    {
        CELER_DEVICE_CALL_PREFIX(DeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
