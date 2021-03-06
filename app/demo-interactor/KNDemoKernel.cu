//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KNDemoKernel.cu
//---------------------------------------------------------------------------//
#include "KNDemoKernel.hh"

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "base/ArrayUtils.hh"
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "physics/base/ParticleTrackView.hh"
#include "base/StackAllocator.hh"
#include "physics/em/detail/KleinNishinaInteractor.hh"
#include "random/cuda/RngEngine.hh"
#include "physics/grid/PhysicsGridCalculator.hh"
#include "Detector.hh"
#include "KernelUtils.hh"

using namespace celeritas;
using celeritas::detail::KleinNishinaInteractor;

namespace demo_interactor
{
namespace
{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//
/*!
 * Kernel to initialize particle data.
 */
__global__ void initialize_kernel(ParamsDeviceRef const params,
                                  StateDeviceRef const  states,
                                  InitialPointers const init)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size())
    {
        return;
    }

    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    particle = init.particle;

    // Particles begin alive and in the +z direction
    states.direction[tid] = {0, 0, 1};
    states.position[tid]  = {0, 0, 0};
    states.time[tid]      = 0;
    states.alive[tid]     = true;
}

//---------------------------------------------------------------------------//
/*!
 * Sample cross sections and move to the collision point.
 */
__global__ void
move_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    // Move to collision
    PhysicsGridCalculator calc_xs(params.tables.xs, params.tables.reals);
    demo_interactor::move_to_collision(particle,
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
__global__ void
interact_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);
    unsigned int           tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Exit if out of range or already dead
    if (tid >= states.size() || !states.alive[tid])
    {
        return;
    }

    // Construct particle accessor from immutable and thread-local data
    ParticleTrackView particle(params.particle, states.particle, ThreadId(tid));
    RngEngine         rng(states.rng, ThreadId(tid));

    Detector detector(params.detector, states.detector);

    Hit h;
    h.pos    = states.position[tid];
    h.dir    = states.direction[tid];
    h.thread = ThreadId(tid);
    h.time   = states.time[tid];

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
    CELER_ASSERT(interaction);

    // Deposit energy from the secondary (effectively, an infinite energy
    // cutoff)
    {
        const auto& secondary = interaction.secondaries.front();
        h.dir                 = secondary.direction;
        h.energy_deposited    = secondary.energy;
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
__global__ void
process_hits_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    Detector        detector(params.detector, states.detector);
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
__global__ void
cleanup_kernel(ParamsDeviceRef const params, StateDeviceRef const states)
{
    unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    Detector     detector(params.detector, states.detector);
    StackAllocator<Secondary> allocate_secondaries(states.secondaries);

    if (thread_idx == 0)
    {
        allocate_secondaries.clear();
        detector.clear_buffer();
    }
}

} // namespace

//---------------------------------------------------------------------------//
// KERNEL INTERFACES
//---------------------------------------------------------------------------//
/*!
 * Initialize particle states.
 */
void initialize(const CudaGridParams&  opts,
                const ParamsDeviceRef& params,
                const StateDeviceRef&  states,
                const InitialPointers& initial)
{
    static const KernelParamCalculator calc_kernel_params(
        initialize_kernel, "initialize", opts.block_size);
    auto grid = calc_kernel_params(states.size());

    CELER_EXPECT(states.alive.size() == states.size());
    CELER_EXPECT(states.rng.size() == states.size());
    initialize_kernel<<<grid.grid_size, grid.block_size>>>(
        params, states, initial);
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Run an iteration.
 */
void iterate(const CudaGridParams&  opts,
             const ParamsDeviceRef& params,
             const StateDeviceRef&  states)
{
    //// Move to the collision site ////

    static const KernelParamCalculator calc_kernel_params(
        move_kernel, "move", opts.block_size);
    auto grid = calc_kernel_params(states.size());

    move_kernel<<<grid.grid_size, grid.block_size>>>(params, states);
    CELER_CUDA_CHECK_ERROR();

    //// Perform the interaction ////

    static const KernelParamCalculator calc_interact_params(
        interact_kernel, "interact", opts.block_size);
    grid = calc_interact_params(states.size());
    interact_kernel<<<grid.grid_size, grid.block_size>>>(params, states);
    CELER_CUDA_CHECK_ERROR();

    if (opts.sync)
    {
        // Note: the device synchronize is useful for debugging and necessary
        // for timing diagnostics.
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Clean up after an iteration.
 */
void cleanup(const CudaGridParams&  opts,
             const ParamsDeviceRef& params,
             const StateDeviceRef&  states)
{
    //// Process hits ////

    static const KernelParamCalculator calc_process_hits_params(
        process_hits_kernel, "process_hits", opts.block_size);
    auto grid = calc_process_hits_params(states.detector.capacity());
    cleanup_kernel<<<grid.grid_size, grid.block_size>>>(params, states);
    CELER_CUDA_CHECK_ERROR();

    //// Clear buffers ////

    static const KernelParamCalculator calc_cleanup_params(
        cleanup_kernel, "cleanup", 32);
    grid = calc_cleanup_params(1);
    cleanup_kernel<<<grid.grid_size, grid.block_size>>>(params, states);
    CELER_CUDA_CHECK_ERROR();

    if (opts.sync)
    {
        // Note: the device synchronize is useful for debugging and necessary
        // for timing diagnostics.
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Sum the total number of living particles.
 */
size_type reduce_alive(const CudaGridParams& grid, Span<const bool> alive)
{
    size_type result = thrust::reduce(
        thrust::device_pointer_cast(alive.data()),
        thrust::device_pointer_cast(alive.data() + alive.size()),
        size_type(0),
        thrust::plus<size_type>());
    CELER_CUDA_CHECK_ERROR();

    if (grid.sync)
    {
        CELER_CUDA_CALL(cudaDeviceSynchronize());
    }
    return result;
}

//---------------------------------------------------------------------------//
} // namespace demo_interactor
